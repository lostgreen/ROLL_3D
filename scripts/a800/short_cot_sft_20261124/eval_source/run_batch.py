"""Native v2 closed-loop inference. Private GT is never read by this runner."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, socket, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

def read(p): return json.loads(Path(p).read_text())
def dump(p, obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def get(url):
    with urlopen(url,timeout=10) as r: return json.load(r)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--variant',choices=['base','short_cot_step303'],required=True)
    ap.add_argument('--port',type=int,required=True);ap.add_argument('--ids',default='')
    ap.add_argument('--workers',type=int);ap.add_argument('--probe',action='store_true')
    a=ap.parse_args();root=a.root.resolve();cfg=read(root/'config.json')
    os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ['RAW_COT_CONTEXT_RUNTIME']=cfg['runtime_root']
    sys.path.insert(0,cfg['train_source'])
    from gpu_probe import bootstrap
    bootstrap(Path(cfg['repo']))
    cpu_vendor=root/'cpu_vendor'
    sys.path.insert(0,str(cpu_vendor))
    os.environ['PYTHONPATH']=str(cpu_vendor)+os.pathsep+os.environ.get('PYTHONPATH','')
    from pack_raw_cot import setup
    processor=setup(cfg['runtime_root'],expected_version='raw_cot_context_v2')
    from v2s_integration.envs.reconstruction import ReconstructionEnv,VENDOR
    from v2s_integration.agent.messages import render_request,record_request
    import torch
    from mcp_client import BlenderMCPClient

    class EvaluationEnv(ReconstructionEnv):
        def _start(self):
            process=Path(cfg['cpu_runtime'])/'headless/blender_process.py'
            spec=importlib.util.spec_from_file_location('_eval_cpu_blender',process)
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
            self.blender=mod.HeadlessBlender(port=port,log_path=self.episode/'blender.log',blender_bin=self.task['blender_bin']).start()
            self.mcp=BlenderMCPClient(command=sys.executable,
                args=['-c','from blender_mcp.server import main; main()'],cwd=str(VENDOR/'blender-mcp'),
                env={'BLENDER_PORT':str(port),'PYTHONPATH':os.pathsep.join([str(cpu_vendor),str(VENDOR/'blender-mcp/src'),os.environ.get('PYTHONPATH','')]),'PYTHONDONTWRITEBYTECODE':'1'})
            self.mcp.start()
            self._code('import bpy\nbpy.ops.wm.open_mainfile(filepath='+repr(self.task['initial_blend'])+',use_scripts=False)\n')

        def finish_episode(self,reason):
            # Native termination normally closes Blender immediately. Retain the
            # owned session until the caller has saved the final state/audit.
            try:
                self.save_scene(self.episode/'scene.blend')
            except Exception as exc:
                dump(self.episode/'save_error.json',dict(type=type(exc).__name__,message=str(exc)[:400]))
                reason='environment_error'
            self.recorder.event('episode.end',reason=reason,steps=self.step_count)
            dump(self.episode/'result.json',dict(task_id=self.task['id'],steps=self.step_count,termination_reason=reason,quality_evaluated=False))
            return reason

        def save_scene(self,path):
            self._code('import bpy\nbpy.context.preferences.filepaths.save_version=0\nbpy.ops.wm.save_as_mainfile(filepath='+repr(str(path))+',copy=True)\n')

    base=f'http://127.0.0.1:{a.port}'
    rows=cfg['rows']
    if a.ids:rows=[r for r in rows if r['id'] in set(a.ids.split(','))]
    # A known training case verifies engineering without consuming extra test cases.
    rows=sorted(rows,key=lambda r:(r['subset']!='train10_diagnostic',r['id']))
    batch=root/('probe' if a.probe else 'runs')/a.variant
    batch.mkdir(parents=True,exist_ok=True)
    if a.probe:raise ValueError('Use a separate run-root for probes; official request_root is immutable')
    batch_status=batch/('batch_status_'+(hashlib.sha256(a.ids.encode()).hexdigest()[:8] if a.ids else 'all')+'.json')
    dump(batch_status,dict(state='waiting_for_model',variant=a.variant,total=len(rows),pid=os.getpid(),started_at=time.time()))
    deadline=time.monotonic()+3600
    while True:
        try:
            health=get(base+'/health')
            if health.get('ready'):break
            if health.get('stage') in {'failed','load_failed'}:raise RuntimeError('MODEL_LOAD_FAILED')
        except (OSError,ValueError):pass
        if time.monotonic()>deadline:raise TimeoutError('Model did not become ready')
        time.sleep(10)

    if len(rows)>1 and all(r['id']!='eval_021' for r in rows):
        # Queue the complete batch now, and release it only after the same
        # variant has exercised native parsing/tool feedback and a next prefix.
        dump(batch_status,dict(state='waiting_for_startup_check',variant=a.variant,total=len(rows),pid=os.getpid(),started_at=time.time()))
        startup=batch/'eval_021';deadline=time.monotonic()+10800
        while True:
            first=read(startup/'status.json') if (startup/'status.json').exists() else {}
            metrics=read(startup/'metrics.json') if (startup/'metrics.json').exists() else []
            valid=any(x.get('tool') and x.get('error_type')!='invalid_action' for x in metrics)
            requests=sorted(startup.glob('decision_*/context_audit.json'))
            next_prefix=any(read(p).get('history_turns',0)>=1 and (p.parent/'input_ids.pt').exists() for p in requests)
            finished=first.get('state')=='complete' and first.get('termination_reason')=='finish'
            if valid and (next_prefix or finished):
                dump(batch/'startup_acceptance.json',dict(passed=True,variant=a.variant,first_scene='eval_021',
                    actual_native_call=True,next_visual_prefix_materialized=next_prefix,first_finished=finished,at=time.time()))
                break
            if first.get('state') in {'failed','complete'}:
                dump(batch_status,dict(state='startup_check_failed',variant=a.variant,total=len(rows),first=first,at=time.time()))
                return
            if time.monotonic()>deadline:
                dump(batch_status,dict(state='startup_check_timeout',variant=a.variant,total=len(rows),at=time.time()))
                return
            time.sleep(15)

    def episode(row):
        sid=row['id'];out=batch/sid;out.mkdir(exist_ok=True)
        old=out/'status.json'
        if old.exists():
            previous=read(old)
            if previous.get('state') in {'complete','failed','context_budget_exceeded'}:return previous
            raise RuntimeError('Refusing to overwrite interrupted trajectory '+sid)
        env=None;t0=time.time();step=0;metrics=[]
        state=dict(state='waiting_for_task',variant=a.variant,id=sid,subset=row['subset'],started_at=t0,step=0)
        dump(old,state)
        try:
            task=root/'public'/sid/'task.json'
            while not task.exists():
                prep=read(root/'preparation_status.json') if (root/'preparation_status.json').exists() else {}
                if prep.get('state') in {'failed','partial_failure'}:raise RuntimeError('TASK_PREPARATION_FAILED:'+sid)
                if time.time()-t0>7200:raise TimeoutError('Task preparation deadline')
                time.sleep(10)
            state.update(state='initializing');dump(old,state)
            env=EvaluationEnv(task,out/'environment',max_steps=cfg['max_steps'],
                max_output_tokens=cfg['max_new_tokens'],context_profile=cfg['context_profile'])
            seed=cfg['seed']+int(sid.rsplit('_',1)[1]);env.reset(seed=seed)
            assert env.allow_camera_light_edits
            assert {t['function']['name'] for t in env.model_tools}=={'execute_blender_code','reference.crop','scene.list_objects','render_scene_view','finish'}
            binding=dict(variant=a.variant,task_sha256=sha(task),initial_blend_sha256=sha(env.task['initial_blend']),
                references_sha256=[sha(r['image']) for r in env.task['references']],context_profile_sha256=env.context_profile_sha256,
                selection_sha256=cfg['selection_sha256'],system_sha256=hashlib.sha256(env.system_prompt.encode()).hexdigest(),seed=seed,
                max_steps=cfg['max_steps'],max_new_tokens=cfg['max_new_tokens'],max_input_tokens=cfg['max_input_tokens'],
                backend=cfg['backend'],private_gt_read=False)
            dump(out/'binding.json',binding)
            while env.step_count<cfg['max_steps']:
                step=env.step_count+1;req=out/f'decision_{step:03d}';req.mkdir()
                prompt,messages,pictures,refs,ctx=render_request(processor.tokenizer,env)
                for marker in ('gt.blend','hidden_0','hidden_1','3D-FUTURE-model','source_house_id','source_layout_hash','category_counts'):
                    if marker in prompt:raise ValueError('PRIVATE_METADATA_IN_MODEL_INPUT:'+marker)
                (req/'prompt.txt').write_text(prompt)
                images=[]
                cache=out/'image_cache';cache.mkdir(exist_ok=True)
                for i,img in enumerate(pictures):
                    digest=hashlib.sha256(str((img.mode,img.size)).encode()+img.tobytes()).hexdigest()
                    saved=cache/(digest+'.png')
                    if not saved.exists():img.save(saved)
                    path=req/f'image_{i:02d}.png';os.link(saved,path);images.append(str(path))
                dump(req/'images.json',images)
                dump(req/'context_audit.json',dict(state=ctx,images=refs,history_turns=len(env.turns)))
                dump(req/'messages.json',messages);dump(req/'tool_schemas.json',env.model_tools)
                state.update(state='generating',step=step,request_dir=str(req),updated_at=time.time());dump(old,state)
                payload=json.dumps(dict(request_dir=str(req),seed=seed+step*100000)).encode()
                request=Request(base+'/generate',data=payload,headers={'Content-Type':'application/json'},method='POST')
                with urlopen(request,timeout=14400) as response:reply=json.load(response)
                result=read(req/'output.json')
                if result.get('error') or not result.get('passed',False):raise RuntimeError('MODEL_GENERATION_ERROR:'+str(result.get('error',reply))[:300])
                input_ids=torch.load(req/'input_ids.pt',map_location='cpu',weights_only=True)
                if hasattr(input_ids,'tolist'):input_ids=input_ids.tolist()
                if input_ids and isinstance(input_ids[0],list):input_ids=input_ids[0]
                record_request(req,prompt,messages,env.model_tools,refs,input_ids,input_ids,ctx)
                raw=(req/'generated.txt').read_text()
                # Prevent any explicit attempt to import private evaluation data.
                # This is a visible-code audit, not a filesystem sandbox claim.
                from v2s_integration.agent.actions import parse_action
                try:call=parse_action(raw.strip().removesuffix('<|im_end|>').removesuffix('<|endoftext|>').strip(),step,env.session.registry)
                except ValueError:call=None
                if call and call.name=='execute_blender_code':
                    code=call.arguments.get('code','')
                    if any(s in code for s in ('gt.blend','hidden_0','hidden_1','3D-FUTURE-model','ground_truth/','klink.key')):
                        raise ValueError('PRIVATE_FILESYSTEM_ACCESS_ATTEMPT')
                state.update(state='executing_tool',updated_at=time.time());dump(old,state)
                _,_,done,truncated,info=env.step(raw)
                turn=env.turns[-1]
                metrics.append(dict(step=step,tool=turn['feedback']['tool_name'],tool_status=turn['feedback']['status'],
                    error_type=turn['feedback']['error_type'],generation={k:result.get(k) for k in ('input_tokens','generated_tokens','generation_seconds','hit_output_limit')}))
                dump(out/'metrics.json',metrics)
                state.update(step=step,last_tool=metrics[-1]['tool'],last_error=metrics[-1]['error_type'],updated_at=time.time());dump(old,state)
                if done or truncated:break
            if not (env.episode/'scene.blend').exists():env.save_scene(env.episode/'scene.blend')
            dump(out/'final_public_state.json',env.public_state)
            state.update(state='complete',termination_reason=info['termination_reason'],steps=env.step_count,
                scene_blend=str(env.episode/'scene.blend'),episode=str(env.episode),finished_at=time.time(),elapsed_seconds=time.time()-t0,
                invalid_actions=sum(x['error_type']=='invalid_action' for x in metrics),tool_failures=sum(bool(x['error_type']) for x in metrics),quality_evaluated=False)
            dump(old,state);return state
        except BaseException as exc:
            state.update(state='failed',step=step,error_type=type(exc).__name__,message=str(exc)[:500],finished_at=time.time(),elapsed_seconds=time.time()-t0)
            dump(old,state)
            with (out/'exception.log').open('w') as f:traceback.print_exc(file=f)
            if env and not env.closed:
                try:env.save_scene(out/'partial.blend')
                except Exception:pass
            return state
        finally:
            if env:env.close()

    with ThreadPoolExecutor(max_workers=a.workers or cfg['concurrency_per_variant']) as pool:
        futures=[pool.submit(episode,row) for row in rows]
        for future in as_completed(futures):
            r=future.result()
            states=[read(batch/x['id']/'status.json') for x in rows if (batch/x['id']/'status.json').exists()]
            dump(batch_status,dict(state='running',variant=a.variant,total=len(rows),
                complete=sum(s['state']=='complete' for s in states),failed=sum(s['state']=='failed' for s in states),
                active=sum(s['state'] not in {'complete','failed'} for s in states),updated_at=time.time(),last_id=r['id']))
    states=[read(batch/r['id']/'status.json') for r in rows]
    dump(batch_status,dict(state='complete',variant=a.variant,total=len(rows),
        complete=sum(s['state']=='complete' for s in states),failed=sum(s['state']=='failed' for s in states),finished_at=time.time()))

if __name__=='__main__':main()
