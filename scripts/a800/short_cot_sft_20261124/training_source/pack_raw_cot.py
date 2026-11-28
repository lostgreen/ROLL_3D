"""Pack immutable trajectories through the same context builder as the harness.

No API, Blender or GPU work. Large records stay compressed on shared storage.
"""
from __future__ import annotations
import argparse
import base64
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

REPO = Path('/m2v_intern/xuboshen/zgw/ROLL')
MODEL = Path('/m2v_intern/xuboshen/models/Qwen3.5-9B')
ROOT = REPO/'artifacts/raw_cot_sft_preflight_20261121'

def dump(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def read(path):
    path=Path(path)
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:return json.load(f)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def configure_context_runtime(runtime=None, expected_version=None):
    # Prefer the uploaded source snapshot; do not modify the shared remote repo.
    overlay=Path(runtime or os.environ.get('RAW_COT_CONTEXT_RUNTIME', str(ROOT/'runtime')))
    names=['context_policy.py','messages.py','actions.py']
    if expected_version=='raw_cot_context_v2':
        names.append('observation_format.py')
    for name in names:
        expected=(overlay/'v2s_integration/agent'/name).resolve()
        assert expected.is_file(), 'Missing context runtime source: '+str(expected)
        cached=sys.modules.get('v2s_integration.agent.'+name[:-3])
        if expected_version=='raw_cot_context_v2' and cached is not None:
            assert Path(cached.__file__).resolve()==expected, 'Context module cache belongs to another runtime: '+name
    import v2s_integration
    v2s_integration.__path__.insert(0,str(overlay/'v2s_integration'))
    import v2s_integration.agent
    v2s_integration.agent.__path__.insert(0,str(overlay/'v2s_integration/agent'))
    import v2s_integration.envs
    v2s_integration.envs.__path__.insert(0,str(overlay/'v2s_integration/envs'))
    return overlay

def setup(runtime=None, expected_version=None):
    configure_context_runtime(runtime, expected_version)
    import torch
    torch.set_num_threads(2)
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(str(MODEL),local_files_only=True)

def record_sources(scope):
    root=REPO/'artifacts/indoor_qwen_200rollout_20261121'
    cfg=read(root/'config.json');pilot=Path(cfg['pilot_root'])
    if scope=='pilot':return [(p.parent.name,p.parent,'pilot') for p in sorted((pilot/'runs').glob('scene_*/status.json'))]
    gallery=read(root/'gallery_export/gallery_data.json')
    assert len(gallery['cases'])==200 and gallery['merged_retry']
    selected=None
    if scope=='review':selected={x['id'] for x in read(REPO/'artifacts/raw_cot_baseline_review_20261121/sample_manifest.json')['samples']}
    result=[]
    for case in gallery['cases']:
        sid=case['id']
        if selected is not None and sid not in selected:continue
        if case.get('attempt')=='retry17_v1':source=root/'retry17_v1';tag='retry17_v1'
        elif sid in cfg['reused_ids']:source=pilot;tag='pilot_reused'
        else:source=root;tag='original'
        result.append((sid,source/'runs'/sid,tag))
    return result

def initial_context(run, cache):
    status=read(run/'status.json')
    original_episodes=sorted((run/'environment').glob('blender_*'))
    assert len(original_episodes)==1
    ep=original_episodes[0]
    assert status['status']=='complete'
    attempts=sorted((run/'decision_001').glob('attempt_*/assistant.json'))
    req=read(attempts[-1].parent/'request.json.gz')
    from v2s_integration.agent.api_context import CONTEXT_RULES
    system=req['messages'][0]['content']
    assert system.endswith('\n'+CONTEXT_RULES),'Unknown teacher context suffix'
    system=system[:-(len(CONTEXT_RULES)+1)]
    # API aliases map to the true tools, not new tool names.
    aliases={'reference_crop':'reference.crop','scene_list_objects':'scene.list_objects'}
    tools=copy.deepcopy(req['tools'])
    for t in tools:t['function']['name']=aliases.get(t['function']['name'],t['function']['name'])
    refs=read(ep/'manifest.json')['references']
    assert len(refs)==2
    prompt_parts=req['messages'][1]['content']
    task=next(p['text'] for p in prompt_parts if p['type']=='text')
    snapshot_parts=req['messages'][2]['content']
    snap=json.loads(next(p['text'] for p in snapshot_parts if p['type']=='text'))
    image_paths={sha(p):p for p in (ep/'images').glob('*.png')}
    initial=[]
    for part in snapshot_parts:
        if part['type']!='image_url':continue
        data=base64.b64decode(part['image_url']['url'].split(',',1)[1]);h=hashlib.sha256(data).hexdigest()
        path=image_paths.get(h)
        if path is None:
            path=cache/(h+'.png');path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():path.write_bytes(data)
        initial.append({'path':str(path),'label':'Initial scene S0','sha256':h,'generated_step':0,'scene_version':0})
    # Automatic views can change when teacher edits s.camera. Without persisted
    # reliable camera metadata, use the documented latest-two fallback.
    env=SimpleNamespace(system_prompt=system,task_prompt=task,reference_refs=refs,initial_images=initial,
        public_state=snap['public_scene_state'],turns=[],model_tools=tools,task={},
        initial_scene_description='Initial scene S0 contains only cameras, lights and a neutral world; no room or furniture geometry.')
    profile={'policy_version':'raw_cot_context_v1','system_prompt':system,'tools':tools,
             'initial_scene_description':env.initial_scene_description,'task_prompt_mode':'public_instruction_camera_v1',
             'allow_camera_light_edits':True,'render_mode':'active_camera_640x480_cycles_v1'}
    from v2s_integration.agent.context_policy import load_context_profile
    env.context_profile=load_context_profile(profile)
    return env,status,ep,aliases

def event_source(run,status,step):
    current=Path(status['episode'])/f'step_{step:03d}/event.json'
    if current.exists():return current
    original=list((run/'environment').glob(f'blender_*/step_{step:03d}/event.json'))
    assert len(original)==1,('Missing or ambiguous pre-recovery event',step)
    return original[0]

RESERVED=re.compile(r'<(?:/?(?:tool_call|function|parameter|think)(?:[=>\s])|\|[^>]+\|>)')
def escape_body(text):
    # Reversible lexical escaping only for reserved protocol tokens in content.
    replacements=[]
    def sub(m):
        s=m.group(0);escaped=s.replace('<','&lt;').replace('>','&gt;')
        replacements.append({'offset':m.start(),'original':s,'escaped':escaped});return escaped
    return RESERVED.sub(sub,text),replacements

class Registry:
    def __init__(self,tools):self.schemas={t['function']['name']:t['function']['parameters'] for t in tools}
    def lookup(self,name):
        return (SimpleNamespace(input_schema=self.schemas[name]),None) if name in self.schemas else None

def target_message(assistant,event,aliases,tools,step):
    calls=assistant.get('tool_calls') or [];assert len(calls)==1
    call=event['call'];assert call is not None
    assert event['step']==step
    name=calls[0]['function']['name']
    assert aliases.get(name,name)==call['name'],'Native/executed tool names differ'
    assert call['name'] in {t['function']['name'] for t in tools}
    assert event['feedback']['call_id']==call['id'],'Execution/feedback call binding differs'
    args=calls[0]['function']['arguments'];args=json.loads(args) if isinstance(args,str) else args
    assert args==call['arguments'],'Native/executed arguments differ'
    content,changes=escape_body(assistant.get('content') or '')
    cot=assistant.get('reasoning_content') or assistant.get('reasoning') or ''
    assert isinstance(cot,str) and cot
    target={'role':'assistant','reasoning_content':cot,'content':content,'tool_calls':[{'id':call['id'],'type':'function','function':{'name':call['name'],'arguments':copy.deepcopy(args)}}]}
    return target,changes

def tensor_features(processor,record,verify=True):
    import torch
    from PIL import Image
    messages=record['messages'];tools=record['tools'];target=record['target']
    prompt=processor.apply_chat_template(messages,tools=tools,tokenize=False,add_generation_prompt=True,enable_thinking=True)
    full=processor.apply_chat_template(messages+[target],tools=tools,tokenize=False,add_generation_prompt=False,enable_thinking=True)
    assert full.startswith(prompt),'Official template is not a generation-prefix extension'
    images=[]
    for ref in record['images']:
        with Image.open(ref['path']) as image:images.append(image.convert('RGB'))
    encoded=processor(text=[full],images=images,return_tensors='pt')
    prefix=processor(text=[prompt],images=images,return_tensors='pt')
    n=prefix['input_ids'].shape[1];ids=encoded['input_ids'];length=ids.shape[1]
    assert torch.equal(ids[:,:n],prefix['input_ids']),'Tokenized train prefix differs from inference prefix'
    assert torch.equal(encoded['image_grid_thw'],prefix['image_grid_thw'])
    assert torch.equal(encoded['pixel_values'],prefix['pixel_values'])
    suffix=full[len(prompt):]
    tail=processor.tokenizer(suffix,add_special_tokens=False,return_offsets_mapping=True)
    assert tail['input_ids']==ids[0,n:].tolist(),'Target boundary changes tokenization'
    labels=ids.clone();labels[:,:n]=-100
    from v2s_integration.agent.actions import action_text,parse_action
    response=suffix.removesuffix('<|im_end|>\n').removesuffix('<|im_end|>').strip()
    envelope=action_text(response)
    action_start=suffix.rfind(envelope);assert action_start>=0
    cot_end=suffix.rfind('</think>',0,action_start);assert cot_end>=0,'Current raw reasoning missing from official template'
    masks={k:torch.full_like(ids,-100) for k in ['cot','body','action']}
    for i,(begin,end) in enumerate(tail['offset_mapping']):
        group='cot' if begin<cot_end+len('</think>') else 'body' if end<=action_start else 'action'
        masks[group][0,n+i]=ids[0,n+i]
    def shift(x):return torch.cat([x[:,1:],torch.full((1,1),-100,dtype=torch.long)],dim=1)
    labels=shift(labels)
    batch={'input_ids':ids,'attention_mask':encoded['attention_mask'],'labels':labels,'labels_for_loss':labels.clone()}
    for k,v in masks.items():batch['labels_'+k]=shift(v)
    mm={k:v for k,v in encoded.items() if k not in ['input_ids','attention_mask','token_type_ids']}
    visual=int((ids==processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')).sum())
    parsed=parse_action(response,record['step'],Registry(tools))
    expected=target['tool_calls'][0]['function'];assert parsed.name==expected['name'] and parsed.arguments==expected['arguments'],'Target parser mismatch'
    row={'sample_id':record['sample_id'],'scene_id':record['scene_id'],'step':record['step'],'source':record['source'],
         'prompt_tokens':n,'visual_tokens':visual,'prompt_nonvisual_tokens':n-visual,'target_tokens':length-n,'total_tokens':length,
         'cot_tokens':int((batch['labels_cot']!=-100).sum()),'body_tokens':int((batch['labels_body']!=-100).sum()),
         'action_tokens':int((batch['labels_action']!=-100).sum()),'images':len(images),'image_grid':mm['image_grid_thw'].tolist(),
         'history_turns':record['step']-1,'prefix_equal':True,'parser_equal':True,'content_escapes':len(record['content_escapes']),
         'quality_hold':record['scene_id']=='scene_171','record_path':record.get('record_path'),
         'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'prompt_ids_sha256':hashlib.sha256(prefix['input_ids'].numpy().tobytes()).hexdigest(),
         'image_hashes':[r.get('sha256') or sha(r['path']) for r in record['images']]}
    metadata={'sample_id':row['sample_id'],'prompt_length':n,'source_length':length,'synthetic':False,'policy':record.get('context_policy_version','raw_cot_context_v1'),'record_path':record.get('record_path')}
    return {'batch':batch,'multi_modal_inputs':mm,'metadata':metadata},row,{'prompt_tail':prompt[-80:],'target_prefix':suffix[:min(30,cot_end)],'historical_think_tags':prompt[:-len('<think>\n')].count('<think>')}

def materialize_record(path,processor=None):
    processor=processor or setup()
    return tensor_features(processor,read(path))[0]

def summarize(rows,failures,scope):
    def stats(key):
        values=sorted(x[key] for x in rows)
        if not values:return {}
        return {'min':values[0],'p50':values[math.ceil(len(values)*.50)-1],'p95':values[math.ceil(len(values)*.95)-1],'p99':values[math.ceil(len(values)*.99)-1],'max':values[-1]}
    return {'scope':scope,'samples':len(rows),'scenes':len({x['scene_id'] for x in rows}),'failures':len(failures),
            'statistics':{k:stats(k) for k in ['prompt_tokens','visual_tokens','target_tokens','cot_tokens','action_tokens','total_tokens','images']},
            'longest':[{k:r[k] for k in ['sample_id','prompt_tokens','target_tokens','total_tokens','record_path']} for r in sorted(rows,key=lambda x:x['total_tokens'],reverse=True)[:5]],
            'content_escaped_samples':sum(bool(r['content_escapes']) for r in rows),'prefix_equal':all(r['prefix_equal'] for r in rows),'parser_equal':all(r['parser_equal'] for r in rows),
            'held_decisions':sum(r['quality_hold'] for r in rows),'units':'actual Qwen3.5-9B tokenizer and processor output positions'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--scope',choices=['pilot','review','all'],default='pilot');p.add_argument('--limit',type=int);p.add_argument('--materialize');p.add_argument('--batch-out');a=p.parse_args()
    processor=setup();out=ROOT/a.scope;out.mkdir(parents=True,exist_ok=True)
    if a.materialize:
        import torch
        value=materialize_record(a.materialize,processor);torch.save(value,a.batch_out);print('BATCH_READY',json.dumps(value['metadata']));return
    sources=record_sources(a.scope);rows=[];failures=[];template_examples=[];started=time.time()
    from v2s_integration.agent.messages import build_messages,render_request
    manifest={'version':'raw_cot_context_v1','model':str(MODEL),'model_file_hashes':{n:sha(MODEL/n) for n in ['chat_template.jinja','tokenizer_config.json','preprocessor_config.json','config.json']},
              'context_sources':{n:sha(ROOT/'runtime/v2s_integration/agent'/n) for n in ['context_policy.py','messages.py','actions.py']},
              'thinking':True,'image_processor':processor.image_processor.to_dict(),'scope':a.scope,'source_count':len(sources),'legacy_teacher_context_removed':True,
              'history_truncation':False,'supervision':'current raw reasoning + escaped content + native call','content_escape_policy':'reserved protocol tokens in assistant.content only; raw record preserved'}
    dump(out/'context_policy_manifest.json',manifest)
    with (out/'lengths.jsonl').open('w') as table:
        for sid,run,source in sources:
            env,status,ep,aliases=initial_context(run,ROOT/'initial_image_cache')
            profile_path=out/'context_profile.json'
            if profile_path.exists():assert read(profile_path)==env.context_profile,'Context profile differs across scenes'
            else:dump(profile_path,env.context_profile)
            for rec in status['records']:
                step=rec['step'];sample_id=f'{sid}:{step:03d}'
                try:
                    ap=sorted((run/f'decision_{step:03d}').glob('attempt_*/assistant.json'))[-1]
                    assistant=read(ap);event_path=event_source(run,status,step);event=read(event_path)
                    target,escapes=target_message(assistant,event,aliases,env.model_tools,step)
                    messages,pictures,refs,state=build_messages(env)
                    assert len(env.turns)==step-1
                    # Invoke the actual harness rendering entry independently.
                    harness_prompt,harness_messages,harness_pictures,harness_refs,_=render_request(processor,env)
                    assert messages==harness_messages and [r['path'] for r in refs]==[r['path'] for r in harness_refs]
                    prompt=processor.apply_chat_template(messages,tools=env.model_tools,tokenize=False,add_generation_prompt=True,enable_thinking=True)
                    assert prompt==harness_prompt,'Harness Thinking/template settings differ'
                    path=out/'records'/sid/f'{step:03d}.json.gz';path.parent.mkdir(parents=True,exist_ok=True)
                    record={'sample_id':sample_id,'scene_id':sid,'step':step,'source':source,'source_run':str(run),'source_assistant_sha256':sha(ap),
                            'source_event_path':str(event_path),'source_event_sha256':sha(event_path),'messages':messages,'tools':env.model_tools,'target':target,'images':refs,
                            'content_escapes':escapes,'context_policy':state['context_policy'],'record_path':str(path)}
                    features,row,example=tensor_features(processor,record)
                    with gzip.open(path,'wt') as f:json.dump(record,f,ensure_ascii=False)
                    table.write(json.dumps(row,ensure_ascii=False)+'\n');table.flush();rows.append(row)
                    if len(template_examples)<3:template_examples.append({'sample_id':sample_id,**example})
                    if len(rows)==1:
                        import torch
                        torch.save(features,out/'first_batch.pt')
                    del features,pictures,harness_pictures
                except Exception as e:
                    import traceback
                    failures.append({'sample_id':sample_id,'type':type(e).__name__,'message':str(e)[:250]})
                    (out/'error_tracebacks.log').open('a').write(sample_id+'\n'+traceback.format_exc()+'\n')
                    dump(out/'failures.json',failures)
                    dump(out/'status.json',{'stage':'failed','samples':len(rows),'failure':failures[-1]})
                    raise RuntimeError('Packaging failed; inspect compact failures.json') from e
                # Build next decision only from the completed event; never from final state.
                env.turns.append(event);env.public_state=copy.deepcopy(event['scene_state'])
                dump(out/'status.json',{'stage':'packing','samples':len(rows),'failures':len(failures),'latest':sample_id,'elapsed_seconds':round(time.time()-started,1)})
                if a.limit and len(rows)>=a.limit:break
            if a.limit and len(rows)>=a.limit:break
    summary=summarize(rows,failures,a.scope);summary['elapsed_seconds']=round(time.time()-started,1)
    dump(out/'length_summary.json',summary);dump(out/'template_probe.json',template_examples)
    dump(out/'prefix_equivalence_report.json',{'scope':a.scope,'checked':len(rows),'messages_and_image_order_equal':True,'expanded_prompt_ids_and_pixels_equal':summary['prefix_equal'],'parser_round_trip_equal':summary['parser_equal'],'failures':failures})
    dump(out/'status.json',{'stage':'complete',**summary});print('PACK_COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
