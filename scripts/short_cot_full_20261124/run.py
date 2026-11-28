"""Scene-sequential ordinary API annotation; all retries share the adaptive cap."""
from collections import Counter,deque
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import gzip
import json
import math
from pathlib import Path
import random
import sys
import threading
import time
from common import ROOT,OUT,CONFIG,EXCLUSIONS,dump,read,sha,digest,cpu_processor
from prepare import load_packet
from presentation import payload
from validate import strict_json,validate,narrow_cleanup

sys.path.insert(0,str(ROOT))
from v2s_integration.agent.api_transport import invoke

def p95(values):return sorted(values)[max(0,math.ceil(len(values)*.95)-1)] if values else None

class Gate:
    def __init__(self,total):
        self.cv=threading.Condition();self.limit=8;self.active=0;self.peak=0;self.queue=deque()
        self.window=[];self.events=[];self.selected={};self.errors=[];self.total=total
        self.started=time.time();self.last_change=self.started;self.changes=[dict(time=self.started,limit=8,reason='initial')]
        self.aborted=False;self.cooldown_until=0
        for path in (OUT/'rewrites').glob('*/attempt_*/event.json'):self.events.append(read(path))
        for path in (OUT/'rewrites').glob('*/selection.json'):self.selected[read(path)['sample_id']]=read(path)
        self.publish()

    def acquire(self):
        ticket=object()
        with self.cv:
            self.queue.append(ticket)
            while not self.aborted and (self.queue[0] is not ticket or self.active>=self.limit or time.time()<self.cooldown_until):self.cv.wait(timeout=5)
            if self.aborted:
                self.queue.remove(ticket);raise RuntimeError('pipeline_stopped_after_engineering_failure')
            self.queue.popleft();self.active+=1;self.peak=max(self.peak,self.active);self.cv.notify_all()
            return self.limit

    def release(self,event):
        with self.cv:
            self.active-=1;self.events.append(event);self.window.append(event)
            with (OUT/'request_events.jsonl').open('a') as f:f.write(json.dumps(event,separators=(',',':'))+'\n')
            now=time.time();recent=self.window[-48:];bad=[x for x in recent if not x['success']]
            throttled=event.get('http_status')==429
            if throttled:self.cooldown_until=max(self.cooldown_until,now+30)
            if throttled or len(recent)>=12 and len(bad)/len(recent)>.10:
                new=max(8,self.limit-2)
                if new!=self.limit:self.limit=new;self.changes.append(dict(time=now,limit=new,reason='429_or_network_error_rate'));self.last_change=now;self.window=[]
            successful=[x for x in self.window if x['success']]
            if self.limit<12 and len(successful)>=24 and not any(not x['success'] for x in self.window) and now-self.last_change>=30 and p95([x['seconds'] for x in successful])<=180:
                self.limit+=2;self.changes.append(dict(time=now,limit=self.limit,reason='24_successes_no_errors_p95_le_180'));self.last_change=now;self.window=[]
            self.publish();self.cv.notify_all()

    def done(self,result):
        with self.cv:self.selected[result['sample_id']]=result;self.publish()

    def fail(self,scene,exc):
        with self.cv:
            self.aborted=True;self.errors.append(dict(scene_id=scene,error_type=type(exc).__name__,reason=str(exc)[:250]));self.publish();self.cv.notify_all()

    def publish(self):
        groups={}
        for label,subset in [('all',self.events),('initial',[x for x in self.events if x['content_attempt']==1 and x['network_attempt']==1]),
                ('retry',[x for x in self.events if x['content_attempt']!=1 or x['network_attempt']!=1]),
                ('scene_first',[x for x in self.events if x['scene_first']]),('scene_later',[x for x in self.events if not x['scene_first']])]:
            good=[x for x in subset if x['success']];incoming=sum(x.get('input_tokens') or 0 for x in good);cached=sum(x.get('cached_tokens') or 0 for x in good)
            outgoing=sum(x.get('output_tokens') or 0 for x in good)
            groups[label]=dict(requests=len(subset),successful=len(good),input_tokens=incoming,cached_tokens=cached,
                cache_ratio=cached/incoming if incoming else None,output_tokens=outgoing,
                missing_usage=sum(x.get('input_tokens') is None or x.get('cached_tokens') is None for x in good),
                latency_p95=p95([x['seconds'] for x in good]),
                estimated_usd=((incoming-cached)*1.76686+cached*.35337+outgoing*5.30059)/1e6)
        states=Counter('ok' if x.get('accepted_contract') else 'flagged' if x.get('model_requested_review') else 'failed' for x in self.selected.values())
        dump(OUT/'progress.json',dict(stage='engineering_failure' if self.aborted else 'annotating',updated_at=time.time(),
            run_started_at=self.started,total=self.total,completed=len(self.selected),states=dict(states),
            concurrency=self.limit,active_requests=self.active,peak_active=self.peak,
            concurrency_changes=self.changes,request_metrics=groups,
            failed_http_statuses=dict(Counter(str(x.get('http_status')) for x in self.events if not x['success'])),
            pipeline_errors=self.errors,occupancy_enabled=False,semantic_repair_calls=0,api_checker_calls=0,training_release=False))

def check_response(call,message,packet,tokenizer):
    (call/'native_reasoning.txt').write_text(message.get('reasoning_content') or message.get('reasoning') or '')
    (call/'content.txt').write_text(message.get('content') or '')
    try:
        obj=strict_json(message.get('content') or '');raw=validate(obj,packet,tokenizer);dump(call/'rewrite_output_raw.json',obj)
        clean,changes=narrow_cleanup(obj);result=validate(clean,packet,tokenizer)
        assert clean.get('short_cot')==obj.get('short_cot');dump(call/'rewrite_output.json',clean)
        if changes:dump(call/'local_cleanup.json',dict(changes=changes,short_cot_changed=False,original_sha256=digest(obj),effective_sha256=digest(clean)))
    except Exception as exc:
        raw=result=dict(passed=False,status='invalid_json_or_schema',errors=[type(exc).__name__+':'+str(exc)[:200]],warnings=[]);changes=[]
    dump(call/'validation_raw.json',raw);dump(call/'validation.json',result)
    return raw,result,changes

def verify_source(row,packet,processor,tensor_features,processor_lock):
    folder=OUT/'packages'/row['sample_id'].replace(':','_');path=folder/'prefix_validation.json'
    receipt=read(folder/'input_receipt.json');old=receipt['eligible_row']
    assert sha(old['record_path'])==old['record_sha256'] and digest(packet)==row['packet_sha256']
    record=json.load(gzip.open(old['record_path'],'rt'))
    if not path.exists():
        with processor_lock:feature,metrics,_=tensor_features(processor,record)
        for k in ['prompt_tokens','prompt_sha256','prompt_ids_sha256','image_hashes','images','target_tokens','cot_tokens','action_tokens','body_tokens']:
            assert metrics[k]==old[k],(row['sample_id'],k)
        assert metrics['prefix_equal'] and metrics['parser_equal'];del feature
        dump(path,dict(passed=True,metrics=metrics,source_record_sha256=old['record_sha256'],packet_sha256=digest(packet)))
    else:
        previous=read(path);assert previous['passed'] and previous['source_record_sha256']==old['record_sha256'] and previous['packet_sha256']==digest(packet)

def one(row,first,processor,tensor_features,processor_lock,key,gate):
    sid=row['sample_id'];name=sid.replace(':','_');packet=load_packet(name)
    verify_source(row,packet,processor,tensor_features,processor_lock)
    identity=digest(dict(sample_id=sid,packet=digest(packet),config=CONFIG))
    target=OUT/'rewrites'/name;selection=target/'selection.json'
    if selection.exists():
        choice=read(selection);assert choice['idempotency_key']==identity;return choice
    feedback=None;calls=[]
    for content in range(1,CONFIG['content_retries']+2):
        response_seen=False
        for network in range(1,CONFIG['network_retries']+2):
            call=target/('attempt_%02d_network_%02d'%(content,network));changes=[]
            if (call/'validation.json').exists():
                raw=read(call/'validation_raw.json');result=read(call/'validation.json')
                changes=read(call/'local_cleanup.json')['changes'] if (call/'local_cleanup.json').exists() else []
            elif (call/'response.json').exists():
                message=read(call/'response.json')['choices'][0]['message'];raw,result,changes=check_response(call,message,packet,processor.tokenizer)
            elif call.exists():
                calls.append(str(call));continue
            else:
                request,equivalence=payload(packet,feedback);limit=gate.acquire();started=time.monotonic()
                event=dict(sample_id=sid,scene_id=row['scene_id'],scene_first=first,content_attempt=content,network_attempt=network,concurrency_limit=limit)
                try:
                    message,metadata=invoke(CONFIG['base_url'],key,request,call,timeout=900)
                    event.update({k:metadata.get(k) for k in ['seconds','http_status','input_tokens','cached_tokens','output_tokens','reasoning_tokens','response_model']});event['success']=True
                    dump(call/'request_receipt.json',dict(sample_id=sid,idempotency_key=identity,config_sha256=digest(CONFIG),packet_sha256=digest(packet),
                        equivalence=equivalence,thinking_budget=2048,content_attempt=content,network_attempt=network,scene_first=first,usage=metadata))
                except Exception as exc:
                    failure=read(call/'failure.json') if (call/'failure.json').exists() else {}
                    event.update(success=False,seconds=round(time.monotonic()-started,3),http_status=failure.get('http_status'))
                    dump(call/'network_failure.json',dict(error_type=type(exc).__name__,message=str(exc).replace(key,'<REDACTED>')[:250]))
                finally:
                    gate.release(event)
                dump(call/'event.json',event);calls.append(str(call))
                if not event['success']:
                    if event.get('http_status') in [400,401,403,404,413,422]:raise RuntimeError('API_request_engineering_error:%s:%s'%(event['http_status'],sid))
                    time.sleep(min(60,5*2**(network-1)+random.random()*3));continue
                raw,result,changes=check_response(call,message,packet,processor.tokenizer)
            if str(call) not in calls:calls.append(str(call))
            response_seen=True
            if result.get('passed') or result.get('schema_valid_review_flag'):
                choice=dict(sample_id=sid,idempotency_key=identity,attempt=str(call),calls=calls,accepted_contract=bool(result.get('passed')),
                    model_requested_review=bool(result.get('schema_valid_review_flag')),content_attempt=content,network_attempt=network,
                    raw_format_valid=bool(raw.get('passed') or raw.get('schema_valid_review_flag')),effective_format_valid=True,
                    local_cleanup=bool(changes),semantic_reviewed=False,training_release=False)
                dump(selection,choice);return choice
            feedback=result.get('errors') or [result.get('status')];break
        if not response_seen:
            choice=dict(sample_id=sid,idempotency_key=identity,accepted_contract=False,reason='network_retries_exhausted',calls=calls,training_release=False)
            dump(selection,choice);return choice
    choice=dict(sample_id=sid,idempotency_key=identity,accepted_contract=False,reason='contract_retries_exhausted',last_errors=feedback,calls=calls,training_release=False)
    dump(selection,choice);return choice

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=read(OUT/'manifest.json');assert manifest['total']==3277-len(EXCLUSIONS) and not manifest['unresolved_count']
        assert manifest['config_sha256']==digest(CONFIG)
        for name,h in manifest['source_files'].items():assert sha(Path(__file__).parent/name)==h,('source_changed',name)
        import torch
        torch.set_num_threads(1)
        processor,tensor_features=cpu_processor();processor_lock=threading.Lock()
        source=read(ROOT/'artifacts/indoor_qwen_200rollout_20261121/config.json')
        assert source['model']==CONFIG['model'] and source['base_url']==CONFIG['base_url']
        key=Path(source['key_file']).read_text().strip();gate=Gate(manifest['total'])
        scenes={}
        for row in manifest['rows']:scenes.setdefault(row['scene_id'],[]).append(row)
        def scene_run(scene,rows):
            for i,row in enumerate(sorted(rows,key=lambda x:x['step'])):
                if gate.aborted:return
                result=one(row,i==0,processor,tensor_features,processor_lock,key,gate);gate.done(result)
        with ThreadPoolExecutor(max_workers=12) as pool:
            futures={pool.submit(scene_run,s,rows):s for s,rows in scenes.items()}
            for future in as_completed(futures):
                try:future.result()
                except Exception as exc:gate.fail(futures[future],exc)
        if gate.aborted:raise RuntimeError('Engineering failure; see progress.json; no training export released')
        assert len(gate.selected)==manifest['total']
        dump(OUT/'annotation_complete.json',dict(total=manifest['total'],completed_at=time.time(),training_release=False))
    from export import main as export_main
    export_main()

if __name__=='__main__':main()
