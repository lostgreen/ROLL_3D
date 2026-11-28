"""Run both A800s on the unchanged thirty-case evaluation protocol."""
import os
import time
from chain_common import *

def main():
    wait_receipt(ROOT/'native_upstream_integrity.json')
    wait_receipt(ROOT/'vllm_preflight_receipt.json')
    wait_receipt(ROOT/'chain_cpu_preflight.json')
    wait_receipt(ROOT/'merge_receipt.json')
    cfg=read(EVAL/'config.json');rows=cfg['rows']
    assert len(rows)==30 and len({x['id'] for x in rows})==30
    assert cfg['max_steps']==25 and cfg['max_new_tokens']==32768 and cfg['max_input_tokens']==98304
    assert cfg['selection_sha256']=='8e7eba3e66c686a0013bdde08d320ed0b827e854fbf655c1871fbfe617fe0d38'
    ordered=sorted(rows,key=lambda r:(r['id']!='eval_021',r['subset']!='train10_diagnostic',r['id']))
    shards=[ordered[::2],ordered[1::2]]
    assert shards[0][0]['id']=='eval_021'
    plan=dict(passed=True,selection_sha256=cfg['selection_sha256'],variant=VARIANT,
              shards=[[r['id'] for r in rr] for rr in shards],workers_per_gpu=2,ports=[26790,26791],
              base_rerun=False,patches=[],backend='vllm0.17.1',at=time.time())
    dump(EVAL/'dispatch_plan.json',plan)
    servers=[];clients=[]
    try:
        for gpu,rr in enumerate(shards):
            port=26790+gpu;out=EVAL/'servers'/('gpu'+str(gpu))
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
            servers.append(spawn([VLLM,CODE/'eval_source/native_server.py','--gpu',gpu,'--port',port,
                           '--variant','sft','--output-root',out,'--config',EVAL/'config.json'],
                           ROOT/('eval_server_gpu'+str(gpu)+'.log'),env))
            clients.append(spawn([TRAIN,CODE/'eval_source/run_batch.py','--root',EVAL,'--variant',VARIANT,
                           '--port',port,'--workers','2','--ids',','.join(r['id'] for r in rr)],
                           ROOT/('eval_client_gpu'+str(gpu)+'.log'),dict(os.environ,CUDA_VISIBLE_DEVICES='')))
        dump(EVAL/'driver_status.json',dict(stage='running',servers=[p.pid for p in servers],clients=[p.pid for p in clients],at=time.time()))
        deadline=time.monotonic()+48*3600
        while any(p.poll() is None for p in clients):
            for i,p in enumerate(servers):
                state=EVAL/'servers'/('gpu'+str(i))/'status.json'
                if p.poll() is not None:raise RuntimeError('SERVER_EXIT gpu='+str(i)+' code='+str(p.returncode))
                if state.exists() and read(state).get('stage')=='load_failed':raise RuntimeError('SERVER_LOAD_FAILED gpu='+str(i))
            if time.monotonic()>deadline:raise TimeoutError('EVAL_48H_DEADLINE')
            time.sleep(15)
        codes=[p.returncode for p in clients]
        assert codes==[0,0],('EVAL_CLIENT_FAILURE',codes)
        states=[]
        for row in rows:
            p=EVAL/'runs'/VARIANT/row['id']/'status.json'
            states.append(dict((read(p) if p.exists() else {'state':'missing'}),id=row['id'],subset=row['subset']))
        assert all(s['state'] in ('complete','failed','context_budget_exceeded') for s in states),'INCOMPLETE_DISPATCH'
        dump(EVAL/'driver_status.json',dict(stage='complete',cases=states,
             complete=sum(x['state']=='complete' for x in states),failed=sum(x['state']!='complete' for x in states),at=time.time()))
    finally:
        for p in clients+servers:stop_owned(p)

if __name__=='__main__':main()
