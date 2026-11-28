"""Small status only; never stream job logs or model responses."""
from chain_common import *
out={}
for key,file in [('chain','chain_status.json'),('data','prepare_status.json'),('vllm','vllm_setup_status.json')]:
    p=ROOT/file
    if p.exists():out[key]=read(p)
for name in ['smoke_long_001']+['runs/attempt_'+str(i).zfill(2) for i in range(1,5)]:
    p=ROOT/name
    if (p/'status.json').exists():
        v=read(p/'status.json')
        if (p/'metrics.jsonl').exists():
            lines=(p/'metrics.jsonl').read_text().splitlines()
            if lines:
                m=json.loads(lines[-1]);v.update(completed_step=m['global_step'],last_wall_seconds=m['wall_seconds'],
                    metrics={k:x for k,x in m['metrics'].items() if any(z in k for z in ['loss','grad','memory'])})
        out[name]=v
p=EVAL/'preparation_status.json'
if p.exists():out['eval_inputs']={k:v for k,v in read(p).items() if k!='public_references_sha_verified'}
p=EVAL/'driver_status.json'
if p.exists():out['evaluation']={k:v for k,v in read(p).items() if k!='cases'}
print(json.dumps(out,ensure_ascii=False))
