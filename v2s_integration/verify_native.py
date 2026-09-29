"""Bounded, attempt-specific evidence; never prints policy responses."""
import argparse
import hashlib
import json
import math
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--attempt',required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parent
since=(root/f'{args.attempt}.pid').stat().st_mtime
out={'attempt':args.attempt}
manifest=json.loads((root/'source_manifest.json').read_text())
source=Path(manifest['source'])
changed=[name for name,digest in manifest['sha256'].items()
         if not (source/name).is_file() or hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest]
out['source_audit']={'checked':len(manifest['sha256']),'changed_count':len(changed),'changed':changed[:10]}
state=json.loads((root/'native_status.json').read_text())
out['pipeline']={k:state[k] for k in ('state','seconds','error_type') if k in state}
episodes=[]
for path in (root/'native_episodes').glob('*/result.json'):
    if path.stat().st_mtime<since:continue
    r=json.loads(path.read_text());ev=json.loads((path.parent/'events.json').read_text())
    cp=path.parent/'training_contract.json'
    c=json.loads(cp.read_text()) if cp.exists() else {}
    episodes.append({'id':path.parent.name,'reward':r['reward'],'steps':r['steps'],
                     'errors':sum(bool(e.get('error')) for e in ev),
                     'tool_executions':sum(bool(e.get('tool')) for e in ev),
                     'contract_pass':bool(c.get('exact_generated_token_match')) and c.get('image_tokens_in_loss')==0
                     and all(t['prompt_match'] and t['response_match'] for t in c.get('turns',[]))})
rewards=[e['reward'] for e in episodes]
out['episodes']={'count':len(episodes),'reward_min':min(rewards) if rewards else None,
                 'reward_max':max(rewards) if rewards else None,'contracts_pass':sum(e['contract_pass'] for e in episodes),
                 'tool_executions':sum(e['tool_executions'] for e in episodes),'errors':sum(e['errors'] for e in episodes),
                 'items':episodes[:8]}
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
metrics={}
for path in (root/'native_run/tensorboard').rglob('events.out.tfevents.*'):
    if path.stat().st_mtime<since:continue
    ea=EventAccumulator(str(path));ea.Reload()
    for tag in ea.Tags()['scalars']:
        if any(s in tag.lower() for s in ('grad_norm','pg_loss','learning_rate','system/samples','episode_reward')):
            vals=[e for e in ea.Scalars(tag) if e.wall_time>=since]
            if vals:metrics[tag]={'step':vals[-1].step,'value':vals[-1].value,'finite':math.isfinite(vals[-1].value)}
out['metrics']=metrics
files=[p for p in (root/'native_run').rglob('*') if p.is_file() and p.stat().st_mtime>=since and
       ('checkpoint' in str(p) or p.name=='adapter_model.safetensors')]
out['checkpoint']={'files':len(files),'bytes':sum(p.stat().st_size for p in files)}
from safetensors import safe_open
adapters=[]
for path in files:
    if path.name!='adapter_model.safetensors':continue
    with safe_open(str(path),framework='pt',device='cpu') as f:
        names=[n for n in f.keys() if 'lora_B' in n]
        tensors=[f.get_tensor(n).float() for n in names]
        adapters.append({'path':str(path),'lora_B_tensors':len(names),
                         'lora_B_nonzero':sum(bool(t.count_nonzero()) for t in tensors),
                         'lora_B_all_finite':all(bool(t.isfinite().all()) for t in tensors),
                         'lora_B_max_abs':max((float(t.abs().max()) for t in tensors),default=0.)})
out['adapters']=adapters
(root/f'{args.attempt}_verification.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
