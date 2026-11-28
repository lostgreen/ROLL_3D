"""Audit the one-update capacity smoke, including zero-LR warmup semantics."""
import math
from chain_common import *

def main():
    import torch
    from train_raw_cot import state_digest
    root=ROOT/'smoke_long_001';s=read(root/'status.json')
    assert s['stage']=='bounded_run_complete' and s['global_step']==1
    cp=Path(s['checkpoint']);c=validate_checkpoint(cp,step=1)
    metrics=[json.loads(x) for x in (root/'metrics.jsonl').read_text().splitlines()]
    assert len(metrics)==1 and all(math.isfinite(x) for x in metrics[0]['metrics'].values())
    mm=metrics[0]['metrics']
    assert any('grad' in k and v>0 for k,v in mm.items())
    assert mm['audit/optimizer_updates@max']==1
    audits=list(root.rglob('kernel_audit_rank_*.json'))
    assert len(audits)==2 and all(read(p)['passed'] for p in audits)
    manifest=read(root/'training_manifest.json') if (root/'training_manifest.json').exists() else read(root/'manifest.json')
    optimizer=[]
    for x in c['rank_files']:
        payload=torch.load(cp/Path(x['path']).name,map_location='cpu',weights_only=False)
        assert payload['checkpoint_schema_version']==2 and payload['global_step']==1
        assert state_digest(payload['optimizer'])==x['training_state_hashes']['optimizer_sha256']
        assert payload['scheduler']['last_epoch']==1
        stats={'finite_tensors':0,'nonzero_tensors':0}
        def walk(v):
            if torch.is_tensor(v):
                assert bool(torch.isfinite(v).all());stats['finite_tensors']+=1
                stats['nonzero_tensors']+=bool(torch.count_nonzero(v))
            elif isinstance(v,dict):
                for xx in v.values():walk(xx)
            elif isinstance(v,(tuple,list)):
                for xx in v:walk(xx)
        states=payload['optimizer']['state'];assert states
        walk(states);assert stats['nonzero_tensors']>0
        optimizer.append(dict(rank=x['rank'],states=len(states),**stats))
    # The exact production schedule starts at LR=0, so its first optimizer
    # update changes Adam moments rather than parameters. This is expected.
    if any(p['delta_l2']==0 for p in c['parameters']):assert manifest['warmup_steps']==1
    receipt=read(ROOT/'prepare_receipt.json')
    dump(ROOT/'smoke_receipt.json',dict(passed=True,ids=receipt['smoke_ids'],lengths=receipt['smoke_lengths'],
         metrics=metrics[0],full_checkpoint=str(cp),fast_kernel_audits=[str(p) for p in audits],
         optimizer_audit=optimizer,first_update_zero_lr_warmup=True,
         fresh_base_for_formal=True,at=time.time()))
    print('LONGEST_PAIR_SMOKE_VERIFIED_WITH_ADAM')

if __name__=='__main__':main()
