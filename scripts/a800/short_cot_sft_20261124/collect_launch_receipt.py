"""Compact, source-bound launch evidence for the project record."""
import math
from chain_common import *

chain=read(ROOT/'chain_status.json');runroot=Path(chain['run'])
manifest=read(runroot/'manifest.json');prepared=read(ROOT/'prepare_receipt.json')
assert manifest['manifest_hash']==prepared['manifest_hash']
assert manifest['eligible_decisions']==3231 and manifest['eligible_scenes']==199 and manifest['total_updates']==303
assert chain.get('resume') is None
metrics=[json.loads(x) for x in (runroot/'metrics.jsonl').read_text().splitlines()]
assert metrics and all(math.isfinite(v) for v in metrics[-1]['metrics'].values())
checks={}
for name in ['prepare_receipt','fast_kernel_receipt','smoke_receipt','chain_cpu_preflight','adapter_export_preflight',
             'native_upstream_integrity','vllm_preflight_receipt']:
    v=read(ROOT/(name+'.json'));assert v['passed'];checks[name]=sha(ROOT/(name+'.json'))
code=read(CODE/'code_manifest.json')
assert all(sha(CODE/name)==digest for name,digest in code['files'].items())
result=dict(passed=True,machine_id='100004879',task_root=str(ROOT),source_reviewed_index_sha256=prepared['reviewed_index_sha256'],
    formal_run=str(runroot),formal_pid=read(runroot/'status.json')['pid'],supervisor_pid=chain['pid'],
    from_original_base=True,manifest_hash=manifest['manifest_hash'],config_sha256=sha(ROOT/'train_config.json'),
    train_decisions=3231,train_scenes=199,total_updates=303,completed_step=metrics[-1]['global_step'],
    latest_loss=metrics[-1]['metrics']['sft_train/loss@sum'],latest_wall_seconds=metrics[-1]['wall_seconds'],
    checkpoint_interval=25,full_adam_scheduler_rng=True,bounded_training_attempts=4,
    evaluation_cases=30,evaluation_selection_sha256=read(EVAL/'config.json')['selection_sha256'],
    automatic_after_success=True,base_rerun=False,occupancy_enabled=False,h800_contacted=False,
    evidence_sha256=checks,code_manifest_sha256=sha(CODE/'code_manifest.json'),
    vllm_cuda_preflight={k:v for k,v in read(ROOT/'vllm_preflight_receipt.json').items() if k!='upstream_gdn_selection_evidence'},at=time.time())
dump(ROOT/'launch_receipt.json',result)
print(json.dumps({k:result[k] for k in ['passed','formal_run','completed_step','latest_loss','automatic_after_success','occupancy_enabled']}))
