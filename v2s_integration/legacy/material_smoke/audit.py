"""Compact evidence only: source preservation, episodes, training metrics, checkpoints."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
manifest=json.loads((root/'source_manifest.json').read_text())
source=Path(manifest['source'])
changed=[name for name,digest in manifest['sha256'].items() if not (source/name).is_file() or hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest]
result={'source_files_checked':len(manifest['sha256']),'source_changed':changed}
for dirname in ('episodes','native_episodes','component_episodes'):
    episodes=[]
    for path in sorted((root/dirname).glob('*/result.json')):
        r=json.loads(path.read_text());events=json.loads((path.parent/'events.json').read_text())
        episodes.append({'episode':path.parent.name,'steps':r['steps'],'reward':r['reward'],'initial_reward':r['initial']['reward'],'invalid_actions':sum(e.get('error')=='invalid_action' for e in events),'tool_errors':sum(bool(e.get('error')) for e in events),'truncated':r['truncated']})
    result[dirname]=episodes
metrics={}
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    for path in (root/'native_run/tensorboard').rglob('events.out.tfevents.*'):
        ea=EventAccumulator(str(path));ea.Reload()
        for tag in ea.Tags()['scalars']:
            if any(x in tag.lower() for x in ('grad_norm','pg_loss','episode_reward','quality','learning_rate','response_length')):
                metrics[tag]=[{'step':e.step,'value':e.value} for e in ea.Scalars(tag)[-3:]]
except ImportError:metrics['unavailable']='tensorboard import missing'
result['native_metrics']=metrics
checkpoints=[p for p in (root/'native_run/checkpoints').rglob('*') if p.is_file()]
result['checkpoint_files']=len(checkpoints);result['checkpoint_bytes']=sum(p.stat().st_size for p in checkpoints)
(root/'audit_summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
