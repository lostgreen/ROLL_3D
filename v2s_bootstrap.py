"""Create an isolated harness snapshot and probe ROLL without changing source envs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

root = Path('/home/xuboshen/zgw/ROLL')
work = root / 'v2s_integration'
work.mkdir(exist_ok=True)
source = Path('/m2v_intern/xuboshen/projects/video2scene_segmented_texture_20260922/code')
snapshot = work / 'harness'
snapshot.mkdir(exist_ok=True)
manifest = {}
for name in ('src', 'headless', 'blender-mcp', 'metrics', 'examples'):
    for path in sorted((source / name).rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        if any(part in ('.git', '.venv', 'runs', 'artifacts', 'node_modules') for part in path.relative_to(source).parts):
            continue
        rel = path.relative_to(source)
        dest = snapshot / rel
        data = path.read_bytes()
        manifest[str(rel)] = hashlib.sha256(data).hexdigest()
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(data)
        elif hashlib.sha256(dest.read_bytes()).hexdigest() != manifest[str(rel)]:
            raise RuntimeError(f'Existing snapshot differs: {rel}')
(work / 'source_manifest.json').write_text(json.dumps({'source':str(source),'sha256':manifest},indent=2))
print('SNAPSHOT files=', len(manifest), flush=True)
venv = work / 'venv'
if not venv.exists():
    subprocess.run(['/home/xuboshen/Anaconda/envs/roll-t28/bin/python','-m','venv','--system-site-packages',str(venv)],check=True)
env = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
checks = {
 'torch': 'import torch; print(torch.__version__,torch.cuda.is_available(),torch.cuda.device_count())',
 'vl_manager': 'from roll.pipeline.agentic.env_manager.vl_traj_env_manager import VLTrajEnvManager; print("OK")',
 'pipeline': 'from roll.pipeline.agentic.agentic_pipeline import AgenticPipeline; print("OK")',
 'fsdp2': 'from roll.distributed.strategy.fsdp2_strategy import FSDP2TrainStrategy; print("OK")',
 'hf': 'from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor; from peft import LoraConfig,get_peft_model; print("OK")',
}
results = {}
for name, code in checks.items():
    start = time.time()
    with (work / f'probe_{name}.log').open('w') as log:
        p = subprocess.run([str(venv/'bin/python'),'-c',code],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=150)
    lines=(work/f'probe_{name}.log').read_text().splitlines()
    errors=[x for x in lines if any(t in x for t in ('Error:', 'Exception:', 'No module named'))]
    results[name]={'exit_code':p.returncode,'seconds':round(time.time()-start,1),'error':errors[-1:]}
    print('PROBE',name,json.dumps(results[name]),flush=True)
(work/'bootstrap_results.json').write_text(json.dumps(results,indent=2))
print('DONE bootstrap',flush=True)
