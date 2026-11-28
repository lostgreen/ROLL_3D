import hashlib
import json
from pathlib import Path
import os
import signal
import subprocess
import time

ROOT=Path('/data/zgw/short_cot_sft_20261124')
CODE=ROOT/'code'
TRAIN=CODE/'run_train_python.sh'
VLLM=CODE/'run_vllm_python.sh'
EVAL=ROOT/'eval30'
VARIANT='short_cot_step303'

def read(p):return json.loads(Path(p).read_text())
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
    t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def validate_checkpoint(p,manifest=None,step=None):
    p=Path(p);v=read(p/'progress.json')
    if manifest is not None:assert v['manifest_hash']==manifest
    if step is not None:assert v['global_step']==step
    assert len(v['rank_files'])==2 and {x['rank'] for x in v['rank_files']}=={0,1}
    for x in v['rank_files']:
        f=p/Path(x['path']).name
        assert f.is_file() and sha(f)==x['sha256'],f
        assert set(x['training_state_hashes'])>={'optimizer_sha256','scheduler_sha256','rng_sha256'}
    assert (p/'adapter/adapter_model.safetensors').is_file()
    assert (p/'adapter/adapter_config.json').is_file()
    assert all(x['sha256']==v['adapter_sha256'] for x in v['parameters'])
    return v

def spawn(cmd,log,env=None):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as f:
        return subprocess.Popen([str(x) for x in cmd],stdout=f,stderr=subprocess.STDOUT,
                                start_new_session=True,env=env)
def stop_owned(p):
    if p and p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=30)
def run(cmd,log,env=None,timeout=None):
    p=spawn(cmd,log,env)
    try:code=p.wait(timeout=timeout)
    except BaseException:
        stop_owned(p);raise
    if code:raise RuntimeError('SUBPROCESS_FAILED code='+str(code)+' log='+str(log))

def wait_receipt(path,key='passed',timeout=7200):
    until=time.monotonic()+timeout
    while time.monotonic()<until:
        if Path(path).exists():
            v=read(path)
            if v.get(key):return v
            if v.get('stage')=='failed' or v.get('state')=='failed':raise RuntimeError('PREPARATION_FAILED:'+str(path))
        time.sleep(10)
    raise TimeoutError('receipt:'+str(path))
