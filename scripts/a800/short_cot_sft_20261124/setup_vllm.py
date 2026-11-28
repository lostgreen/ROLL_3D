"""Install pinned upstream vLLM in a separate task-local environment."""
import base64
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path('/data/zgw/short_cot_sft_20261124')
PY = '/m2v_intern/xuboshen/zgw/venvs/roll3d_native_20261113/bin/python'


def dump(value):
    p = ROOT / 'vllm_setup_status.json'
    t = p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)


def main():
    dump(dict(stage='installing',at=time.time(),pid=os.getpid()))
    try:
        envdir = ROOT / 'env0171'
        if not (envdir / 'bin/python').exists():
            subprocess.run([PY,'-m','venv',str(envdir)],check=True)
        env = dict(os.environ,http_proxy='http://10.66.28.231:11080',https_proxy='http://10.66.28.231:11080',
                   no_proxy='localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com',
                   PIP_CACHE_DIR=str(ROOT/'pip_cache'),PYTHONDONTWRITEBYTECODE='1')
        env.pop('PYTHONPATH',None)
        with (ROOT/'vllm_install.log').open('a') as f:
            subprocess.run([str(envdir/'bin/python'),'-m','pip','install','-i','https://pypi.tuna.tsinghua.edu.cn/simple',
                            'vllm==0.17.1','transformers==4.57.6'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            subprocess.run([str(envdir/'bin/python'),'-m','pip','check'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        site = next((envdir/'lib').glob('python*/site-packages'))
        record = site/'vllm-0.17.1.dist-info/RECORD'
        wanted = ['vllm/model_executor/models/qwen3_5.py','vllm/model_executor/models/qwen3_vl.py',
                  'vllm/entrypoints/llm.py','vllm/config/compilation.py']
        rows={x[0]:x[1] for x in csv.reader(record.open())}
        verified=[]
        for n in wanted:
            if not (site/n).is_file():continue
            actual=base64.urlsafe_b64encode(hashlib.sha256((site/n).read_bytes()).digest()).decode().rstrip('=')
            assert rows[n]=='sha256='+actual,n
            verified.append(n)
        assert len(verified)>=3
        receipt=dict(passed=True,vllm='0.17.1',transformers='4.57.6',source_record_verified=verified,
                     model_patches=[],env=str(envdir))
        (ROOT/'native_upstream_integrity.json').write_text(json.dumps(receipt,indent=2)+'\n')
        dump(dict(stage='complete',at=time.time(),**receipt))
        print('UPSTREAM_VLLM_ENV_READY',flush=True)
    except BaseException as e:
        (ROOT/'vllm_setup_traceback.log').write_text(traceback.format_exc())
        dump(dict(stage='failed',error_type=type(e).__name__,message=str(e)[:240],at=time.time()))
        raise


if __name__=='__main__':main()
