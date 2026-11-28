"""Finish isolated environment verification alongside training."""
from chain_common import *

def main():
    deadline=time.monotonic()+3*3600
    retry=False
    while not (ROOT/'native_upstream_integrity.json').exists():
        p=ROOT/'vllm_setup_status.json'
        if p.exists() and read(p).get('stage')=='failed':
            assert not retry,'VLLM_SETUP_RETRY_EXHAUSTED'
            retry=True
            run([TRAIN,CODE/'setup_vllm.py'],ROOT/'setup_vllm_retry.log',timeout=7200)
        if time.monotonic()>deadline:raise TimeoutError('VLLM_ENV_INSTALL_DEADLINE')
        time.sleep(15)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1')
    run([VLLM,CODE/'preflight_vllm.py'],ROOT/'vllm_preflight.log',env,timeout=600)

if __name__=='__main__':main()
