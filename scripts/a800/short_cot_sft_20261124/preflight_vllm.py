"""Tiny A800 CUDA/import check, without loading an extra model during SFT."""
import os
from pathlib import Path
import subprocess
from chain_common import *

def main():
    import torch,transformers,vllm
    import importlib.metadata as meta
    import csv,base64
    pins=read(CODE/'inference_protocol_versions.json')
    actual={name:meta.version(name) for name in pins}
    assert actual==pins,('INFERENCE_PROTOCOL_DEPENDENCY_DRIFT',actual,pins)
    assert torch.__version__.split('+')[0]=='2.10.0'
    assert transformers.__version__=='4.57.6' and vllm.__version__=='0.17.1'
    x=torch.ones((32,32),dtype=torch.bfloat16,device='cuda')
    assert float((x@x).mean())==32
    torch.cuda.synchronize()
    site=Path(vllm.__file__).parent
    record=site.parent/'vllm-0.17.1.dist-info/RECORD'
    hashes={r[0]:r[1] for r in csv.reader(record.open())}
    kernel_sources=0
    for rel,expected in hashes.items():
        if not rel.endswith('.py') or not any(x in rel for x in ['/layers/mamba/','/layers/fla/']):continue
        digest=base64.urlsafe_b64encode(bytes.fromhex(sha(site.parent/rel))).decode().rstrip('=')
        assert expected=='sha256='+digest,rel
        kernel_sources+=1
    assert kernel_sources>0
    choices=[]
    for p in (site/'model_executor/layers').glob('**/*gdn*.py'):
        for number,line in enumerate(p.read_text().splitlines(),1):
            if any(s in line for s in ['is_device_capability','FLASHINFER','flashinfer','use_triton']):
                if len(choices)<30:choices.append(dict(file=str(p.relative_to(site)),line=number,text=line.strip()[:180]))
    lock=ROOT/'locks';lock.mkdir(exist_ok=True)
    with (lock/'vllm_native0171.freeze.txt').open('w') as f:
        subprocess.run([os.sys.executable,'-m','pip','freeze'],stdout=f,check=True)
    dump(ROOT/'vllm_preflight_receipt.json',dict(passed=True,torch=torch.__version__,transformers=transformers.__version__,
        vllm=vllm.__version__,gpu=torch.cuda.get_device_name(),sm=torch.cuda.get_device_capability(),
        protocol_versions=actual,kernel_sources_record_verified=kernel_sources,cuda_matmul_bf16=True,extra_model_loaded=False,upstream_gdn_selection_evidence=choices,
        compiler_root=os.environ.get('CUDA_HOME'),cxx=os.environ.get('CXX'),at=time.time()))
    print('ISOLATED_UPSTREAM_VLLM_A800_CUDA_PASS')

if __name__=='__main__':main()
