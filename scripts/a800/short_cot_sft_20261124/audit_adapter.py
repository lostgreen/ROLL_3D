"""Audit exported PEFT tensors without loading another 9B model."""
from chain_common import *

def main():
    import torch
    from safetensors.torch import load_file
    cp=Path(read(ROOT/'smoke_long_001/status.json')['checkpoint'])
    value=read(cp/'progress.json')
    tensors=load_file(str(cp/'adapter/adapter_model.safetensors'),device='cpu')
    h=hashlib.sha256();names=[]
    for key,tensor in tensors.items():
        name=key.replace('.lora_A.weight','.lora_A.default.weight').replace('.lora_B.weight','.lora_B.default.weight')
        assert '.lora_' in name
        names.append((name,tensor))
    for name,tensor in sorted(names):
        h.update(name.encode());h.update(tensor.float().contiguous().numpy().tobytes())
    assert h.hexdigest()==value['adapter_sha256'],('PEFT_EXPORT_HASH',h.hexdigest(),value['adapter_sha256'])
    dump(ROOT/'adapter_export_preflight.json',dict(passed=True,checkpoint=str(cp),tensors=len(tensors),
         adapter_sha256=value['adapter_sha256'],peft_export_roundtrip_names_verified=True,model_loaded=False,at=time.time()))
    print('PEFT_EXPORT_ROUNDTRIP_PASS')

if __name__=='__main__':main()
