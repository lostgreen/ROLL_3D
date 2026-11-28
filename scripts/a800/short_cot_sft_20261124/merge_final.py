"""CPU merge, bound to the final trusted training checkpoint."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from chain_common import *

def main():
    import torch
    from transformers import AutoModelForImageTextToText,AutoProcessor
    from peft import PeftModel
    torch.set_num_threads(8)
    pointer=read(ROOT/'final_checkpoint.json');cp=Path(pointer['path'])
    v=validate_checkpoint(cp,read(ROOT/'training_manifest.json')['manifest_hash'],303)
    model=AutoModelForImageTextToText.from_pretrained('/m2v_intern/xuboshen/models/Qwen3.5-9B',
            torch_dtype=torch.bfloat16,device_map='cpu',local_files_only=True)
    model=PeftModel.from_pretrained(model,str(cp/'adapter'),is_trainable=False)
    h=hashlib.sha256();n=0
    for name,param in sorted(model.named_parameters()):
        if 'lora_' not in name:continue
        h.update(name.encode());h.update(param.detach().float().cpu().numpy().tobytes());n+=1
    assert h.hexdigest()==v['adapter_sha256'],('ADAPTER_PARAMETER_HASH',n,h.hexdigest(),v['adapter_sha256'])
    model=model.merge_and_unload(safe_merge=True)
    assert not any('lora_' in name for name,_ in model.named_parameters())
    dest=ROOT/'merged_final'
    model.save_pretrained(dest,safe_serialization=True,max_shard_size='4GB')
    proc=AutoProcessor.from_pretrained('/m2v_intern/xuboshen/models/Qwen3.5-9B',local_files_only=True)
    proc.save_pretrained(dest)
    files={p.name:sha(p) for p in dest.iterdir() if p.is_file()}
    dump(ROOT/'merge_receipt.json',dict(passed=True,checkpoint=str(cp),step=303,
         adapter_sha256=v['adapter_sha256'],lora_tensors=n,device='cpu',dtype='bfloat16',files=files,at=time.time()))
    cfg=read(EVAL/'config.json');cfg['adapter_sha256']=v['adapter_sha256'];dump(EVAL/'config.json',cfg)

if __name__=='__main__':main()
