"""Extract only upstream hardware dispatch facts from the pinned cached wheel."""
import json
from pathlib import Path
import zipfile
from chain_common import ROOT,dump

for p in (ROOT/'pip_cache').rglob('*'):
    if not p.is_file():continue
    if not 400_000_000<p.stat().st_size<500_000_000:continue
    if not zipfile.is_zipfile(p):continue
    with zipfile.ZipFile(p) as z:
        if 'vllm/__init__.py' not in z.namelist():continue
        names=[n for n in z.namelist() if 'gdn' in n.lower() and n.endswith('.py')]
        snippets=[]
        for name in names:
            if '/layers/mamba/' not in name:continue
            lines=z.read(name).decode().splitlines()
            for i,line in enumerate(lines):
                if any(s in line for s in ['is_device_capability','use_flashinfer','gdn_backend','GDN_BACKEND']):
                    snippets.append(dict(file=name,line=i+1,text='\n'.join(lines[max(0,i-1):i+3])[:550]))
        dump(ROOT/'vllm_wheel_dispatch_inspection.json',dict(wheel_files=names,dispatch_evidence=snippets))
        print(json.dumps(dict(files=names,evidence=snippets[:10]),ensure_ascii=False))
        break
else:raise RuntimeError('PINNED_WHEEL_NOT_FOUND_IN_TASK_CACHE')
