"""Recreate the frozen thirty public starts on A800 from public metadata only."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from prepare import ROOT, REPO, BASE, dump, read, sha

OLD = REPO/'artifacts/raw_cot_eval30_20261123/official_chat_eos_v2'
EVAL = ROOT/'eval30'
SELECTION = '8e7eba3e66c686a0013bdde08d320ed0b827e854fbf655c1871fbfe617fe0d38'


def main():
    EVAL.mkdir(parents=True,exist_ok=True)
    cfg = read(OLD/'config.json')
    assert cfg['selection_sha256']==SELECTION
    assert len(cfg['rows'])==30 and len({r['id'] for r in cfg['rows']})==30
    assert sum(r['subset']=='test20' for r in cfg['rows'])==20
    assert sum(r['subset']=='train10_diagnostic' for r in cfg['rows'])==10
    checks=[]
    for row in cfg['rows']:
        sid=row['id']
        for i in (0,1):
            source=OLD/'public'/sid/('reference_'+str(i)+'.jpg')
            dest=EVAL/'public_refs'/sid/('public_'+str(i)+'.jpg')
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
            assert sha(source)==sha(dest)
            checks.append(dict(id=sid,view=i,sha256=sha(dest)))
        row['reference_files']=[str(EVAL/'public_refs'/sid/('public_'+str(i)+'.jpg')) for i in (0,1)]
        if row['subset']=='test20':
            metadata=read(OLD/'ground_truth/scenes'/sid/'status.json')
            views=metadata.get('views4',metadata.get('views',[]))
            row['public_views']=[x for x in views if x['name'] in ('public_0','public_1')]
        assert len(row['public_views'])==2
    cfg.update(root=str(EVAL),request_root=str(EVAL/'runs'),
               shared_root=str(ROOT/'results'),backend='vllm_native0171_cuda_graph',
               train_source=str(ROOT/'code/training_source'),concurrency_per_variant=2,
               inference_patches=[],cohort='short_cot_epoch3',evaluation_variant='short_cot_step303',
               serving_model=str(ROOT/'merged_final'),gpu_memory_utilization=.70)
    dump(EVAL/'config.json',cfg)
    shutil.copy2(OLD/'selection_manifest.json',EVAL/'selection_manifest.json')
    if not (EVAL/'ground_truth').exists():
        (EVAL/'ground_truth').symlink_to(OLD/'ground_truth',target_is_directory=True)
    vendor=EVAL/'cpu_vendor'
    if not (vendor/'mcp').exists() and importlib.util.find_spec('mcp') is None:
        env=dict(os.environ,http_proxy='http://10.66.28.231:11080',https_proxy='http://10.66.28.231:11080',
                 PIP_CACHE_DIR=str(ROOT/'pip_cache'))
        with (ROOT/'cpu_vendor_install.log').open('a') as f:
            subprocess.run([sys.executable,'-m','pip','install','--target',str(vendor),
                            '-i','https://pypi.tuna.tsinghua.edu.cn/simple','mcp==1.27.0'],
                           stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
    # prepare_one uses factory-empty Blender, public cameras and these 60 JPEGs;
    # it never loads a GT scene, furniture metadata or held-out images.
    dump(EVAL/'preparation_config.json',dict(rows=cfg['rows'],blender=cfg['blender']))
    with (ROOT/'prepare_public_starts.log').open('a') as f:
        subprocess.run([cfg['blender'],'--background','--factory-startup','--threads','4','--python',
                        str(ROOT/'code/eval_source/prepare_batch_blender.py'),'--',str(EVAL)],
                       stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True,timeout=1800)
    for row in cfg['rows']:
        sid=row['id'];new=read(EVAL/'public'/sid/'task.json');old=read(OLD/'public'/sid/'task.json')
        assert new['public_camera_protocol']==old['public_camera_protocol'],sid
        assert new['instruction']==old['instruction'] and new['budget']==old['budget'],sid
        receipt=read(EVAL/'public'/sid/'preparation_receipt.json')
        assert receipt['passed'] and receipt['after_reload']['meshes']==0
    dump(EVAL/'preparation_status.json',dict(state='complete',train_ready=10,test_ready=20,
         public_cameras_identical=True,public_references_sha_verified=checks,source_gt_opened=False,
         regenerated_empty_starts=True,at=time.time()))
    print('EVAL30_PUBLIC_INPUTS_READY train10+test20',flush=True)


if __name__=='__main__':main()
