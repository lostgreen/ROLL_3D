"""Validate chained evaluation dependencies before the long training completes."""
import ast
import os
import sys
from chain_common import *

def main():
    cfg=read(EVAL/'config.json');checks={}
    for key in ['repo','runtime_root','cpu_runtime','blender','context_profile','train_source']:
        p=Path(cfg[key]);assert p.exists(),(key,str(p));checks[key]=str(p)
    assert (Path(cfg['cpu_runtime'])/'headless/blender_process.py').is_file()
    source=Path(cfg['repo'])/'artifacts/indoor_qwen_200rollout_20261121/source/cutaway_preview.py'
    assert source.is_file()
    for p in CODE.rglob('*.py'):ast.parse(p.read_text())
    from gpu_probe import bootstrap
    bootstrap(Path(cfg['repo']))
    sys.path.insert(0,str(EVAL/'cpu_vendor'))
    from pack_raw_cot import setup
    processor=setup(cfg['runtime_root'],expected_version='raw_cot_context_v2')
    from v2s_integration.envs.reconstruction import ReconstructionEnv,VENDOR
    from v2s_integration.agent.messages import render_request
    from mcp_client import BlenderMCPClient
    import mcp
    assert len(cfg['rows'])==30
    for r in cfg['rows']:
        task=read(EVAL/'public'/r['id']/'task.json')
        assert Path(task['initial_blend']).is_file()
        assert all(Path(x['image']).is_file() for x in task['references'])
        if r['subset']=='train10_diagnostic':assert len(r['private_views'])==2
    dump(ROOT/'chain_cpu_preflight.json',dict(passed=True,dependencies=checks,cases=30,
         agent_imports=True,blender_mcp_imports=True,patched_model_sources=False,at=time.time()))
    print('CHAIN_CPU_PREFLIGHT_PASS')

if __name__=='__main__':main()
