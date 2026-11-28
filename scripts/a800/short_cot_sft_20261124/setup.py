"""Task-local A800 environment setup; never modifies the existing Python env."""
import importlib.metadata as meta
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path('/data/zgw/short_cot_sft_20261124')
PY = '/m2v_intern/xuboshen/zgw/venvs/roll3d_native_20261113/bin/python'


def dump(name, value):
    ROOT.mkdir(parents=True, exist_ok=True)
    p = ROOT / name
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    t.replace(p)


def run(cmd, log, env=None):
    with (ROOT / log).open('a') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, check=True)


def main():
    dump('setup_status.json', dict(stage='cuda_probe', pid=os.getpid(), at=time.time()))
    try:
        import torch
        x = torch.ones((32, 32), device='cuda', dtype=torch.bfloat16)
        assert float((x @ x).mean()) == 32
        torch.cuda.synchronize()
        dump('cuda_probe.json', dict(passed=True, torch=torch.__version__, cuda=torch.version.cuda,
             gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
             python=sys.executable))
        del x
        env = dict(os.environ, http_proxy='http://10.66.28.231:11080',
                   https_proxy='http://10.66.28.231:11080',
                   no_proxy='localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com',
                   MAX_JOBS='4', TORCH_CUDA_ARCH_LIST='8.0', CUDA_HOME='/usr/local/cuda',
                   PIP_CACHE_DIR=str(ROOT / 'pip_cache'), PYTHONDONTWRITEBYTECODE='1')
        vendor = ROOT / 'vendor_fast'
        if not (ROOT / 'fast_install.ok').exists():
            dump('setup_status.json', dict(stage='install_fast_kernels', pid=os.getpid(), at=time.time()))
            run([PY, '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '--target', str(vendor),
                 '-i', 'https://pypi.tuna.tsinghua.edu.cn/simple', 'flash-linear-attention==0.4.1',
                 'fla-core==0.4.1', 'causal-conv1d==1.5.3.post1'], 'fast_install.log', env)
            (ROOT / 'fast_install.ok').write_text('installed\n')
        sys.path.insert(0, str(vendor))
        import fla
        import causal_conv1d
        from transformers.models.qwen3_5 import modeling_qwen3_5 as q
        assert q.is_fast_path_available
        # Exercise the compiled extension on the actual SM80 device.
        xx = torch.randn(1, 32, 32, device='cuda', dtype=torch.bfloat16)
        ww = torch.randn(32, 4, device='cuda', dtype=torch.bfloat16)
        yy = causal_conv1d.causal_conv1d_fn(xx, ww, activation='silu')
        assert yy.shape == xx.shape and bool(torch.isfinite(yy).all())
        torch.cuda.synchronize()
        dump('fast_kernel_receipt.json', dict(passed=True, target=str(vendor),
             fast_path_available=bool(q.is_fast_path_available),
             versions={n: meta.version(n) for n in ['torch', 'transformers', 'flash-linear-attention', 'fla-core', 'causal-conv1d']},
             causal_conv_sm80_passed=True))
        dump('setup_status.json', dict(stage='complete', passed=True, pid=os.getpid(), at=time.time()))
        print('A800_TRAIN_ENV_READY', flush=True)
    except BaseException as e:
        (ROOT / 'setup_traceback.log').write_text(traceback.format_exc())
        dump('setup_status.json', dict(stage='failed', error_type=type(e).__name__, message=str(e)[:240], at=time.time()))
        raise


if __name__ == '__main__':
    main()
