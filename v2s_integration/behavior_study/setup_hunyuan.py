"""Prepare a separate shape environment reusing the verified B300 torch wheel."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import importlib.metadata


def main(args):
    target = Path(args.target)
    if target.exists():
        raise RuntimeError('Use a fresh environment path; never modify an existing environment')
    subprocess.run([sys.executable, '-m', 'venv', str(target)], check=True)
    python = target / 'bin/python'
    site = next((target / 'lib').glob('python*/site-packages'))
    base = [p for p in sys.path if p.endswith('site-packages') and Path(p).is_dir()]
    (site / 'verified_b300_runtime.pth').write_text('\n'.join(base) + '\n')
    # Proxy and package-index settings belong to the deployment environment.
    env = dict(os.environ)
    pinned = {name: importlib.metadata.version(name) for name in ('torch', 'torchvision')}
    constraints = target / 'cuda_constraints.txt'
    constraints.write_text('\n'.join(f'{k}=={v}' for k, v in pinned.items()) + '\n')
    requirements = ['transformers==4.46.0', 'diffusers==0.30.0', 'accelerate==1.1.1',
        'huggingface-hub==0.30.2', 'safetensors==0.4.5', 'tokenizers==0.20.3',
        'numpy==1.26.4', 'einops==0.8.0', 'trimesh==4.4.7', 'omegaconf==2.3.0',
        'scikit-image==0.24.0', 'rembg==2.0.65', 'onnxruntime==1.20.1',
        'timm==1.0.12', 'torchdiffeq==0.2.5', 'pymeshlab==2023.12.post3',
        'peft==0.13.2', 'pytorch-lightning==2.5.0']
    subprocess.run([str(python), '-m', 'pip', 'install', '-c', str(constraints), *requirements],
                   env=env, check=True, timeout=1200)
    # Do not let dependency resolution replace the CUDA runtime used by the policy.
    check = subprocess.check_output([str(python), '-c',
        'import torch,json; print(json.dumps({"torch":torch.__version__,"path":torch.__file__}))'], text=True)
    verified = json.loads(check)
    if verified['torch'] != pinned['torch'] or not any(verified['path'].startswith(p + '/') for p in base):
        raise RuntimeError('Shape environment did not preserve the verified B300 PyTorch')
    (target / 'setup_summary.json').write_text(json.dumps({'python': str(python),
        'base_packages': base, 'requirements': requirements, 'torch': json.loads(check)}, indent=2))
    print(json.dumps({'status': 'prepared', 'python': str(python)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--target', required=True)
    main(p.parse_args())
