"""Run the original Hunyuan backend in its own Python/CUDA environment."""
import fcntl
from functools import partial
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid


def terminate_with_parent(parent):
    # Linux B300 worker: kill CUDA work if the lock-owning driver disappears.
    import ctypes
    if ctypes.CDLL(None).prctl(1, signal.SIGKILL) != 0 or os.getppid() != parent:
        os._exit(1)


class IsolatedHunyuanBackend:
    supports_texture = False

    def __init__(self, config, root):
        from tools.hunyuan import HunyuanLocalBackend
        self.config, self.root = dict(config), Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        local = HunyuanLocalBackend(config['model_path'], config['subfolder'], 'cuda:0')
        self.presets = local.presets
        self.version = local.version + ':isolated-process-v1'
        self.last_metadata = {}

    def generate(self, payload, timeout=None):
        start = time.monotonic()
        limit = min(self.config.get('timeout', 1200), timeout or 1200)
        folder = self.root / uuid.uuid4().hex
        folder.mkdir()
        (folder / 'request.json').write_text(json.dumps(payload))
        (folder / 'config.json').write_text(json.dumps(self.config))
        self.last_metadata = {'directory': str(folder), 'device': self.config['gpu'],
                              'cold_load_per_call': True}
        lock_path = Path(self.config['lock_path'])
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open('a') as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() - start >= limit:
                        raise TimeoutError('Hunyuan queue timeout; request not submitted')
                    time.sleep(.25)
            self.last_metadata['queue_seconds'] = round(time.monotonic() - start, 3)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self.config['gpu']),
                       PYTHONPATH=os.pathsep.join([self.config['code_path'], self.config['code_path'] + '/hy3dshape',
                                                  os.environ.get('PYTHONPATH', '')]))
            if self.config.get('rembg_path'):
                env['U2NET_HOME'] = self.config['rembg_path']
            with (folder / 'worker.log').open('w') as log:
                proc = subprocess.Popen([self.config['python'], '-m', 'v2s_integration.runtime.isolated_hunyuan',
                                         str(folder)], env=env, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True, preexec_fn=partial(terminate_with_parent, os.getpid()),
                                        pass_fds=(lock.fileno(),))
                try:
                    proc.wait(timeout=max(.01, limit - (time.monotonic() - start)))
                except BaseException:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    raise
                if proc.returncode:
                    raise RuntimeError(f'Hunyuan worker exit={proc.returncode}; see {folder / "worker.log"}')
            self.last_metadata['seconds'] = round(time.monotonic() - start, 3)
            return (folder / 'mesh.glb').read_bytes()


def worker(folder):
    from . import runtime  # Resolve the frozen harness modules in the child.
    from tools.hunyuan import HunyuanLocalBackend
    folder = Path(folder)
    cfg = json.loads((folder / 'config.json').read_text())
    backend = HunyuanLocalBackend(cfg['model_path'], cfg['subfolder'], 'cuda:0')
    mesh = backend.generate(json.loads((folder / 'request.json').read_text()))
    (folder / 'mesh.glb').write_bytes(mesh)


if __name__ == '__main__':
    import sys
    worker(sys.argv[1])
