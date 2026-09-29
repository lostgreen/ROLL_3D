"""B300 four-arm collection coordinator. Logs stay on disk; status is atomic."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

from .demo import write_json
from .resources import gpus

GROUPS = ('blender', 'hunyuan', 'assets', 'mixed')


def freeze_catalog(source, destination):
    source = Path(source)
    raw = json.loads(source.read_text())
    items = raw.get('assets', raw.get('items', [])) if isinstance(raw, dict) else raw
    normalized = []
    for item in items:
        record = dict(item)
        value = next((record.get(k) for k in ('glb_path', 'path', 'file', 'source_relative_path') if record.get(k)), None)
        if not value:
            raise ValueError('Catalog item has no mesh path')
        mesh = Path(value)
        mesh = mesh if mesh.is_absolute() else source.parent / mesh
        if not mesh.is_file():
            raise ValueError('Catalog mesh unavailable: ' + str(mesh))
        record['glb_path'] = str(mesh.resolve())
        record['sha256'] = hashlib.sha256(mesh.read_bytes()).hexdigest()
        license = record.get('license')
        if isinstance(license, dict) and 'name' not in license and license.get('spdx'):
            record['license'] = {**license, 'name': license['spdx']}
        normalized.append(record)
    if not normalized:
        raise ValueError('Catalog is empty')
    write_json(destination, {'assets': normalized, 'source_index': str(source),
               'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()})


def prepare(config, out):
    repo, code = Path(config['repo']), Path(__file__).resolve().parents[1]
    work = repo / 'v2s_integration'
    harness = out / 'harness'
    shutil.copytree(work / 'harness', harness, ignore=shutil.ignore_patterns('__pycache__', '.git'))
    addon = harness / 'blender-mcp/addon.py'
    before = addon.read_text()
    needle = '            temporary.render.engine = "BLENDER_EEVEE_NEXT" if bpy.app.version >= (4, 2, 0) else "BLENDER_EEVEE"'
    if before.count(needle) != 1:
        raise RuntimeError('Original inspection renderer changed; inspect before patching')
    after = before.replace(needle, '            temporary.render.engine = "CYCLES"\n'
                           '            temporary.cycles.device = "CPU"\n            temporary.cycles.samples = 16')
    addon.write_text(after)
    overlay = out / 'roll_overlay'
    shutil.copytree(repo / 'roll', overlay / 'roll', ignore=shutil.ignore_patterns('__pycache__'))
    converter = overlay / 'roll/distributed/strategy/vllm_strategy.py'
    original = converter.read_text()
    seed_line = '        seed=gen_kwargs.get("seed"),'
    if seed_line not in original:
        needle = '        logprobs=gen_kwargs.get("logprobs", 0),'
        if original.count(needle) != 1:
            raise RuntimeError('Sampling converter changed; inspect before patching')
        converter.write_text(original.replace(needle, needle + '\n' + seed_line))
    converter.write_text(converter.read_text().replace('VLLM_PORT_START = 20000',
        'VLLM_PORT_START = int(os.environ.get("V2S_VLLM_PORT_START", "20000"))'))
    write_json(out / 'runtime_patches.json', {
        'renderer': {'original': hashlib.sha256(before.encode()).hexdigest(),
                     'runtime': hashlib.sha256(after.encode()).hexdigest()},
        'sampling_seed': {'original': hashlib.sha256(original.encode()).hexdigest(),
                          'runtime': hashlib.sha256(converter.read_bytes()).hexdigest()}})
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, (code, overlay, repo, work))),
               V2S_HARNESS_ROOT=str(harness), PYTHONDONTWRITEBYTECODE='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               ROLL_OTEL_ENABLED='0', OMP_NUM_THREADS='4')
    env.pop('ROLL_TORCHINDUCTOR_CUDA_ARCH', None)
    libs = sorted(Path(config['cuda_libraries']).glob('*/lib'))
    if not libs:
        raise RuntimeError('Verified CUDA libraries not found')
    env['LD_LIBRARY_PATH'] = ':'.join([*map(str, libs), env.get('LD_LIBRARY_PATH', '')])
    tasks = {}
    source = json.loads(Path(config['source_task']).read_text())
    catalog = out / 'asset_index.json'
    freeze_catalog(config['asset_index'], catalog)
    for group in GROUPS:
        task = {'id': 'scene_pilot_behavior_128k', 'group': group,
                'blender_bin': config['blender_bin'], 'initial_blend': source['initial_blend'],
                'observation_camera': 'EvaluationCamera',
                'references': [{'id': 'ref:0', 'image': source['reference_image']}],
                'image_size': 512, 'budget': {'max_wall_seconds': config.get('episode_wall_seconds', 14400)},
                'asset_index': str(catalog), 'generation': config['generation'],
                'probe': {'bbox': [0.40, 0.03, 0.67, 0.42]}}
        path = out / f'task_{group}.json'
        write_json(path, task)
        tasks[group] = path
    manifest = {str(p.relative_to(code)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (code / 'behavior_study').iterdir() if p.is_file()}
    write_json(out / 'code_manifest.json', manifest)
    # Record actual public input/catalog hashes; no hidden data enters model prompts.
    write_json(out / 'input_hashes.json', {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in (source['reference_image'], source['initial_blend'], config['asset_index'])})
    return env, tasks


def terminate(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def command(config, task, target, gpu, episodes=None, steps=None):
    return [config['python'], '-m', 'behavior_study.demo', '--repo', config['repo'],
            '--task', str(task), '--output', str(target), '--model', config['model'],
            '--gpu', str(gpu), '--seed-start', str(config.get('seed_start', 42)),
            '--context', str(config.get('context', 131072)),
            '--output-tokens', str(config.get('output_tokens', 8192)),
            '--steps', str(config.get('steps', 40) if steps is None else steps),
            '--episodes', str(config.get('episodes', 8) if episodes is None else episodes),
            '--memory-utilization', str(config.get('memory_utilization', .80))]


def run(config, out):
    if config.get('expected_host') and socket.getfqdn() != config['expected_host']:
        raise RuntimeError('Wrong cluster hostname')
    rows = gpus()
    if len(rows) != 8 or any('B300' not in r['name'] for r in rows):
        raise RuntimeError('Expected eight B300 devices')
    concurrent = config.get('allow_background_load', False)
    if not concurrent and any(r['memory_mib'] > 1024 or r['utilization'] for r in rows if r['index'] <= 4):
        raise RuntimeError('Reserved GPUs0..4 are busy')
    if concurrent and any(r['memory_mib'] > config.get('background_memory_limit_mib', 30000)
                          for r in rows if r['index'] <= 4):
        raise RuntimeError('Background allocation exceeds approved coexistence budget')
    write_json(out / 'resource_condition.json', {'background_load': concurrent,
        'initial_gpus': rows, 'timing_is_contended': concurrent,
        'memory_includes_background': concurrent})
    env, tasks = prepare(config, out)
    peak = {r['index']: r['memory_mib'] for r in rows}
    children, handles = {}, []
    started = time.monotonic()

    def start(label, argv, deadline):
        handle = (out / f'{label}.log').open('w')
        handles.append(handle)
        child_env = dict(env, V2S_VLLM_PORT_START=str(20000 + (GROUPS.index(label) if label in GROUPS else 0) * 1000))
        proc = subprocess.Popen(argv, cwd=out, env=child_env, stdout=handle,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        children[label] = (proc, time.monotonic(), deadline)
        return proc

    def tick(stage):
        rows = gpus()
        for row in rows:
            peak[row['index']] = max(peak[row['index']], row['memory_mib'])
        snapshot = {'stage': stage, 'pid': os.getpid(), 'updated_at': time.time(),
                    'elapsed_seconds': round(time.monotonic() - started, 2),
                    'peak_memory_mib': peak, 'gpus': rows,
                    'jobs': {name: {'pid': p.pid, 'returncode': p.poll(),
                                    'seconds': round(time.monotonic() - s, 2)}
                             for name, (p, s, _) in children.items()}}
        with (out / 'gpu_samples.jsonl').open('a') as f:
            f.write(json.dumps(snapshot) + '\n')
        write_json(out / 'status.json', snapshot)
        for name, (proc, start_time, deadline) in children.items():
            if proc.poll() is None and time.monotonic() - start_time > deadline:
                terminate(proc)
                raise TimeoutError(f'{name} exceeded {deadline}s')

    def wait_one(label, argv, deadline):
        proc = start(label, argv, deadline)
        while proc.poll() is None:
            tick(label)
            time.sleep(5)
        if proc.returncode:
            raise RuntimeError(f'{label} exit={proc.returncode}; see {out / (label + ".log")}')

    try:
        # Real mixed probe includes all backend paths; each group also gets a
        # separate contract probe so missing group-specific availability fails early.
        for group in GROUPS:
            wait_one('probe_' + group, [config['python'], '-m', 'behavior_study.backend_probe',
                     '--task', str(tasks[group]), '--output', str(out / 'probes' / group)], config.get('probe_timeout', 2400))
        wait_one('context_preflight', command(config, tasks['blender'], out / 'preflight', 0, 1, 2), config.get('preflight_timeout', 2400))
        write_json(out / 'preflight_passed.json', {'context': config.get('context', 131072),
                   'meaning': 'Model loaded and real requests worked; not a full128K input stress test',
                   'backend_probes': list(GROUPS), 'time': time.time()})
        for gpu, group in enumerate(GROUPS):
            start(group, command(config, tasks[group], out / 'groups' / group, gpu),
                  config.get('group_wall_seconds', 32 * 3600))
            time.sleep(3)
        while any(children[group][0].poll() is None for group in GROUPS):
            tick('collecting')
            time.sleep(10)
        # The CLI is the stable aggregation interface, independent of its Python signature.
        subprocess.run([config['python'], '-m', 'behavior_study.summarize', '--root', str(out / 'groups'),
                        '--output', str(out / 'behavior_summary.json')], env=env, check=True,
                       stdout=subprocess.DEVNULL)
        failures = {g: children[g][0].returncode for g in GROUPS if children[g][0].returncode}
        tick('complete' if not failures else 'completed_with_failures')
        write_json(out / 'result.json', {'state': 'complete' if not failures else 'completed_with_failures',
                   'failures': failures, 'seconds': time.monotonic() - started, 'peak_memory_mib': peak})
        if failures:
            raise RuntimeError('Collection groups failed; inspect result.json and group outcomes')
    finally:
        for proc, _, _ in children.values():
            terminate(proc)
        for handle in handles:
            handle.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--detach', action='store_true')
    p.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    out = Path(args.output).resolve()
    if not args.child:
        out.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(args.config, out / 'launch_config.json')
    if args.detach:
        with (out / 'coordinator.log').open('w') as log:
            proc = subprocess.Popen([sys.executable, '-m', 'behavior_study.run_study',
                '--config', str(out / 'launch_config.json'), '--output', str(out), '--child'],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(out / 'coordinator_pid.json', {'pid': proc.pid, 'started_at': time.time()})
        print(json.dumps({'state': 'launched', 'pid': proc.pid, 'output': str(out)}))
        return
    try:
        run(json.loads(Path(args.config).read_text()), out)
    except BaseException as exc:
        write_json(out / 'failure.json', {'type': type(exc).__name__, 'error': str(exc)[-1000:]})
        raise


if __name__ == '__main__':
    main()
