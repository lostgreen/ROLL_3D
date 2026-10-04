"""Focused coordinator checks using temporary repositories and real subprocesses."""
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v2s_integration.ops.batch import command, prepare, terminate


def test_prepare_patches_only_run_copy_and_overlay_is_importable(tmp_path):
    repo, out = tmp_path / 'repo', tmp_path / 'output'
    addon = repo / 'v2s_integration/runtime/vendor/blender-mcp/addon.py'
    addon.parent.mkdir(parents=True)
    renderer = '            temporary.render.engine = "BLENDER_EEVEE_NEXT" if bpy.app.version >= (4, 2, 0) else "BLENDER_EEVEE"'
    addon.write_text(renderer)
    strategy = repo / 'roll/distributed/strategy/vllm_strategy.py'
    strategy.parent.mkdir(parents=True)
    strategy.write_text('        logprobs=gen_kwargs.get("logprobs", 0),')
    (repo / 'roll/__init__.py').write_text('')
    cuda = tmp_path / 'nvidia'
    (cuda / 'cuda_runtime/lib').mkdir(parents=True)
    image, blend, index = [tmp_path / n for n in ('ref.png', 'initial.blend', 'index.json')]
    for p in (image, blend, index):
        p.write_bytes(b'test fixture')
    mesh = tmp_path / 'candidate.glb'
    mesh.write_bytes(b'fixture mesh')
    index.write_text(json.dumps({'assets': [{'asset_id': 'fixture', 'source_relative_path': mesh.name,
                                            'license': {'spdx': 'CC0-1.0'}}]}))
    source = tmp_path / 'source.json'
    source.write_text(json.dumps({'initial_blend': str(blend), 'reference_image': str(image)}))
    out.mkdir()
    config = dict(repo=str(repo), cuda_libraries=str(cuda), source_task=str(source),
                  blender_bin='/test/blender', asset_index=str(index), generation={'device': 'cuda:0'})
    env, tasks = prepare(config, out)
    assert addon.read_text() == renderer
    assert 'seed=' not in strategy.read_text()
    assert 'CYCLES' in (out / 'runtime/blender-mcp/addon.py').read_text()
    assert 'seed=gen_kwargs.get("seed")' in (out / 'roll_overlay/roll/distributed/strategy/vllm_strategy.py').read_text()
    assert set(tasks) == {'blender', 'hunyuan', 'assets', 'mixed'}
    frozen = json.loads((out / 'asset_index.json').read_text())['assets'][0]
    assert frozen['glb_path'] == str(mesh)
    assert frozen['license']['name'] == 'CC0-1.0'
    assert frozen['sha256']
    actual = subprocess.check_output([sys.executable, '-c', 'import roll; print(roll.__file__)'],
                                     cwd=out, env=env, text=True).strip()
    assert actual == str(out / 'roll_overlay/roll/__init__.py')


def test_default_study_budgets_are_explicit():
    argv = command({'python': '/test/python', 'repo': '/test/repo', 'model': '/test/model'},
                   '/test/task.json', '/test/out', 3)
    for name, value in (('--gpu', '3'), ('--context', '131072'), ('--steps', '40'),
                        ('--episodes', '8'), ('--output-tokens', '8192')):
        assert argv[argv.index(name) + 1] == value


def test_terminate_reaps_live_process_and_tolerates_finished_process():
    proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                            start_new_session=True)
    try:
        terminate(proc)
        assert proc.poll() is not None
        terminate(proc)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_study_config_controls_sampling_and_preflight_overrides():
    config = dict(python='/test/python', repo='/test/repo', model='/test/model',
                  episodes=3, steps=12, seed_start=50, output_tokens=2048, context=32768)
    argv = command(config, '/task.json', '/output', 2)
    for flag, value in (('--episodes', '3'), ('--steps', '12'), ('--seed-start', '50'),
                        ('--output-tokens', '2048'), ('--context', '32768')):
        assert argv[argv.index(flag) + 1] == value
    preflight = command(config, '/task.json', '/preflight', 0, episodes=1, steps=2)
    assert preflight[preflight.index('--episodes') + 1] == '1'
    assert preflight[preflight.index('--steps') + 1] == '2'
