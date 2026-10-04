"""Collection contracts that do not require torch/Ray or a model download."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v2s_integration.rollout.collect import episode_status, validate_args, validate_gpu_visibility, write_json
from v2s_integration.rollout.seeded_proxy import SeededPolicyProxy, sampling_seed, verify_seed_converter


def args(**changes):
    return SimpleNamespace(**dict(dict(gpu=3, context=131072, output_tokens=8192,
        steps=40, episodes=8, max_images=8, memory_utilization=.65, seed_start=42), **changes))


def test_study_config_validation():
    validate_args(args())
    for changes in ({'steps': 41}, {'steps': 0}, {'gpu': 8}, {'output_tokens': 0},
                    {'context': 8192}, {'max_images': 0}, {'episodes': 0}, {'seed_start': -1}):
        with pytest.raises(ValueError):
            validate_args(args(**changes))


def test_gpu_mapping_refuses_ambiguous_parent_mask():
    validate_gpu_visibility({})
    validate_gpu_visibility({'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'})
    for mask in ('3', '', '3,2,1,0,4,5,6,7', 'GPU-uuid'):
        with pytest.raises(ValueError):
            validate_gpu_visibility({'CUDA_VISIBLE_DEVICES': mask})


def test_seed_pairing_reaches_native_proxy_and_is_recorded(tmp_path):
    calls = []
    delegate = SimpleNamespace(generate=lambda **kw: calls.append(kw) or 'response')
    original = {'temperature': .7}
    manager = SimpleNamespace(sampling_episode_seed=42, env=SimpleNamespace(step_count=3), request_dir=tmp_path)
    proxy = SeededPolicyProxy(delegate, manager)
    assert proxy.generate([], 'native-data', original) == 'response'
    assert calls[0]['generation_config']['seed'] == 42003
    assert calls[0]['lm_input'] == 'native-data'
    assert 'seed' not in original
    assert json.loads((tmp_path / 'sampling_seed.json').read_text())['sampling_seed'] == 42003
    manager.sampling_episode_seed = 43
    proxy.generate([], 'native-data', original)
    assert calls[-1]['generation_config']['seed'] == 43003
    assert len({sampling_seed(s, t) for s in range(42, 50) for t in range(40)}) == 320


def test_stale_converter_is_rejected():
    verify_seed_converter(lambda cfg: {'seed': cfg['seed']})
    with pytest.raises(RuntimeError, match='does not forward'):
        verify_seed_converter(lambda cfg: {})


def test_failed_or_shortened_collection_is_not_silently_successful():
    for reason in ('finish', 'max_steps'):
        assert episode_status({'termination_reason': reason}) == 'collected'
    for reason in ('context_budget_exhausted', 'budget_exceeded'):
        assert episode_status({'termination_reason': reason}) == 'limited'
    for reason in ('environment_error', 'generation_aborted', 'environment_contract_violation', None):
        assert episode_status({'termination_reason': reason}) == 'failed'


def test_progress_replacement_is_complete_json(tmp_path):
    path = tmp_path / 'status.json'
    write_json(path, {'stage': 'first'})
    write_json(path, {'stage': 'second', 'completed': 8})
    assert json.loads(path.read_text()) == {'stage': 'second', 'completed': 8}
    assert not path.with_suffix('.json.tmp').exists()
