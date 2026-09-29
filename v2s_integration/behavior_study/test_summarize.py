"""Synthetic behavior traces cover successful, failing, and interrupted collection."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from behavior_study.summarize import action_hash, atomic_write, summarize


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def episode(root, group='mixed', seed=42):
    path = root / group / 'episodes' / f'{group}_{seed}'
    write(path / 'manifest.json', {'group': group, 'seed': seed, 'max_output_tokens': 4})
    return path


def step(path, n, name, status='succeeded', error=None, objects=1, argument='a'):
    write(path / f'step_{n:03}' / 'event.json', {
        'step': n, 'call': {'id': str(n), 'name': name, 'arguments': {'code': argument}},
        'feedback': {'tool_name': name, 'status': status, 'error_type': error,
                     'artifact_ids': ['asset:1'] if status == 'succeeded' else []},
        'scene_state': {'objects': [{}] * objects}})
    write(path / 'requests' / f'{n:03}' / 'input_manifest.json', {
        'input_token_count': n * 10, 'image_count': 2,
        'state': {'omitted_history_turns': 0, 'public_scene_state': {'objects': []}}})
    write(path / 'requests' / f'{n:03}' / 'generation.json', {
        'response_ids': [1, 2, 3, 4], 'decision_seconds': 2})


def test_completed_trace_aggregates_without_exposing_arguments(tmp_path):
    path = episode(tmp_path)
    step(path, 1, 'asset.generate_3d_from_image', argument='PRIVATE_MODEL_CODE')
    step(path, 2, 'scene.import_artifact', objects=2)
    step(path, 3, 'finish', objects=2)
    write(path / 'result.json', {'termination_reason': 'finish', 'steps': 3})
    summary = summarize(tmp_path)
    trace = summary['trajectories'][0]
    assert trace['state'] == 'collected'
    assert trace['object_delta'] == 2
    assert trace['artifact_id_count'] == 1
    assert trace['generation_tool_successes'] == trace['import_tool_successes'] == 1
    assert trace['response_id_at_output_cap_approx'] == 3
    assert trace['response_ids_per_decision_second'] == 2
    assert trace['input_tokens_max'] == 30
    assert summary['groups']['mixed']['finish_succeeded'] == 1
    assert summary['groups']['blender']['episodes_started'] == 0
    assert 'PRIVATE_MODEL_CODE' not in json.dumps(summary)
    assert 'response_ids"' not in json.dumps(summary)


def test_repeated_failure_recovery_and_parse_errors_are_separate(tmp_path):
    path = episode(tmp_path, 'blender')
    step(path, 1, 'execute_blender_code', 'failed', 'blender_error')
    step(path, 2, 'execute_blender_code', 'failed', 'blender_error', argument='a  \n')
    step(path, 3, 'execute_blender_code')
    step(path, 4, None, 'rejected', 'invalid_action')
    step(path, 5, 'finish')
    write(path / 'result.json', {'termination_reason': 'finish', 'steps': 5})
    trace = summarize(tmp_path)['trajectories'][0]
    assert trace['repeated_identical_failed_actions'] == 1
    assert trace['execution_failures'] == 2
    assert trace['invalid_actions'] == 1
    assert trace['immediate_recovery']['same_tool_success'] == 1
    assert trace['immediate_recovery']['different_tool_success'] == 1


def test_partial_files_and_missing_results_never_claim_success(tmp_path):
    path = episode(tmp_path, 'assets')
    step(path, 1, 'asset.search')
    (path / 'step_002').mkdir()
    (path / 'step_002' / 'event.json').write_text('{')
    (path / 'result.json').write_text('{')
    trace = summarize(tmp_path)['trajectories'][0]
    assert trace['state'] == 'in_progress_or_interrupted'
    assert trace['observed_steps'] == 1
    assert trace['incomplete_step_records'] == 1
    assert trace['unreadable_records'] == 2
    assert not trace['result_present']
    write(path.parents[1] / 'outcomes.json', [{'episode': str(path), 'status': 'failed'}])
    assert summarize(tmp_path)['trajectories'][0]['state'] == 'failed'


def test_normalization_ignores_call_id_but_preserves_code_indentation():
    a = {'id': '1', 'name': 'code', 'arguments': {'code': 'if True:\n x()'}}
    b = {'id': '2', 'name': 'code', 'arguments': {'code': 'if True:\n x()  \n'}}
    assert action_hash(a) == action_hash(b)
    b['arguments']['code'] = 'if True:\nx()'
    assert action_hash(a) != action_hash(b)


def test_cleanup_failure_overrides_completed_result(tmp_path):
    path = episode(tmp_path)
    step(path, 1, 'finish')
    write(path / 'result.json', {'termination_reason': 'finish', 'steps': 1})
    write(path.parents[1] / 'outcomes.json', [{'episode': str(path), 'status': 'failed',
                                            'cleanup_error': {'type': 'RuntimeError'}}])
    trace = summarize(tmp_path)['trajectories'][0]
    assert trace['state'] == 'failed'
    assert trace['finish_succeeded'] == 1


def test_atomic_output_replaces_and_leaves_no_temporary_files(tmp_path):
    path = tmp_path / 'summary.json'
    atomic_write(path, '{"first": true}')
    atomic_write(path, '{"second": true}')
    assert json.loads(path.read_text()) == {'second': True}
    assert list(tmp_path.iterdir()) == [path]
