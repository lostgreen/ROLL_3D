"""Context v2 observation boundaries and live/archive agreement, without GPUs."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v2s_integration.agent.context_policy import load_context_profile
from v2s_integration.agent.messages import build_messages
from v2s_integration.agent.observation_format import (
    format_feedback, format_observation_messages, format_public_scene_state,
    observation_contract, serialize_observation,
)
from v2s_integration.tests.test_context_policy import environment, turn
from v2s_integration.tests.test_context_profile import profile_inputs, start_env


def test_round4_dedup_coordinates_and_preserve_unknown_camera_fields():
    raw = {'objects': [
        {'name': 'ThinShelf', 'parent': None, 'location': [1.0000001, -0.0, 0.0005],
         'world_location': [1.0000002, -0.0000001, 0.0005],
         'dimensions': [4.900000095367432, 0.0005, 1.0],
         'world_bounds': [[-0.0000001, 0.0005, 1.11111111], [4.9000001, 2.0, 3.0]],
         'matrix_world': [[1.123456789]], 'unknown': 2.123456789},
        {'name': 'Parented', 'parent': 'Root', 'location': [1.0], 'world_location': [1.0]},
        {'name': 'UnknownParent', 'location': [1.0], 'world_location': [1.0]},
        {'name': 'Constrained', 'parent': None, 'location': [1.0], 'world_location': [2.0]},
        {'name': 'PublicCamera_A', 'parent': None, 'location': [1.123456789],
         'world_location': [1.123456789], 'dimensions': [0.123456789]},
        {'name': 'OtherCamera', 'type': 'CAMERA', 'location': [1.123456789]},
    ], 'public_camera_protocol': {'location': [1.123456789]}}
    before = copy.deepcopy(raw)
    result = format_public_scene_state(raw)
    assert raw == before
    thin = result['objects'][0]
    assert thin['location'] == [1, 0, 0.0005]
    assert thin['dimensions'] == [4.9, 0.0005, 1]
    assert 'world_location' not in thin
    assert thin['world_bounds'] == [[0, 0.0005, 1.1111], [4.9, 2, 3]]
    assert thin['matrix_world'] == before['objects'][0]['matrix_world']
    assert thin['unknown'] == before['objects'][0]['unknown']
    assert all('world_location' in obj for obj in result['objects'][1:4])
    assert result['objects'][4:] == before['objects'][4:]
    assert result['public_camera_protocol'] == before['public_camera_protocol']
    rendered = serialize_observation(thin)
    assert '-0.0' not in rendered and ': ' not in rendered
    assert format_public_scene_state(result) == result


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_reports_field_path_even_in_unrounded_fields(value):
    with pytest.raises(ValueError, match=r'public_scene_state.objects\[0\].matrix_world\[0\]'):
        format_public_scene_state({'objects': [{'matrix_world': [value]}]})


def test_feedback_formats_only_canonical_structured_delta_and_keeps_free_text():
    stdout = 'ERROR 1.123456789; stdout JSON {"location":[2.123456789]}\n'
    delta = {'changed': [{'name': 'Thin', 'delta_location': [-0.0000001, 0.0005, 1.11111111],
                          'new_dimensions': [2.00000001, 3.123456789]}],
             'counts': {'added': 0}, 'unknown_number': 9.123456789}
    payload = {'structured': {'nested': {'scene_delta': delta}, 'other': {'location': [8.123456789]}}}
    original = {'text': stdout + serialize_observation(payload), 'status': 'failed',
                'error_type': 'tool_error', 'call_id': 'call_1', 'artifact_ids': ['artifact:7']}
    before = copy.deepcopy(original)
    result = format_feedback(original)
    assert original == before
    assert result['text'].startswith(stdout)
    result_payload = json.loads(result['text'][len(stdout):])
    changed = result_payload['structured']['nested']['scene_delta']['changed'][0]
    assert changed['delta_location'] == [0, 0.0005, 1.1111]
    assert changed['new_dimensions'] == [2, 3.1235]
    assert result_payload['structured']['other'] == payload['structured']['other']
    assert result_payload['structured']['nested']['scene_delta']['unknown_number'] == 9.123456789
    assert {k: v for k, v in result.items() if k != 'text'} == {k: v for k, v in original.items() if k != 'text'}
    assert format_feedback(result) == result
    for text in [stdout, '{"structured":{"scene_delta":{"location":[1.123456789]}}}',
                 'literal {"structured":{"scene_delta":{"location":[1.123456789]}}} trailing',
                 'ordinary error\n{"structured":not JSON}']:
        assert format_feedback({'text': text})['text'] == text


def test_live_v2_matches_transformed_archive_without_changing_calls_or_images(profile_inputs, tmp_path):
    _, _, v1 = profile_inputs
    v2 = copy.deepcopy(v1)
    v2['policy_version'] = 'raw_cot_context_v2'
    v2['observation_format'] = observation_contract()
    env = environment(tmp_path)
    env.context_profile = v1
    env.public_state = {'objects': [{'name': 'Box', 'parent': None, 'location': [1.123456789, -0.0, 0.0005],
                                    'world_location': [1.123456789, -0.0, 0.0005]}]}
    env.task_prompt += '\nPublic camera protocol: {"matrix_world":[[1.123456789]]}'
    env.turns = [turn(1, tool='execute_blender_code', feedback='raw 1.123456789\n' +
                      serialize_observation({'structured': {'scene_delta': {
                          'changed': [{'name': 'Box', 'delta_location': [0.123456789]}]}}}))]
    env.turns[0]['call']['arguments']['code'] = 'x=1.123456789; print("-0.0")'
    original = build_messages(env)
    env.context_profile = v2
    before = copy.deepcopy(env.__dict__)
    live = build_messages(env)
    assert env.__dict__ == before
    assert live[0] == format_observation_messages(original[0])
    assert live[0][:3] == original[0][:3]
    assert [m for m in live[0] if m['role'] == 'assistant'] == [m for m in original[0] if m['role'] == 'assistant']
    assert live[2] == original[2]
    assert [image.tobytes() for image in live[1]] == [image.tobytes() for image in original[1]]
    assert live[3]['context_policy']['version'] == 'raw_cot_context_v2'
    assert original[3]['context_policy']['version'] == 'raw_cot_context_v1'
    assert 'raw_cot_context_v2' not in json.dumps(live[0])


def test_v2_profile_requires_current_policy_and_all_shared_source_hashes(profile_inputs, tmp_path):
    _, _, v1 = profile_inputs
    v2 = copy.deepcopy(v1)
    v2.update(policy_version='raw_cot_context_v2', observation_format=observation_contract())
    assert load_context_profile(v1) == v1
    assert load_context_profile(v2) == v2
    for mutate in [lambda p: p['observation_format'].update(decimals=3),
                   lambda p: p['observation_format']['context_sources'].update({'messages.py': 'stale'}),
                   lambda p: p.update(policy_version='raw_cot_context_v3'),
                   lambda p: p.pop('observation_format')]:
        invalid = copy.deepcopy(v2)
        mutate(invalid)
        with pytest.raises(ValueError):
            load_context_profile(invalid)
    env = environment(tmp_path)
    env.context_profile = v2
    env.context_policy_version = 'raw_cot_context_v1'
    with pytest.raises(ValueError, match='differs from its explicit profile'):
        build_messages(env)


def test_real_environment_reset_and_tool_feedback_use_v2(profile_inputs, tmp_path, monkeypatch):
    _, task_path, profile = profile_inputs
    profile.update(policy_version='raw_cot_context_v2', observation_format=observation_contract())
    env = start_env(tmp_path, monkeypatch, task_path, profile)
    try:
        from v2s_integration.tests.test_integration import action
        env.step(action('execute_blender_code', {'code': 'print("1.123456789")'}))
        env.public_state = {'objects': [{'name': 'Thin', 'parent': None, 'location': [0.0005, -0.0, 1.123456789],
                                        'world_location': [0.0005, -0.0, 1.123456789]}]}
        env.turns[-1]['feedback']['text'] = 'error details 1.123456789\n' + serialize_observation({
            'structured': {'scene_delta': {'changed': [{'name': 'Thin', 'delta_location': [0.123456789]}]}}})
        messages, _, _, state = build_messages(env)
        assert state['context_policy']['version'] == 'raw_cot_context_v2'
        visible = json.loads(messages[-1]['content'][len('Current public scene state: '):])
        assert visible['public_scene_state']['objects'][0]['location'] == [0.0005, 0, 1.1235]
        assert 'world_location' not in visible['public_scene_state']['objects'][0]
        feedback = json.loads(next(m['content'] for m in messages if m['role'] == 'tool'))
        assert feedback['text'].startswith('error details 1.123456789\n')
        assert '"delta_location":[0.1235]' in feedback['text']
    finally:
        env.close()
