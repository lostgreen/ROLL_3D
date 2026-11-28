"""Raw-CoT context contracts without models, torch, Ray or Blender."""
import ast
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v2s_integration.agent.messages import build_messages, record_request


def image(tmp_path, name, camera=None, color=None, **metadata):
    path = tmp_path / (name + '.png')
    Image.new('RGB', (8, 8), color or (len(list(tmp_path.glob('*.png'))) * 20, 0, 0)).save(path)
    return dict(path=str(path), label=name, **({'camera_id': camera} if camera else {}), **metadata)


def turn(n, *, tool='render_scene_view', images=(), scene=(), camera='A', feedback='ok', **extra):
    return dict(step=n, call={'id': f'call_{n}', 'name': tool,
                'arguments': {'camera': camera, 'view': 'camera'} if tool == 'render_scene_view' else {'code': f'code_{n}'}},
                feedback={'text': feedback, 'status': 'succeeded', 'call_id': f'call_{n}'},
                tool_images=list(images), scene_images=list(scene),
                reasoning_content=f'OLD_PRIVATE_COT_{n}', content=f'OLD_ASSISTANT_BODY_{n}', **extra)


def environment(tmp_path):
    return SimpleNamespace(system_prompt='System', task_prompt='Task with fixed initial scene state',
        task={'observation_camera': 'A'}, reference_refs=[image(tmp_path, 'refA'), image(tmp_path, 'refB')],
        initial_images=[image(tmp_path, 'S0', camera='A')], turns=[], public_state={'objects': []})


def test_latest_each_camera_crop_and_all_history(tmp_path):
    env = environment(tmp_path)
    a1, b2, a3, crop = [image(tmp_path, name) for name in ('A1', 'B2', 'A3', 'crop4')]
    env.turns = [turn(1, images=[a1]), turn(2, images=[b2], camera='B'),
                 turn(3, images=[a3]), turn(4, tool='reference.crop', images=[crop])]
    env.public_state = {'objects': [{'name': 'cabinet'}]}
    before = copy.deepcopy(env.__dict__)
    messages, pictures, refs, state = build_messages(env)
    assert env.__dict__ == before
    assert [r['label'] for r in refs] == ['refA', 'refB', 'B2', 'A3', 'crop4']
    assert len(pictures) == 5
    assert state['context_policy']['render_selection'] == 'latest_per_camera'
    assert state['omitted_history_turns'] == 0
    assert len([m for m in messages if m['role'] == 'tool']) == 4
    text = json.dumps(messages, ensure_ascii=False)
    assert 'OLD_PRIVATE_COT' not in text and 'OLD_ASSISTANT_BODY' not in text
    assert '历史渲染图已省略' in text and 'fixed initial scene state' in text
    assert 'cabinet' in text
    assert 'raw_cot_context_v1' not in text  # audit-only metadata
    assert str(tmp_path) not in text
    assert json.loads(messages[-1]['content'].split(': ', 1)[1])['public_scene_state'] == env.public_state
    for a, feedback in zip([m for m in messages if m['role'] == 'assistant'],
                           [m for m in messages if m['role'] == 'tool']):
        assert a['content'] == '' and feedback['tool_call_id'] == a['tool_calls'][0]['id']


def test_crop_expires_after_next_step_without_dropping_its_result(tmp_path):
    env = environment(tmp_path)
    crop = image(tmp_path, 'crop')
    env.turns = [turn(1, tool='reference.crop', images=[crop])]
    assert 'crop' in [r['label'] for r in build_messages(env)[2]]
    env.turns.append(turn(2, tool='scene.list_objects'))
    messages, _, refs, _ = build_messages(env)
    assert 'crop' not in [r['label'] for r in refs]
    assert '历史参考裁图已省略' in json.dumps(messages, ensure_ascii=False)
    assert [m['tool_call_id'] for m in messages if m['role'] == 'tool'] == ['call_1', 'call_2']


def test_repeated_current_image_does_not_become_new_or_evict_view_b(tmp_path):
    env = environment(tmp_path)
    a, b = image(tmp_path, 'A1', camera='A'), image(tmp_path, 'B2')
    env.turns = [turn(1, tool='execute_blender_code', scene=[a]),
                 turn(2, camera='B', images=[b], scene=[a]),
                 turn(3, tool='scene.list_objects', scene=[a])]
    _, _, refs, state = build_messages(env)
    assert [r['label'] for r in refs] == ['refA', 'refB', 'A1', 'B2']
    a_ref = next(r for r in refs if r['label'] == 'A1')
    assert a_ref['generated_step'] == 1
    assert len(state['context_policy']['images']) == 3  # S0, A1, B2


def test_fallback_latest_two_distinct_images_and_no_fake_camera(tmp_path):
    env = environment(tmp_path)
    env.task = {}
    a, duplicate, b = (image(tmp_path, 'a', color='red'), image(tmp_path, 'dup', color='red'),
                       image(tmp_path, 'b', color='green'))
    env.turns = [turn(1, tool='execute_blender_code', scene=[a]),
                 turn(2, tool='execute_blender_code', scene=[duplicate]),
                 turn(3, tool='execute_blender_code', scene=[b])]
    _, _, refs, state = build_messages(env)
    assert [r['label'] for r in refs] == ['refA', 'refB', 'dup', 'b']
    assert state['context_policy']['fallback_reason'] == 'missing_camera_metadata'


def test_configured_camera_does_not_change_archive_vs_live_selection(tmp_path):
    live = environment(tmp_path)
    live.initial_images[0].pop('camera_id')
    b, a2, a3 = [image(tmp_path, name) for name in ('B1', 'unknown2', 'unknown3')]
    live.turns = [turn(1, camera='B', images=[b]),
                  turn(2, tool='execute_blender_code', scene=[a2]),
                  turn(3, tool='execute_blender_code', scene=[a3])]
    archive = copy.deepcopy(live)
    archive.task = {}
    live_messages, _, live_refs, live_state = build_messages(live)
    archive_messages, _, archive_refs, archive_state = build_messages(archive)
    assert live_messages == archive_messages
    assert live_refs == archive_refs and live_state == archive_state
    assert [r['label'] for r in live_refs] == ['refA', 'refB', 'unknown2', 'unknown3']
    assert live_state['context_policy']['render_selection'] == 'latest_two_fallback'


def test_reference_slots_not_deduplicated_and_initial_is_visible(tmp_path):
    env = environment(tmp_path)
    env.reference_refs[1] = dict(env.reference_refs[0], label='refB')
    messages, _, refs, _ = build_messages(env)
    assert [r['label'] for r in refs] == ['refA', 'refB', 'S0']
    assert 'Initial scene S0' in json.dumps(messages)


def test_feedback_is_complete_including_rejected_turn_and_no_new_failed_image(tmp_path):
    env = environment(tmp_path)
    long_result = 'ERROR_PREFIX_' + 'x' * 9000 + '_ERROR_SUFFIX'
    first = turn(1, feedback=long_result)
    first['feedback']['status'] = 'failed'
    second = turn(2, feedback='y' * 2500 + '_REJECT_SUFFIX')
    second['call'] = None
    second['feedback']['status'] = 'rejected'
    env.turns = [first, second]
    messages, _, refs, _ = build_messages(env)
    text = json.dumps(messages, ensure_ascii=False)
    assert long_result in text and '_REJECT_SUFFIX' in text
    assert [r['label'] for r in refs] == ['refA', 'refB', 'S0']
    assert '历史渲染图已省略' not in text
    with pytest.raises(ValueError, match='complete tool history'):
        build_messages(env, history_limit=1)


def test_reject_future_image_or_missing_history(tmp_path):
    env = environment(tmp_path)
    future = image(tmp_path, 'future', generated_step=2)
    env.turns = [turn(1, images=[future])]
    with pytest.raises(ValueError, match='outside the visible history'):
        build_messages(env)
    env.turns = [turn(2)]
    with pytest.raises(ValueError, match='all completed turns'):
        build_messages(env)


def test_manager_budget_failure_records_full_request_without_retrying_shorter_history(tmp_path):
    # Execute the production method in isolation from Ray/torch imports. The
    # collator output represents the already-expanded text+vision token length.
    path = Path(__file__).parents[1] / 'rollout' / 'manager.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'ReconstructionManager')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'format_messages')
    calls, events = [], []
    class TokenRow:
        def tolist(self):
            return list(range(70))
    class TokenMatrix:
        shape = (1, 70)
        def __getitem__(self, key):
            return TokenRow()
    data = SimpleNamespace(batch={'input_ids': TokenMatrix()}, non_tensor_batch={})
    class Collator:
        prompt_key = 'prompt'
        image_key = 'image'
        def __call__(self, features):
            return features
    def render(*args):
        calls.append(args)
        return 'full prompt', [{'role': 'user', 'content': 'full history'}], [], [], {'omitted_history_turns': 0}
    error = type('ContextBudgetExceeded', (RuntimeError,), {})
    namespace = {'render_request': render, 'record_request': record_request, 'json': json,
                 'DataProto': SimpleNamespace(from_single_dict=lambda _: data), 'ContextBudgetExceeded': error}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    manager = SimpleNamespace(env=SimpleNamespace(max_output_tokens=40, turns=[{}, {}, {}],
        episode=tmp_path, step_count=3, model_tools=[], recorder=SimpleNamespace(event=lambda *a, **kw: events.append((a, kw)))),
        env_config={'max_tokens_per_step': 40}, worker_config=SimpleNamespace(generating_args=SimpleNamespace(max_new_tokens=40)),
        pipeline_config=SimpleNamespace(sequence_length=100), tokenizer=object(), collator=Collator())
    with pytest.raises(error, match='Complete history'):
        namespace['format_messages'](manager, None)
    assert len(calls) == 1 and len(calls[0]) == 2  # no shortened-history attempt
    receipt = json.loads((tmp_path / 'requests/004/context_budget_exceeded.json').read_text())
    assert receipt['prompt_tokens'] == 70 and receipt['omitted_history_turns'] == 0
    assert receipt['history_turns'] == 3 and events[0][0] == ('context_budget_exceeded',)
