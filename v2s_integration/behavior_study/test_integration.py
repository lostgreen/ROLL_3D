"""Exercise the real providers with bounded fake external Blender/generation IO."""
import base64
import io
import json
from pathlib import Path
import sys
import struct

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from behavior_study.env import ReconstructionEnv
from behavior_study.inputs import render_request, record_request
from behavior_study.runtime import create_session, execute_validated, model_tools, GROUPS, COMMON
from behavior_study.actions import parse_action


def image_payload():
    buffer = io.BytesIO()
    Image.new('RGB', (32, 32), 'blue').save(buffer, format='PNG')
    return {'data': base64.b64encode(buffer.getvalue()).decode(), 'mime': 'image/png'}


class MCP:
    def __init__(self):
        self.calls = []
        self.fail_next_code = False

    def list_tools(self):
        return [{'name': 'execute_blender_code', 'description': 'Execute Blender Python',
                 'input_schema': {'type': 'object', 'properties': {'code': {'type': 'string'}}, 'required': ['code']}},
                {'name': 'render_scene_view', 'description': 'Render scene', 'input_schema': {'type': 'object'}}]

    def call_tool_rich(self, name, args):
        self.calls.append((name, args))
        code = args.get('code', '')
        if name == 'render_scene_view':
            return {'text': 'Current view', 'images': [image_payload()]}
        if 'blendermcp_use_' in code:
            return {'text': 'Code executed successfully: {}'}
        if "'world_bounds'" in code:
            return {'text': 'Code executed successfully: {"objects": []}'}
        if 'sorted(items' in code:
            return {'text': 'Code executed successfully: [{"name":"PublicCamera","type":"CAMERA"}]'}
        if self.fail_next_code:
            self.fail_next_code = False
            return {'text': 'ERROR: Python execution failed', 'isError': True}
        return {'text': 'Code executed successfully: done'}

    def close(self):
        pass


class Backend:
    version = 'test-io-only'
    supports_texture = False

    def generate(self, payload, timeout=None):
        self.payload = payload
        document = json.dumps({'asset': {'version': '2.0'}, 'meshes': [{'primitives': []}]}).encode()
        document += b' ' * (-len(document) % 4)
        return struct.pack('<4sIIII', b'glTF', 2, 20 + len(document), len(document), 0x4E4F534A) + document


@pytest.fixture
def setup(tmp_path):
    reference = tmp_path / 'ref.png'
    Image.new('RGB', (32, 32), 'blue').save(reference)
    index = tmp_path / 'index.json'
    index.write_text(json.dumps({'assets': [{'id': 'test', 'name': 'vase'}]}))
    task = {'id': 'contract', 'group': 'blender', 'asset_index': str(index),
            'budget': {'max_agent_steps': 40, 'max_tool_calls': 40},
            'references': [{'id': 'ref:0', 'image': str(reference)}],
            'observation_camera': 'PublicCamera'}
    return task, {'ref:0': str(reference)}, tmp_path


@pytest.mark.parametrize('group', list(GROUPS))
def test_group_schemas_and_executor_boundaries(setup, group):
    task, refs, root = setup
    task['group'] = group
    mcp = MCP()
    session = create_session(task, mcp, root / 'artifacts', refs,
                             backend=Backend() if group in ('hunyuan', 'mixed') else None)
    names = {x['function']['name'] for x in model_tools(session)}
    assert names == set(COMMON) | set(GROUPS[group])
    before = len(mcp.calls)
    result, log = execute_validated(session, 'download_sketchfab_model', {})
    assert result.error_type == 'tool_not_allowed' and not log['executed']
    assert len(mcp.calls) == before
    result, _ = execute_validated(session, 'execute_blender_code', {'code': 'import bpy\nbpy.ops.mesh.primitive_uv_sphere_add()'})
    assert (result.status == 'succeeded') == (group in ('blender', 'mixed'))
    if group in ('hunyuan', 'mixed'):
        result, log = execute_validated(session, 'asset.generate_3d_from_image', {'reference_image': 'ref:0', 'texture': True})
        assert result.error_type == 'invalid_arguments' and not log['executed']


def test_missing_backend_and_index_fail_closed(setup, monkeypatch):
    task, refs, root = setup
    for name in ('V2S_HUNYUAN_ENDPOINT', 'V2S_HUNYUAN_MODEL_PATH', 'V2S_HUNYUAN_SUBFOLDER'):
        monkeypatch.delenv(name, raising=False)
    task['group'] = 'hunyuan'
    with pytest.raises(ValueError, match='requires an endpoint'):
        create_session(task, MCP(), root / 'a', refs)
    task.update(group='assets', asset_index=str(root / 'missing'))
    with pytest.raises(ValueError, match='existing asset_index'):
        create_session(task, MCP(), root / 'b', refs)


def test_schema_rejection_happens_before_mcp(setup):
    task, refs, root = setup
    mcp = MCP()
    session = create_session(task, mcp, root / 'a', refs)
    count = len(mcp.calls)
    for name, args in [('execute_blender_code', {}), ('finish', {'unknown': True}),
                       ('reference.crop', {'reference_image': 'ref:0', 'bbox': [0, 0, 2, 1]})]:
        result, log = execute_validated(session, name, args)
        assert result.error_type == 'invalid_arguments' and not log['executed']
    assert len(mcp.calls) == count


def test_native_qwen_function_tags_and_typed_arguments(setup):
    task, refs, root = setup
    session = create_session(task, MCP(), root / 'a', refs)
    code = 'import bpy\nfor i in range(2):\n    bpy.ops.mesh.primitive_cube_add()'
    raw = '<tool_call>\n<function=execute_blender_code>\n<parameter=code>\n' + code + '\n</parameter>\n</function>\n</tool_call>'
    assert parse_action('reasoning</think>' + raw, 1, session.registry).arguments['code'] == code
    crop = '<tool_call><function=reference.crop><parameter=reference_image>ref:0</parameter><parameter=bbox>[0,0,1,1]</parameter></function></tool_call>'
    call = parse_action(crop, 2, session.registry)
    assert call.arguments == {'reference_image': 'ref:0', 'bbox': [0, 0, 1, 1]}
    assert parse_action('<tool_call><function=finish></function></tool_call>', 3, session.registry).arguments == {}
    for bad in ['<think>' + raw, raw + ' extra', raw + raw, raw[:-12], crop.replace('</function>', '<parameter=bbox>[]</parameter></function>')]:
        with pytest.raises(ValueError):
            parse_action(bad, 1, session.registry)


def test_crop_reference_reaches_generation_backend(setup):
    task, refs, root = setup
    task['group'] = 'hunyuan'
    backend = Backend()
    session = create_session(task, MCP(), root / 'a', refs, backend=backend)
    crop, _ = execute_validated(session, 'reference.crop', {'reference_image': 'ref:0', 'bbox': [0, 0, .5, .5]})
    ref = crop.structured['reference_image']
    result, _ = execute_validated(session, 'asset.generate_3d_from_image', {'reference_image': ref, 'texture': False, 'seed': 7})
    assert result.status == 'succeeded'
    assert result.artifact_ids and backend.payload['seed'] == 7
    assert base64.b64decode(backend.payload['image']) == Path(session.generation.references[ref]).read_bytes()
    manifest = json.loads((root / 'a' / (result.artifact_ids[0] + '.manifest.json')).read_text())
    assert manifest['evidence_refs'] == [ref]


def make_env(setup, monkeypatch, steps=5):
    task, _, root = setup
    manifest = root / 'task.json'
    manifest.write_text(json.dumps(task))
    env = ReconstructionEnv(manifest, root / 'episodes', max_steps=steps)
    monkeypatch.setattr(env, '_start', lambda: setattr(env, 'mcp', MCP()))
    env.reset(seed=42)
    return env


def action(name, arguments):
    return '<tool_call>' + json.dumps({'name': name, 'arguments': arguments}) + '</tool_call>'


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['add_generation_prompt'] is True
        assert kwargs['tools']
        return json.dumps({'messages': messages, 'tools': kwargs['tools']})


def test_multiturn_crop_images_and_finish(setup, monkeypatch):
    env = make_env(setup, monkeypatch)
    try:
        assert 'no color-only or two-statement restriction' in env.system_prompt
        # First an invalid action, then actual provider crop, then finish.
        _, _, done, truncated, info = env.step('thinking without an action')
        assert not done and not truncated and info['metrics']['invalid_action'] == 1
        _, _, done, truncated, info = env.step(action('reference.crop', {'reference_image': 'ref:0', 'bbox': [0, 0, 1, 1]}))
        assert not done and not truncated and info['metrics']['tool_success'] == 1
        prompt, messages, images, refs, state = render_request(Tokenizer(), env)
        tool = next(m for m in messages if m['role'] == 'tool')
        assistant = next(m for m in messages if m.get('tool_calls'))
        assert tool['tool_call_id'] == assistant['tool_calls'][0]['id']
        assert len(images) == len(refs) == 3  # reference, real crop, current scene
        crop_id = env.turns[-1]['feedback']['artifact_ids'][0]
        assert crop_id in prompt
        folder = env.episode / 'requests' / 'test'
        record_request(folder, prompt, messages, env.model_tools, refs, [1, 2], [1], state)
        assert json.loads((folder / 'input_manifest.json').read_text())['image_count'] == 3
        _, _, done, truncated, _ = env.step(action('finish', {'summary': 'Stop'}))
        assert done and not truncated and env.closed
        assert json.loads((env.episode / 'result.json').read_text())['termination_reason'] == 'finish'
    finally:
        env.close()


def test_error_retry_budget_and_reasoning_not_executed(setup, monkeypatch):
    env = make_env(setup, monkeypatch, steps=3)
    try:
        bad = '<think>' + action('execute_blender_code', {'code': 'DO_NOT_EXECUTE'})
        env.step(bad)
        assert not any('DO_NOT_EXECUTE' in args.get('code', '') for _, args in env.mcp.calls)
        env.mcp.fail_next_code = True
        _, _, done, truncated, _ = env.step(action('execute_blender_code', {'code': 'bad code'}))
        assert not done and not truncated
        _, _, done, truncated, _ = env.step(action('execute_blender_code', {'code': 'import bpy\nbpy.ops.mesh.primitive_cube_add()'}))
        assert not done and truncated
        assert json.loads((env.episode / 'result.json').read_text())['termination_reason'] == 'max_steps'
    finally:
        env.close()


def test_history_pruning_keeps_call_result_pairs(setup, monkeypatch):
    env = make_env(setup, monkeypatch)
    try:
        for _ in range(3):
            env.step(action('scene.list_objects', {}))
        _, messages, _, _, state = render_request(Tokenizer(), env, history_limit=1)
        assert sum(m['role'] == 'tool' for m in messages) == 1
        assert sum(bool(m.get('tool_calls')) for m in messages) == 1
        assert state['omitted_history_turns'] == 2
        _, messages, _, _, _ = render_request(Tokenizer(), env, history_limit=0)
        assert not any(m['role'] == 'tool' for m in messages)
        assert 'public_scene_state' in str(messages)
    finally:
        env.close()


def test_default_history_retains_more_than_six_turns(setup, monkeypatch):
    env = make_env(setup, monkeypatch, steps=12)
    try:
        for _ in range(8):
            env.step(action('scene.list_objects', {}))
        _, messages, _, _, state = render_request(Tokenizer(), env)
        assert sum(m['role'] == 'tool' for m in messages) == 8
        assert state['omitted_history_turns'] == 0
    finally:
        env.close()


def test_template_cannot_discard_parameter_schema(setup, monkeypatch):
    env = make_env(setup, monkeypatch)
    class NamesOnly:
        def apply_chat_template(self, messages, **kwargs):
            return ' '.join(t['function']['name'] for t in kwargs['tools'])
    try:
        with pytest.raises(ValueError, match='full tool schema'):
            render_request(NamesOnly(), env)
    finally:
        env.close()


def test_empty_scene_observation_does_not_add_placeholder(setup, monkeypatch):
    env = make_env(setup, monkeypatch)
    commands = []
    try:
        monkeypatch.setattr(env.mcp, 'call_tool_rich', lambda *a: {
            'text': 'ERROR: no finite visible geometry bounds to inspect', 'isError': True})
        def render(code):
            commands.append(code)
            Image.new('RGB', (32, 32)).save(env.episode / 'empty_observation_000.png')
        monkeypatch.setattr(env, '_code', render)
        refs = env._render()
        assert len(refs) == 1 and Path(refs[0]['path']).is_file()
        assert 'bpy.ops.render.render' in commands[0]
        assert 'primitive_' not in commands[0]
    finally:
        env.close()


def test_roll_manager_collates_images_and_reserves_output(setup, monkeypatch):
    # Load only the integration adapter, replacing unavailable heavy ROLL imports.
    # Its real format_messages method and providers run; this is not a GPU test.
    import importlib.util
    from types import ModuleType, SimpleNamespace
    import numpy as np
    base = ModuleType('roll.pipeline.agentic.env_manager.vl_traj_env_manager')
    base.VLTrajEnvManager = object
    protocol = ModuleType('roll.distributed.scheduler.protocol')
    protocol.DataProto = SimpleNamespace(from_single_dict=lambda data: data)
    monkeypatch.setitem(sys.modules, base.__name__, base)
    monkeypatch.setitem(sys.modules, protocol.__name__, protocol)
    spec = importlib.util.spec_from_file_location('behavior_study._test_manager', Path(__file__).with_name('manager.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = make_env(setup, monkeypatch)
    try:
        env.step(action('scene.list_objects', {}))
        class Collator:
            prompt_key, image_key = 'prompt', 'images'
            def __call__(self, features):
                assert len(features[0]['images']) >= 2
                payload = json.loads(features[0]['prompt'])
                count = sum(m['role'] == 'tool' for m in payload['messages'])
                ids = np.arange(100 + count * 300)[None, :]
                return SimpleNamespace(batch={'input_ids': ids},
                    non_tensor_batch={'multi_modal_data': [{'prompt_token_ids': [1, 2, 3]}]})
        manager = module.ReconstructionManager.__new__(module.ReconstructionManager)
        manager.env, manager.tokenizer, manager.collator = env, Tokenizer(), Collator()
        manager.env_config = {'max_tokens_per_step': 8192}
        manager.worker_config = SimpleNamespace(generating_args=SimpleNamespace(max_new_tokens=8192))
        manager.pipeline_config = SimpleNamespace(sequence_length=8392)
        data, messages = manager.format_messages(None)
        assert data.batch['input_ids'].shape[1] == 100
        assert not any(m['role'] == 'tool' for m in messages)
        manifest = json.loads((manager.request_dir / 'input_manifest.json').read_text())
        assert manifest['state']['omitted_history_turns'] == 1
        assert json.loads((manager.request_dir / 'inference_prompt_ids.json').read_text()) == [1, 2, 3]
        manager.pipeline_config.sequence_length = 8200
        with pytest.raises(module.ContextBudgetExceeded):
            manager.format_messages(None)
        with pytest.raises(RuntimeError, match='training'):
            manager.run()
    finally:
        env.close()
