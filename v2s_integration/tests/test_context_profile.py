"""Exercise the explicit context profile through the real environment reset."""
import ast
import copy
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v2s_integration.agent.context_policy import PROFILE_TOOLS, context_profile_sha256, load_context_profile
from v2s_integration.agent.messages import build_messages, render_request
from v2s_integration.envs.reconstruction import ReconstructionEnv
from v2s_integration.runtime.session import create_session, model_tools
from v2s_integration.tests.test_integration import MCP, Tokenizer, action


class RenderingMCP(MCP):
    def call_tool_rich(self, name, args):
        result = super().call_tool_rich(name, args)
        code = args.get('code', '')
        if name == 'execute_blender_code' and 'bpy.ops.render.render(write_still=True)' in code:
            path = ast.literal_eval(re.search(r'^s\.render\.filepath=(.+)$', code, re.M).group(1))
            Image.new('RGB', (640, 480), 'blue').save(path)
        return result


@pytest.fixture
def profile_inputs(tmp_path):
    paths = []
    for n in range(2):
        path = tmp_path / f'reference_{n}.png'
        Image.new('RGB', (32, 32), ('red', 'green')[n]).save(path)
        paths.append(path)
    task = {'id': 'context-profile', 'group': 'blender',
            'budget': {'max_agent_steps': 8, 'max_tool_calls': 8},
            'references': [{'id': f'ref:{n}', 'image': str(path)} for n, path in enumerate(paths)],
            'observation_camera': 'ConfiguredCamera', 'instruction': 'Build the public room.',
            'public_camera_protocol': {'cameras': [{'name': 'CameraA'}, {'name': 'CameraB'}]},
            'private_gt': '/never-include/private.blend'}
    session = create_session(task, MCP(), tmp_path / 'tool_schema',
                             {f'ref:{n}': str(path) for n, path in enumerate(paths)})
    tools = [t for t in model_tools(session) if t['function']['name'] in PROFILE_TOOLS]
    for tool in tools:
        tool['function']['description'] = 'Public teacher description for ' + tool['function']['name']
    profile = {'policy_version': 'raw_cot_context_v1', 'system_prompt': 'Frozen Raw-CoT system instructions.',
               'tools': tools, 'initial_scene_description': 'Initial scene S0 has cameras and lights only.',
               'task_prompt_mode': 'public_instruction_camera_v1', 'allow_camera_light_edits': True,
               'render_mode': 'active_camera_640x480_cycles_v1'}
    manifest = tmp_path / 'task.json'
    manifest.write_text(json.dumps(task))
    return task, manifest, profile


def start_env(tmp_path, monkeypatch, task_path, profile=None, suffix='live'):
    env = ReconstructionEnv(task_path, tmp_path / suffix, max_steps=8, context_profile=profile)
    monkeypatch.setattr(env, '_start', lambda: setattr(env, 'mcp', RenderingMCP()))
    env.reset(seed=7)
    return env


def test_real_reset_profile_matches_offline_context_and_exports_hash(profile_inputs, tmp_path, monkeypatch):
    task, task_path, profile = profile_inputs
    profile_path = tmp_path / 'context_profile.json'
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=4))
    env = start_env(tmp_path, monkeypatch, task_path, profile_path)
    try:
        assert env.system_prompt == profile['system_prompt']
        assert env.model_tools == profile['tools']
        assert env.task_prompt == task['instruction'] + '\nPublic camera protocol:\n' + json.dumps(task['public_camera_protocol'], ensure_ascii=False)
        offline = SimpleNamespace(system_prompt=profile['system_prompt'], task_prompt=env.task_prompt,
            model_tools=copy.deepcopy(profile['tools']), initial_scene_description=profile['initial_scene_description'],
            reference_refs=copy.deepcopy(env.reference_refs), initial_images=copy.deepcopy(env.initial_images),
            turns=[], public_state=copy.deepcopy(env.public_state), task={})
        live = render_request(Tokenizer(), env)
        packed = render_request(Tokenizer(), offline)
        assert live[0] == packed[0] and live[1] == packed[1] and live[3:] == packed[3:]
        assert [im.tobytes() for im in live[2]] == [im.tobytes() for im in packed[2]]
        assert task['private_gt'] not in live[0]
        assert str(profile_path) not in live[0] and 'context_profile_sha256' not in live[0]
        manifest = json.loads((env.episode / 'manifest.json').read_text())
        assert manifest['context_profile_sha256'] == context_profile_sha256(profile)
        assert manifest['context_profile'] == profile
        assert manifest['prompt_version'] == 'raw_cot_context_v1'
        # One real tool exchange then the same actual prefix on both entry paths.
        env.step(action('execute_blender_code', {'code': 'print("build")'}))
        offline.turns = copy.deepcopy(env.turns)
        offline.public_state = copy.deepcopy(env.public_state)
        assert render_request(Tokenizer(), env)[0] == render_request(Tokenizer(), offline)[0]
    finally:
        env.close()


def test_profile_renderer_matches_teacher_active_camera_protocol(profile_inputs, tmp_path, monkeypatch):
    _, task_path, profile = profile_inputs
    env = start_env(tmp_path, monkeypatch, task_path, profile)
    try:
        scripts = [args['code'] for name, args in env.mcp.calls
                   if name == 'execute_blender_code' and 'bpy.ops.render.render(write_still=True)' in args.get('code', '')]
        assert len(scripts) == 1
        source = scripts[0]
        for text in ['if s.camera is None:', 's.render.engine="CYCLES"', 's.cycles.device="CPU"',
                     's.cycles.samples=12', 's.cycles.seed=20261120', 's.cycles.use_denoising=True',
                     's.render.threads=4', 's.render.resolution_x=640', 's.render.resolution_y=480']:
            assert text in source
        assert 'ConfiguredCamera' not in source
        assert not any(name == 'render_scene_view' for name, _ in env.mcp.calls)
        assert Image.open(env.initial_images[0]['path']).size == (640, 480)
        assert env.initial_images[0]['label'] == 'Initial scene S0'
        assert 'camera_id' not in env.initial_images[0]  # no guessed metadata
    finally:
        env.close()


def test_camera_edits_allowed_only_for_explicit_profile(profile_inputs, tmp_path, monkeypatch):
    _, task_path, profile = profile_inputs
    for chosen, suffix in [(None, 'default'), (profile, 'profile')]:
        env = start_env(tmp_path, monkeypatch, task_path, chosen, suffix)
        try:
            mcp = env.mcp
            monkeypatch.setattr(env, '_protected_state', lambda: [{'name': 'ChangedCamera'}])
            _, _, _, truncated, info = env.step(action('execute_blender_code', {'code': 'print("camera changed")'}))
            if chosen is None:
                assert truncated and info['termination_reason'] == 'environment_contract_violation'
                assert any(name == 'render_scene_view' for name, _ in mcp.calls)
            else:
                assert not truncated and info['termination_reason'] is None
        finally:
            env.close()


def test_profile_rejects_hidden_sixth_tool_before_execution(profile_inputs, tmp_path, monkeypatch):
    _, task_path, profile = profile_inputs
    env = start_env(tmp_path, monkeypatch, task_path, profile)
    try:
        assert len(env.model_tools) == 5
        _, _, _, _, info = env.step(action('scene.set_transform', {'object_name': 'chair', 'location': [0, 0, 0]}))
        assert info['metrics']['invalid_action'] == 1
        assert env.turns[-1]['feedback']['status'] == 'rejected'
        assert 'active context profile' in env.turns[-1]['feedback']['text']
        assert env.turns[-1]['call']['name'] == 'scene.set_transform'
        assert json.loads((env.episode / 'step_001/event.json').read_text())['execution'] is None
    finally:
        env.close()


def test_profile_cannot_change_parameters_add_tools_or_load_history(profile_inputs, tmp_path, monkeypatch):
    _, task_path, profile = profile_inputs
    invalid = copy.deepcopy(profile)
    invalid['tools'][0]['function']['parameters'] = {'type': 'object', 'properties': {'arbitrary': {'type': 'string'}}}
    env = ReconstructionEnv(task_path, tmp_path / 'invalid', context_profile=invalid)
    monkeypatch.setattr(env, '_start', lambda: setattr(env, 'mcp', RenderingMCP()))
    with pytest.raises(ValueError, match='schema differs from registered tool'):
        env.reset()
    assert env.closed
    for mutate in [lambda p: p['tools'].append(copy.deepcopy(p['tools'][0])),
                   lambda p: p.update(task_prompt='Teacher future history'),
                   lambda p: p.update(render_mode='unverified_renderer')]:
        invalid = copy.deepcopy(profile)
        mutate(invalid)
        with pytest.raises(ValueError):
            load_context_profile(invalid)
