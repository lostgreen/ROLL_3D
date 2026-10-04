"""Keep model-facing examples executable against each real provider schema."""
import json
from pathlib import Path
import sys

import pytest
from jsonschema import Draft202012Validator
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v2s_integration.agent.actions import parse_action
from v2s_integration.ops.backend_probe import generation_arguments, run_probe, catalog_probe_query
from v2s_integration.runtime.session import create_session, model_tools, COMMON, GROUPS
from v2s_integration.tests.test_integration import MCP, Backend


@pytest.fixture
def session_factory(tmp_path):
    ref = tmp_path / 'reference.png'
    Image.new('RGB', (16, 16), 'white').save(ref)
    index = tmp_path / 'catalog.json'
    index.write_text(json.dumps({'assets': [{'id': 'test', 'name': 'vase'}]}))
    def factory(group, texture=False):
        backend = Backend() if group in ('hunyuan', 'mixed') else None
        if backend:
            backend.supports_texture = texture
        task = {'id': 'guidance', 'group': group, 'asset_index': str(index), 'budget': {}}
        return create_session(task, MCP(), tmp_path / group, {'ref:0': str(ref)}, backend=backend)
    return factory


@pytest.mark.parametrize('group', GROUPS)
def test_each_native_example_matches_exposed_schema(session_factory, group):
    session = session_factory(group)
    tools = model_tools(session)
    assert {t['function']['name'] for t in tools} == set(COMMON) | set(GROUPS[group])
    for tool in tools:
        fn = tool['function']
        description = fn['description']
        if fn['name'] == 'render_scene_view':
            assert description == 'Render scene'
            continue
        assert '<tool_call>' in description
        raw = description[description.index('<tool_call>'):]
        parsed = parse_action(raw, 1, session.registry)
        assert parsed.name == fn['name']
        Draft202012Validator(fn['parameters']).validate(parsed.arguments)
        assert fn['parameters'] == session.registry.lookup(fn['name'])[0].input_schema


@pytest.mark.parametrize('texture', [False, True])
def test_generation_help_and_probe_respect_actual_backend_capability(session_factory, texture):
    session = session_factory('hunyuan', texture=texture)
    fn = next(t['function'] for t in model_tools(session) if t['function']['name'] == 'asset.generate_3d_from_image')
    example = parse_action(fn['description'][fn['description'].index('<tool_call>'):], 1, session.registry)
    assert example.arguments['texture'] is texture
    assert 'scene.import_artifact' in fn['description']
    assert 'no polling tool is needed' in fn['description']
    assert 'not a path, URL' in fn['description']
    args = generation_arguments(fn['parameters'], 'actual_crop_id')
    Draft202012Validator(fn['parameters']).validate(args)
    assert args['texture'] is texture and args['reference_image'] == 'actual_crop_id'


def test_probe_failure_produces_compact_result(tmp_path):
    result = run_probe(tmp_path / 'missing-task.json', tmp_path / 'probe')
    saved = json.loads((tmp_path / 'probe/probe_result.json').read_text())
    assert saved == result
    assert result['status'] == 'failed'
    assert result['error']['type'] == 'FileNotFoundError'
    assert result['model_inference_tested'] is False


def test_probe_query_uses_only_existing_importable_catalog_payload(tmp_path):
    (tmp_path / 'real.glb').write_bytes(b'probe catalog selection only')
    index = tmp_path / 'index.json'
    index.write_text(json.dumps({'assets': [
        {'id': 'missing', 'name': 'Unavailable', 'path': 'missing.glb'},
        {'id': 'real', 'name': 'Actual Vase', 'path': 'real.glb'}]}))
    assert catalog_probe_query({'asset_index': str(index)}) == 'Actual Vase'
