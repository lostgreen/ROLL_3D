"""Wire the frozen Video2Scene providers into explicit experimental tool groups."""
import json
import os
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[1]
HARNESS = Path(os.getenv('V2S_HARNESS_ROOT', str(WORK / 'harness')))
for directory in (WORK, HARNESS / 'src', HARNESS / 'src/harness', HARNESS / 'headless'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from environment.session import RunSession
from environment.task_spec import TaskSpec
from tools.base import ToolResult

COMMON = ('reference.crop', 'scene.list_objects', 'scene.set_transform',
          'render_scene_view', 'finish')
RETRIEVAL = ('asset.search', 'asset.inspect', 'asset.import')
GROUPS = {
    'blender': ('execute_blender_code',),
    'hunyuan': ('asset.generate_3d_from_image', 'scene.import_artifact'),
    'assets': (*RETRIEVAL, 'scene.import_artifact'),
    'mixed': ('execute_blender_code', 'asset.generate_3d_from_image',
              *RETRIEVAL, 'scene.import_artifact'),
}
SCOPES = {
    'blender': 'Blender Python construction',
    'hunyuan': 'image-conditioned 3D asset generation',
    'assets': 'local asset retrieval',
    'mixed': 'Blender Python construction, image-conditioned 3D asset generation, and local asset retrieval',
}
FINISH_SCHEMA = {'type': 'object', 'properties': {'summary': {'type': 'string'}},
                 'additionalProperties': False}


def generation_backend(task, artifact_root):
    """Same backend selection as the original harness; load only for B/D."""
    if task['group'] not in ('hunyuan', 'mixed'):
        return None
    from tools.hunyuan import HunyuanHTTPBackend, HunyuanLocalBackend
    cfg = task.get('generation', {})
    if cfg.get('python'):
        from .isolated_hunyuan import IsolatedHunyuanBackend
        return IsolatedHunyuanBackend(cfg, Path(artifact_root) / 'generation_workers')
    endpoint = cfg.get('endpoint') or os.getenv('V2S_HUNYUAN_ENDPOINT')
    if endpoint:
        return HunyuanHTTPBackend(endpoint, timeout=cfg.get('timeout', 900))
    model = cfg.get('model_path') or os.getenv('V2S_HUNYUAN_MODEL_PATH')
    subfolder = cfg.get('subfolder') or os.getenv('V2S_HUNYUAN_SUBFOLDER')
    if not model or not subfolder:
        raise ValueError('Hunyuan group requires an endpoint or model_path and subfolder')
    backend = HunyuanLocalBackend(model, subfolder, cfg.get('device', os.getenv('V2S_HUNYUAN_DEVICE', 'cuda:0')))
    paint = cfg.get('paint_code') or os.getenv('V2S_HUNYUAN_PAINT_CODE')
    if paint:
        from tools.hunyuan_paint import HunyuanTexturedBackend
        backend = HunyuanTexturedBackend(backend, paint, Path(artifact_root) / 'generation_intermediates',
                    python=cfg.get('paint_python') or os.getenv('V2S_HUNYUAN_PAINT_PYTHON'))
    return backend


def create_session(task, mcp, artifact_root, references, backend=None, segmenter=None):
    group = task['group']
    if group not in GROUPS:
        raise ValueError(f'Unknown tool group: {group}')
    if not references:
        raise ValueError('At least one actual public reference image is required')
    if group in ('assets', 'mixed'):
        index = Path(task.get('asset_index', ''))
        if not index.is_file():
            raise ValueError('Asset group requires an existing asset_index')
        payload = json.loads(index.read_text())
        items = payload.get('assets', payload.get('items', [])) if isinstance(payload, dict) else payload
        if not isinstance(items, list) or not any(isinstance(item, dict) for item in items):
            raise ValueError('Asset index contains no candidates')
    if group in ('hunyuan', 'mixed') and backend is None:
        backend = generation_backend(task, artifact_root)
    if group not in ('hunyuan', 'mixed') and backend is not None:
        raise ValueError('Generation backend is not permitted for this group')
    allowed = [*COMMON, *GROUPS[group]]
    if segmenter is not None:
        allowed.append('reference.segment')
    spec = TaskSpec.from_task({
        'id': task['id'], 'tool_profile': 'adaptive', 'allowed_tools': allowed,
        'asset_index': task.get('asset_index') if group in ('assets', 'mixed') else None,
        'budget': task['budget'],
    })
    session = RunSession(spec, mcp, artifact_root=artifact_root, reference_images=references,
                         generation_backend=backend, segmentation_backend=segmenter,
                         extra_tools={'finish': {
                             'description': 'End this episode with an optional short summary of unresolved limitations.',
                             'schema': FINISH_SCHEMA,
                             'handler': lambda args: ToolResult(args.get('summary', 'Episode finished.')),
                         }})
    visible = {tool['name'] for tool in session.registry.schemas_for_model()}
    if visible != set(allowed):
        raise ValueError(f'Tools not available: {sorted(set(allowed) - visible)}')
    return session


def model_tools(session):
    from .tool_guidance import guidance
    schemas = session.registry.schemas_for_model()
    visible = {tool['name'] for tool in schemas}
    return [{'type': 'function', 'function': {
        'name': tool['name'],
        'description': guidance(tool['name'], tool['input_schema'], visible) or tool['description'],
        'parameters': tool['input_schema']}} for tool in schemas]


def execute_validated(session, name, arguments):
    """Validate before invoking the original executor, including finish."""
    from jsonschema import Draft202012Validator
    entry = session.registry.lookup(name)
    if entry and session.registry.permitted(name):
        errors = sorted(Draft202012Validator(entry[0].input_schema).iter_errors(arguments), key=lambda e: str(e.path))
        if errors:
            result = ToolResult('Invalid arguments: ' + errors[0].message,
                                status='rejected', error_type='invalid_arguments')
            return result, {'name': name, 'arguments': arguments, 'executed': False,
                            'status': result.status, 'error_type': result.error_type}
    return session.executor.execute(name, arguments)
