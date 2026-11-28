"""Deterministic Raw-CoT context selection shared by SFT and the student harness.

Input turns must contain only completed decisions before the prediction step.
This module selects observations; it never truncates tool calls or feedback.
"""
import copy
import hashlib
import json
from pathlib import Path

POLICY_VERSION = 'raw_cot_context_v1'
V2_POLICY_VERSION = 'raw_cot_context_v2'
SUPPORTED_POLICY_VERSIONS = frozenset({POLICY_VERSION, V2_POLICY_VERSION})
PROFILE_TOOLS = frozenset({'execute_blender_code', 'reference.crop', 'scene.list_objects',
                           'render_scene_view', 'finish'})
INITIAL_DESCRIPTION = (
    'Initial scene S0 is the initial scene state specified in the fixed task above. '
    'This description stays unchanged; later observations describe subsequent states.'
)


def load_context_profile(profile):
    """Read an explicit profile without importing task data or teacher history."""
    if profile is None:
        return None
    if isinstance(profile, (str, Path)):
        profile = json.loads(Path(profile).read_text())
    if not isinstance(profile, dict):
        raise ValueError('context_profile must be a JSON object or JSON path')
    profile = copy.deepcopy(profile)
    fields = {'policy_version', 'system_prompt', 'tools', 'initial_scene_description',
              'task_prompt_mode', 'allow_camera_light_edits', 'render_mode'}
    version = profile.get('policy_version')
    if version not in SUPPORTED_POLICY_VERSIONS:
        raise ValueError('Unsupported context policy version')
    if version == V2_POLICY_VERSION:
        fields.add('observation_format')
    if set(profile) != fields:
        raise ValueError('context_profile fields must exactly match the public context contract')
    if version == V2_POLICY_VERSION:
        from v2s_integration.agent.observation_format import observation_contract
        if profile['observation_format'] != observation_contract():
            raise ValueError('context_profile observation policy/source hash mismatch')
    if profile['task_prompt_mode'] != 'public_instruction_camera_v1':
        raise ValueError('Unsupported context task prompt mode')
    if profile['render_mode'] != 'active_camera_640x480_cycles_v1':
        raise ValueError('Unsupported context render mode')
    if profile['allow_camera_light_edits'] is not True:
        raise ValueError('Raw-CoT context profiles explicitly permit camera and light edits')
    for key in ('system_prompt', 'initial_scene_description'):
        if not isinstance(profile[key], str) or not profile[key].strip():
            raise ValueError(f'context_profile {key} must be nonempty text')
    tools = profile['tools']
    if not isinstance(tools, list) or len(tools) != len(PROFILE_TOOLS):
        raise ValueError('context_profile must contain exactly the five Raw-CoT tools')
    names = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise ValueError('context_profile tools require native function schemas')
        function = tool.get('function', {})
        if not isinstance(function, dict):
            raise ValueError('context_profile tools require native function schemas')
        if tool.get('type') != 'function' or not isinstance(function.get('parameters'), dict):
            raise ValueError('context_profile tools require native function schemas')
        if not isinstance(function.get('name'), str):
            raise ValueError('context_profile tool names must be strings')
        names.append(function.get('name'))
    if len(set(names)) != len(names) or set(names) != PROFILE_TOOLS:
        raise ValueError('context_profile must contain exactly the five Raw-CoT tools')
    return profile


def context_profile_sha256(profile):
    """Canonical object hash, independent of the profile file's formatting."""
    return hashlib.sha256(json.dumps(profile, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def active_context_policy(env):
    """Resolve an explicit profile, rejecting stale v2 source fingerprints."""
    profile = load_context_profile(getattr(env, 'context_profile', None))
    declared = getattr(env, 'context_policy_version', None)
    version = profile['policy_version'] if profile else POLICY_VERSION
    if declared is not None and declared != version:
        raise ValueError('Environment context policy differs from its explicit profile')
    return version


def apply_context_profile(env):
    """Apply a validated profile after the real environment registers its tools."""
    profile = env.context_profile
    if profile is None:
        return
    if len(env.reference_refs) != 2:
        raise ValueError('raw_cot_context_v1 requires two public reference images')
    registered = {tool['function']['name']: tool['function']['parameters'] for tool in env.model_tools}
    for tool in profile['tools']:
        function = tool['function']
        if function['name'] not in registered or function['parameters'] != registered[function['name']]:
            raise ValueError('context_profile schema differs from registered tool: ' + function['name'])
    instruction = env.task.get('instruction')
    cameras = env.task.get('public_camera_protocol')
    if not isinstance(instruction, str) or not instruction.strip() or not isinstance(cameras, dict):
        raise ValueError('context_profile requires public task instruction and camera protocol')
    env.system_prompt = profile['system_prompt']
    env.task_prompt = instruction + '\nPublic camera protocol:\n' + json.dumps(cameras, ensure_ascii=False)
    env.model_tools = copy.deepcopy(profile['tools'])
    env.initial_scene_description = profile['initial_scene_description']


def _arguments(call):
    value = (call or {}).get('arguments', {})
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    return value if isinstance(value, dict) else {}


def _camera(ref, call):
    explicit = ref.get('camera_id') or ref.get('camera')
    if explicit:
        return str(explicit)
    if call and call.get('name') == 'render_scene_view':
        args = _arguments(call)
        # Non-camera views can use an adaptive fit; their labels are not a
        # reliable persistent camera identity.
        if args.get('view', 'camera') == 'camera' and args.get('camera'):
            return str(args['camera'])
    # task.observation_camera is a configured default, not evidence of the
    # actual camera used after an agent edits s.camera or rendering behavior.
    return None


def select_observations(env):
    """Return occurrence lookup, chosen source IDs and an audit-only manifest.

    Optional image fields: image_id, sha256, camera_id/camera, generated_step
    (or step), scene_version. Repeated references to the same source path retain
    their first generation time, even if current_images repeats in later turns.
    """
    turns = env.turns
    occurrences, sources, source_keys = {}, {}, {}

    def add(ref, key, kind, step, call=None, scene_version=None):
        ref = copy.deepcopy(ref)
        generated = ref.get('generated_step', ref.get('step', step))
        if not isinstance(generated, int) or generated < 0 or generated > step:
            raise ValueError('Image generation step is outside the visible history')
        source_key = (kind, str(ref.get('image_id') or Path(ref['path'])))
        if source_key in source_keys:
            occurrences[key] = source_keys[source_key]
            return
        source_id = kind + ':' + str(ref.get('image_id') or f'observation_{len(sources)}')
        source_keys[source_key] = source_id
        occurrences[key] = source_id
        sha = ref.get('sha256') or hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest()
        source = dict(source_id=source_id, kind=kind, generated_step=generated,
                      scene_version=ref.get('scene_version', scene_version),
                      camera_id=_camera(ref, call),
                      sha256=sha, ref=ref, occurrence=key, ordinal=len(sources))
        sources[source_id] = source

    for i, ref in enumerate(env.initial_images):
        add(ref, ('initial', i), 'render', 0, scene_version=0)
    for index, turn in enumerate(turns):
        step = turn.get('step', index + 1)
        if step != index + 1:
            raise ValueError('History must contain all completed turns in execution order')
        call = turn.get('call')
        name = (call or {}).get('name')
        for i, ref in enumerate(turn.get('tool_images', [])):
            kind = 'crop' if name == 'reference.crop' else 'render' if name == 'render_scene_view' else 'tool'
            add(ref, (index, 'tool', i), kind, step, call, turn.get('scene_version'))
        for i, ref in enumerate(turn.get('scene_images', [])):
            add(ref, (index, 'scene', i), 'render', step,
                scene_version=turn.get('scene_version'))

    renders = sorted((s for s in sources.values() if s['kind'] == 'render'),
                     key=lambda s: (s['generated_step'], s['ordinal']))
    fallback = bool(renders) and any(s['camera_id'] is None for s in renders)
    if fallback:
        latest_by_hash = {}
        for source in renders:
            latest_by_hash[source['sha256']] = source
        selected_renders = sorted(latest_by_hash.values(),
                                  key=lambda s: (s['generated_step'], s['ordinal']))[-2:]
    else:
        latest = {}
        for source in renders:
            latest[source['camera_id']] = source
        selected_renders = sorted(latest.values(), key=lambda s: (s['generated_step'], s['ordinal']))[-2:]
    selected = {s['source_id'] for s in selected_renders}
    if turns and (turns[-1].get('call') or {}).get('name') == 'reference.crop':
        for i, _ in enumerate(turns[-1].get('tool_images', [])):
            selected.add(occurrences[(len(turns) - 1, 'tool', i)])

    manifest = dict(version=active_context_policy(env), history_turns=len(turns), omitted_history_turns=0,
                    render_selection='latest_two_fallback' if fallback else 'latest_per_camera',
                    fallback_reason='missing_camera_metadata' if fallback else None,
                    images=[{k: v for k, v in s.items() if k not in ('ref', 'occurrence')} |
                            {'path': s['ref']['path'], 'selected': s['source_id'] in selected}
                            for s in sources.values()])
    return occurrences, sources, selected, manifest


def observation_label(source):
    parts = [source['ref'].get('label', 'Observation'),
             'image_id=' + source['source_id'], 'generated_step=' + str(source['generated_step'])]
    if source['camera_id'] is not None:
        parts.append('camera=' + source['camera_id'])
    if source['scene_version'] is not None:
        parts.append('scene_version=' + str(source['scene_version']))
    return '; '.join(parts)


def omitted_label(source, selected, sources):
    category = {'render': '历史渲染图', 'crop': '历史参考裁图', 'tool': '历史工具图'}[source['kind']]
    newer = any(sources[key]['kind'] == source['kind'] and
                sources[key]['generated_step'] > source['generated_step'] for key in selected)
    instruction = '；请结合保留的较新观察判断当前状态' if newer and source['kind'] == 'render' else ''
    return f'[{category}已省略{instruction}。image_id={source["source_id"]}; generated_step={source["generated_step"]}]'
