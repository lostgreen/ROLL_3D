"""Pure observation serialization for the explicit Raw-CoT context v2 profile.

Only public state and the structured scene-delta payload are observation data.
Code, reasoning, camera protocols and free tool-output text are never rewritten.
"""
import copy
import hashlib
import json
import math
from pathlib import Path


FORMAT_VERSION = 'raw_cot_observation_round4_v1'
GEOMETRY_FIELDS = frozenset({
    'location', 'world_location', 'dimensions', 'world_bounds',
    'delta_location', 'delta_rotation', 'delta_scale', 'new_dimensions',
    'rotation', 'rotation_euler', 'scale', 'world_rotation', 'world_scale',
})
PROTECTED_FIELDS = frozenset({
    'matrix', 'matrix_world', 'projection_matrix', 'view_matrix',
    'public_camera_protocol', 'camera_protocol',
})
JSON_SEPARATORS = (',', ':')
STATE_PREFIX = 'Current public scene state: '
NOTICE_PREFIX = 'Environment notice: '


def observation_contract():
    """Return the exact serialization policy and loaded module's source hash."""
    return {
        'version': FORMAT_VERSION,
        'geometry_fields': sorted(GEOMETRY_FIELDS),
        'protected_fields': sorted(PROTECTED_FIELDS),
        'decimals': 4,
        'rounding': 'python_round_half_even',
        'integer_values': 'int',
        'negative_zero': '0',
        'nonfinite': 'reject_with_field_path',
        'drop_world_location': 'parent_explicitly_null_and_formatted_location_equal',
        'camera_objects': 'preserve_type_CAMERA_or_name_prefix_PublicCamera',
        'feedback_scope': 'structured.scene_delta_and_final_for_model_structured_JSON_block',
        'free_text': 'preserve',
        'json_separators': list(JSON_SEPARATORS),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'context_sources': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ('observation_format.py', 'context_policy.py', 'messages.py', 'actions.py')},
    }


def observation_contract_sha256():
    return hashlib.sha256(json.dumps(observation_contract(), sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()


def _validate_finite(value, path):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite observation number at ' + path)
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite(item, path + '.' + str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite(item, f'{path}[{index}]')


def _round_geometry(value):
    if isinstance(value, float):
        rounded = round(value, 4)
        return int(rounded) if rounded.is_integer() else rounded
    if isinstance(value, (list, tuple)):
        return [_round_geometry(item) for item in value]
    # Unknown schemas inside a geometry-valued field are copied, not guessed.
    return copy.deepcopy(value)


def _geometry(value):
    if isinstance(value, list):
        return [_geometry(item) for item in value]
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    if value.get('type') == 'CAMERA' or str(value.get('name', '')).startswith('PublicCamera'):
        return copy.deepcopy(value)
    result = {}
    for key, item in value.items():
        if key in PROTECTED_FIELDS:
            result[key] = copy.deepcopy(item)
        elif key in GEOMETRY_FIELDS:
            result[key] = _round_geometry(item)
        else:
            result[key] = _geometry(item)
    if (result.get('parent', ...) is None and 'location' in result
            and 'world_location' in result and result['world_location'] == result['location']):
        del result['world_location']
    return result


def format_public_scene_state(state):
    """Format a copied public scene state (or its public_scene_state wrapper)."""
    _validate_finite(state, 'public_scene_state')
    return _geometry(state)


def _structured(value):
    """Visit named scene_delta regions while preserving other numeric payloads."""
    if isinstance(value, list):
        return [_structured(item) for item in value]
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    return {key: (copy.deepcopy(item) if key in PROTECTED_FIELDS else
                  _geometry(item) if key == 'scene_delta' else _structured(item))
            for key, item in value.items()}


def _feedback_text(text):
    # ToolResult.for_model appends precisely one canonical JSON block on the
    # final line. Earlier stdout, including JSON-looking output, remains bytes.
    marker = '\n{"structured":'
    start = text.rfind(marker)
    if start < 0:
        return text
    start += 1
    try:
        payload, end = json.JSONDecoder().raw_decode(text[start:])
    except ValueError:
        return text
    if (text[start + end:].strip() or not isinstance(payload, dict)
            or set(payload) != {'structured'}):
        return text
    _validate_finite(payload, 'feedback.text.structured')
    formatted = {'structured': _structured(payload['structured'])}
    return text[:start] + serialize_observation(formatted) + text[start + end:]


def format_feedback(feedback):
    """Copy feedback; format only its declared structured scene-delta regions."""
    result = copy.deepcopy(feedback)
    if not isinstance(result, dict):
        return result
    _validate_finite(result, 'feedback')
    if 'structured' in result:
        result['structured'] = _structured(result['structured'])
    if isinstance(result.get('text'), str):
        result['text'] = _feedback_text(result['text'])
    return result


def serialize_observation(value):
    return json.dumps(value, ensure_ascii=False, separators=JSON_SEPARATORS, allow_nan=False)


def format_observation_messages(messages):
    """Derive v2 messages from archived v1 messages using the live serializer.

    This explicit v2 operation only edits tool feedback, environment notices and
    the final public-state message. Assistant calls and task/camera text stay raw.
    """
    result = copy.deepcopy(messages)
    for message in result:
        content = message.get('content')
        if message.get('role') == 'tool' and isinstance(content, str):
            message['content'] = serialize_observation(format_feedback(json.loads(content)))
        elif message.get('role') == 'user' and isinstance(content, str):
            if content.startswith(STATE_PREFIX):
                state = json.loads(content[len(STATE_PREFIX):])
                message['content'] = STATE_PREFIX + serialize_observation(format_public_scene_state(state))
            elif content.startswith(NOTICE_PREFIX):
                feedback = json.loads(content[len(NOTICE_PREFIX):])
                message['content'] = NOTICE_PREFIX + serialize_observation(format_feedback(feedback))
    return result
