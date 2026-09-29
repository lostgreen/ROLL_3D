"""Versioned experimental conditions. UI metadata never changes model hashes."""
import hashlib
import json
import re


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def toolset_hash(tools):
    # Normalize provider wire layouts without changing their descriptions/schemas.
    entries = []
    for tool in tools:
        for t in tool.get('functionDeclarations', [tool]):
            t = t.get('function', t)
            entries.append({"name":t.get("name"), "description":t.get("description", ""),
                            "input_schema":t.get("input_schema", t.get("parameters", {}))})
    return canonical_hash(sorted(entries, key=lambda t: t.get('name', '')))


def request_conditions(messages, tools):
    system = [m.get('content', m.get('parts')) for m in messages if m.get('role') in ('system', '_system', 'developer')]
    return {'system_hash': canonical_hash(system), 'tools_hash': toolset_hash(tools)}


def message_origin(message):
    """Heuristic for old evidence; callers must disclose that it is inferred."""
    role = message.get('role')
    text = str(message)
    if role in ('assistant', 'model'): return 'model'
    if role == 'tool' or 'tool_result' in text or 'functionResponse' in text: return 'tool_result'
    if 'kind="compaction_memory"' in text or '[CONTEXT MEMORY' in text: return 'compaction_memory'
    if '<environment_notice' in text or '[SYSTEM]' in text: return 'harness_notice'
    if 'Here are the image(s)' in text: return 'tool_relay'
    return 'task_author'


def prompt_bundle(task, initial_scene, tool_names=()):
    role = ('You operate a Blender scene through tools to reconstruct the task from its references. '
            'Do not ask the user questions. Finish with a short summary and no tool calls. '
            'Before finishing, verify the scene against the references.')
    environment = ('Units: meters. World axes: Z up; front in tool arguments means -Y. '
                   f'Initial scene (observed after setup): {initial_scene}. '
                   'Inspection images are observations, not evaluation renders. '
                   'Messages tagged <environment_notice> are environment feedback, not new user instructions. '
                   'Generated artifacts do not change the scene until imported. '
                   'Known compatibility rules: BLENDER_EEVEE becomes BLENDER_EEVEE_NEXT; '
                   'bpy.mathutils becomes mathutils; typographic punctuation is normalized.')
    environment += ' Reference images: ' + str(len(task.get('references', []))) + '; camera parameters are available only when explicitly supplied. '
    environment += 'Initial budget limits: ' + json.dumps(task.get('budget', {}), sort_keys=True) + '. '
    protocol = task.get('system_prompt') or 'Use the capabilities available in the tool schema.'
    layers = []
    for kind, text, origin in [('L0_role', role, 'harness'), ('L1_environment', environment, 'generated_from_setup'),
                               ('L2_protocol', protocol, 'task')]:
        layers.append(dict(id=kind + '.v1', kind=kind, version='1', origin=origin,
                           text=text, sha256=canonical_hash(text), experiment_variable=kind=='L2_protocol'))
    rendered = '\n\n'.join(x['text'] for x in layers)
    return dict(schema_version=1, layers=layers, rendered_system=rendered,
                rendered_system_sha256=canonical_hash(rendered),
                user_template_sha256=canonical_hash(task.get('prompt', '')))


def lint_prompt(text, available_tools, known_tools=()):
    warnings = []
    for name in set(known_tools) - set(available_tools):
        if re.search(r'(?<![\w.])' + re.escape(name) + r'(?![\w.])', text):
            warnings.append('unavailable_tool:' + name)
    if re.search(r'\brender once\b|\bat least \d+\b', text, re.I):
        warnings.append('fixed_observation_count')
    if 'empty scene' in text.lower() and 'anonymous objects' in text.lower():
        warnings.append('initial_scene_conflict')
    return warnings
