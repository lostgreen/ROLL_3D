"""Append-only API history, independent of training prompt/token packing.

Keep the bytes of previously sent messages intact. Scene changes are JSON Patch
operations against a versioned snapshot; full states remain in environment logs.
"""
import base64
import copy
import hashlib
import json
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def state_patch(before, after, path=''):
    """Deterministic RFC 6902 subset; also captures partial edits on tool failure."""
    if type(before) is not type(after):
        return [{'op': 'replace', 'path': path, 'value': copy.deepcopy(after)}]
    if isinstance(before, dict):
        escape = lambda key: key.replace('~', '~0').replace('/', '~1')
        result = [{'op': 'remove', 'path': path + '/' + escape(k)}
                  for k in sorted(before.keys() - after.keys())]
        for key in sorted(after):
            child = path + '/' + escape(key)
            if key not in before:
                result.append({'op': 'add', 'path': child, 'value': copy.deepcopy(after[key])})
            else:
                result.extend(state_patch(before[key], after[key], child))
        return result
    if isinstance(before, list):
        common = min(len(before), len(after))
        result = []
        for index in range(common):
            result.extend(state_patch(before[index], after[index], path + '/' + str(index)))
        for index in range(len(before) - 1, common - 1, -1):
            result.append({'op': 'remove', 'path': path + '/' + str(index)})
        for item in after[common:]:
            result.append({'op': 'add', 'path': path + '/-', 'value': copy.deepcopy(item)})
        return result
    return [] if before == after else [{'op': 'replace', 'path': path, 'value': copy.deepcopy(after)}]


def image_content(text, refs, seen=None):
    parts = [{'type': 'text', 'text': text}]
    for ref in refs:
        raw = Path(ref['path']).read_bytes()
        identity = hashlib.sha256(raw).hexdigest()
        if seen is not None:
            if identity in seen:
                continue
            seen.add(identity)
        parts.extend([{'type': 'text', 'text': ref.get('label', 'Observation')},
                      {'type': 'image_url', 'image_url': {
                          'url': 'data:image/png;base64,' + base64.b64encode(raw).decode(),
                          'detail': 'high'}}])
    return parts


CONTEXT_RULES = '''API context protocol: the initial public_scene_state is a snapshot.
Later scene_patch entries are JSON Patch operations applied in order to that snapshot.
An unchanged scene_version means no scene change. Previously shown images remain
in history; no new image means reuse the previous observation. Tool errors can still
partially change the scene; trust scene_patch. Issue exactly one native function
call per decision. Do not encode tool calls in XML or Markdown.'''


class APIContext:
    def __init__(self, env, *, system_prompt=None):
        self.system = (system_prompt or env.system_prompt) + '\n' + CONTEXT_RULES
        self.reference = {'role': 'user', 'content': image_content(env.task_prompt, env.reference_refs)}
        self.seen_images = set()
        self.scene = copy.deepcopy(env.public_state)
        self.version = self.steps = self.checkpoints = 0
        self.events = []
        self.messages = [{'role': 'system', 'content': self.system}, copy.deepcopy(self.reference),
                         self._snapshot(env.initial_images)]
        if getattr(env, 'backend_repair_notice', None):
            self.messages.append({'role': 'user', 'content': env.backend_repair_notice})

    def _snapshot(self, refs, summary=None):
        state = {'scene_version': self.version, 'public_scene_state': self.scene}
        if summary is not None:
            state['checkpoint'] = {'completed_steps': self.steps, 'prior_actions': summary,
                                   'reason': 'explicit_context_budget_boundary'}
        return {'role': 'user', 'content': image_content(canonical(state), refs, self.seen_images)}

    def append(self, original, turn):
        if turn['step'] != self.steps + 1:
            raise ValueError('Append exactly once, in execution order')
        replay = {k: copy.deepcopy(original[k]) for k in (
            'role', 'content', 'tool_calls', 'reasoning', 'reasoning_content',
            'reasoning_summary', '_responses_output') if k in original}
        feedback = copy.deepcopy(turn['feedback'])
        feedback['text'] = feedback.get('text', '')[:8000]
        calls = original.get('tool_calls') or []
        if len(calls) > 1 and turn.get('call') is not None:
            raise ValueError('Multiple calls must all be rejected before recording feedback')
        ids = [c['id'] for c in calls]
        if len(set(ids)) != len(ids) or any(not i for i in ids):
            raise ValueError('Native tool call IDs must be unique and nonempty')
        block = [replay]
        if calls:
            # Resolve every native call, including rejected multi-call responses.
            for call in calls:
                result = {**feedback, 'call_id': call['id']}
                block.append({'role': 'tool', 'tool_call_id': call['id'], 'content': canonical(result)})
        else:
            block.append({'role': 'user', 'content': 'Environment notice: ' + canonical(feedback)})
        changes = state_patch(self.scene, turn['scene_state'])
        version = self.version + bool(changes)
        observation = {'scene_version': version, 'base_scene_version': self.version,
                       'scene_patch': changes}
        seen = set(self.seen_images)
        block.append({'role': 'user', 'content': image_content(canonical(observation),
                      turn['tool_images'] + turn['scene_images'], seen)})
        self.messages.extend(block)
        self.events.append({'step': turn['step'], 'tool': feedback.get('tool_name'),
                            'status': feedback.get('status'), 'error_type': feedback.get('error_type')})
        self.scene, self.version = copy.deepcopy(turn['scene_state']), version
        self.steps, self.seen_images = turn['step'], seen

    def checkpoint(self, env):
        """Explicit lossy boundary: current snapshot + bounded action ledger.

        Called only after an entire tool exchange. The caller records the boundary;
        a snapshot that still exceeds the budget is rejected, never silently sent.
        """
        self.seen_images = set()
        self.messages = [{'role': 'system', 'content': self.system}, copy.deepcopy(self.reference),
                         self._snapshot(env.current_images, self.events[-32:])]
        if getattr(env, 'backend_repair_notice', None):
            self.messages.append({'role': 'user', 'content': env.backend_repair_notice})
        self.checkpoints += 1


def api_payload(messages, tools, *, model, protocol, max_output_tokens, cache_key,
                effort=None, parameters=None):
    """Encode native replay without reconstructing or renumbering returned items."""
    messages = copy.deepcopy(messages)
    if protocol == 'responses':
        items = []
        for message in messages[1:]:
            role = message['role']
            if role == 'assistant':
                if '_responses_output' not in message:
                    raise ValueError('Responses replay requires original native output items')
                items.extend(copy.deepcopy(message['_responses_output']))
            elif role == 'tool':
                items.append({'type': 'function_call_output', 'call_id': message['tool_call_id'],
                              'output': message['content']})
            else:
                content = message['content']
                if isinstance(content, str):
                    content = [{'type': 'text', 'text': content}]
                parts = []
                for part in content:
                    if part['type'] == 'text':
                        parts.append({'type': 'input_text', 'text': part['text']})
                    elif part['type'] == 'image_url':
                        parts.append({'type': 'input_image', 'image_url': part['image_url']['url'],
                                      'detail': part['image_url']['detail']})
                    else:
                        raise ValueError('Unsupported input content')
                items.append({'role': role, 'content': parts})
        payload = {'model': model, 'instructions': messages[0]['content'], 'input': items,
                   'tools': [{'type': 'function', **t['function'], 'strict': False} for t in tools],
                   'max_output_tokens': max_output_tokens, 'store': False,
                   'include': ['reasoning.encrypted_content']}
        if effort is not None:
            payload['reasoning'] = {'effort': effort, 'summary': 'auto'}
    elif protocol == 'chat':
        for message in messages:
            message.pop('_responses_output', None)
            message.pop('reasoning_summary', None)
        payload = {'model': model, 'messages': messages, 'tools': copy.deepcopy(tools),
                   'max_tokens': max_output_tokens, 'stream_options': {'include_usage': True}}
        if effort is not None:
            payload['reasoning_effort'] = effort
    else:
        raise ValueError('Unsupported API protocol')
    payload.update(tool_choice='auto', parallel_tool_calls=False, stream=True, prompt_cache_key=cache_key)
    if parameters:
        if payload.keys() & parameters.keys():
            raise ValueError('Provider extras cannot overwrite core request fields')
        payload.update(copy.deepcopy(parameters))
    return payload


def check_budget(payload, *, max_text_bytes=100000, max_wire_bytes=4000000, max_images=12):
    """Final-payload guard, not a claim of exact provider tokenization.

    Counts opaque reasoning and tool schemas after native conversion, and images
    separately after base64 expansion. Server usage remains the token authority.
    """
    reduced = copy.deepcopy(payload)
    images = 0

    def strip(value):
        nonlocal images
        if isinstance(value, dict):
            if value.get('type') == 'input_image':
                images += 1
                value['image_url'] = '<image>'
            elif value.get('type') == 'image_url':
                images += 1
                value['image_url']['url'] = '<image>'
            else:
                for child in value.values():
                    strip(child)
        elif isinstance(value, list):
            for child in value:
                strip(child)
    strip(reduced)
    metrics = {'text_bytes': len(canonical(reduced).encode()),
               'wire_bytes': len(json.dumps(payload, ensure_ascii=False).encode()), 'images': images}
    if metrics['text_bytes'] > max_text_bytes or metrics['wire_bytes'] > max_wire_bytes or images > max_images:
        raise ValueError('Final API payload exceeds budget: ' + canonical(metrics))
    return metrics
