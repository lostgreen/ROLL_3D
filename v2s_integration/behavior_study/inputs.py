"""Auditable model inputs; keep complete tool-call/result pairs when pruning."""
import hashlib
import json
import re
from PIL import Image


def build_messages(env, history_limit=None):
    if history_limit is None:
        history_limit = len(env.turns)
    turns = env.turns[-history_limit:] if history_limit else []
    scene_refs = {r['path'] for t in env.turns[-2:] for r in t['scene_images']}
    if not env.turns:
        scene_refs.update(r['path'] for r in env.initial_images)
    active_refs = [r['path'] for t in turns for r in t['tool_images']][-2:]
    selected = scene_refs | set(active_refs)
    pictures, refs = [], []

    def content(text, image_refs, always=False):
        value = [{'type': 'text', 'text': text}]
        for ref in image_refs:
            if not always and ref['path'] not in selected:
                continue
            with Image.open(ref['path']) as source:
                pictures.append(source.convert('RGB'))
            refs.append(ref)
            value.extend([{'type': 'text', 'text': ref['label']}, {'type': 'image'}])
        return value

    messages = [{'role': 'system', 'content': env.system_prompt},
                {'role': 'user', 'content': content(env.task_prompt, env.reference_refs, always=True)}]
    if not env.turns:
        messages.append({'role': 'user', 'content': content('Initial scene observation', env.initial_images)})
    for turn in turns:
        call = turn['call']
        # Reasoning text is archived separately; never reinterpret it as an action.
        if call:
            messages.append({'role': 'assistant', 'content': '', 'tool_calls': [{
                'id': call['id'], 'type': 'function',
                'function': {'name': call['name'], 'arguments': call['arguments']}}]})
            feedback = dict(turn['feedback'])
            feedback['text'] = feedback['text'][:8000]
            messages.append({'role': 'tool', 'tool_call_id': call['id'], 'name': call['name'],
                             'content': json.dumps(feedback, ensure_ascii=False)})
        else:
            messages.append({'role': 'user', 'content': 'Environment notice: ' + turn['feedback']['text'][:2000]})
        # Repeat the current render only once below; retain actual tool images here.
        messages.append({'role': 'user', 'content': content(
            f'Environment observation after turn {turn["step"]}', turn['tool_images'])})
    state = {'turn_index': env.step_count, 'turns_remaining': env.max_steps - env.step_count,
             'public_scene_state': env.public_state, 'omitted_history_turns': len(env.turns) - len(turns)}
    # Include the preceding distinct scene view once for visual comparison.
    views = []
    for turn in env.turns[-2:]:
        for ref in turn['scene_images']:
            if ref['path'] not in {v['path'] for v in views}:
                views.append(ref)
    if env.turns:
        messages.append({'role': 'user', 'content': content('Environment notice: ' + json.dumps(state), views)})
    return messages, pictures, refs, state


def render_request(tokenizer, env, history_limit=None):
    messages, pictures, refs, state = build_messages(env, history_limit)
    prompt = tokenizer.apply_chat_template(messages, tools=env.model_tools,
        add_generation_prompt=True, tokenize=False, return_dict=False)
    # The Qwen tool template embeds JSON schemas. Verify the entire definition,
    # not just a tool name that might occur in the task or previous response.
    decoder = json.JSONDecoder()
    embedded = []
    for match in re.finditer(r'\{', prompt):
        try:
            value, _ = decoder.raw_decode(prompt[match.start():])
        except ValueError:
            continue
        if isinstance(value, dict):
            embedded.append(value.get('function', value))
    for tool in env.model_tools:
        if tool['function'] not in embedded:
            raise ValueError('Chat template omitted or altered full tool schema: ' + tool['function']['name'])
    return prompt, messages, pictures, refs, state


def record_request(directory, prompt, messages, tools, refs, input_ids, inference_ids, state):
    directory.mkdir(parents=True, exist_ok=True)
    if hasattr(inference_ids, 'tolist'):
        inference_ids = inference_ids.tolist()
    for name, value in [('messages.json', messages), ('tool_schemas.json', tools),
                        ('input_ids.json', input_ids), ('inference_prompt_ids.json', inference_ids)]:
        (directory / name).write_text(json.dumps(value, ensure_ascii=False))
    (directory / 'rendered_prompt.txt').write_text(prompt)
    manifest = {'input_token_count': len(input_ids), 'image_count': len(refs), 'images': refs,
                'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                'tools_sha256': hashlib.sha256(json.dumps(tools, sort_keys=True).encode()).hexdigest(),
                'state': state, 'training_contract_verified': False}
    (directory / 'input_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
