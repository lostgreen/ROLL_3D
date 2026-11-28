"""Auditable model inputs shared by Raw-CoT SFT and the student harness."""
import copy
import hashlib
import json
import re
from PIL import Image
from v2s_integration.agent.context_policy import (
    INITIAL_DESCRIPTION, V2_POLICY_VERSION, active_context_policy,
    observation_label, omitted_label, select_observations,
)
from v2s_integration.agent.observation_format import (
    format_observation_messages, format_public_scene_state, observation_contract_sha256,
)


def build_messages(env, history_limit=None):
    version = active_context_policy(env)
    if history_limit is not None and history_limit != len(env.turns):
        raise ValueError('Raw-CoT context requires complete tool history; history pruning is disabled')
    turns = env.turns
    occurrences, sources, selected, policy = select_observations(env)
    pictures, refs = [], []
    emitted = set()

    def append_image(value, ref, label):
        with Image.open(ref['path']) as source:
            pictures.append(source.convert('RGB'))
        refs.append(copy.deepcopy(ref))
        value.extend([{'type': 'text', 'text': label}, {'type': 'image'}])

    def observation(text, keys):
        value = [{'type': 'text', 'text': text}]
        for key in keys:
            identity = occurrences[key]
            source = sources[identity]
            if identity in selected and identity not in emitted:
                ref = dict(source['ref'], generated_step=source['generated_step'],
                           camera_id=source['camera_id'], scene_version=source['scene_version'],
                           sha256=source['sha256'], source_kind=source['kind'])
                append_image(value, ref, observation_label(source))
                emitted.add(identity)
            elif identity in emitted:
                value.append({'type': 'text', 'text': '[同一观察已在历史中保留，不重复附图。image_id=' + identity + ']'})
            else:
                value.append({'type': 'text', 'text': omitted_label(source, selected, sources)})
        return value

    initial = getattr(env, 'initial_scene_description', None) or INITIAL_DESCRIPTION
    references = [{'type': 'text', 'text': env.task_prompt + '\n\n' + initial}]
    for ref in env.reference_refs:
        # Public references are semantic slots and must not be deduplicated.
        append_image(references, ref, ref.get('label', 'Reference'))
    messages = [{'role': 'system', 'content': env.system_prompt},
                {'role': 'user', 'content': references},
                {'role': 'user', 'content': observation('Initial scene observation S0',
                    [('initial', i) for i in range(len(env.initial_images))])}]
    for index, turn in enumerate(turns):
        call = turn['call']
        # Reasoning text is archived separately; never reinterpret it as an action.
        if call:
            messages.append({'role': 'assistant', 'content': '', 'tool_calls': [{
                'id': call['id'], 'type': 'function',
                'function': {'name': call['name'], 'arguments': copy.deepcopy(call['arguments'])}}]})
            feedback = copy.deepcopy(turn['feedback'])
            messages.append({'role': 'tool', 'tool_call_id': call['id'], 'name': call['name'],
                             'content': json.dumps(feedback, ensure_ascii=False)})
        else:
            messages.append({'role': 'user', 'content': 'Environment notice: ' +
                             json.dumps(turn['feedback'], ensure_ascii=False)})
        keys = [(index, 'tool', i) for i in range(len(turn.get('tool_images', [])))]
        keys += [(index, 'scene', i) for i in range(len(turn.get('scene_images', [])))]
        messages.append({'role': 'user', 'content': observation('Environment observation', keys)})
    state = {'public_scene_state': copy.deepcopy(env.public_state),
             'omitted_history_turns': 0, 'context_policy': policy}
    # The audit manifest is returned/saved, not inserted into model context.
    messages.append({'role': 'user', 'content': 'Current public scene state: ' +
                     json.dumps({'public_scene_state': env.public_state}, ensure_ascii=False)})
    if version == V2_POLICY_VERSION:
        messages = format_observation_messages(messages)
        state['public_scene_state'] = format_public_scene_state(env.public_state)
        state['context_policy']['observation_contract_sha256'] = observation_contract_sha256()
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
