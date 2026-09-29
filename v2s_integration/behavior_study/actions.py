"""Parse Qwen3.5's native function/parameter tags and legacy explicit JSON."""
import json
import re

from . import runtime  # establish the frozen harness import paths
from adapters import ToolCall
from action_response import final_action_text
from local_vlm import parse_tool_calls

CALL = re.compile(r'\s*<tool_call>\s*<function=([\w.]+)>\s*(.*?)\s*</function>\s*</tool_call>\s*', re.S)
PARAMETER = re.compile(r'\s*<parameter=([\w]+)>(.*?)</parameter>', re.S)


def parse_action(response, step, registry):
    text = response.strip()
    match = CALL.fullmatch(text)
    if match is None and '</think>' in text:
        # Qwen may prefill the opening think tag in the input. Never execute
        # examples before its explicit closing boundary.
        text = text.rsplit('</think>', 1)[1].strip()
        match = CALL.fullmatch(text)
    if match is None:
        calls = parse_tool_calls(final_action_text(response), step)
        if len(calls) != 1:
            raise ValueError('Exactly one explicit final tool call is required')
        return calls[0]
    name, body = match.groups()
    entry = registry.lookup(name)
    schema = entry[0].input_schema if entry else {}
    properties = schema.get('properties', {})
    args = {}
    while body.strip():
        parameter = PARAMETER.match(body)
        if parameter is None:
            raise ValueError('Malformed or truncated native tool parameter')
        key, value = parameter.groups()
        if key in args:
            raise ValueError('Duplicate native tool parameter')
        # Only framing newlines are removed; preserve Python code indentation.
        value = value.removeprefix('\n').removesuffix('\n')
        if properties.get(key, {}).get('type') == 'string':
            args[key] = value
        else:
            try:
                args[key] = json.loads(value)
            except ValueError:
                # Unknown tools/keys are rejected by the allowlist/schema layer.
                args[key] = value
        body = body[parameter.end():]
    return ToolCall(f'local_{step}_0', name, args)
