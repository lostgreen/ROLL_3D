"""Parse one native JSON/XML tool envelope; reasoning is never executable."""
from dataclasses import dataclass
import json
import re

CALL = re.compile(r'\s*<tool_call>\s*<function=([\w.]+)>\s*(.*?)\s*</function>\s*</tool_call>\s*', re.S)
PARAMETER = re.compile(r'\s*<parameter=([\w]+)>(.*?)</parameter>', re.S)
JSON_CALL = re.compile(r'\s*<tool_call>\s*(\{.*\})\s*</tool_call>\s*', re.S)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


def action_text(response):
    text = response.strip()
    # Literal thinking markers inside a complete tool envelope are argument data.
    if CALL.fullmatch(text) or JSON_CALL.fullmatch(text):
        return text
    for boundary in reversed(list(re.finditer('</think>', text))):
        candidate = text[boundary.end():].strip()
        if CALL.fullmatch(candidate) or JSON_CALL.fullmatch(candidate):
            return candidate
    if not (CALL.fullmatch(text) or JSON_CALL.fullmatch(text)):
        raise ValueError('Exactly one complete native tool_call envelope is required')
    return text


def parse_action(response, step, registry):
    text = action_text(response)
    match = CALL.fullmatch(text)
    if match is None:
        request = json.loads(JSON_CALL.fullmatch(text).group(1))
        name, arguments = request.get('name'), request.get('arguments', {})
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise ValueError('Tool call requires a string name and object arguments')
        return ToolCall(f'local_{step}_0', name, arguments)
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
