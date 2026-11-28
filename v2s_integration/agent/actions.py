"""Parse one native JSON/XML tool envelope; reasoning is never executable."""
from dataclasses import dataclass
import io
import json
import re
import tokenize

CALL = re.compile(r'\s*<tool_call>\s*<function=([\w.]+)>\s*(.*?)\s*</function>\s*</tool_call>\s*', re.S)
PARAMETER = re.compile(r'\s*<parameter=([\w]+)>(.*?)</parameter>', re.S)
JSON_CALL = re.compile(r'\s*<tool_call>\s*(\{.*\})\s*</tool_call>\s*', re.S)
PARAMETER_START = re.compile(r'\s*<parameter=([\w]+)>')
RESERVED_PREFIX = re.compile(r'</?(?:tool_call|function|parameter|think)(?=[\s=>/])|<\|[^>]*\|>')


def _python_string_spans(text):
    """Keep XML-looking bytes inside Blender code strings/comments as data."""
    starts, offset = [], 0
    for line in text.splitlines(keepends=True):
        starts.append(offset)
        offset += len(line)
    starts.append(offset)
    spans = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type in (tokenize.STRING, tokenize.COMMENT):
                spans.append((starts[token.start[0] - 1] + token.start[1],
                              starts[token.end[0] - 1] + token.end[1]))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # Erroneous teacher code must remain a trainable failed action. Complete
        # string tokens obtained before the syntax error still protect literals.
        pass
    return spans


def _xml_parameters(body):
    fields = {}
    while body.strip():
        opening = PARAMETER_START.match(body)
        if opening is None:
            raise ValueError('Malformed or truncated native tool parameter')
        key = opening.group(1)
        if key in fields:
            raise ValueError('Duplicate native tool parameter')
        remaining = body[opening.end():]
        protected = _python_string_spans(remaining) if key == 'code' else []
        closing = next((m for m in re.finditer('</parameter>', remaining)
                        if not any(start <= m.start() < end for start, end in protected)), None)
        if closing is None:
            raise ValueError('Malformed or truncated native tool parameter')
        fields[key] = remaining[:closing.start()].removeprefix('\n').removesuffix('\n')
        body = remaining[closing.end():]
    return fields


def _envelope(text):
    """Decode the entire envelope, rejecting concatenated calls and tails."""
    text = text.strip()
    if not text.startswith('<tool_call>'):
        raise ValueError('Exactly one complete native tool_call envelope is required')
    payload = text[len('<tool_call>'):].lstrip()
    if payload.startswith('{'):
        request, end = json.JSONDecoder().raw_decode(payload)
        if payload[end:].strip() != '</tool_call>':
            raise ValueError('Exactly one complete native tool_call envelope is required')
        name, arguments = request.get('name'), request.get('arguments', {})
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise ValueError('Tool call requires a string name and object arguments')
        return name, arguments, 'json'
    match = CALL.fullmatch(text)
    if match is None:
        raise ValueError('Exactly one complete native tool_call envelope is required')
    name, body = match.groups()
    # Parsing the entire parameter body prevents the regex from joining two
    # adjacent envelopes into an apparently valid single function match.
    return name, _xml_parameters(body), 'xml'


def _terminal_action(text):
    """Allow ordinary prose followed by one terminal native envelope."""
    start = text.find('<tool_call>')
    if start < 0 or RESERVED_PREFIX.search(text[:start]):
        raise ValueError('Exactly one complete native tool_call envelope is required')
    candidate = text[start:].strip()
    _envelope(candidate)
    return candidate


def _earlier_closed_action(text):
    """A second thinking delimiter cannot hide an already completed action."""
    for boundary in re.finditer('</think>', text):
        remainder = text[boundary.end():]
        for end in re.finditer('</tool_call>', remainder):
            try:
                _terminal_action(remainder[:end.end()])
                return True
            except ValueError:
                pass
    return False


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


def action_text(response):
    text = response.strip()
    # Literal thinking markers inside a complete tool envelope are argument data.
    try:
        _envelope(text)
        return text
    except ValueError:
        pass
    for boundary in reversed(list(re.finditer('</think>', text))):
        if _earlier_closed_action(text[:boundary.start()]):
            continue
        candidate = text[boundary.end():].strip()
        try:
            return _terminal_action(candidate)
        except ValueError:
            pass
    return _terminal_action(text)


def parse_action(response, step, registry):
    text = action_text(response)
    name, fields, format_name = _envelope(text)
    if format_name == 'json':
        arguments = fields
        return ToolCall(f'local_{step}_0', name, arguments)
    entry = registry.lookup(name)
    schema = entry[0].input_schema if entry else {}
    properties = schema.get('properties', {})
    args = {}
    for key, value in fields.items():
        if properties.get(key, {}).get('type') == 'string':
            args[key] = value
        else:
            try:
                args[key] = json.loads(value)
            except ValueError:
                # Unknown tools/keys are rejected by the allowlist/schema layer.
                args[key] = value
    return ToolCall(f'local_{step}_0', name, args)
