"""Audited SSE transport for Chat and Responses, without automatic retries."""
import gzip
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def invoke(base_url, key, payload, folder, timeout=240):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    wire = json.dumps(payload, ensure_ascii=False).encode()
    with gzip.open(folder / 'request.json.gz', 'wb') as handle:
        handle.write(wire)
    responses = 'input' in payload
    path = '/v1/responses' if responses else '/v1/chat/completions'
    request = urllib.request.Request(base_url.rstrip('/') + path, data=wire,
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    started = time.monotonic()
    dump(folder / 'attempt.json', {'started_at': time.time(), 'request_sha256': hashlib.sha256(wire).hexdigest()})
    first_event = first_delta = None
    native = None
    usage, calls = {}, {}
    message = {'role': 'assistant', 'content': ''}
    finish = response_model = identifier = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, (folder / 'response.sse').open('wb') as frames:
            for line in response:
                frames.write(line)
                frames.flush()
                if not line.startswith(b'data:'):
                    continue
                data = line[5:].strip()
                if not data or data == b'[DONE]':
                    continue
                event = json.loads(data)
                now = time.monotonic() - started
                if first_event is None:
                    first_event = now
                if event.get('error'):
                    raise RuntimeError(str(event['error'])[:400])
                if responses:
                    kind = event.get('type', '')
                    if kind.endswith('.delta') and event.get('delta') and first_delta is None:
                        first_delta = now
                    if kind in ('response.completed', 'response.incomplete'):
                        native = event['response']
                        break
                    if kind in ('response.failed', 'error'):
                        raise RuntimeError(str((event.get('response') or {}).get('error', event))[:400])
                else:
                    usage.update(event.get('usage') or {})
                    identifier = event.get('id', identifier)
                    response_model = event.get('model', response_model)
                    for choice in event.get('choices') or []:
                        if choice.get('index', 0) != 0:
                            continue
                        finish = choice.get('finish_reason') or finish
                        delta = choice.get('delta') or {}
                        if any(delta.get(k) for k in ('content', 'reasoning_content', 'reasoning', 'tool_calls')) and first_delta is None:
                            first_delta = now
                        for k in ('content', 'reasoning_content', 'reasoning'):
                            if isinstance(delta.get(k), str):
                                message[k] = message.get(k, '') + delta[k]
                        for call in delta.get('tool_calls') or []:
                            item = calls.setdefault(call['index'], {'id': '', 'type': 'function',
                                                    'function': {'name': '', 'arguments': ''}})
                            if call.get('id'):
                                if item['id'] and item['id'] != call['id']:
                                    raise ValueError('Native call ID changed inside stream')
                                item['id'] = call['id']
                            for k, v in (call.get('function') or {}).items():
                                if k in ('name', 'arguments') and v:
                                    if k != 'name' or item['function'][k] != v:
                                        item['function'][k] += v
        if responses:
            if native is None:
                raise RuntimeError('Responses stream incomplete')
            message['_responses_output'] = native.get('output', [])
            native_calls = []
            for item in message['_responses_output']:
                if item['type'] == 'function_call':
                    native_calls.append({'id': item['call_id'], 'type': 'function',
                                         'function': {'name': item['name'], 'arguments': item['arguments']}})
                if item['type'] == 'message':
                    message['content'] += ''.join(p.get('text', '') for p in item.get('content', []))
            if native_calls:
                message['tool_calls'] = native_calls
            usage = native.get('usage') or {}
            response_model, identifier = native.get('model'), native.get('id')
            finish = 'length' if native.get('status') == 'incomplete' else ('tool_calls' if native_calls else 'stop')
        else:
            if finish is None:
                raise RuntimeError('Chat stream incomplete')
            if calls:
                message['tool_calls'] = [calls[k] for k in sorted(calls)]
        dump(folder / 'response.json', native if responses else {'id': identifier, 'model': response_model,
             'choices': [{'message': message, 'finish_reason': finish}], 'usage': usage})
        dump(folder / 'assistant.json', message)
        incoming = usage.get('input_tokens', usage.get('prompt_tokens'))
        outgoing = usage.get('output_tokens', usage.get('completion_tokens'))
        details = usage.get('input_tokens_details') or usage.get('prompt_tokens_details') or {}
        cached = details.get('cached_tokens')
        record = {'http_status': 200, 'response_model': response_model, 'finish_reason': finish,
                  'seconds': round(time.monotonic() - started, 3),
                  'first_event_seconds': first_event, 'first_content_delta_seconds': first_delta,
                  'input_tokens': incoming, 'cached_tokens': cached, 'output_tokens': outgoing,
                  'reasoning_tokens': (usage.get('output_tokens_details') or usage.get('completion_tokens_details') or {}).get('reasoning_tokens'),
                  'usage': usage, 'request_sha256': hashlib.sha256(wire).hexdigest()}
        dump(folder / 'metadata.json', record)
        return message, record
    except Exception as exc:
        if isinstance(exc, urllib.error.HTTPError):
            body = exc.read().decode(errors='replace').replace(key, '<REDACTED>')
            (folder / 'http_error.txt').write_text(body)
        dump(folder / 'failure.json', {'type': type(exc).__name__, 'message': str(exc).replace(key, '<REDACTED>')[:400],
             'http_status': getattr(exc, 'code', None), 'seconds': time.monotonic() - started,
             'usage_known': False, 'automatic_retries': 0})
        raise RuntimeError(str(exc).replace(key, '<REDACTED>')[:400]) from None
