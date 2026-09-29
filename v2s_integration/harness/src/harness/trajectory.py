"""Append-only interaction evidence; media and large payloads are content addressed."""
import base64
import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path


class TrajectoryRecorder:
    def __init__(self, root, metadata=None):
        self.root = Path(root) / ('episode_' + uuid.uuid4().hex)
        (self.root / 'blobs').mkdir(parents=True)
        self.seq = 0
        self.step = None
        self._lock = threading.RLock()
        self.event('episode.start', schema='video2scene.trajectory.v2', episode_id=self.root.name,
                   trajectory_id=self.root.name.removeprefix('episode_'), **(metadata or {}))

    def blob(self, data, suffix):
        digest = hashlib.sha256(data).hexdigest()
        path = self.root / 'blobs' / (digest + suffix)
        with self._lock:
            if not path.exists():
                with path.open('wb') as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
        return {'artifact_ref': str(path.relative_to(self.root)), 'sha256': digest}

    def pack(self, value):
        if isinstance(value, str) and value.startswith('data:') and ';base64,' in value:
            head, data = value.split(';base64,', 1)
            return dict(self.blob(base64.b64decode(data, validate=True), '.media'),
                        encoding='data_uri', mime=head[5:])
        if isinstance(value, dict):
            # Native tool images and Anthropic/Gemini blocks.
            mime = value.get('mime') or value.get('media_type') or value.get('mimeType')
            if mime and isinstance(value.get('data'), str):
                packed = {k:self.pack(v) for k,v in value.items() if k != 'data'}
                packed['data'] = dict(self.blob(base64.b64decode(value['data'], validate=True), '.media'), encoding='base64')
                return packed
            return {k:self.pack(v) for k,v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.pack(v) for v in value]
        return value

    def event(self, kind, event_step=None, **payload):
        # Full contents stay on disk; events carry only a small pointer.
        with self._lock:
            body = json.dumps(self.pack(payload), ensure_ascii=False).encode()
            ref = self.blob(body, '.json')
            row = dict(seq=self.seq, time=time.time(),
                       step=self.step if event_step is None else event_step,
                       event=kind, **ref)
            with (self.root / 'events.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
                f.flush()
                os.fsync(f.fileno())
            self.seq += 1
        return ref

    def restore(self, value):
        if isinstance(value, dict) and 'artifact_ref' in value and value.get('encoding'):
            path = (self.root / value['artifact_ref']).resolve()
            if not path.is_relative_to(self.root.resolve()): raise ValueError('artifact escapes episode')
            data = base64.b64encode(path.read_bytes()).decode()
            return f"data:{value['mime']};base64,{data}" if value['encoding']=='data_uri' else data
        if isinstance(value, dict): return {k:self.restore(v) for k,v in value.items()}
        if isinstance(value, list): return [self.restore(v) for v in value]
        return value


class RecordingAdapter:
    """Capture every adapter invocation, including retries and compaction requests.

    This is the adapter-boundary input, not a claim to capture HTTP headers or
    every provider-specific wire parameter. Credentials are never serialized.
    """
    def __init__(self, adapter, recorder):
        self.adapter, self.recorder = adapter, recorder
        self._origins = {}

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def _mark(self, message, origin):
        self._origins[id(message)] = (message, origin)
        return message

    def system_message(self, content):
        return self._mark(self.adapter.system_message(content), 'task_author')

    def user_message(self, content, **kwargs):
        origin = 'task_author'
        if content.startswith('<environment_notice kind="compaction_memory">'): origin = 'compaction_memory'
        elif content.startswith('<environment_notice'): origin = 'harness_notice'
        message = self._mark(self.adapter.user_message(content, **kwargs), origin)
        if origin != 'task_author': self.recorder.event('environment.notice', origin=origin, message=message)
        return message

    def tool_result_messages(self, calls, results):
        messages = self.adapter.tool_result_messages(calls, results)
        for message in messages:
            role = message.get('role')
            self._mark(message, 'tool_relay' if role == 'user' and any(text in str(message) for text in ('Here are the image(s)', 'Tool returned image(s):')) else 'tool_result')
        return messages

    def run(self, messages, tools, purpose="main", attempt=0):
        from contracts import request_conditions, canonical_hash, message_origin
        if purpose == "main":
            active = {id(m) for m in messages}
            self._origins = {key:value for key,value in self._origins.items() if key in active}
        started = time.monotonic()
        request_id = uuid.uuid4().hex
        self.recorder.event('model.request', request_id=request_id, messages=messages, tools=tools,
                            purpose=purpose, attempt=attempt, conditions=request_conditions(messages, tools),
                            message_meta=[dict(sha256=canonical_hash(m),
                                origin=self._origins.get(id(m), (None, message_origin(m)))[1],
                                inferred=id(m) not in self._origins and m.get("role") not in ("assistant", "model")) for m in messages],
                            model=getattr(self.adapter,'model',None),
                            parameters={k:getattr(self.adapter,k,None) for k in
                                        ('temperature','max_tokens','reasoning_effort','supports_vision','seed','parallel_tool_calls')})
        try:
            turn = self.adapter.run(messages, tools)
        except Exception as exc:
            self.recorder.event('model.error', request_id=request_id, error_type=type(exc).__name__, error=str(exc))
            raise
        reasoning = getattr(self.adapter, 'last_visible_reasoning', None)
        reasoning_ref = self.recorder.event('model.reasoning_summary', request_id=request_id, evidence='self_report', content=reasoning) if reasoning else None
        self.recorder.event('model.response', request_id=request_id, raw=turn.raw, text=turn.text,
                            reasoning_available=bool(reasoning), reasoning_ref=reasoning_ref,
                            tool_calls=[vars(c) for c in turn.tool_calls],
                            usage=getattr(self.adapter,'last_api_usage',None), latency_sec=time.monotonic()-started,
                            empty=not turn.tool_calls and not (turn.text or '').strip())
        return turn
