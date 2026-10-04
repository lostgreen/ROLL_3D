"""Lossless evidence projection, independent of viewer and training frameworks.

state_hash is deliberately unavailable until a structured environment state was
recorded; an image hash or a prompt hash is not an environment state identifier.
"""
import argparse
import json
from pathlib import Path
from contracts import canonical_hash

SCHEMA = 'video2scene.trajectory.v2'


def read_events(root):
    root = Path(root).resolve()
    with (root / 'events.jsonl').open() as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except ValueError:
                if not line.endswith('\n'): break  # in-flight append only
                raise
            path = (root / event['artifact_ref']).resolve()
            if not path.is_relative_to(root):
                raise ValueError('event payload escapes episode')
            yield event, json.loads(path.read_text())


def media_refs(value):
    found = []
    def visit(v):
        if isinstance(v, dict):
            if str(v.get('artifact_ref', '')).endswith('.media'): found.append(v)
            else:
                for item in v.values(): visit(item)
        elif isinstance(v, list):
            for item in v: visit(item)
    visit(value)
    return found


def stop_reason(end):
    if not end: return None
    if end.get('stop_reason'): return end['stop_reason']
    if end.get('finished'): return 'model_stop'
    error = str(end.get('error', '')).lower()
    if 'max_steps' in error: return 'max_steps'
    if 'max_wall' in error or 'timeout' in error: return 'timeout'
    if 'budget' in error: return 'budget_exceeded'
    if 'crash' in error or 'disconnect' in error: return 'environment_crash'
    if 'cancel' in error: return 'cancelled'
    return 'unknown'  # never invent a tool failure from an arbitrary exception


def from_events(root, run_id=None, events=None):
    root = Path(root)
    metadata, end, steps, evidence, requests = {}, {}, {}, [], {}
    for event, payload in (read_events(root) if events is None else events):
        evidence.append(event)
        kind, number = event['event'], event.get('step')
        if kind == 'episode.start': metadata = payload
        if kind == 'episode.end': end = payload
        if not isinstance(number, int): continue
        step = steps.setdefault(number, dict(step=number, state_hash=None, request_ids=[],
            observation_before={'message_refs':[], 'media_refs':[]},
            agent_action={'assistant_text':None, 'tool_calls':[]}, tool_transitions=[],
            environment_notices=[], reward=None, done=False, truncated=False,
            train_meta={'response_mask_available':False, 'advantage':None, 'return':None}))
        if kind == 'model.request':
            requests[payload['request_id']] = (event, payload)
            step['request_ids'].append(payload['request_id'])
        elif kind == 'model.response':
            req_event, req = requests.get(payload.get('request_id'), ({}, {}))
            purpose = req.get('purpose') or ('compaction' if 'context compaction assistant' in json.dumps(req.get('messages', [])) else 'main')
            if purpose != 'main' or payload.get('empty'): continue
            step['observation_before'] = {'message_refs':[req_event.get('artifact_ref')],
                                          'media_refs':media_refs(req.get('messages', []))}
            step['agent_action'] = {'assistant_text':payload.get('text'), 'tool_calls':payload.get('tool_calls', [])}
        elif kind == 'tool.end':
            result, log = payload.get('result', {}), payload.get('log', {})
            step['tool_transitions'].append(dict(call_id=payload.get('call_id'), tool=payload.get('name',log.get('name')),
                status=log.get('status'), observation={'text_ref':event['artifact_ref'], 'media_refs':media_refs(result),
                                                     'artifact_ids':log.get('artifact_ids', [])},
                scene_delta=log.get('structured', {}).get('scene_delta'),
                usage={k:log.get(k) for k in ('gpu_sec','external_cost_usd','generated_assets','imported_assets','latency_sec')}))
        elif kind == 'environment.notice': step['environment_notices'].append(event['artifact_ref'])
        elif kind == 'step.reward': step['reward'] = payload.get('reward')
        elif kind == 'scene.state':
            step['state_hash'] = canonical_hash({'version':1, 'state':payload, 'conditions':metadata.get('conditions', {})})
    ordered = [steps[i] for i in sorted(steps)]
    reason = stop_reason(end)
    if ordered and end:
        ordered[-1]['done'] = reason == 'model_stop'
        ordered[-1]['truncated'] = reason != 'model_stop'
    return dict(schema=SCHEMA, trajectory_id=metadata.get('trajectory_id',root.name),
        run_id=run_id or metadata.get('run_id'), episode_id=metadata.get('episode_id',root.name),
        group_id=metadata.get('group_id'), task_id=metadata.get('task_id'), protocol=metadata.get('protocol'),
        seed=metadata.get('seed'), model=metadata.get('model'), conditions=metadata.get('conditions', {}),
        prompt_bundle=metadata.get('prompt_bundle'), tool_contracts=metadata.get('tool_contracts', []),
        steps=ordered, evidence=evidence,
        episode_result=dict(status='Running' if not end else 'Finished' if end.get('finished') else 'Truncated',
            reward=end.get('reward'), step_rewards=[s['reward'] for s in ordered], stop_reason=reason,
            metrics=end.get('metrics', {})))


def main():
    parser = argparse.ArgumentParser(description='Export canonical trajectory IR without embedding media bytes')
    parser.add_argument('episode', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(from_events(args.episode), ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
