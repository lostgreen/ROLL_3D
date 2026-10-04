"""Compact, partial-run-safe behavior metrics; never emits model text or arguments."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

SUCCESS = {'succeeded', 'success', 'ok'}
COLLECTED = {'finish', 'max_steps'}
LIMITED = {'context_budget_exhausted', 'budget_exceeded'}
GROUPS = ('blender', 'hunyuan', 'assets', 'mixed')


def read_json(path, issues):
    """Readers can race non-atomic environment writes; report rather than invent data."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        if path.exists():
            issues.append(path.name)
        return None


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def action_hash(call):
    def normalize(value):
        if isinstance(value, str):
            return '\n'.join(line.rstrip() for line in value.replace('\r\n', '\n').split('\n')).rstrip()
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value
    value = normalize({'name': call.get('name'), 'arguments': call.get('arguments', {})})
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def object_count(state):
    objects = state.get('objects') if isinstance(state, dict) else None
    return len(objects) if isinstance(objects, (list, dict)) else None


def summarize_episode(path, outcome=None):
    issues = []
    manifest = read_json(path / 'manifest.json', issues) or {}
    result = read_json(path / 'result.json', issues) or {}
    collection = read_json(path / 'collection.json', issues) or {}
    outcome = outcome or {}
    group = manifest.get('group', result.get('group', path.name.split('_')[0]))
    reason = result.get('termination_reason', outcome.get('termination_reason'))
    state = ('failed' if outcome.get('status') == 'failed' or outcome.get('cleanup_error') else
             'collected' if reason in COLLECTED else 'limited' if reason in LIMITED else
             'failed' if reason else 'in_progress_or_interrupted')
    tools, errors, successes, failed_hashes = Counter(), Counter(), Counter(), Counter()
    artifacts, imported = set(), set()
    events = []
    repeated, invalid, execution_failed, unknown = 0, 0, 0, 0
    scene_counts = []
    for folder in sorted(path.glob('step_*')):
        event = read_json(folder / 'event.json', issues)
        if not isinstance(event, dict):
            unknown += 1
            continue
        call, feedback = event.get('call') or {}, event.get('feedback') or {}
        name = call.get('name') or feedback.get('tool_name')
        status, error = feedback.get('status'), feedback.get('error_type')
        ok = status in SUCCESS
        if name:
            tools[name] += 1
        if ok:
            successes[name or 'unknown'] += 1
        elif status is None:
            unknown += 1
        elif error == 'invalid_action':
            invalid += 1
        else:
            execution_failed += 1
        if not ok and status is not None:
            errors[error or 'unspecified_tool_failure'] += 1
            if call:
                digest = action_hash(call)
                repeated += int(failed_hashes[digest] > 0)
                failed_hashes[digest] += 1
        ids = feedback.get('artifact_ids') or []
        artifacts.update(str(item) for item in ids)
        if ok and name in ('scene.import_artifact', 'asset.import'):
            imported.update(str(item) for item in ids)
        count = object_count(event.get('scene_state'))
        if count is not None:
            scene_counts.append(count)
        events.append({'tool': name, 'ok': ok, 'failed': status is not None and not ok,
                       'step': event.get('step')})
    recovery = Counter()
    for before, after in zip(events, events[1:]):
        if before['failed'] and after['step'] == (before['step'] or 0) + 1:
            route = 'same_tool' if before['tool'] == after['tool'] else 'different_tool'
            recovery[route + ('_success' if after['ok'] else '_not_success')] += 1
    if events and events[-1]['failed']:
        recovery['no_next_observed_step'] += 1
    inputs, outputs, times, images, omissions = [], [], [], [], []
    requests, generations, capped = 0, 0, 0
    initial_count = None
    max_output = manifest.get('max_output_tokens')
    for request in sorted((path / 'requests').glob('*')):
        if not request.is_dir():
            continue
        requests += 1
        data = read_json(request / 'input_manifest.json', issues)
        if isinstance(data, dict):
            for value, values in ((data.get('input_token_count'), inputs), (data.get('image_count'), images),
                                  ((data.get('state') or {}).get('omitted_history_turns'), omissions)):
                if number(value) is not None:
                    values.append(value)
            if (data.get('state') or {}).get('turn_index') == 0 or request.name == '001':
                initial_count = object_count((data.get('state') or {}).get('public_scene_state'))
        generation = read_json(request / 'generation.json', issues)
        if isinstance(generation, dict):
            generations += 1
            ids = generation.get('response_ids')
            if isinstance(ids, list):
                outputs.append(len(ids))
                capped += int(max_output is not None and len(ids) >= max_output)
            if number(generation.get('decision_seconds')) is not None:
                times.append(generation['decision_seconds'])
    final_count = scene_counts[-1] if scene_counts else initial_count
    return {'episode_id': path.name, 'group': group,
            'seed': manifest.get('seed', collection.get('environment_seed', outcome.get('environment_seed'))),
            'state': state, 'termination_reason': reason, 'result_present': bool(result),
            'reported_steps': result.get('steps'), 'observed_steps': len(events),
            'incomplete_step_records': unknown, 'requests': requests, 'generations': generations,
            'tool_calls': dict(tools), 'tool_successes': dict(successes), 'error_types': dict(errors),
            'invalid_actions': invalid, 'execution_failures': execution_failed,
            'repeated_identical_failed_actions': repeated, 'immediate_recovery': dict(recovery),
            'finish_called': tools['finish'], 'finish_succeeded': successes['finish'],
            'initial_objects': initial_count, 'last_observed_objects': final_count,
            'peak_objects': max(scene_counts) if scene_counts else initial_count,
            'object_delta': final_count - initial_count if final_count is not None and initial_count is not None else None,
            'artifact_id_count': len(artifacts), 'import_returned_artifact_id_count': len(imported),
            'generation_tool_successes': successes['asset.generate_3d_from_image'],
            'import_tool_successes': successes['scene.import_artifact'] + successes['asset.import'],
            'input_tokens_sum': sum(inputs), 'input_tokens_max': max(inputs, default=None),
            'image_count_max': max(images, default=None), 'history_omitted_turns_max': max(omissions, default=None),
            'response_id_count_sum': sum(outputs), 'response_id_count_max': max(outputs, default=None),
            'response_id_at_output_cap_approx': capped, 'decision_seconds': round(sum(times), 3),
            'response_ids_per_decision_second': round(sum(outputs) / sum(times), 3) if sum(times) else None,
            'scene_blend_present': (path / 'scene.blend').is_file(),
            'unreadable_records': len(issues)}


def aggregate(traces):
    summary = {'episodes_started': len(traces), 'states': dict(Counter(t['state'] for t in traces)),
               'termination_reasons': dict(Counter(t['termination_reason'] or 'not_recorded' for t in traces))}
    for key in ('observed_steps', 'requests', 'generations', 'invalid_actions', 'execution_failures',
                'repeated_identical_failed_actions', 'finish_called', 'finish_succeeded',
                'artifact_id_count', 'generation_tool_successes', 'import_tool_successes',
                'input_tokens_sum', 'response_id_count_sum', 'response_id_at_output_cap_approx',
                'decision_seconds', 'incomplete_step_records', 'unreadable_records'):
        summary[key] = sum(t[key] for t in traces)
    for key in ('tool_calls', 'tool_successes', 'error_types', 'immediate_recovery'):
        counts = Counter()
        for trace in traces:
            counts.update(trace[key])
        summary[key] = dict(counts)
    for key in ('input_tokens_max', 'response_id_count_max', 'image_count_max', 'history_omitted_turns_max'):
        summary[key] = max((t[key] for t in traces if t[key] is not None), default=None)
    seconds = summary['decision_seconds']
    summary['response_ids_per_decision_second'] = round(summary['response_id_count_sum'] / seconds, 3) if seconds else None
    return summary


def summarize(root):
    root = Path(root).resolve()
    outcomes = {}
    issues = []
    for path in root.glob('**/outcomes.json'):
        items = read_json(path, issues)
        for item in items if isinstance(items, list) else []:
            if isinstance(item, dict) and item.get('episode'):
                outcomes[Path(item['episode']).name] = item
    paths = sorted({p for p in root.glob('**/episodes/*') if p.is_dir()})
    if (root / 'manifest.json').exists() and (root / 'images').is_dir() and root not in paths:
        paths.append(root)
    traces = [summarize_episode(path, outcomes.get(path.name)) for path in paths]
    return {'schema_version': 1, 'root': str(root), 'quality_evaluated': False,
            'notes': ['Response ID lengths may include padding; cap flags are approximate, not proven truncation.',
                      'Decision timing includes inference request processing; rate is not pure decode throughput.',
                      'Tool success and termination do not establish reconstruction quality.',
                      'Missing result means in progress or interrupted, never successful completion.',
                      'Artifact counts use returned IDs, not verified generated/imported geometry.'],
            'overall': aggregate(traces),
            'groups': {group: aggregate([t for t in traces if t['group'] == group])
                       for group in sorted(set(GROUPS) | {t['group'] for t in traces})},
            'trajectories': traces, 'unreadable_outcome_files': len(issues)}


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    atomic_write(args.output, json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': args.output, 'episodes': result['overall']['episodes_started'],
                      'states': result['overall']['states']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
