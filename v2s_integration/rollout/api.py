"""Bounded native API reconstruction using immutable history.

Run against a versioned source snapshot with ``python -m v2s_integration.rollout.api``.
Credentials are read from an external file and never included in request artifacts.
"""
import argparse
import copy
import json
from pathlib import Path
import re
import time

from v2s_integration.agent.api_context import APIContext, api_payload, check_budget, digest
from v2s_integration.agent.api_transport import dump, invoke


def native_system(text):
    if 'INTERACTION\n' in text and 'When several construction methods' in text:
        start, rest = text.split('INTERACTION\n', 1)
        return start + 'INTERACTION\nUse native function calls.\n' + rest[rest.index('When several construction methods'):]
    return text


def tools_for_api(model_tools):
    tools, aliases = copy.deepcopy(model_tools), {}
    for tool in tools:
        function = tool['function']
        original = function['name']
        alias = re.sub(r'[^a-zA-Z0-9_-]', '_', original)
        if alias in aliases or len(alias) > 64:
            raise ValueError('Ambiguous or overlong tool alias')
        aliases[alias] = original
        function['name'] = alias
        function['description'] = function['description'].split(' Example syntax')[0]
    return tools, aliases


def decode_action(message, aliases):
    calls = message.get('tool_calls') or []
    if len(calls) != 1:
        raise ValueError('Exactly one native tool call is required; no action was executed')
    call = calls[0]
    if call.get('type') != 'function' or not call.get('id'):
        raise ValueError('Invalid native call identity; no action was executed')
    function = call['function']
    name = aliases[function['name']]
    args = json.loads(function['arguments'])
    if not isinstance(args, dict):
        raise ValueError('Tool arguments must be a JSON object; no action was executed')
    return '<tool_call>' + json.dumps({'name': name, 'arguments': args}, ensure_ascii=False) + '</tool_call>'


def collect(env, *, output, model, protocol, base_url, key, effort=None,
            parameters=None, max_text_bytes=100000, max_wire_bytes=4000000, max_images=12):
    """The environment owns execution; provider-specific history stays in this layer."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    status = {'status': 'running', 'model': model, 'protocol': protocol, 'effort': effort,
              'context_policy': 'append_only_snapshot_delta_v1', 'steps': 0}
    budget_options = dict(max_text_bytes=max_text_bytes, max_wire_bytes=max_wire_bytes, max_images=max_images)
    cache_key = 'v2s-' + digest(str(output.resolve()))[:32]
    started = time.monotonic()
    try:
        env.reset(seed=0)
        context = APIContext(env, system_prompt=native_system(env.system_prompt))
        tools, aliases = tools_for_api(env.model_tools)
        for step in range(1, env.max_steps + 1):
            def prepare():
                return api_payload(context.messages, tools, model=model, protocol=protocol,
                    max_output_tokens=env.max_output_tokens, cache_key=cache_key, effort=effort, parameters=parameters)
            payload = prepare()
            checkpointed = False
            try:
                budget = check_budget(payload, **budget_options)
            except ValueError:
                if context.steps == 0:
                    raise
                context.checkpoint(env)
                checkpointed = True
                payload = prepare()
                budget = check_budget(payload, **budget_options)
            status.update(phase='api', current_step=step)
            dump(output / 'status.json', status)
            message, record = invoke(base_url, key, payload, output / f'api_{step:03d}')
            record.update(step=step, budget=budget, checkpointed=checkpointed, checkpoints=context.checkpoints)
            records.append(record)
            # Preserve partial replies in artifacts; do not replay incomplete tool arguments.
            if record['finish_reason'] == 'length':
                raise RuntimeError('Output limit reached; truncated reply was not executed')
            try:
                action = decode_action(message, aliases)
            except (ValueError, KeyError, TypeError) as exc:
                action = 'Invalid native API action: ' + str(exc)
            status.update(phase='tool')
            dump(output / 'status.json', status)
            _, _, done, truncated, info = env.step(action)
            turn = env.turns[-1]
            turn['scene_state'] = copy.deepcopy(env.public_state)
            context.append(message, turn)
            record.update(tool_status=turn['feedback']['status'], scene_version=context.version,
                          scene_hash=digest(env.public_state))
            dump(output / f'api_{step:03d}' / 'metadata.json', record)
            status.update(steps=step)
            if done or truncated:
                status.update(status='finished' if done else 'stopped', termination_reason=info['termination_reason'])
                break
        if status['status'] == 'running':
            status.update(status='stopped', termination_reason='max_steps')
    except Exception as exc:
        status.update(status='failed', error_type=type(exc).__name__, error=str(exc).replace(key, '<REDACTED>')[:400])
    finally:
        if not env.closed:
            env.finish_episode('api_collector_end')
        env.close()
        status.update(records=records, elapsed_seconds=time.monotonic() - started)
        dump(output / 'summary.json', status)
        dump(output / 'status.json', status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--protocol', choices=['responses', 'chat'], required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--key-file', required=True)
    parser.add_argument('--effort')
    parser.add_argument('--parameters', default='{}', help='Provider extras, e.g. thinking.type; never credentials')
    parser.add_argument('--max-steps', type=int, default=12)
    parser.add_argument('--max-output-tokens', type=int, default=4096)
    parser.add_argument('--max-text-bytes', type=int, default=100000)
    args = parser.parse_args()
    from v2s_integration.envs.reconstruction import ReconstructionEnv
    env = ReconstructionEnv(args.task, Path(args.output).parent / (Path(args.output).name + '_environment'),
                            max_steps=args.max_steps, max_output_tokens=args.max_output_tokens)
    key = Path(args.key_file).read_text().strip()
    if not key:
        raise ValueError('Empty API credential')
    result = collect(env, output=args.output, model=args.model, protocol=args.protocol,
        base_url=args.base_url, key=key, effort=args.effort, parameters=json.loads(args.parameters),
        max_text_bytes=args.max_text_bytes)
    print(json.dumps({k: result.get(k) for k in ('status', 'steps', 'termination_reason', 'error')}))
    return int(result['status'] == 'failed')


if __name__ == '__main__':
    raise SystemExit(main())
