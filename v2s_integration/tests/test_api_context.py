"""Offline API history regression tests; no model, Blender, or credentials."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from v2s_integration.agent.api_context import APIContext, api_payload, check_budget, state_patch


def apply_patch(value, ops):
    value = copy.deepcopy(value)
    for op in ops:
        if op['path'] == '':
            value = copy.deepcopy(op['value'])
            continue
        keys = [p.replace('~1', '/').replace('~0', '~') for p in op['path'].split('/')[1:]]
        parent = value
        for key in keys[:-1]:
            parent = parent[int(key)] if isinstance(parent, list) else parent[key]
        key = keys[-1]
        if isinstance(parent, list):
            if op['op'] == 'remove':
                parent.pop(int(key))
            elif op['op'] == 'add':
                parent.insert(len(parent) if key == '-' else int(key), copy.deepcopy(op['value']))
            else:
                parent[int(key)] = copy.deepcopy(op['value'])
        elif op['op'] == 'remove':
            del parent[key]
        else:
            parent[key] = copy.deepcopy(op['value'])
    return value


class APIContextTests(unittest.TestCase):
    def setUp(self):
        self.env = SimpleNamespace(system_prompt='system', task_prompt='task', reference_refs=[],
            initial_images=[], current_images=[], public_state={'objects': [{'name': 'cube', 'x': 0}]})
        self.context = APIContext(self.env)
        self.native = {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'call_native_1', 'type': 'function', 'function': {'name': 'inspect', 'arguments': '{}'}}],
            '_responses_output': [{'type': 'reasoning', 'id': 'rs1', 'encrypted_content': 'opaque'},
                                  {'type': 'function_call', 'call_id': 'call_native_1', 'id': 'fc1', 'name': 'inspect', 'arguments': '{}'}]}

    def turn(self, step=1, state=None):
        return {'step': step, 'call': {'name': 'inspect'}, 'feedback': {'text': 'ok', 'status': 'success'},
                'scene_state': copy.deepcopy(self.env.public_state if state is None else state),
                'scene_images': [], 'tool_images': []}

    def payload(self):
        return api_payload(self.context.messages, [], model='test', protocol='responses',
                           cache_key='stable', max_output_tokens=1024, effort='low')

    def test_readonly_keeps_entire_native_prefix(self):
        first = self.payload()
        self.context.append(self.native, self.turn())
        second = self.payload()
        self.assertEqual(first['input'], second['input'][:len(first['input'])])
        self.assertEqual(second['input'][len(first['input']):len(first['input']) + 2], self.native['_responses_output'])
        self.assertEqual(self.context.version, 0)
        self.assertEqual(second['reasoning']['effort'], 'low')
        self.assertEqual(second['input'][-2]['call_id'], 'call_native_1')
        self.native['_responses_output'][0]['encrypted_content'] = 'mutated'
        self.assertIn('opaque', json.dumps(self.context.messages))

    def test_patch_roundtrip_deletion_lists_escaped_keys_and_types(self):
        cases = [({'a/b': {'~': [1, 2, 3]}, 'gone': True}, {'a/b': {'~': [4]}, 'new': None}),
                 ([{'x': 1}], [{'x': 2}, {'y': 3}]), ({'x': False}, {'x': 0}), ([], {'new': []})]
        for before, after in cases:
            self.assertEqual(apply_patch(before, state_patch(before, after)), after)

    def test_failed_tool_partial_edit_is_observed(self):
        changed = {'objects': [{'name': 'cube', 'x': 2}]}
        turn = self.turn(state=changed)
        turn['feedback']['status'] = 'failed'
        self.context.append(self.native, turn)
        data = json.loads(self.context.messages[-1]['content'][0]['text'])
        self.assertEqual(apply_patch(self.env.public_state, data['scene_patch']), changed)
        self.assertEqual(self.context.version, 1)

    def test_reject_duplicate_append(self):
        self.context.append(self.native, self.turn())
        before = copy.deepcopy(self.context.messages)
        with self.assertRaises(ValueError):
            self.context.append(self.native, self.turn())
        self.assertEqual(before, self.context.messages)

    def test_resolve_all_rejected_native_calls(self):
        native = copy.deepcopy(self.native)
        native['tool_calls'].append({**copy.deepcopy(native['tool_calls'][0]), 'id': 'second'})
        turn = self.turn()
        turn['call'] = None
        turn['feedback']['status'] = 'rejected'
        self.context.append(native, turn)
        self.assertEqual([m['tool_call_id'] for m in self.context.messages if m['role'] == 'tool'], ['call_native_1', 'second'])

    def test_checkpoint_snapshot_and_new_phase(self):
        self.context.append(self.native, self.turn())
        self.context.checkpoint(self.env)
        state = json.loads(self.context.messages[-1]['content'][0]['text'])
        self.assertEqual(state['public_scene_state'], self.env.public_state)
        self.assertEqual(state['checkpoint']['completed_steps'], 1)
        self.assertEqual(self.context.checkpoints, 1)
        self.assertFalse(any(m['role'] == 'tool' for m in self.context.messages))

    def test_final_payload_budget_includes_native_reasoning(self):
        self.native['_responses_output'][0]['encrypted_content'] = 'x' * 10001
        self.context.append(self.native, self.turn())
        with self.assertRaises(ValueError):
            check_budget(self.payload(), max_text_bytes=10000)

    def test_identical_image_not_repeated_or_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image.png'
            path.write_bytes(b'image fixture')
            ref = {'path': str(path), 'label': 'same'}
            self.env.initial_images = [ref]
            context = APIContext(self.env)
            before = copy.deepcopy(context.messages)
            turn = self.turn()
            turn['scene_images'] = [ref]
            context.append(self.native, turn)
            self.assertEqual(context.messages[:len(before)], before)
            self.assertEqual(len(context.messages[-1]['content']), 1)


if __name__ == '__main__':
    unittest.main()
