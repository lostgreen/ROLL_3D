import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from v2s_integration.rollout.api import collect


class Environment:
    max_steps = 2
    max_output_tokens = 512
    system_prompt = 'system'
    task_prompt = 'reconstruct'
    reference_refs = initial_images = current_images = []
    model_tools = [{'type': 'function', 'function': {'name': 'scene.list_objects', 'description': 'inspect',
                   'parameters': {'type': 'object', 'properties': {}}}}]

    def __init__(self):
        self.closed = True
        self.turns = []
        self.public_state = {'objects': []}

    def reset(self, seed):
        self.closed = False

    def step(self, action):
        turn = {'step': len(self.turns) + 1, 'call': {'name': 'scene.list_objects'},
                'feedback': {'text': 'observed', 'status': 'success'}, 'scene_images': [], 'tool_images': []}
        self.turns.append(turn)
        return None, 0, len(self.turns) == 2, False, {'termination_reason': 'finish'}

    def finish_episode(self, reason):
        self.closed = True

    def close(self):
        self.closed = True


class CollectorTests(unittest.TestCase):
    def run_fixture(self, reasoning='', finish='tool_calls'):
        requests = []
        def invoke(base, key, payload, folder):
            requests.append(copy.deepcopy(payload))
            number = len(requests)
            message = {'role': 'assistant', 'content': '', 'tool_calls': [{'id': f'c{number}', 'type': 'function',
                       'function': {'name': 'scene_list_objects', 'arguments': '{}'}}],
                       '_responses_output': [{'type': 'reasoning', 'encrypted_content': reasoning},
                       {'type': 'function_call', 'call_id': f'c{number}', 'name': 'scene_list_objects', 'arguments': '{}'}]}
            return message, {'finish_reason': finish}
        env = Environment()
        with tempfile.TemporaryDirectory() as directory, patch('v2s_integration.rollout.api.invoke', side_effect=invoke):
            status = collect(env, output=Path(directory) / 'run', model='model', protocol='responses',
                             base_url='http://fixture', key='secret', effort='low', max_text_bytes=2500)
        return status, requests, env

    def test_general_collector_retains_prior_input(self):
        status, requests, env = self.run_fixture()
        self.assertEqual(status['status'], 'finished')
        self.assertEqual(requests[1]['input'][:len(requests[0]['input'])], requests[0]['input'])
        self.assertEqual(requests[1]['reasoning']['effort'], 'low')
        self.assertEqual(requests[0]['prompt_cache_key'], requests[1]['prompt_cache_key'])
        self.assertEqual(len(env.turns), 2)

    def test_oversized_history_has_explicit_checkpoint(self):
        status, requests, _ = self.run_fixture(reasoning='x' * 8000)
        self.assertEqual(status['status'], 'finished')
        self.assertTrue(status['records'][1]['checkpointed'])
        self.assertEqual(status['records'][1]['checkpoints'], 1)
        self.assertNotIn('x' * 8000, str(requests[1]))
        self.assertIn('explicit_context_budget_boundary', str(requests[1]))

    def test_incomplete_output_never_executes(self):
        status, requests, env = self.run_fixture(finish='length')
        self.assertEqual(status['status'], 'failed')
        self.assertEqual(len(requests), 1)
        self.assertEqual(env.turns, [])


if __name__ == '__main__':
    unittest.main()
