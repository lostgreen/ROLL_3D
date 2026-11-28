import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from v2s_integration.agent.api_transport import invoke


def stream(events):
    return io.BytesIO(b''.join(b'data: ' + json.dumps(e).encode() + b'\n\n' for e in events))


class TransportTests(unittest.TestCase):
    def test_chat_assembles_arguments_and_usage_after_finish(self):
        events = [
            {'model': 'cheap', 'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'id1', 'function': {'name': 'inspect', 'arguments': '{'}}]}}]},
            {'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': '}'}}]}, 'finish_reason': 'tool_calls'}]},
            {'choices': [], 'usage': {'prompt_tokens': 2048, 'completion_tokens': 8, 'prompt_tokens_details': {'cached_tokens': 1024}}},
        ]
        with tempfile.TemporaryDirectory() as directory, patch('urllib.request.urlopen', return_value=stream(events)):
            message, metrics = invoke('http://fixture', 'private-key', {'messages': [], 'stream': True}, Path(directory) / 'call')
            self.assertEqual(message['tool_calls'][0]['function']['arguments'], '{}')
            self.assertEqual(metrics['cached_tokens'], 1024)
            self.assertIsNotNone(metrics['first_content_delta_seconds'])
            self.assertNotIn('private-key', (Path(directory) / 'call/metadata.json').read_text())

    def test_responses_preserves_opaque_output_and_distinguishes_event_latency(self):
        output = [{'type': 'reasoning', 'id': 'r', 'encrypted_content': 'opaque'},
                  {'type': 'function_call', 'call_id': 'c', 'id': 'f', 'name': 'inspect', 'arguments': '{}'}]
        events = [{'type': 'response.created'}, {'type': 'response.function_call_arguments.delta', 'delta': '{}'},
                  {'type': 'response.completed', 'response': {'model': 'astra', 'output': output, 'status': 'completed',
                   'usage': {'input_tokens': 4096, 'output_tokens': 10, 'input_tokens_details': {'cached_tokens': 3072}}}}]
        with tempfile.TemporaryDirectory() as directory, patch('urllib.request.urlopen', return_value=stream(events)):
            message, metrics = invoke('http://fixture', 'private-key', {'input': [], 'stream': True}, Path(directory) / 'call')
            self.assertEqual(message['_responses_output'], output)
            self.assertEqual(metrics['cached_tokens'], 3072)
            self.assertLessEqual(metrics['first_event_seconds'], metrics['first_content_delta_seconds'])

    def test_incomplete_stream_is_not_retried_or_counted_as_success(self):
        with tempfile.TemporaryDirectory() as directory, patch('urllib.request.urlopen', return_value=stream([{'type': 'response.created'}])) as request:
            path = Path(directory) / 'call'
            with self.assertRaises(RuntimeError):
                invoke('http://fixture', 'private-key', {'input': [], 'stream': True}, path)
            request.assert_called_once()
            self.assertFalse((path / 'metadata.json').exists())
            self.assertFalse(json.loads((path / 'failure.json').read_text())['usage_known'])


if __name__ == '__main__':
    unittest.main()
