import json
import unittest
from action_response import final_action_text


class FinalActionTests(unittest.TestCase):
    def test_supported_final_formats(self):
        call = json.dumps({'name':'execute_blender_code','arguments':{'code':'print(1)'}})
        for final in (call, '```json\n'+call+'\n```', '<tool_call>'+call+'</tool_call>'):
            for prefix in ('', '<think>reasoning</think>', 'prefilled reasoning</think>',
                           'example '+call+'</think>more reasoning</think>'):
                with self.subTest(prefix=prefix, final=final):
                    self.assertEqual(final_action_text(prefix+final), final)

    def test_never_execute_reasoning_examples(self):
        call = '{"name":"finish","arguments":{}}'
        for text in ('<think>'+call, '<think><tool_call>'+call+'</tool_call>',
                     '<think>'+call+'</think>', 'explanation '+call,
                     'reasoning</think>not JSON '+call,
                     'reasoning</think>'+call+' trailing prose',
                     'reasoning</think><tool_call>'+call,
                     'reasoning</think><think>'+call):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    final_action_text(text)

    def test_literal_markers_in_json_are_preserved(self):
        call = json.dumps({'name':'execute_blender_code','arguments':{'code':'print("</think>")'}})
        self.assertEqual(final_action_text(call), call)
        self.assertEqual(final_action_text('reasoning</think>'+call), call)


if __name__ == '__main__':
    unittest.main()
