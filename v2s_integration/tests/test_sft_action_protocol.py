"""The current SFT target may contain ordinary prose before its sole action."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v2s_integration.agent.actions import action_text, parse_action


class Registry:
    def lookup(self, name):
        return (SimpleNamespace(input_schema={'properties': {
            'code': {'type': 'string'}, 'summary': {'type': 'string'},
            'reference_image': {'type': 'string'}, 'bbox': {'type': 'array'},
        }}), None)


def xml_call(code='print("ok")'):
    return '<tool_call>\n<function=execute_blender_code>\n<parameter=code>\n' + code + '\n</parameter>\n</function>\n</tool_call>'


def json_call(code='print("ok")'):
    return '<tool_call>' + json.dumps({'name': 'execute_blender_code', 'arguments': {'code': code}}) + '</tool_call>'


@pytest.mark.parametrize('envelope', [xml_call(), json_call()])
@pytest.mark.parametrize('prefix', ['', 'I will now build the table.\n',
    'Check the dimensions.</think>创建桌面。\n', '<think>Inspect proportions.</think>Now execute:\n'])
def test_ordinary_content_before_single_action(prefix, envelope):
    parsed = parse_action(prefix + envelope, 7, Registry())
    assert parsed.name == 'execute_blender_code' and parsed.id == 'local_7_0'
    assert parsed.arguments == {'code': 'print("ok")'}
    assert action_text(prefix + envelope) == envelope


@pytest.mark.parametrize('factory', [xml_call, json_call])
def test_complete_and_prefixed_envelopes_keep_literal_protocol_markers(factory):
    code = '''markers = "<think></think><tool_call></tool_call><function=finish></function><parameter=x></parameter><|im_end|>"
print(markers)
print("</parameter></function></tool_call>")
# A code comment may also contain </parameter> and </think>.
'''.rstrip('\n')
    envelope = factory(code)
    for prefix in ['', 'reasoning</think>Apply this change:\n']:
        assert parse_action(prefix + envelope, 1, Registry()).arguments['code'] == code


@pytest.mark.parametrize('factory', [xml_call, json_call])
def test_thinking_examples_are_not_executed(factory):
    example = factory('DO_NOT_EXECUTE')
    actual = factory('EXECUTE_ONLY_THIS')
    response = '<think>One possible call would be:\n' + example + '\nChoose the final action.</think>Executing now.\n' + actual
    assert parse_action(response, 1, Registry()).arguments['code'] == 'EXECUTE_ONLY_THIS'
    with pytest.raises(ValueError):
        parse_action('<think>Example only:\n' + example, 1, Registry())


@pytest.mark.parametrize('first,second', [(xml_call(), xml_call()), (json_call(), json_call()),
    (xml_call(), json_call()), (json_call(), xml_call())])
def test_multiple_actions_and_second_thinking_boundary_do_not_hide_first(first, second):
    for response in [first + second, 'reasoning</think>Doing both:\n' + first + second,
                     'reasoning</think>' + first + '</think>Now another call:\n' + second]:
        with pytest.raises(ValueError):
            parse_action(response, 1, Registry())


@pytest.mark.parametrize('factory', [xml_call, json_call])
def test_reject_incomplete_envelopes_and_any_trailing_content(factory):
    envelope = factory()
    for suffix in [' trailing explanation', '\n<tool_call>', '\n<|im_start|>user']:
        with pytest.raises(ValueError):
            parse_action('Here is the action:\n' + envelope + suffix, 1, Registry())
    for fragment in [envelope[:-5], envelope.replace('</tool_call>', ''), '<tool_call>']:
        with pytest.raises(ValueError):
            parse_action('reasoning</think>Now act:\n' + fragment, 1, Registry())


def test_body_reserved_markup_requires_explicit_escaping_but_code_does_not():
    envelope = xml_call()
    for body in ['<function=finish>example</function>', '<parameter=x>example</parameter>', '<think>unclosed']:
        with pytest.raises(ValueError):
            parse_action('reasoning</think>' + body + envelope, 1, Registry())
    body = '&lt;function=finish&gt; is only an example, not the executed action.\n'
    assert parse_action('reasoning</think>' + body + envelope, 1, Registry()).name == 'execute_blender_code'


def test_multiple_xml_parameters_are_typed_and_duplicates_rejected():
    envelope = '<tool_call><function=reference.crop><parameter=reference_image>ref:0</parameter><parameter=bbox>[0,0,1,1]</parameter></function></tool_call>'
    call = parse_action('Inspect this crop:\n' + envelope, 2, Registry())
    assert call.arguments == {'reference_image': 'ref:0', 'bbox': [0, 0, 1, 1]}
    duplicate = envelope.replace('</function>', '<parameter=bbox>[0,0,1,1]</parameter></function>')
    with pytest.raises(ValueError):
        parse_action(duplicate, 2, Registry())
