"""Select an explicit final action without altering sampled training tokens."""
import json
import re


def final_action_text(response):
    text = response.strip()

    def explicit_json(value):
        fenced = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```', value, re.S)
        candidate = fenced.group(1).strip() if fenced else value
        try:
            return isinstance(json.loads(candidate), dict)
        except ValueError:
            return False

    # Literal thinking markers inside a complete JSON string are ordinary data.
    if explicit_json(text):
        return text
    # Qwen can prefill <think> in the prompt, so its response need not open it.
    # Repeated closing markers were observed; everything before the last is
    # non-executable. Never search that prefix for a tempting JSON/tool example.
    if '</think>' in text:
        suffixes = [text[m.end():].strip() for m in re.finditer('</think>', text)]
        # A marker inside the final JSON string is not a boundary. Choose only
        # a suffix that forms a complete explicit object through end-of-input.
        for suffix in reversed(suffixes):
            if explicit_json(suffix):
                return suffix
        text = suffixes[-1]
    if not text:
        raise ValueError('empty final action after reasoning')
    if explicit_json(text):
        return text
    # Preserve explicit tool wrappers, but reject prose, unfinished reasoning,
    # and trailing unparsed text. The underlying parser validates each JSON.
    blocks = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.S)
    if blocks and re.fullmatch(r'(?:\s*<tool_call>.*?</tool_call>\s*)+', text, re.S):
        if text.count('<tool_call>') == len(blocks) == text.count('</tool_call>'):
            for block in blocks:
                if not isinstance(json.loads(block), dict):
                    raise ValueError('tool call must be a JSON object')
            return text
    raise ValueError('final action must be explicit JSON or complete tool_call wrappers')
