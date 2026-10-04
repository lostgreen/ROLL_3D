"""Small offline Qwen2.5-VL adapter for real environment smoke tests."""

import base64
import io
import json
from pathlib import Path
import re
import time

from adapters import AssistantTurn, ToolCall
from environment.budget import Usage


def parse_tool_calls(response, turn_id):
    """Accept explicit JSON calls only; never evaluate generated Python here."""
    blocks = re.findall(r"<tool_call>(.*?)</tool_call>", response, flags=re.S)
    if response.count("<tool_call>") != len(blocks) or response.count("</tool_call>") != len(blocks):
        raise ValueError("truncated tool call; no action executed")
    requests = [json.loads(block) for block in blocks]
    if not blocks:
        stripped = response.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", stripped, flags=re.S)
        if fenced:
            stripped = fenced.group(1).strip()
        if stripped.startswith("{"):
            decoder = json.JSONDecoder()
            # Some local models emit consecutive JSON calls despite the one-call hint.
            # Parse the entire stream before returning any executable action.
            while stripped:
                request, end = decoder.raw_decode(stripped)
                if not isinstance(request, dict):
                    raise ValueError("tool call must be a JSON object")
                if "name" in request or "tool_name" in request:
                    requests.append(request)
                else:
                    raise ValueError("JSON response lacks a function name")
                stripped = stripped[end:].lstrip()
    calls = []
    for i, request in enumerate(requests):
        if not isinstance(request, dict):
            raise ValueError("tool call must be a JSON object")
        name = request.get("name", request.get("tool_name"))
        arguments = request.get("arguments", {})
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise ValueError("invalid tool call JSON")
        calls.append(ToolCall(f"local_{turn_id}_{i}", name, arguments))
    return calls


class LocalQwenAdapter:
    supports_vision = True
    supports_video = False

    def __init__(self, model_path, output_dir, max_tokens=1200):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_tokens = max_tokens
        self.calls = 0
        self.last_usage = Usage(None, 0.0, 0, 0)
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True,
                                                       max_pixels=256 * 28 * 28)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"}, attn_implementation="sdpa",
        ).eval()

    def convert_tools(self, tools):
        return [{"name": t["name"], "description": t["description"],
                 "parameters": t["input_schema"]} for t in tools]

    def system_message(self, content):
        return {"role": "system", "content": content}

    def user_message(self, content, images=None, videos=None):
        if videos:
            raise ValueError("local smoke adapter uses images, not videos")
        return {"role": "user", "content": [{"type": "text", "text": content}] +
                [{"type": "image", "image": image} for image in (images or [])]}

    def tool_result_messages(self, calls, results):
        return [self.user_message(f"Tool {call.name} returned:\n{result['text']}", result.get("images"))
                for call, result in zip(calls, results)]

    def run(self, messages, tools):
        from PIL import Image

        self.last_usage = Usage(None, 0.0, 0, 0)
        self.calls += 1
        prepared, images = [], []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                blocks = []
                for block in content:
                    if block["type"] == "image":
                        data = block["image"]
                        im = Image.open(io.BytesIO(base64.b64decode(data["data"]))).convert("RGB")
                        images.append(im)
                        blocks.append({"type": "image"})
                    else:
                        blocks.append(block)
                content = blocks
            prepared.append({"role": message["role"], "content": content})
        instructions = (
            "\nAvailable functions:\n" + json.dumps(tools) +
            '\nTo call a function, respond with one JSON object: {"name": "FUNCTION_NAME", "arguments": {}}. '
            "Use the exact function name and required parameters from the signatures above. "
            "Code is a JSON string; escape its newlines. Make one function call per response. "
            "When the requested work is complete, respond with a short final statement and no tool call."
        )
        if tools:
            prepared[0] = {**prepared[0], "content": str(prepared[0]["content"]) + instructions}
        prompt = self.processor.apply_chat_template(prepared, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=images or None, return_tensors="pt", padding=True).to("cuda:0")
        self.torch.cuda.synchronize()
        started = time.monotonic()
        try:
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=False)
            self.torch.cuda.synchronize()
        finally:
            self.last_usage = Usage(time.monotonic() - started, 0.0, 0, 0)
        response = self.processor.batch_decode(output[:, inputs.input_ids.shape[1]:],
                                               skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        (self.output_dir / f"turn_{self.calls:03}.txt").write_text(response)
        try:
            calls = parse_tool_calls(response, self.calls)
        except ValueError as exc:
            (self.output_dir / f"turn_{self.calls:03}_error.json").write_text(
                json.dumps({"error_type": type(exc).__name__, "message": str(exc)}) + "\n")
            raise
        return AssistantTurn(response, calls, {"role": "assistant", "content": response})
