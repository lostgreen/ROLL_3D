"""Model adapters: normalize different model APIs to one interface for the agent loop.

Every adapter exposes:
  - convert_tools(mcp_tools) -> provider-specific tool schema
  - run(messages, tools) -> AssistantTurn

AssistantTurn:
  - text: str | None         (assistant's natural-language output, if any)
  - tool_calls: list[ToolCall]
  - raw: provider message    (to append back into history)

The agent loop only deals with these neutral objects + a small set of helpers
to append tool results back into the message history per provider.

Two dialects cover everything we need:
  - AnthropicAdapter  -> Claude (Messages API)
  - OpenAIAdapter     -> GLM (Zhipu), GPT, DeepSeek, Qwen, Kimi, vLLM/Ollama ...
"""

from dataclasses import dataclass, field
from typing import Any
import os


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class AssistantTurn:
    text: str | None
    tool_calls: list[ToolCall]
    raw: Any  # provider-native assistant message, appended to history verbatim


def _record_token_usage(adapter, prompt=None, completion=None, total=None):
    if total is None and isinstance(prompt, int) and isinstance(completion, int):
        total = prompt + completion
    usage = {"prompt_tokens": prompt, "completion_tokens": completion,
             "total_tokens": total}
    adapter.last_api_usage = usage
    totals = getattr(adapter, "api_usage_totals", None) or {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if isinstance(usage[name], int):
            totals[name] += usage[name]
    totals["calls"] += 1
    adapter.api_usage_totals = totals


# =========================================================================
# OpenAI-compatible (GLM / GPT / DeepSeek / Qwen / Kimi / vLLM / Ollama)
# =========================================================================
class OpenAIAdapter:
    def __init__(self, model: str, api_key: str, base_url: str | None = None,
                 temperature: float | None = None, max_tokens: int = 4096,
                 supports_vision: bool = False, supports_video: bool = False,
                 reasoning_effort: str | None = None, extra_body: dict | None = None,
                 timeout: float | None = 600.0, cache_task_id: str | None = None):
        from openai import OpenAI
        # timeout defaults to 600s (10min) + max_retries=0 (retries handled by the agent loop).
        # * Evolution (2026-07-15): 60/150/300s were all too short — kimi T5 step0 injects 11
        #   multi-view reference images + the full context in one call takes >300s, gets cut off
        #   into a 9-retry death loop -> steps=0 false failure.
        #   So timeout was removed (None) for a while, but that hit a bigger pitfall: when the
        #   gateway severs an established connection mid-way (TCP CLOSED), an httpx client with no
        #   timeout waits forever on that dead connection, and all 40 workers hang with CPU at 0
        #   (whole pool froze for 20min in the early hours of 2026-07-15).
        #   Compromise at 600s: enough for kimi multi-image step0 (measured worst case <5min); a
        #   dead connection errors out after at most 600s -> the agent's 10 retries rebuild the
        #   connection. Neither cuts off slow requests nor hangs forever.
        # * cache_task_id: WOA gateway task-dimension affinity policy (standard protocol). Requests
        #   with the same cache_task_id route to the same underlying resource -> prompt cache hit;
        #   different ids -> spread across multiple healthy accounts -> higher success rate.
        #   per-sample uses md5(model+task_id): one 150-step sample shares one client -> same id ->
        #   large speedup from per-step cache hits. Appended after the Bearer as a query, the SDK
        #   auto-assembles Authorization: Bearer {cred}?cache_task_id=xxx. Verified the gateway
        #   accepts it and the cache hits.
        if cache_task_id:
            api_key = f"{api_key}?cache_task_id={cache_task_id}"
        self.client = OpenAI(api_key=api_key, base_url=base_url,
                             timeout=timeout, max_retries=0)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.supports_vision = supports_vision
        self.supports_video = supports_video
        self.reasoning_effort = reasoning_effort
        self.extra_body = extra_body or None
        self.last_api_usage = None
        self.api_usage_totals = None

    def convert_tools(self, mcp_tools: list[dict]) -> list[dict]:
        # Logical provider names use dots; OpenAI function names cannot.
        self._tool_names = {t['name'].replace('.', '__'): t['name'] for t in mcp_tools}
        if len(self._tool_names) != len(mcp_tools):
            raise ValueError('tool wire-name collision')
        return [{
            "type": "function",
            "function": {
                "name": t["name"].replace('.', '__'),
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        } for t in mcp_tools]

    def system_message(self, content: str) -> dict:
        return {"role": "system", "content": content}

    def user_message(self, content: str, images: list[dict] | None = None,
                     videos: list[dict] | None = None) -> dict:
        if not images and not videos:
            return {"role": "user", "content": content}
        blocks = [{"type": "text", "text": content}]
        for img in (images or []):
            blocks.append({"type": "image_url", "image_url": {
                "url": f"data:{img['mime']};base64,{img['data']}"}})
        # Video path: placeholder implementation. No OpenAI-compatible video provider is wired in
        # currently; when one is, the specific provider's adapter (e.g. GeminiAdapter) overrides this branch.
        if videos:
            raise NotImplementedError(
                "OpenAIAdapter does not support video blocks yet; use a specific adapter with supports_video=True")
        return {"role": "user", "content": blocks}

    def run(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        kwargs = dict(model=self.model, messages=messages, tools=tools or None)
        if "gpt-5" in self.model.casefold():
            kwargs["max_completion_tokens"] = self.max_tokens
        else:
            kwargs["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        resp = self.client.chat.completions.create(**kwargs)
        usage = getattr(resp, "usage", None)
        _record_token_usage(self, getattr(usage, "prompt_tokens", None),
                            getattr(usage, "completion_tokens", None),
                            getattr(usage, "total_tokens", None))
        msg = resp.choices[0].message
        self.last_visible_reasoning = getattr(msg, "reasoning_summary", None)
        calls = []
        for tc in (msg.tool_calls or []):
            import json
            args = json.loads(tc.function.arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError(f"tool arguments must be a JSON object: {tc.function.name}")
            name = getattr(self, '_tool_names', {}).get(tc.function.name, tc.function.name)
            calls.append(ToolCall(id=tc.id, name=name, arguments=args))
        # Append assistant message verbatim (must keep tool_calls for the API).
        raw = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            raw["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        return AssistantTurn(text=msg.content, tool_calls=calls, raw=raw)

    def tool_result_messages(self, calls: list[ToolCall], results: list[dict]) -> list[dict]:
        """results[i] = {"text": str, "images": [{"data": b64, "mime": str}]}.

        OpenAI tool messages are text-only, so images (if any, and if this model
        supports vision) are appended afterwards as a separate user message with
        image_url data-URI blocks.
        """
        msgs = []
        pending_images = []
        for c, r in zip(calls, results):
            text = r["text"] if isinstance(r, dict) else str(r)
            imgs = r.get("images", []) if isinstance(r, dict) else []
            if imgs and self.supports_vision:
                text = (text + "\n[image returned — shown below]").strip()
                pending_images.extend((c.id, c.name, img) for img in imgs)
            elif imgs:
                text = (text + f"\n[{len(imgs)} image(s) returned; this model "
                        "has no vision, image not shown]").strip()
            msgs.append({"role": "tool", "tool_call_id": c.id, "content": text})

        if pending_images:
            content = [{"type": "text",
                        "text": "Here are the image(s) returned by the tool:"}]
            for call_id, tool_name, img in pending_images:
                content.append({"type": "text", "text":
                    f"Tool {tool_name}, call {call_id}: {img.get('label', 'image')}"})
                content.append({"type": "image_url", "image_url": {
                    "url": f"data:{img['mime']};base64,{img['data']}"}})
            msgs.append({"role": "user", "content": content})
        return msgs


class KuaishouGatewayAdapter(OpenAIAdapter):
    """OpenAI chat schema over the internal Kigress header contract."""

    def __init__(self, model: str, api_key: str, user_key: str,
                 base_url: str, biz_scene: str = "offline",
                 temperature: float | None = None, max_tokens: int = 4096,
                 supports_vision: bool = True, supports_video: bool = False,
                 timeout: float = 120.0):
        import httpx
        if not api_key or not user_key:
            raise ValueError("Kigress api_key and user_key are required")
        self.model = model
        self.api_key = api_key
        self.user_key = user_key
        self.base_url = base_url.rstrip("/")
        self.biz_scene = biz_scene
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.supports_vision = supports_vision
        self.supports_video = supports_video
        self.timeout = timeout
        self.http = httpx.Client(timeout=timeout)
        self.last_api_usage = None
        self.api_usage_totals = None
        self.last_response_metadata = None

    def user_message(self, content: str, images: list[dict] | None = None,
                     videos: list[dict] | None = None) -> dict:
        if videos:
            raise NotImplementedError(
                "Kigress OpenAI chat adapter uses sampled images; native video is not enabled")
        return super().user_message(content, images=images)

    def run(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        import json
        body = {"model": self.model, "messages": messages,
                "tools": tools or None, "stream": False}
        if "gpt-5" in self.model.casefold():
            body["max_completion_tokens"] = self.max_tokens
        else:
            body["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            body["temperature"] = self.temperature
        headers = {
            "Content-Type": "application/json", "Accept": "application/json",
            "x-api-key": self.api_key, "x-ks-user-key": self.user_key,
            "x-ks-llm-model": self.model, "x-ks-biz-scene": self.biz_scene,
        }
        response = self.http.post(f"{self.base_url}/chat/completions",
                                  headers=headers, json=body)
        if response.status_code >= 400:
            snippet = response.text[:500].replace(self.api_key, "<redacted>")
            snippet = snippet.replace(self.user_key, "<redacted>")
            raise RuntimeError(f"Kigress API {response.status_code}: {snippet}")
        payload = response.json()
        choice = payload["choices"][0]
        message = choice.get("message") or {}
        self.last_visible_reasoning = message.get("reasoning_summary")
        usage = payload.get("usage") or {}
        _record_token_usage(self, usage.get("prompt_tokens"),
                            usage.get("completion_tokens"), usage.get("total_tokens"))
        self.last_response_metadata = {
            "finish_reason": choice.get("finish_reason"),
            "request_id": response.headers.get("x-request-id") or payload.get("id"),
            "response_chars": len(message.get("content") or ""),
        }
        calls = []
        raw_calls = []
        for index, tool_call in enumerate(message.get("tool_calls") or []):
            function = tool_call.get("function") or {}
            arguments = json.loads(function.get("arguments") or "{}")
            if not isinstance(arguments, dict):
                raise ValueError(f"tool arguments must be a JSON object: {function.get('name')}")
            call_id = tool_call.get("id") or f"call_{index}"
            name = getattr(self, '_tool_names', {}).get(function['name'], function['name'])
            calls.append(ToolCall(call_id, name, arguments))
            raw_calls.append({"id": call_id, "type": "function",
                              "function": {"name": function["name"],
                                           "arguments": function.get("arguments") or "{}"}})
        raw = {"role": "assistant", "content": message.get("content") or ""}
        if raw_calls:
            raw["tool_calls"] = raw_calls
        return AssistantTurn(message.get("content"), calls, raw)


# =========================================================================
# Anthropic (Claude)
# =========================================================================
class AnthropicAdapter:
    def __init__(self, model: str, api_key: str, base_url: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 4096,
                 supports_vision: bool = True, supports_video: bool = False):
        from anthropic import Anthropic
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = Anthropic(**kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.supports_vision = supports_vision
        self.supports_video = supports_video
        self.last_api_usage = None
        self.api_usage_totals = None

    def convert_tools(self, mcp_tools: list[dict]) -> list[dict]:
        return [{
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["input_schema"],
        } for t in mcp_tools]

    def system_message(self, content: str) -> dict:
        # Anthropic takes system as a top-level param, not a message.
        return {"role": "_system", "content": content}  # filtered out in run()

    def user_message(self, content: str, images: list[dict] | None = None,
                     videos: list[dict] | None = None) -> dict:
        if not images and not videos:
            return {"role": "user", "content": content}
        blocks = [{"type": "text", "text": content}]
        for img in (images or []):
            blocks.append({"type": "image", "source": {
                "type": "base64", "media_type": img["mime"], "data": img["data"]}})
        # Anthropic's current API has no native video block. Only a supports_video=True adapter
        # (e.g. a future GeminiAdapter) actually injects video.
        if videos:
            raise NotImplementedError(
                "AnthropicAdapter does not support native video blocks; supports_video should be False")
        return {"role": "user", "content": blocks}

    def run(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        api_messages = [m for m in messages if m["role"] != "_system"]
        system = "\n\n".join(m.get("content", "") for m in messages if m.get("role") == "_system")
        kwargs = dict(model=self.model, system=system or None,
                      messages=api_messages, tools=tools or None,
                      max_tokens=self.max_tokens)
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        resp = self.client.messages.create(**kwargs)
        usage = getattr(resp, "usage", None)
        _record_token_usage(self, getattr(usage, "input_tokens", None),
                            getattr(usage, "output_tokens", None))
        text_parts, calls, content_blocks = [], [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
                content_blocks.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input or {}))
                content_blocks.append({"type": "tool_use", "id": block.id,
                                       "name": block.name, "input": block.input})
        raw = {"role": "assistant", "content": content_blocks}
        text = "\n".join(text_parts) if text_parts else None
        return AssistantTurn(text=text, tool_calls=calls, raw=raw)

    def tool_result_messages(self, calls: list[ToolCall], results: list[dict]) -> list[dict]:
        """All tool_results for one assistant turn go in a single user message.

        Anthropic lets images live INSIDE a tool_result block, so a render's
        image rides along with its own tool call (no separate message needed).
        """
        content = []
        for c, r in zip(calls, results):
            text = r["text"] if isinstance(r, dict) else str(r)
            imgs = r.get("images", []) if isinstance(r, dict) else []
            block_content = [{"type": "text", "text": text or "(no text)"}]
            if imgs and self.supports_vision:
                for img in imgs:
                    if img.get("label"):
                        block_content.append({"type": "text", "text": img["label"]})
                    block_content.append({"type": "image", "source": {
                        "type": "base64", "media_type": img["mime"],
                        "data": img["data"]}})
            elif imgs:
                block_content[0]["text"] += (
                    f"\n[{len(imgs)} image(s) returned; vision disabled]")
            content.append({"type": "tool_result", "tool_use_id": c.id,
                            "content": block_content})
        return [{"role": "user", "content": content}]


# =========================================================================
# Gemini NACI passthrough protocol (bypasses the litellm parsing bug)
# =========================================================================
class GeminiNACIAdapter:
    """Calls the NACI passthrough protocol directly (Gemini's native generateContent format).
    Bypasses the litellm parsing layer bug in the standard protocol (choice.message=None crash)."""

    DEFAULT_URL = "http://trpc-gpt-eval.production.polaris:8080/v1beta/models/{model}:generateContent"

    def __init__(self, model: str, api_key: str, base_url: str | None = None,
                 temperature: float | None = None, max_tokens: int = 16384,
                 supports_vision: bool = True, supports_video: bool = False):
        import httpx
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.supports_vision = supports_vision
        self.supports_video = supports_video
        url_template = base_url or os.environ.get("NACI_PASSTHROUGH_URL") or self.DEFAULT_URL
        if "{model}" in url_template:
            self.url = url_template.format(model=model)
        else:
            self.url = f"{url_template.rstrip('/')}/v1beta/models/{model}:generateContent"
        self.auth = f"Bearer {api_key}?provider=naci_default&model={model}"
        self.http = httpx.Client(timeout=600.0)  # 600s, same as OpenAIAdapter (dead connections don't hang forever)
        self.last_api_usage = None
        self.api_usage_totals = None

    def convert_tools(self, mcp_tools: list[dict]) -> list[dict]:
        decls = []
        for t in mcp_tools:
            decls.append({
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            })
        return [{"functionDeclarations": decls}]

    def system_message(self, content: str) -> dict:
        return {"role": "_system", "content": content}

    def user_message(self, content: str, images: list[dict] | None = None,
                     videos: list[dict] | None = None) -> dict:
        parts = [{"text": content}]
        for img in (images or []):
            parts.append({"inlineData": {
                "mimeType": img["mime"], "data": img["data"]}})
        if videos and not self.supports_video:
            raise NotImplementedError("Gemini video input is disabled for this model config")
        for video in (videos or []):
            parts.append({"inlineData": {
                "mimeType": video["mime"], "data": video["data"]}})
        return {"role": "user", "parts": parts}

    def run(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        import json
        contents = []
        for m in messages:
            if m.get("role") == "_system":
                continue
            role = m.get("role", "user")
            if role == "assistant":
                role = "model"
            parts = m.get("parts")
            if parts:
                contents.append({"role": role, "parts": parts})
            else:
                c = m.get("content", "")
                if isinstance(c, str):
                    contents.append({"role": role, "parts": [{"text": c}]})
                elif isinstance(c, list):
                    ps = []
                    for block in c:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                ps.append({"text": block["text"]})
                            elif block.get("type") == "image_url":
                                url = block["image_url"]["url"]
                                if url.startswith("data:"):
                                    mime, b64 = url.split(";base64,", 1)
                                    mime = mime.replace("data:", "")
                                    ps.append({"inlineData": {"mimeType": mime, "data": b64}})
                            elif block.get("type") == "function_response":
                                ps.append({"functionResponse": block["functionResponse"]})
                        elif isinstance(block, str):
                            ps.append({"text": block})
                    if ps:
                        contents.append({"role": role, "parts": ps})

        body = {
            "contents": contents,
            "generationConfig": {"maxOutputTokens": self.max_tokens},
        }
        system = "\n\n".join(m.get("content", "") for m in messages if m.get("role") == "_system")
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if self.temperature is not None:
            body["generationConfig"]["temperature"] = self.temperature
        if tools:
            body["tools"] = tools
            body["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}

        headers = {"Content-Type": "application/json", "Authorization": self.auth}
        resp = self.http.post(self.url, headers=headers, json=body)
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini API {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        usage = data.get("usageMetadata")
        _record_token_usage(self,
                            usage.get("promptTokenCount") if isinstance(usage, dict) else None,
                            usage.get("candidatesTokenCount") if isinstance(usage, dict) else None,
                            usage.get("totalTokenCount") if isinstance(usage, dict) else None)
        candidate = data.get("candidates", [{}])[0]
        parts_out = candidate.get("content", {}).get("parts", [])

        text_parts = []
        calls = []
        raw_parts = []
        for p in parts_out:
            if "text" in p and not p.get("thought"):
                text_parts.append(p["text"])
                clean = {"text": p["text"]}
                raw_parts.append(clean)
            elif "functionCall" in p:
                fc = p["functionCall"]
                call_id = fc.get("id", f"call_{fc['name']}_{len(calls)}")
                calls.append(ToolCall(id=call_id, name=fc["name"],
                                      arguments=fc.get("args", {})))
                clean = {"functionCall": fc}
                if "thoughtSignature" in p:
                    clean["thoughtSignature"] = p["thoughtSignature"]
                raw_parts.append(clean)

        text = "\n".join(text_parts) if text_parts else None
        raw = {"role": "model", "parts": raw_parts}
        return AssistantTurn(text=text, tool_calls=calls, raw=raw)

    def tool_result_messages(self, calls: list[ToolCall], results: list[dict]) -> list[dict]:
        parts = []
        pending_images = []
        for c, r in zip(calls, results):
            text = r["text"] if isinstance(r, dict) else str(r)
            imgs = r.get("images", []) if isinstance(r, dict) else []
            if len(text) > 30000:
                import sys
                print(f"  [warn] tool result truncated: {c.name} {len(text)} -> 30000 chars",
                      file=sys.stderr)
            resp_content = {"content": text[:30000]}
            parts.append({"functionResponse": {"name": c.name, "response": resp_content}})
            if imgs and self.supports_vision:
                pending_images.extend(imgs)

        msgs = [{"role": "function", "parts": parts}]
        if pending_images:
            img_parts = [{"text": "Tool returned image(s):"}]
            for img in pending_images:
                img_parts.append({"inlineData": {"mimeType": img["mime"], "data": img["data"]}})
            msgs.append({"role": "user", "parts": img_parts})
        return msgs


def build_adapter(cfg: dict):
    """cfg = {provider, model, api_key, base_url?, temperature?, max_tokens?,
              vision?, video?}"""
    provider = cfg["provider"]
    common = dict(model=cfg["model"], api_key=cfg["api_key"],
                  base_url=cfg.get("base_url"),
                  temperature=cfg.get("temperature"),
                  max_tokens=cfg.get("max_tokens", 4096))
    # video defaults to False, set True only in a specific video-capable adapter (e.g. a future
    # GeminiAdapter). Currently neither OpenAI nor Anthropic supports native video.
    sv = cfg.get("video", False)
    if provider == "anthropic":
        # Claude models are vision-capable by default; allow opt-out.
        return AnthropicAdapter(supports_vision=cfg.get("vision", True),
                                supports_video=sv, **common)
    if provider == "gemini_naci":
        return GeminiNACIAdapter(supports_vision=cfg.get("vision", True),
                                 supports_video=sv, **common)
    if provider == "kuaishou_gateway":
        return KuaishouGatewayAdapter(
            model=cfg["model"], api_key=cfg["api_key"], user_key=cfg.get("user_key", ""),
            base_url=cfg["base_url"], biz_scene=cfg.get("biz_scene", "offline"),
            temperature=cfg.get("temperature"), max_tokens=cfg.get("max_tokens", 4096),
            supports_vision=cfg.get("vision", True), supports_video=sv,
            timeout=cfg.get("timeout", 120.0))
    if provider == "openai":
        # OpenAI-compatible models vary; default OFF, opt in per model config.
        return OpenAIAdapter(supports_vision=cfg.get("vision", False),
                             supports_video=sv,
                             reasoning_effort=cfg.get("reasoning_effort"),
                             extra_body=cfg.get("extra_body"),
                             timeout=cfg.get("timeout", 600.0),
                             cache_task_id=cfg.get("cache_task_id"),
                             **common)
    raise ValueError(f"unknown provider: {provider}")
