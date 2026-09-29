"""Thin wrapper around the official MCP Python SDK stdio client.

Spawns the Blender MCP server (`uv run blender-mcp`) as a subprocess and exposes
two simple sync-ish helpers: list_tools() and call_tool(). The Blender MCP server
talks to a running Blender instance over a TCP socket internally, so Blender must
be open with the addon "Connect to MCP server" enabled.

Because the MCP SDK is async, we run everything inside a single persistent event
loop on a background thread, and expose blocking methods to the agent loop.
"""

import json
import asyncio
import os
import shutil
import sys
import threading
from concurrent.futures import Future
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class BlenderMCPClient:
    def __init__(self, command: str = "uv", args: list[str] | None = None,
                 env: dict[str, str] | None = None, cwd: str | None = None):
        if command == "uv" and shutil.which("uv") is None:
            # KML images may have the Python dependencies but omit uv. The
            # vendored server is executable directly in that environment.
            command = sys.executable
            args = ["-c", "from blender_mcp.server import main; main()"]
            if cwd:
                src_path = os.path.join(cwd, "src")
                existing = (env or {}).get("PYTHONPATH") or os.environ.get("PYTHONPATH", "")
                env = {**(env or {}), "PYTHONPATH": os.pathsep.join(
                    value for value in (src_path, existing) if value)}
        # Default: launch the bundled blender-mcp via its own uv project.
        # `cwd` should point at the blender-mcp/ subdirectory so that
        # `uv run blender-mcp` resolves against its pyproject.toml.
        self.params = StdioServerParameters(
            command=command,
            args=args if args is not None else ["run", "blender-mcp"],
            env={**os.environ, **(env or {}), "BLENDER_MCP_DISABLE_TELEMETRY": "true"},
            cwd=cwd,
        )
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._session: ClientSession | None = None

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro) -> Any:
        fut: Future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    # ---- lifecycle ----
    # The MCP SDK uses anyio cancel scopes that MUST be entered and exited in
    # the SAME task. So we run the whole session lifetime inside one persistent
    # task (_session_main): it opens the contexts, signals ready, waits on a
    # shutdown event, then closes the contexts itself.
    def start(self):
        async def _kickoff():
            self._ready = asyncio.Event()
            self._shutdown = asyncio.Event()
            self._session_task = asyncio.create_task(self._session_main())
            await self._ready.wait()
            if self._start_error:
                raise self._start_error
        self._start_error = None
        self._submit(_kickoff())

    async def _session_main(self):
        try:
            async with stdio_client(self.params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._ready.set()
                    await self._shutdown.wait()
        except Exception as e:
            self._start_error = e
            self._ready.set()

    def close(self):
        async def _close():
            if getattr(self, "_shutdown", None):
                self._shutdown.set()
            if getattr(self, "_session_task", None):
                try:
                    await self._session_task
                except Exception:
                    pass
        try:
            self._submit(_close())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ---- tools ----
    def list_tools(self) -> list[dict]:
        """Return tools as plain dicts: {name, description, input_schema}."""
        async def _list():
            resp = await self._session.list_tools()
            out = []
            for t in resp.tools:
                out.append({
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": t.inputSchema or {"type": "object", "properties": {}},
                })
            return out
        return self._submit(_list())

    def call_tool(self, name: str, arguments: dict | None) -> str:
        """Call a tool, return its TEXT result only (images summarized).

        Kept for callers that only care about text (scene setup, snapshots).
        For image-aware calls use call_tool_rich().
        """
        rich = self.call_tool_rich(name, arguments)
        if rich["images"] and not rich["text"].strip():
            return f"[returned {len(rich['images'])} image(s)]"
        return rich["text"]

    def call_tool_rich(self, name: str, arguments: dict | None) -> dict:
        """Call a tool, return {"text": str, "images": [{"data": b64, "mime": str}]}.

        Image content blocks (e.g. from render_scene_view) are preserved as
        base64 so the agent loop can feed them back to vision models.
        """
        async def _call():
            result = await self._session.call_tool(name, arguments or {})
            parts, images, structured = [], [], {}
            image_label = None
            for block in result.content:
                btype = getattr(block, "type", None)
                if btype == "text":
                    try:
                        envelope = json.loads(block.text)
                    except (ValueError, TypeError):
                        envelope = None
                    if isinstance(envelope, dict) and set(envelope) == {"v2s_structured"}:
                        structured.update(envelope["v2s_structured"])
                        continue
                    parts.append(block.text)
                    image_label = block.text.removeprefix("Inspection view: ") if block.text.startswith("Inspection view: ") else None
                elif btype == "image":
                    # MCP ImageContent: .data is base64 str, .mimeType e.g. image/png
                    images.append({"data": block.data,
                                   "mime": getattr(block, "mimeType", "image/png"),
                                   **({"label": image_label} if image_label else {})})
                    image_label = None
                else:
                    parts.append(str(block))
            text = "\n".join(parts)
            if getattr(result, "isError", False):
                text = f"ERROR: {text}"
            output = {"text": text, "images": images}
            if structured: output["structured"] = structured
            if getattr(result, "isError", False): output["isError"] = True
            return output
        return self._submit(_call())
