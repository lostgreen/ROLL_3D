"""Adapt the vendored MCP without changing its modeling tool behavior."""

import hashlib
import json
import re
from pathlib import Path

from tools.base import ToolResult, ToolSpec


INTEGRATIONS = {
    "polyhaven": ("get_polyhaven_status", {
        "get_polyhaven_categories", "search_polyhaven_assets", "download_polyhaven_asset", "set_texture"}),
    "sketchfab": ("get_sketchfab_status", {
        "search_sketchfab_models", "get_sketchfab_model_preview", "download_sketchfab_model"}),
    "hyper3d": ("get_hyper3d_status", {
        "generate_hyper3d_model_via_text", "generate_hyper3d_model_via_images",
        "poll_rodin_job_status", "import_generated_asset"}),
    "hunyuan3d": ("get_hunyuan3d_status", {
        "generate_hunyuan3d_model", "poll_hunyuan_job_status", "import_generated_asset_hunyuan"}),
}


# LLM-generated Blender snippets occasionally contain typographic punctuation
# (especially U+2212) that Python rejects as source code. Normalize only the
# small set of punctuation variants that are unambiguous in Python source.
_PYTHON_PUNCTUATION = str.maketrans({
    "\u2212": "-",  # MINUS SIGN
    "\u2013": "-",  # EN DASH
    "\u2014": "-",  # EM DASH
    "\u2018": "'",  # LEFT SINGLE QUOTATION MARK
    "\u2019": "'",  # RIGHT SINGLE QUOTATION MARK
    "\u201c": '"',  # LEFT DOUBLE QUOTATION MARK
    "\u201d": '"',  # RIGHT DOUBLE QUOTATION MARK
})


def _sanitize_blender_code(code):
    """Make common typographic punctuation valid Python without rewriting code."""
    if not isinstance(code, str):
        return code
    code = code.translate(_PYTHON_PUNCTUATION)
    # Blender 4.x renamed the Eevee enum. Models often emit the old value;
    # normalize only a standalone legacy token and preserve BLENDER_EEVEE_NEXT.
    code = re.sub(r"(?<![A-Z0-9_])BLENDER_EEVEE(?!_NEXT)", "BLENDER_EEVEE_NEXT", code)
    if "bpy.mathutils" in code:
        code = code.replace("bpy.mathutils", "mathutils")
        if "import mathutils" not in code:
            code = "import mathutils\n" + code
    return code


def bundled_version():
    root = Path(__file__).resolve().parents[2] / "blender-mcp"
    digest = hashlib.sha256()
    for relative in ("addon.py", "src/blender_mcp/server.py"):
        digest.update((root / relative).read_bytes())
    return "sha256:" + digest.hexdigest()


class BlenderToolProvider:
    def __init__(self, mcp):
        self.mcp = mcp
        self.version = bundled_version()
        self.probes = {}

    def _probe(self, schemas):
        names = {s["name"] for s in schemas}
        # Check toggles separately: the vendored Sketchfab status can report ready
        # for a valid key even while dispatch is disabled by the scene toggle.
        toggles = {}
        if "execute_blender_code" in names:
            code = ("import bpy, json\nprint(json.dumps({k: bool(getattr(bpy.context.scene, "
                    "'blendermcp_use_' + k, False)) for k in "
                    "['polyhaven','sketchfab','hyper3d','hunyuan3d']}))")
            try:
                raw = self.mcp.call_tool_rich("execute_blender_code", {"code": code})["text"]
                prefix = "Code executed successfully: "
                if raw.startswith(prefix):
                    raw = raw[len(prefix):]
                toggles = json.loads(raw.strip())
                if not isinstance(toggles, dict):
                    toggles = {}
            except Exception:
                toggles = {}
        for provider, (status_tool, _) in INTEGRATIONS.items():
            state = {"available": False, "reason": "toggle_off_or_unverified", "probe": status_tool}
            if toggles.get(provider) is True and status_tool in names:
                try:
                    result = self.mcp.call_tool_rich(status_tool, {})
                    # Exact success phrase from the pinned addon. Never persist the
                    # status text, which can contain account names or key metadata.
                    ready = "integration is enabled and ready to use." in result.get("text", "")
                    state.update(available=ready and not result.get("isError", False),
                                 reason=None if ready else "status_not_ready")
                except Exception:
                    state["reason"] = "probe_failed"
            self.probes[provider] = state

    def register(self, registry, adaptive=False):
        schemas = self.mcp.list_tools()
        if adaptive or any(s["name"] in names for s in schemas for _, names in INTEGRATIONS.values()):
            self._probe(schemas)
        for schema in schemas:
            name = schema["name"]
            state = None
            for provider, (_, names) in INTEGRATIONS.items():
                if name in names:
                    state = self.probes.get(provider)
                    break
            spec = ToolSpec(name, schema.get("description", ""), schema.get("input_schema", {}),
                            "blender-mcp", self.version,
                            available=state["available"] if state else True,
                            unavailable_reason=state["reason"] if state else None)
            def handler(args, name=name):
                payload = dict(args or {})
                if name == "execute_blender_code" and "code" in payload:
                    payload["code"] = _sanitize_blender_code(payload["code"])
                result = ToolResult.from_legacy(self.mcp.call_tool_rich(name, payload))
                result.executed_arguments = payload
                original = (args or {}).get("code", "")
                if name == "execute_blender_code" and payload.get("code") != original:
                    for rule, count in [("python_punctuation", sum(ord(c) in _PYTHON_PUNCTUATION for c in original)),
                                        ("eevee_enum", len(re.findall(r"(?<![A-Z0-9_])BLENDER_EEVEE(?!_NEXT)", original))),
                                        ("mathutils_import", original.count("bpy.mathutils"))]:
                        if count: result.sanitizer_changes.append({"rule":rule, "count":count})
                return result

            registry.register(spec, handler)
