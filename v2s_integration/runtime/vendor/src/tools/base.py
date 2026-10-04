from dataclasses import dataclass, field
import re

from environment.budget import Usage


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    provider: str
    version: str
    available: bool = True
    unavailable_reason: str | None = None
    visibility: str = "agent"
    usage_upper_bound: Usage = field(default_factory=Usage)

    category: str = "generic"
    side_effects: tuple = ()
    async_mode: str = "synchronous"
    ui: dict = field(default_factory=dict)

    def contract(self):
        from dataclasses import asdict
        return asdict(self)

    def schema_for_model(self):
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


@dataclass
class ToolResult:
    text: str
    images: list = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    artifact_ids: list[str] = field(default_factory=list)
    status: str = "succeeded"
    error_type: str | None = None

    structured: dict = field(default_factory=dict)
    executed_arguments: dict | None = None
    sanitizer_changes: list = field(default_factory=list)

    @classmethod
    def from_legacy(cls, value, usage=None):
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            value = {"text": str(value)}
        text = value.get("text", "")
        failed = bool(value.get("isError")) or bool(re.match(r"^\s*error\b", text, re.I))
        return cls(text, value.get("images", []), usage or Usage(),
                   status="failed" if failed else "succeeded",
                   error_type="tool_error" if failed else None, structured=value.get("structured", {}))

    def for_model(self):
        text = self.text
        if self.structured:
            import json
            text += "\n" + json.dumps({"structured":self.structured}, ensure_ascii=False, separators=(',', ':'))
        return {"text": text, "images": self.images}
