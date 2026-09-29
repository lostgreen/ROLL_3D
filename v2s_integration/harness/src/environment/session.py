"""Agent-visible execution state; no evaluator, GT path or coverage labels."""

from environment.budget import BudgetLedger, Usage
from tools.base import ToolResult, ToolSpec
from tools.blender import BlenderToolProvider
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry
from environment.artifacts import ArtifactStore
from tools.retrieval import RetrievalProvider
from tools.scene import SceneProvider


class RunSession:
    def __init__(self, task_spec, mcp, extra_tools=None, artifact_root=None, generation_backend=None, reference_images=None,
                 segmentation_backend=None):
        self.task_spec = task_spec
        self.artifacts = ArtifactStore(artifact_root or "artifacts/runs")
        self.budget = BudgetLedger(task_spec.budget)
        self.registry = ToolRegistry(task_spec.allowed_tools)
        self.blender = BlenderToolProvider(mcp)
        self.blender.register(self.registry, adaptive=task_spec.tool_profile == "adaptive")
        index = getattr(task_spec, "asset_index", None)
        if task_spec.tool_profile == "adaptive" and index:
            RetrievalProvider(index, self.artifacts).register(self.registry)
        self.scene = None
        if task_spec.tool_profile == "adaptive" or generation_backend is not None:
            self.scene = SceneProvider(self.artifacts, mcp)
            self.scene.register(self.registry)
        self.generation = None
        # One run-local map: newly observed crops are immediately available to generation.
        references = dict(reference_images or {})
        if references:
            from tools.reference import ReferenceProvider
            ReferenceProvider(self.artifacts, references, segmentation_backend).register(self.registry)
        if generation_backend is not None:
            from tools.hunyuan import HunyuanProvider
            self.generation = HunyuanProvider(self.artifacts, generation_backend, references, budget=self.budget)
            self.generation.register(self.registry)
        for name, tool in (extra_tools or {}).items():
            usage = Usage.zero() if name == "read_reference_frames" else Usage()
            spec = ToolSpec(name, tool.get("description", ""),
                            tool.get("schema", {"type": "object", "properties": {}}),
                            "harness", "phase0-v1", usage_upper_bound=usage)
            self.registry.register(spec, lambda args, fn=tool["handler"], usage=usage:
                                   ToolResult.from_legacy(fn(args), usage))
        self.registry.validate_allowlist()
        self.executor = ToolExecutor(self.registry, self.budget)

    def replace_blender(self, mcp):
        self.blender.mcp = mcp
        if self.scene is not None: self.scene.mcp = mcp
        # Restarted Blender may have different integration toggles.
        extra_entries = [(spec, fn) for spec, fn in self.registry._entries.values()
                         if spec.provider != "blender-mcp"]
        self.registry = ToolRegistry(self.task_spec.allowed_tools)
        self.blender.register(self.registry, adaptive=self.task_spec.tool_profile == "adaptive")
        for spec, handler in extra_entries:
            self.registry.register(spec, handler)
        self.registry.validate_allowlist()
        self.executor.registry = self.registry

    def snapshot(self):
        return {"schema_version": 1, "task_id": self.task_spec.task_id,
                "tool_profile": self.task_spec.tool_profile,
                "tools": self.registry.snapshot(), "integrations": dict(self.blender.probes),
                "budget": self.budget.snapshot(),
                "artifact_store": str(self.artifacts.root),
                "evaluation_feedback": False,
                "isolation": "contracts only; Blender Python is not a filesystem sandbox"}
