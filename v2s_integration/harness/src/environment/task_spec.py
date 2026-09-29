"""Public agent task fields. Evaluation paths and coverage tiers stay in the runner."""

from dataclasses import dataclass

from environment.budget import BudgetLimits


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    budget: BudgetLimits
    allowed_tools: tuple[str, ...] | None = None
    tool_profile: str = "legacy"
    asset_index: str | None = None

    @classmethod
    def from_task(cls, task, default_steps=30):
        allowed = task.get("allowed_tools")
        if allowed is not None:
            if not isinstance(allowed, list) or not all(isinstance(n, str) for n in allowed):
                raise ValueError("allowed_tools must be a list of names; [] disables all tools")
            if len(allowed) != len(set(allowed)):
                raise ValueError("duplicate allowed_tools")
            allowed = tuple(allowed)
        profile = task.get("tool_profile", "legacy")
        if profile not in ("legacy", "adaptive"):
            raise ValueError("tool_profile must be legacy or adaptive")
        index = task.get("asset_index")
        if index is not None and not isinstance(index, str):
            raise ValueError("asset_index must be a path string")
        return cls(str(task["id"]), BudgetLimits.from_dict(task.get("budget"), default_steps), allowed, profile, index)
