"""Vector accounting. Missing measurements remain unknown, never free."""

from dataclasses import asdict, dataclass, fields
import math
import time


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class Usage:
    gpu_sec: float | None = None
    external_cost_usd: float | None = None
    generated_assets: int | None = None
    imported_assets: int | None = None

    def __post_init__(self):
        for name, value in asdict(self).items():
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value < 0):
                raise ValueError(f"invalid usage: {name}")
            if name.endswith("assets") and value is not None and not isinstance(value, int):
                raise ValueError(f"asset counts must be integers: {name}")

    @classmethod
    def zero(cls):
        return cls(0.0, 0.0, 0, 0)


@dataclass(frozen=True)
class BudgetLimits:
    max_agent_steps: int = 30
    max_tool_calls: int | None = None
    max_wall_seconds: float | None = None
    max_gpu_seconds: float | None = None
    max_external_cost_usd: float | None = None
    max_generated_assets: int | None = None
    max_imported_assets: int | None = None
    max_retries: int | None = None

    def __post_init__(self):
        for name, value in asdict(self).items():
            if value is None and name != "max_agent_steps":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid budget: {name}")
            if name not in ("max_wall_seconds", "max_gpu_seconds", "max_external_cost_usd") and not isinstance(value, int):
                raise ValueError(f"budget must be an integer: {name}")

    @classmethod
    def from_dict(cls, values, default_steps=30):
        values = dict(values or {})
        unknown = values.keys() - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown budget fields: {sorted(unknown)}")
        values.setdefault("max_agent_steps", default_steps)
        return cls(**values)


RESOURCE_LIMITS = {
    "gpu_sec": "max_gpu_seconds",
    "external_cost_usd": "max_external_cost_usd",
    "generated_assets": "max_generated_assets",
    "imported_assets": "max_imported_assets",
}


class BudgetLedger:
    def __init__(self, limits, clock=time.monotonic):
        self.limits = limits
        self.clock = clock
        self.started = clock()
        self.agent_steps = self.tool_calls = self.retries = self.model_calls = 0
        self.known = {name: 0 for name in RESOURCE_LIMITS}
        self.unknown = {name: 0 for name in RESOURCE_LIMITS}

    def check_wall(self):
        cap = self.limits.max_wall_seconds
        if cap is not None and self.clock() - self.started >= cap:
            raise BudgetExceeded("max_wall_seconds reached")

    def begin_step(self):
        self.check_wall()
        if self.agent_steps >= self.limits.max_agent_steps:
            raise BudgetExceeded("max_agent_steps reached")
        self.agent_steps += 1

    def before_model(self):
        self.check_wall()
        # Existing adapters do not expose prices or request cost upper bounds.
        if self.limits.max_external_cost_usd is not None:
            raise BudgetExceeded("budget_unaccountable: model external_cost_usd")

    def record_model_call(self, usage=None):
        self.model_calls += 1
        self.charge(usage if usage is not None else Usage(0.0, None, 0, 0))

    def retry(self, wait_seconds):
        self.check_wall()
        cap = self.limits.max_retries
        if cap is not None and self.retries >= cap:
            raise BudgetExceeded("max_retries reached")
        wall = self.limits.max_wall_seconds
        if wall is not None and self.clock() - self.started + wait_seconds >= wall:
            raise BudgetExceeded("retry would exceed max_wall_seconds")
        self.retries += 1

    def before_tool(self, upper_bound):
        self.check_wall()
        if self.limits.max_tool_calls is not None and self.tool_calls >= self.limits.max_tool_calls:
            raise BudgetExceeded("max_tool_calls reached")
        for name, limit_name in RESOURCE_LIMITS.items():
            cap = getattr(self.limits, limit_name)
            bound = getattr(upper_bound, name)
            if cap is not None:
                if bound is None or self.unknown[name]:
                    raise BudgetExceeded(f"budget_unaccountable: {name}")
                if self.known[name] + bound > cap:
                    raise BudgetExceeded(f"{limit_name} would be exceeded")
        self.tool_calls += 1

    def charge(self, usage):
        for name, value in asdict(usage).items():
            if value is None:
                self.unknown[name] += 1
            else:
                self.known[name] += value

    def check_after(self):
        self.check_wall()
        for name, limit_name in RESOURCE_LIMITS.items():
            cap = getattr(self.limits, limit_name)
            if cap is not None:
                if self.unknown[name]:
                    raise BudgetExceeded(f"budget_unaccountable: {name}")
                if self.known[name] > cap:
                    raise BudgetExceeded(f"{limit_name} exceeded")

    def snapshot(self):
        return {
            "limits": asdict(self.limits), "agent_steps": self.agent_steps,
            "tool_calls": self.tool_calls, "model_calls": self.model_calls,
            "retries": self.retries, "wall_sec": self.clock() - self.started,
            "usage": {name: None if self.unknown[name] else value for name, value in self.known.items()},
            "known_usage": dict(self.known), "unmeasured_calls": dict(self.unknown),
            "enforcement": "between_calls; synchronous calls cannot be preempted",
        }
