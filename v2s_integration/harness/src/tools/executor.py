from dataclasses import asdict
import json
import time

from environment.budget import BudgetExceeded, Usage
from tools.base import ToolResult


class ToolExecutor:
    def __init__(self, registry, budget, clock=time.monotonic):
        self.registry = registry
        self.budget = budget
        self.clock = clock
        self.halted = False

    def execute(self, name, arguments):
        started = self.clock()
        entry = self.registry.lookup(name)
        spec, handler = entry if entry else (None, None)
        executed = False
        try:
            if self.halted:
                raise BudgetExceeded("run tool budget is exhausted")
            if not spec or not self.registry.permitted(name):
                result = ToolResult("ERROR: tool is not allowed", usage=Usage.zero(),
                                    status="rejected", error_type="tool_not_allowed")
            elif not spec.available:
                result = ToolResult("ERROR: tool is unavailable", usage=Usage.zero(),
                                    status="rejected", error_type="tool_unavailable")
            elif arguments is not None and not isinstance(arguments, dict):
                result = ToolResult("ERROR: arguments must be an object", usage=Usage.zero(),
                                    status="rejected", error_type="invalid_arguments")
            else:
                self.budget.before_tool(spec.usage_upper_bound)
                executed = True
                try:
                    result = ToolResult.from_legacy(handler(arguments or {}))
                except Exception as exc:
                    result = ToolResult(f"ERROR calling tool: {exc}", status="failed", error_type="tool_exception")
                self.budget.charge(result.usage)
                self.budget.check_after()
        except BudgetExceeded as exc:
            self.halted = True
            if executed:
                result.status = "budget_exceeded"
                result.error_type = "budget_exceeded"
                result.text += f"\nERROR: {exc}"
            else:
                result = ToolResult(f"ERROR: {exc}", usage=Usage.zero(),
                                    status="rejected", error_type="budget_exceeded")
        log = {
            "name": name, "arguments": arguments, "result": result.text[:2000],
            "images_returned": len(result.images),
            "provider": spec.provider if spec else None, "version": spec.version if spec else None,
            "latency_sec": self.clock() - started, **asdict(result.usage),
            "status": result.status, "error_type": result.error_type,
            "artifact_ids": list(result.artifact_ids), "executed": executed,
            "executed_arguments": result.executed_arguments,
            "sanitizer_changes": result.sanitizer_changes, "structured": result.structured,
        }
        if name == "asset.inspect":
            try:
                evidence = json.loads(result.text)
                log["preview_status"] = evidence.get("preview_status")
                log["preview_views"] = evidence.get("preview_views", [])
                log["preview_errors"] = evidence.get("preview_errors", [])
            except (ValueError, AttributeError):
                pass
        return result, log
