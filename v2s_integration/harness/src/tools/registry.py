"""Registration does not decide which construction path the agent should take."""


class ToolRegistry:
    def __init__(self, allowed_tools=None):
        self.allowed_tools = None if allowed_tools is None else frozenset(allowed_tools)
        self._entries = {}

    def register(self, spec, handler):
        if spec.visibility != "agent":
            raise ValueError("evaluation tools cannot be registered for the agent")
        if spec.name in self._entries:
            raise ValueError(f"duplicate tool: {spec.name}")
        self._entries[spec.name] = (spec, handler)

    def validate_allowlist(self):
        if self.allowed_tools is not None:
            missing = self.allowed_tools - self._entries.keys()
            if missing:
                raise ValueError(f"unknown allowed tools: {sorted(missing)}")

    def permitted(self, name):
        return self.allowed_tools is None or name in self.allowed_tools

    def lookup(self, name):
        return self._entries.get(name)

    def schemas_for_model(self):
        return [spec.schema_for_model() for spec, _ in self._entries.values()
                if spec.available and self.permitted(spec.name)]

    def snapshot(self):
        return [{"name": spec.name, "provider": spec.provider, "version": spec.version,
                 "available": spec.available, "allowed": self.permitted(spec.name),
                 "unavailable_reason": spec.unavailable_reason}
                for spec, _ in self._entries.values()]

    def contracts(self):
        return [dict(spec.contract(), allowed=self.permitted(spec.name))
                for spec, _ in self._entries.values()]

    def contract_hash(self):
        import hashlib
        import json
        contracts = [{k:v for k,v in item.items() if k != 'ui'} for item in self.contracts()]
        return hashlib.sha256(json.dumps(sorted(contracts, key=lambda t:t['name']),
                              sort_keys=True, separators=(',', ':')).encode()).hexdigest()
