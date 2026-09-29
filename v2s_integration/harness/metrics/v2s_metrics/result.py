"""Strict-JSON metric records. Missing measurements never become zero."""
import math

VERSION = "0.1.0"


def metric(value=None, *, direction="lower", unit="unitless", status="ok", reason=None, **details):
    if value is not None:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("metric values must be finite; encode unavailable values as null")
    return {"value": value, "status": status, "reason": reason,
            "direction": direction, "unit": unit, "version": VERSION, **details}


def error_record(exc):
    # No raw traceback, tensors, model response or log content in the report.
    status = "dependency_unavailable" if isinstance(exc, ImportError) else "invalid_input"
    return {"status": status, "reason": f"{type(exc).__name__}: {str(exc)[:240]}"}
