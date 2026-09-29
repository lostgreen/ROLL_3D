"""python -m v2s_metrics --manifest ... --output ..."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from .evaluate import evaluate_manifest, read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline Video2Scene metrics; no Blender or network required for default metrics")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="JSON metric config overrides")
    args = parser.parse_args(argv)
    try:
        overrides = read_json(args.config) if args.config else None
        if overrides is not None and not isinstance(overrides, dict):
            raise ValueError("config overrides must be a JSON object")
        report = evaluate_manifest(args.manifest, overrides)
        if args.config:
            report["config_override_source"] = str(args.config.resolve())
        destination = args.output.resolve()
        if str(destination) in report["inputs"] or (args.config and destination == args.config.resolve()):
            raise ValueError("output must not overwrite a manifest, config, or input artifact")
        payload = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent, delete=False) as stream:
                temporary = stream.name
                stream.write(payload)
            os.replace(temporary, destination)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
        print(f"{report['evaluation_status']}: {destination}")
        return 0 if report["evaluation_status"] == "complete" else 2
    except (OSError, ValueError, TypeError, ImportError) as exc:
        print(f"evaluation failed: {type(exc).__name__}: {str(exc)[:300]}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
