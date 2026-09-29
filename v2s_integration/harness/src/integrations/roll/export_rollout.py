"""python -m integrations.roll.export_rollout INPUT_IR --output OUTPUT_JSON"""
import argparse
import json
from pathlib import Path
from integrations.roll.schema import export_rollout


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.write_text(json.dumps(export_rollout(json.loads(args.input.read_text())), ensure_ascii=False))


if __name__ == '__main__': main()
