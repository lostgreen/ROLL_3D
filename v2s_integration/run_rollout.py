"""Video2Scene collection/training through the same native ROLL task contract."""
import argparse
from pathlib import Path
import runpy
import sys


COMMANDS = {
    'collect': 'v2s_integration.rollout.collect',
    'train': 'v2s_integration.training.launch',
    'batch': 'v2s_integration.ops.batch',
    'probe': 'v2s_integration.ops.backend_probe',
    'summarize': 'v2s_integration.evaluation.summarize',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=COMMANDS)
    parser.add_argument('arguments', nargs=argparse.REMAINDER,
                        help='Arguments for the selected command; use COMMAND --help')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.argv = [sys.argv[0], *args.arguments]
    module = COMMANDS[args.command]
    runpy.run_module(module, run_name='__main__')


if __name__ == '__main__':
    main()
