"""Public CLI for reconstruction collection using the existing harness tools."""
import argparse
import runpy
import sys


COMMANDS = {
    'collect': 'behavior_study.demo',
    'study': 'behavior_study.run_study',
    'probe': 'behavior_study.backend_probe',
    'summarize': 'behavior_study.summarize',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=COMMANDS)
    parser.add_argument('arguments', nargs=argparse.REMAINDER,
                        help='Arguments for the selected command; use COMMAND --help')
    args = parser.parse_args()
    sys.argv = [sys.argv[0], *args.arguments]
    runpy.run_module(COMMANDS[args.command], run_name='__main__')


if __name__ == '__main__':
    main()
