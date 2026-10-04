"""CLI routes only to the current shared task implementation."""
from pathlib import Path
import sys

import pytest
from v2s_integration import run_rollout


@pytest.mark.parametrize('command,module', list(run_rollout.COMMANDS.items()))
def test_public_commands_forward_arguments(monkeypatch, command, module):
    calls = []
    monkeypatch.setattr(sys, 'argv', ['run_rollout.py', command, '--help'])
    monkeypatch.setattr(run_rollout.runpy, 'run_module', lambda name, **kw: calls.append((name, sys.argv[1:])))
    run_rollout.main()
    assert calls == [(module, ['--help'])]


@pytest.mark.parametrize('command', ['train-smoke', 'study'])
def test_retired_commands_are_not_active(monkeypatch, command):
    monkeypatch.setattr(sys, 'argv', ['run_rollout.py', command])
    with pytest.raises(SystemExit) as exc:
        run_rollout.main()
    assert exc.value.code == 2


def test_current_package_excludes_old_manager_and_agent_loop():
    root = Path(run_rollout.__file__).parent
    assert not (root / 'v2s_manager.py').exists()
    assert not (root / 'runtime/vendor/src/harness/agent.py').exists()
