"""Explicit reward plugins, separate from model-visible observations."""
from importlib import import_module
import math


def load_reward(path):
    if not path:
        return None
    if not isinstance(path, str) or ':' not in path:
        raise ValueError('reward_function must be module:function')
    module, name = path.rsplit(':', 1)
    function = getattr(import_module(module), name)
    if not callable(function):
        raise ValueError('reward_function must resolve to a callable')
    return function


def evaluate_reward(function, *, env, turn):
    value = 0.0 if function is None else float(function(env=env, turn=turn))
    if not math.isfinite(value):
        raise ValueError('Reward must be finite')
    return value
