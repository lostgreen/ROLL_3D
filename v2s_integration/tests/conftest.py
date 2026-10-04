"""CPU tensor contracts; model engines and distributed transport are substitutes."""
import ast
from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest


@pytest.fixture
def manager_module(monkeypatch):
    torch = pytest.importorskip('torch')
    from tensordict import TensorDict
    import numpy as np

    class DataProto:
        def __init__(self, batch=None, non_tensor_batch=None, meta_info=None):
            self.batch, self.non_tensor_batch = batch, non_tensor_batch or {}
            self.meta_info = meta_info or {}

        @staticmethod
        def from_single_dict(data):
            return data

        @staticmethod
        def concat(samples):
            keys = samples[0].non_tensor_batch
            return DataProto(torch.cat([s.batch for s in samples]),
                {k: np.concatenate([s.non_tensor_batch[k] for s in samples]) for k in keys})

    @dataclass
    class RolloutCache:
        env_id: int
        group_id: int
        tag: str
        history: list = field(default_factory=list)
        step: int = 0
        terminated: bool = False
        truncated: bool = False

    base = ModuleType('roll.pipeline.agentic.env_manager.vl_traj_env_manager')
    base.VLTrajEnvManager = object
    protocol = ModuleType('roll.distributed.scheduler.protocol')
    protocol.DataProto = DataProto
    cache = ModuleType('roll.pipeline.agentic.env_manager.base_env_manager')
    cache.RolloutCache = RolloutCache
    for module in (base, protocol, cache):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    path = Path(__file__).resolve().parents[1] / 'rollout/manager.py'
    spec = importlib.util.spec_from_file_location('v2s_integration.rollout._contract_manager', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ContractDataProto = DataProto
    return module


@pytest.fixture
def native_postprocess(manager_module):
    import torch
    from tensordict import TensorDict
    root = Path(__file__).resolve().parents[2]
    source = root / 'roll/utils/functionals.py'
    tree = ast.parse(source.read_text())
    names = {'get_pad_mask', 'pad_to_length', 'postprocess_generate'}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {'torch': torch, 'TensorDict': TensorDict, 'Optional': __import__('typing').Optional,
                 'DataProto': manager_module.ContractDataProto}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['postprocess_generate']
