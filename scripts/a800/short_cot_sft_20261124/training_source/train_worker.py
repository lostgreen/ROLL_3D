"""Importable native ROLL Raw-CoT worker; no local factory/decorator closures."""
import time
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from gpu_probe_worker import GPUProbeWorker
from roll.distributed.scheduler.decorator import Dispatch, register
from gpu_probe import file_sha256, dump
from train_raw_cot import state_digest


class RawCoTTrainWorker(GPUProbeWorker):
    @register(Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        require_fast = os.environ.get("RAW_COT_REQUIRE_FAST_KERNELS") == "1"
        if require_fast:
            from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen_kernels
            assert qwen_kernels.is_fast_path_available, "Required Qwen3.5 fast kernels unavailable"
        deterministic = os.environ.get("RAW_COT_DETERMINISTIC") == "1"
        torch.use_deterministic_algorithms(deterministic, warn_only=False)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        super().initialize(pipeline_config)
        if require_fast:
            import importlib.metadata
            modules = []
            for name, module in self.strategy.unwrap_model().named_modules():
                if not hasattr(module, "chunk_gated_delta_rule"):
                    continue
                functions = {key: getattr(module, key, None) for key in (
                    "chunk_gated_delta_rule", "recurrent_gated_delta_rule",
                    "causal_conv1d_fn", "causal_conv1d_update")}
                origins = {key: getattr(value, "__module__", "") for key, value in functions.items()}
                assert all(functions.values()), "Incomplete fast kernels: " + name
                assert origins["chunk_gated_delta_rule"].startswith("fla."), origins
                assert origins["recurrent_gated_delta_rule"].startswith("fla."), origins
                assert origins["causal_conv1d_fn"].startswith("causal_conv1d"), origins
                assert origins["causal_conv1d_update"].startswith("causal_conv1d"), origins
                modules.append({"name": name, "functions": origins})
            assert modules, "No actual Qwen3.5 Gated DeltaNet modules audited"
            dump(Path(pipeline_config.output_dir) / ("kernel_audit_rank_" + str(dist.get_rank()) + ".json"), {
                "passed": True, "required": True, "fast_path_available": True,
                "versions": {name: importlib.metadata.version(name) for name in (
                    "torch", "triton", "transformers", "flash-linear-attention", "fla-core", "causal-conv1d")},
                "gated_delta_modules": len(modules), "modules": modules})
        dump(Path(pipeline_config.output_dir) / ("determinism_rank_" + str(dist.get_rank()) + ".json"), {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "warn_only": False,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "attn_implementation": "sdpa", "forced_attention_backend": None,
            "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled()})
        self._selection_hook.remove()
        self._micro_labels = None
        self._micro_lengths = None
        self._micro_index = 0

        def select_current_micro(module, args, kwargs):
            assert self._micro_labels is not None and self._micro_index < len(self._micro_labels)
            row = self._micro_labels[self._micro_index]
            length = self._micro_lengths[self._micro_index]
            self._micro_index += 1
            assert row[length:].eq(-100).all(), "Cannot drop any supervised token"
            positions = (row[:length] != -100).nonzero(as_tuple=True)[0]
            if not positions.numel():
                positions = row.new_tensor([0])
            self._supervised_positions = positions.detach().cpu()
            result = dict(kwargs)
            # DP2/CP1, micro1: remove only this microbatch's right padding
            # before the model forward. MM patches/grids were never padded;
            # labels keep their -100 tail in DataProto and are selected by
            # the same unchanged target positions in loss_func.
            assert result["input_ids"].shape[0] == 1 and length > 0
            for key in ("input_ids", "attention_mask", "position_ids"):
                assert result[key].shape[-1] == row.shape[-1]
                result[key] = result[key][..., :length].contiguous()
            assert "logits_to_keep" not in result
            result["logits_to_keep"] = positions.to(kwargs["input_ids"].device)
            return args, result

        self._selection_hook = self.strategy.model.register_forward_pre_hook(select_current_micro, with_kwargs=True)

    @register(Dispatch.DP_MP_DISPATCH_FIRST, clear_cache=False)
    def train_step(self, data):
        count = len(data.batch["labels"])
        assert count >= 1 and self.worker_config.training_args.per_device_train_batch_size == 1
        self._probe_stage = "training_" + str(data.meta_info["global_step"])
        self._probe_started = time.monotonic()
        self.worker_config.training_args.gradient_accumulation_steps = count
        self._micro_labels = data.batch["labels"].detach().cpu()
        mask = data.batch["attention_mask"].detach().cpu()
        assert mask.ndim == 2 and ((mask == 0) | (mask == 1)).all()
        self._micro_lengths = mask.sum(dim=-1).to(dtype=torch.long).tolist()
        for index, length in enumerate(self._micro_lengths):
            assert length > 0 and mask[index, :length].eq(1).all() and mask[index, length:].eq(0).all(), "Only contiguous right padding is supported"
        self._micro_index = 0
        previous = int(self.strategy.scheduler.last_epoch)
        torch.cuda.reset_peak_memory_stats()
        try:
            output = super().train_step(data)
            assert self._micro_index == count, "Root forward must occur once per microbatch"
            assert int(self.strategy.scheduler.last_epoch) == previous + 1, "Expected exactly one optimizer/scheduler update"
            output.meta_info["metrics"]["audit/optimizer_updates@max"] = 1.
            output.meta_info["metrics"]["audit/actual_ga_steps@max"] = float(count)
            output.meta_info["metrics"]["audit/removed_right_pad_tokens@sum"] = float(mask.numel() - sum(self._micro_lengths))
            output.meta_info["metrics"]["memory/train_peak_allocated_bytes@max"] = float(torch.cuda.max_memory_allocated())
            output.meta_info["metrics"]["memory/train_peak_reserved_bytes@max"] = float(torch.cuda.max_memory_reserved())
            return output
        finally:
            self._micro_labels = None
            self._micro_lengths = None

    def _encode_state(self, value):
        if isinstance(value, DTensor):
            return {"__dtensor__": True, "local": value.to_local().detach().cpu(),
                    "shape": tuple(value.shape), "stride": tuple(value.stride()),
                    "placements": [str(item) for item in value.placements]}
        if torch.is_tensor(value):
            return {"__tensor__": True, "value": value.detach().cpu(), "device_type": value.device.type}
        if isinstance(value, dict):
            return {key: self._encode_state(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._encode_state(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._encode_state(item) for item in value)
        return value

    def _decode_state(self, value, parameter):
        if isinstance(value, dict) and value.get("__dtensor__"):
            assert isinstance(parameter, DTensor)
            assert value["placements"] == [str(item) for item in parameter.placements]
            assert tuple(value["shape"]) == tuple(parameter.shape)
            return DTensor.from_local(value["local"].to(parameter.device), device_mesh=parameter.device_mesh,
                placements=parameter.placements, run_check=False, shape=torch.Size(value["shape"]), stride=tuple(value["stride"]))
        if isinstance(value, dict) and value.get("__tensor__"):
            return value["value"].to(parameter.device if value["device_type"] == "cuda" else "cpu")
        if isinstance(value, dict):
            return {key: self._decode_state(item, parameter) for key, item in value.items()}
        if isinstance(value, list):
            return [self._decode_state(item, parameter) for item in value]
        if isinstance(value, tuple):
            return tuple(self._decode_state(item, parameter) for item in value)
        return value

    def _optimizer_parameter_names(self):
        names = {id(parameter): name for name, parameter in self.strategy.unwrap_model().named_parameters()}
        groups = []
        for group in self.strategy.optimizer.param_groups:
            row = []
            for parameter in group["params"]:
                assert id(parameter) in names, "Optimizer parameter absent from named model parameters"
                assert parameter.requires_grad, "Frozen parameter unexpectedly present in optimizer"
                row.append(names[id(parameter)])
            groups.append(row)
        flattened = [name for group in groups for name in group]
        assert len(flattened) == len(set(flattened)), "Duplicate optimizer parameter"
        return groups

    def _training_state_hashes(self, optimizer, scheduler, rng, parameter_names):
        return {"optimizer_sha256": state_digest(optimizer),
                "scheduler_sha256": state_digest(scheduler),
                "rng_sha256": state_digest(rng),
                "optimizer_parameter_names_sha256": state_digest(parameter_names)}

    @register(Dispatch.ONE_TO_ALL)
    def save_training_state(self, destination, global_step):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        adapter = self.adapter_checkpoint(str(destination / "adapter"), global_step)
        params = {name: self._encode_state(parameter) for name, parameter in self.strategy.unwrap_model().named_parameters() if parameter.requires_grad}
        optimizer = self._encode_state(self.strategy.optimizer.state_dict())
        scheduler = self.strategy.scheduler.state_dict()
        rng = self.strategy.get_rng_state()
        parameter_names = self._optimizer_parameter_names()
        hashes = self._training_state_hashes(optimizer, scheduler, rng, parameter_names)
        payload = {"checkpoint_schema_version": 2, "world_size": dist.get_world_size(),
                   "rank": dist.get_rank(), "global_step": global_step,
                   "parameters": params, "optimizer": optimizer,
                   "optimizer_parameter_names": parameter_names,
                   "scheduler": scheduler, "rng": rng, "training_state_hashes": hashes}
        path = destination / ("rank_" + str(dist.get_rank()) + ".pt")
        torch.save(payload, path)
        dist.barrier()
        return {"rank": dist.get_rank(), "path": str(path), "sha256": file_sha256(path),
                "adapter": adapter, "training_state_hashes": hashes}

    @register(Dispatch.ONE_TO_ALL)
    def restore_training_state(self, destination, expected_step):
        # Only load this application's trusted checkpoint after the driver
        # verifies its manifest and each rank-file hash. No foreign pickle.
        path = Path(destination) / ("rank_" + str(dist.get_rank()) + ".pt")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved.get("checkpoint_schema_version") == 2, "Checkpoint lacks optimizer-name/state-hash audit; re-run smoke"
        assert saved["world_size"] == 2 and saved["rank"] == dist.get_rank() and saved["global_step"] == expected_step
        parameter_names = self._optimizer_parameter_names()
        assert parameter_names == saved["optimizer_parameter_names"], "Optimizer parameter group/name order changed"
        expected_hashes = self._training_state_hashes(saved["optimizer"], saved["scheduler"], saved["rng"], saved["optimizer_parameter_names"])
        assert expected_hashes == saved["training_state_hashes"], "Checkpoint content/state hashes disagree"
        params = {name: parameter for name, parameter in self.strategy.unwrap_model().named_parameters() if parameter.requires_grad}
        assert params.keys() == saved["parameters"].keys()
        with torch.no_grad():
            for name, parameter in params.items():
                source = self._decode_state(saved["parameters"][name], parameter)
                if isinstance(parameter, DTensor):
                    parameter.to_local().copy_(source.to_local())
                else:
                    parameter.copy_(source)
        current = self.strategy.optimizer.state_dict()
        old = saved["optimizer"]
        assert len(old["param_groups"]) == len(current["param_groups"])
        mapping = {}
        for actual_group, saved_group, current_group in zip(self.strategy.optimizer.param_groups, old["param_groups"], current["param_groups"]):
            assert saved_group["params"] == current_group["params"]
            mapping.update(zip(current_group["params"], actual_group["params"]))
        state = {index: self._decode_state(value, mapping[index]) for index, value in old["state"].items()}
        self.strategy.optimizer.load_state_dict({"state": state, "param_groups": old["param_groups"]})
        self.strategy.scheduler.load_state_dict(saved["scheduler"])
        self.strategy.load_rng_state(saved["rng"])
        # Capture actual live values after restore, not copied checkpoint
        # hashes. The driver exposes expected/observed pairs in its receipt.
        observed_hashes = self._training_state_hashes(
            self._encode_state(self.strategy.optimizer.state_dict()), self.strategy.scheduler.state_dict(),
            self.strategy.get_rng_state(), self._optimizer_parameter_names())
        assert observed_hashes == expected_hashes, "Optimizer/scheduler/RNG restore value mismatch"
        dist.barrier()
        return {"rank": dist.get_rank(), "global_step": expected_step,
                "scheduler_last_epoch": self.strategy.scheduler.last_epoch,
                "optimizer_states": len(self.strategy.optimizer.state), "rng_restored": True,
                "optimizer_parameter_names_match": True,
                "expected_training_state_hashes": expected_hashes,
                "observed_training_state_hashes": observed_hashes,
                "training_state_hashes_match": True}
