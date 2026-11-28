"""Private r64 preflight worker: native FSDP2 and full-current-target loss."""
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

from selective_sft_worker import SelectiveSFTWorker
from roll.distributed.scheduler.decorator import Dispatch, register


class GPUProbeWorker(SelectiveSFTWorker):
    @register(Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        import roll.distributed.strategy.fsdp2_strategy as fsdp2

        original = fsdp2.get_init_weight_context_manager
        fsdp2.get_init_weight_context_manager = lambda use_meta_tensor=True, mesh=None: original(False, mesh)
        try:
            super().initialize(pipeline_config)
        finally:
            fsdp2.get_init_weight_context_manager = original
        self._probe_stage = "initialized"
        self._probe_started = time.monotonic()

    def _record_failure(self, error):
        # Avoid synchronization/collectives after a CUDA OOM; the other rank may
        # already be waiting on FSDP. The driver subsequently kills only its actors.
        try:
            value = {
                "rank": dist.get_rank(), "stage": self._probe_stage,
                "error_type": type(error).__name__, "message": str(error).splitlines()[0][:240],
                "cuda_oom": "out of memory" in str(error).lower(),
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            path = Path(self.pipeline_config.output_dir) / ("rank_failure_" + str(dist.get_rank()) + ".json")
            path.write_text(json.dumps(value, indent=2) + "\n")
        except Exception:
            pass

    def set_positions(self, data):
        super().set_positions(data)
        # ROLL treats only these two named label fields as -100-masked labels;
        # arbitrary labels_* would otherwise be summed as token IDs. Count the
        # optional diagnostic subsets explicitly over this DP2 global batch.
        data.meta_info["loss_mask_keys"] = ["labels", "labels_for_loss"]
        totals = {}
        for name in ("labels_cot", "labels_body", "labels_action"):
            if name in data.batch:
                count = (data.batch[name] != -100).sum().to(device="cuda", dtype=torch.long)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
                totals[name] = int(count.item())
        data.meta_info["diagnostic_token_totals"] = totals

    @register(Dispatch.DP_MP_DISPATCH_FIRST, clear_cache=False)
    def train_step(self, data):
        try:
            return super().train_step(data)
        except BaseException as error:
            self._record_failure(error)
            raise

    @register(Dispatch.DP_MP_DISPATCH_FIRST, clear_cache=False)
    def val_step(self, data):
        try:
            return super().val_step(data)
        except BaseException as error:
            self._record_failure(error)
            raise

    def loss_func(self, data, output_tensor):
        # The packer already shifts labels by one position. Never shift again.
        positions = self._supervised_positions.to(output_tensor.device)
        labels = data.batch["labels"].index_select(1, positions)
        assert torch.equal(data.batch["labels"], data.batch["labels_for_loss"]), "Full-current-target supervision required"
        assert output_tensor.shape[:2] == labels.shape
        assert (labels != -100).sum() == (data.batch["labels"] != -100).sum()
        total = data.meta_info["batch_num_tokens"]["labels"]
        # This is the only differentiated loss: every unmasked current CoT,
        # body, tool call and ending-token position participates in gradients.
        loss, metrics = self.strategy.op_compute_language_loss(output_tensor, labels, total)
        with torch.no_grad():
            metrics.update({
                "diagnostic/target_loss@sum": float(loss.detach()),
                "memory/supervised_logit_positions@max": float(positions.numel()),
                "memory/full_context_positions@max": float(data.batch["input_ids"].shape[-1]),
            })
            for name in ("labels_cot", "labels_body", "labels_action"):
                if name not in data.batch:
                    continue
                subset = data.batch[name].index_select(1, positions)
                assert not ((subset != -100) & (subset != labels)).any()
                subset_total = data.meta_info["diagnostic_token_totals"][name]
                if subset_total:
                    part_loss, _ = self.strategy.op_compute_language_loss(output_tensor.detach(), subset, subset_total)
                    metrics["diagnostic/" + name.removeprefix("labels_") + "_loss@sum"] = float(part_loss)
        return loss, metrics

    @register(Dispatch.ONE_TO_ALL)
    def gpu_audit(self):
        value = torch.tensor(float(dist.get_rank() + 1), device="cuda")
        dist.all_reduce(value)
        assert value.item() == 3 and dist.get_world_size() == 2
        model = self.strategy.unwrap_model()
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        assert trainable and all("lora_" in name for name, _ in trainable)
        assert not any(".visual." in name or ".vision_tower." in name or ".lm_head." in name for name, _ in trainable)
        adapter = model.peft_config["default"]
        assert adapter.r == 64 and adapter.lora_alpha == 128 and adapter.lora_dropout == .05
        free, total = torch.cuda.mem_get_info()
        return {
            "rank": dist.get_rank(), "world_size": dist.get_world_size(),
            "device": torch.cuda.get_device_name(), "total_bytes": total,
            "free_bytes": free, "backend": dist.get_backend(), "all_reduce_sum": value.item(),
            "initialization_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "initialization_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "strategy": type(self.strategy).__name__,
            "fsdp_modules": sum(hasattr(module, "set_requires_gradient_sync") for module in model.modules()),
            "initialization": "cpu_per_rank_then_native_fsdp2", "trainable_tensors": len(trainable),
            "lora_rank": adapter.r, "lora_alpha": adapter.lora_alpha, "lora_dropout": adapter.lora_dropout,
            "vision_and_head_frozen": True, "loss_scope": "all_current_target_positions",
        }

    @register(Dispatch.ONE_TO_ALL)
    def begin_memory_stage(self, stage):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self._probe_stage, self._probe_started = stage, time.monotonic()
        return self.memory_snapshot()

    @register(Dispatch.ONE_TO_ALL)
    def memory_snapshot(self):
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return {
            "rank": dist.get_rank(), "stage": self._probe_stage,
            "seconds": time.monotonic() - self._probe_started,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "free_device_bytes": free, "total_device_bytes": total,
            "other_process_memory_not_in_torch_peaks": True,
        }
