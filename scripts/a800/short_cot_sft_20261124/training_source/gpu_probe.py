"""One real multimodal sample/length per isolated native ROLL DP2 r64 probe.

Run with the previously verified h800_env_check/run_sft_python.sh interpreter.
The parent supplies the already serialized, image-expanded and shifted batch.
No truncation, artificial padding, model downloads, or official training occurs.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def bootstrap(repo):
    old = repo / "artifacts/k_sft_native_thinking_v2_20261116"
    task = Path("/data/zgw/short_cot_sft_20261124")
    source = Path(__file__).resolve().parent
    values = [source, task / "vendor_fast", old / "runtime", old / "protocol_source", old / "data_source", repo]
    sys.path[:0] = [str(x) for x in values]
    os.environ.update(PYTHONPATH=os.pathsep.join(map(str, values)), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", ROLL_OTEL_ENABLED="false")
    vendor = str(old / "vendor")
    if vendor not in sys.path: sys.path.append(vendor)
    return old


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_batch(path, torch):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    single = {key: value.contiguous() for key, value in saved["batch"].items()}
    mm = saved["multi_modal_inputs"]
    metadata = saved.get("metadata", {})
    assert not metadata.get("synthetic", False), "Use real serialized trajectory decisions"
    ids = single["input_ids"]
    assert ids.ndim == 2 and ids.shape[0] == 1
    assert "pixel_values" in mm and "image_grid_thw" in mm and mm["image_grid_thw"].numel() > 0
    assert all(torch.is_tensor(value) for value in mm.values())
    assert ids.dtype == torch.long and single["labels"].dtype == torch.long
    assert single["attention_mask"].shape == ids.shape and single["attention_mask"].bool().all(), "One unpadded real sample required"
    for key, value in single.items():
        if key.startswith("labels"):
            assert value.shape == ids.shape and value.dtype == torch.long
            assert value[:, -1].eq(-100).all(), "Expected pre-shifted labels with ignored last position"
            assert not ((value[:, :-1] != -100) & (value[:, :-1] != ids[:, 1:])).any(), "Incorrectly shifted labels"
    assert (single["labels"] != -100).any()
    assert "labels_for_loss" in single and torch.equal(single["labels"], single["labels_for_loss"])
    for key in ("labels_cot", "labels_body", "labels_action"):
        if key in single:
            assert not ((single[key] != -100) & (single[key] != single["labels"])).any(), key
    if "prompt_length" in metadata:
        prefix = int(metadata["prompt_length"])
        assert 0 < prefix < ids.shape[1]
        assert single["labels"][:, :prefix - 1].eq(-100).all(), "Prompt leak into target loss"
        assert single["labels"][:, prefix - 1].ne(-100).all(), "First target position omitted"
    return single, mm, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--batch-file", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--repo", type=Path, default=Path("/m2v_intern/xuboshen/zgw/ROLL"))
    parser.add_argument("--model", default="/m2v_intern/xuboshen/models/Qwen3.5-9B")
    parser.add_argument("--targets-file", type=Path)
    parser.add_argument("--rpc-timeout", type=int, default=1800)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--skip-validation", action="store_true", help="Capacity probe only; receipt records missing val checks")
    parser.add_argument("--skip-checkpoint", action="store_true", help="Capacity probe only; no independent reload possible")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    run = args.root.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if (run / "status.json").exists() or (run / "adapter").exists():
        raise RuntimeError("Use a fresh output directory; refusing to mix probe runs")
    batch_file = args.batch_file.resolve()
    old = bootstrap(args.repo)
    os.environ["roll_RPC_TIMEOUT"] = str(args.rpc_timeout)
    os.environ["RAY_DEDUP_LOGS"] = "1"
    stage, cluster, ray = "imports", None, None
    memory_stages, updates = [], []

    def status(value, **extra):
        nonlocal stage
        stage = value
        dump(run / "status.json", {"stage": value, "pid": os.getpid(), "at": time.time(), **extra})

    status("imports")
    try:
        import numpy as np
        import torch
        import ray as ray_module
        from transformers import AutoProcessor, set_seed
        from gpu_probe_worker import GPUProbeWorker
        from roll.configs import ModelArguments, TrainingArguments
        from roll.configs.worker_config import StrategyArguments, WorkerConfig
        from roll.distributed.executor.cluster import Cluster
        from roll.distributed.scheduler.protocol import DataProto
        from roll.distributed.scheduler.resource_manager import ResourceManager
        from roll.distributed.scheduler.transfer_backend import init_transfer_backend
        from roll.pipeline.sft.sft_config import SFTConfig
        from roll.utils.constants import RAY_NAMESPACE
        from roll.utils.functionals import reduce_metrics
        from roll.models.model_providers import get_extra_data_provider

        ray = ray_module
        set_seed(20261121)
        torch.set_num_threads(2)
        single, mm, metadata = load_batch(batch_file, torch)
        ray.init(address="local", namespace=RAY_NAMESPACE, include_dashboard=False,
                 log_to_driver=False, num_cpus=8, _temp_dir="/tmp/rawcotprobe_" + str(os.getpid()))
        status("prepare_batch")
        if "position_ids" not in single:
            processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
            single["position_ids"] = get_extra_data_provider(args.model, processor)(
                input_ids=single["input_ids"], attention_mask=single["attention_mask"],
                image_grid_thw=mm["image_grid_thw"])["position_ids"]
        assert single["position_ids"].shape[0] == 1 and single["position_ids"].shape[-1] == single["input_ids"].shape[-1]
        torch.save({"batch": single, "multi_modal_inputs": mm, "metadata": metadata}, run / "probe_batch.pt")
        tokens = int(single["input_ids"].shape[1])
        data_receipt = {
            "synthetic_environment_test": False, "batch_file": str(batch_file),
            "batch_sha256": file_sha256(batch_file), "metadata": metadata,
            "tokens": tokens, "supervised_tokens": int((single["labels"] != -100).sum()),
            "mask_counts": {key: int((value != -100).sum()) for key, value in single.items() if key.startswith("labels")},
            "image_grid": mm["image_grid_thw"].tolist(), "position_shape": list(single["position_ids"].shape),
            "loss_scope": "all_current_target_positions", "replicated_identical_sample_on_both_dp_ranks": True,
        }
        dump(run / "data_receipt.json", data_receipt)
        targets_file = args.targets_file or old / "runs/K_A_paired/prepared/lora_targets.json"
        targets = json.loads(targets_file.read_text())["targets"]
        assert targets and len(targets) == len(set(targets))
        worker = WorkerConfig(
            worker_cls="gpu_probe_worker.GPUProbeWorker", device_mapping="[0, 1]", world_size=2,
            infer_batch_size=1, use_sequence_packing=False, apply_loss_scale=True,
            model_args=ModelArguments(
                model_name_or_path=args.model, dtype="bf16", ulysses_size=1, attn_implementation="sdpa",
                lora_target=",".join(targets), lora_rank=64, lora_alpha=128, lora_dropout=.05,
                disable_gradient_checkpointing=False, gradient_checkpointing_use_reentrant=False),
            training_args=TrainingArguments(
                learning_rate=args.learning_rate, per_device_train_batch_size=1, gradient_accumulation_steps=1,
                max_steps=args.steps, lr_scheduler_type="constant", warmup_steps=0, weight_decay=.01),
            strategy_args=StrategyArguments(strategy_name="fsdp2_train", strategy_config={
                "fsdp_size": 2, "param_dtype": "bf16", "reduce_dtype": "fp32", "offload_policy": False,
                "async_save_ckpt": False, "save_only_model": True, "apply_expert_patch": False,
                "wrap_policy": {"transformer_layer_cls_to_wrap": ["Qwen3_5DecoderLayer", "Qwen3_5VisionBlock"]}}),
            system_envs={key: os.environ[key] for key in (
                "PYTHONPATH", "TOKENIZERS_PARALLELISM", "LD_LIBRARY_PATH", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                "PYTHONDONTWRITEBYTECODE", "roll_RPC_TIMEOUT", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF") if key in os.environ})
        config = SFTConfig(
            pretrain=args.model, sft_train=worker, seed=20261121, num_nodes=1, num_gpus_per_node=2,
            sequence_length=tokens, output_dir=str(run), logging_dir=str(run / "logs"),
            track_with="stdout", max_steps=args.steps, max_grad_norm=1.)
        dump(run / "config.json", config.to_dict())
        dump(run / "invocation.json", {"args": vars(args), "python": sys.executable,
             "torch": torch.__version__, "targets_file": str(targets_file), "targets_sha256": file_sha256(targets_file),
             "runner_sha256": file_sha256(__file__), "worker_sha256": file_sha256(Path(__file__).with_name("gpu_probe_worker.py"))})
        status("initializing_native_dp2", tokens=tokens)
        init_transfer_backend(config.transfer_backend)
        resources = ResourceManager(num_nodes=1, num_gpus_per_node=2)
        cluster = Cluster("sft_train", GPUProbeWorker, resources, config.sft_train)
        ray.get(cluster.initialize(pipeline_config=config, blocking=False), timeout=args.rpc_timeout)
        assert cluster.dp_size == 2 and cluster.cp_size == 1
        audit = cluster.gpu_audit()
        dump(run / "gpu_audit.json", audit)
        before = cluster.parameter_audit("before")
        dump(run / "before.json", before)
        assert len({row["sha256"] for row in before}) == 1
        dump(run / "scheduler.json", cluster.scheduler_audit())

        def batch(step):
            objects = np.empty(2, dtype=object)
            for index in range(2):
                objects[index] = {key: value.clone() for key, value in mm.items()}
            return DataProto.from_dict(
                tensors={key: torch.cat([value, value], dim=0) for key, value in single.items()},
                non_tensors={"multi_modal_inputs": objects},
                meta_info={"loss_mask_keys": [key for key in single if key.startswith("labels")],
                           "global_step": step, "is_offload_states": False, "is_offload_optimizer_states_in_train_step": False})

        def measure(method, step, name):
            status(name, tokens=tokens, step=step)
            start_memory = cluster.begin_memory_stage(name)
            started = time.monotonic()
            data = DataProto.materialize_concat(data_refs=method(batch(step), blocking=False), global_keys={"metrics"})
            metrics = {key: float(value) for key, value in reduce_metrics(data.meta_info["metrics"]).items()}
            assert metrics and all(math.isfinite(value) for value in metrics.values()), "Nonfinite/empty metrics"
            snapshot = cluster.memory_snapshot()
            memory_stages.append({"stage": name, "seconds": time.monotonic() - started, "start": start_memory, "end": snapshot})
            dump(run / "memory_stages.json", memory_stages)
            return metrics

        pre = None if args.skip_validation else measure(cluster.val_step, 0, "validation_before")
        for step in range(1, args.steps + 1):
            metrics = measure(cluster.train_step, step, "training_" + str(step))
            grad = [value for key, value in metrics.items() if "grad_norm" in key]
            assert grad and all(value > 0 for value in grad), "Missing/zero gradient norm"
            updates.append(metrics)
            dump(run / "train_metrics.json", updates)
        post = None if args.skip_validation else measure(cluster.val_step, args.steps, "validation_after")
        after = cluster.parameter_audit("after")
        dump(run / "after.json", after)
        assert len({row["sha256"] for row in after}) == 1 and after[0]["sha256"] != before[0]["sha256"]
        assert all(row["delta_l2"] > 0 for row in after)
        checkpoint, save = None, None
        if not args.skip_checkpoint:
            status("checkpoint", tokens=tokens)
            checkpoint = run / "adapter"
            save = cluster.adapter_checkpoint(str(checkpoint), args.steps)
        dump(run / "receipt.json", {
            "passed": True, "optimizer_steps": args.steps, "visual_input": True, "native_roll": True,
            "tokens": tokens, "loss_scope": "all_current_target_positions", "gpu_audit": audit,
            "parameters": after, "before_metrics": pre, "after_metrics": post, "updates": updates,
            "checkpoint": str(checkpoint) if checkpoint else None, "save": save, "model": args.model,
            "validation_executed": not args.skip_validation, "independent_reload_pending": not args.skip_checkpoint,
            "memory_stages": memory_stages, "data_receipt": data_receipt,
        })
        status("complete", passed=True, tokens=tokens)
        print("RAW_COT_R64_PROBE_PASSED", flush=True)
    except BaseException as error:
        text = str(error)
        fingerprint = {"passed": False, "stage": stage, "error_type": type(error).__name__,
                       "message": text.splitlines()[0][:240] if text else "",
                       "cuda_oom": "out of memory" in text.lower(), "memory_stages": memory_stages}
        (run / "error_traceback.log").write_text(traceback.format_exc())
        dump(run / "receipt.json", fingerprint)
        status("failed", failure=fingerprint)
        print("RAW_COT_R64_PROBE_FAILED " + json.dumps({key: fingerprint[key] for key in ("stage", "error_type", "cuda_oom")}), flush=True)
        raise
    finally:
        if ray is not None and ray.is_initialized():
            if cluster is not None:
                for actor in cluster.workers:
                    try:
                        ray.kill(actor, no_restart=True)
                    except Exception:
                        pass
            ray.shutdown()


if __name__ == "__main__":
    main()
