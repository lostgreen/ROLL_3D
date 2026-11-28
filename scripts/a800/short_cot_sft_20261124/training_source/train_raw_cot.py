"""Explicit-execute Raw-CoT SFT using native ROLL DP2 optimizer semantics.

Dry run requires only stdlib. Model/tokenizer loading starts only with --execute.
"""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

from gpu_probe import bootstrap, dump, file_sha256


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def state_digest(value):
    """Stable content digest for encoded checkpoint state, without pickle bytes.

    Tensor/array shape and dtype participate along with exact bytes. Mapping
    insertion order does not, while list/tuple and parameter order do. Lazy
    imports keep the ordinary dry-run independent of torch.
    """
    result = hashlib.sha256()

    def emit(tag, content=b""):
        result.update(tag.encode() + b":" + str(len(content)).encode() + b":" + content)

    def visit(item):
        if type(item).__module__.startswith("torch"):
            import torch
            assert torch.is_tensor(item), "Unsupported torch checkpoint value"
            tensor = item.detach().cpu().contiguous()
            emit("torch", str(tensor.dtype).encode())
            visit(tuple(tensor.shape))
            emit("bytes", tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif type(item).__module__.startswith("numpy"):
            import numpy as np
            array = np.asarray(item)
            assert array.dtype.kind != "O", "Object arrays are not checkpoint state"
            emit("numpy", array.dtype.str.encode())
            visit(tuple(array.shape))
            emit("bytes", array.tobytes(order="C"))
        elif isinstance(item, dict):
            emit("dict", str(len(item)).encode())
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            emit(type(item).__name__, str(len(item)).encode())
            for child in item:
                visit(child)
        elif item is None or isinstance(item, (str, bool, int, float)):
            emit(type(item).__name__, repr(item).encode())
        elif isinstance(item, bytes):
            emit("bytes", item)
        else:
            raise TypeError("Unsupported checkpoint hash value: " + str(type(item)))

    visit(value)
    return result.hexdigest()


def epoch_batches(rows, effective_batch, seed, epoch):
    order = list(rows)
    random.Random(seed + epoch).shuffle(order)
    return [order[index:index + effective_batch] for index in range(0, len(order), effective_batch)]


def prepare(config, verify_images=True):
    config = dict(config)
    assert config["max_sequence_length"] and int(config["max_sequence_length"]) > 0, "Freeze a measured max_sequence_length first"
    assert config.get("lora_rank", 64) == 64 and config.get("lora_alpha", 128) == 128
    assert config.get("lora_dropout", .05) == .05
    assert config["effective_batch_size"] >= 2 and config["effective_batch_size"] % 2 == 0
    assert config["epochs"] >= 1 and isinstance(config["epochs"], int)
    assert config["learning_rate"] > 0 and 0 <= config["warmup_ratio"] < 1 and config["weight_decay"] >= 0
    lengths = Path(config["lengths_file"]).resolve()
    policy_path = lengths.parent / "context_policy_manifest.json"
    policy = json.loads(policy_path.read_text())
    expected_context = config.get("context_policy_version", "raw_cot_context_v1")
    assert expected_context in ("raw_cot_context_v1", "raw_cot_context_v2")
    assert policy["version"] == expected_context and policy["history_truncation"] is False, "Context policy mismatch"
    assert policy["thinking"] is True
    observation_receipt = None
    if expected_context == "raw_cot_context_v2":
        assert config.get("runtime_root"), "v2 requires an isolated runtime_root"
        sys.path.insert(0, config["repo"])
        from pack_raw_cot import configure_context_runtime
        configure_context_runtime(config["runtime_root"], expected_context)
        from v2s_integration.agent.context_policy import load_context_profile, context_profile_sha256
        from v2s_integration.agent.observation_format import observation_contract
        profile_path = lengths.parent / "context_profile.json"
        profile = load_context_profile(profile_path)
        assert profile["policy_version"] == expected_context
        assert context_profile_sha256(profile) == policy["context_profile_sha256"], "Context profile changed"
        assert profile["observation_format"] == observation_contract(), "Observation implementation changed"
        contract_hash = context_profile_sha256(observation_contract())
        assert contract_hash == policy["observation_contract_sha256"] == config["observation_contract_sha256"]
        observation_path = lengths.parent / "observation_cleanup_manifest.json"
        observation_receipt = json.loads(observation_path.read_text())
        assert observation_receipt["passed"] and observation_receipt["policy_version"] == expected_context
        assert observation_receipt["lengths_sha256"] == file_sha256(lengths)
        assert observation_receipt["context_profile_sha256"] == policy["context_profile_sha256"]
        assert observation_receipt["observation_contract_sha256"] == contract_hash
        observed_records = {entry["sample_id"]: entry for entry in observation_receipt["records"]}
        assert len(observed_records) == observation_receipt["candidate_decisions"]
    cleanup = None
    cleanup_path = lengths.parent / "target_cleanup_manifest.json"
    if config.get("target_body_policy"):
        assert config["target_body_policy"] == "drop_protocol_shaped_body_v1", "Unknown target body policy"
        cleanup = json.loads(cleanup_path.read_text())
        assert cleanup["policy"] == config["target_body_policy"] and cleanup["passed"], "Unverified target cleanup"
        assert cleanup["lengths_sha256"] == file_sha256(lengths), "Cleaned lengths changed"
        cleaned_records = {row["sample_id"]: row for row in cleanup["records"]}
        assert len(cleaned_records) == cleanup["candidate_decisions"], "Incomplete target cleanup receipt"
    for name, expected in policy["model_file_hashes"].items():
        assert file_sha256(Path(config["model"]) / name) == expected, "Model/template changed: " + name
    runtime = Path(config.get("runtime_root", str(Path(config["repo"]) / "artifacts/raw_cot_sft_preflight_20261121/runtime"))) / "v2s_integration/agent"
    for name, expected in policy["context_sources"].items():
        assert file_sha256(runtime / name) == expected, "Context code changed: " + name
    rows, excluded, seen, image_hashes = [], [], set(), {}
    for line in lengths.open():
        row = json.loads(line)
        sid = row["sample_id"]
        assert sid not in seen, "Duplicate decision: " + sid
        seen.add(sid)
        if observation_receipt is not None:
            assert row["context_policy_version"] == expected_context, sid
            assert row["observation_contract_sha256"] == contract_hash, sid
            assert row["harness_equal"] and row["target_ids_equal"] and row["image_order_equal"], sid
        reason = None
        if row.get("quality_hold") or row["scene_id"] in config.get("excluded_scenes", ["scene_171"]):
            reason = "quality_hold"
        elif row.get("eligible") is False:
            reason = "explicit_ineligible"
        elif row["total_tokens"] > config["max_sequence_length"]:
            reason = "whole_decision_overlength"
        if reason:
            excluded.append({"sample_id": sid, "reason": reason, "total_tokens": row["total_tokens"]})
            continue
        assert row["prefix_equal"] and row["parser_equal"], sid
        path = Path(row["record_path"])
        if not path.is_absolute():
            path = lengths.parent / path
        with gzip.open(path, "rt") as handle:
            record = json.load(handle)
        assert record["sample_id"] == sid and record["scene_id"] == row["scene_id"]
        if cleanup is not None:
            expected = cleaned_records[sid]
            assert file_sha256(path) == expected["record_sha256"], "Cleaned record changed: " + sid
            assert not record["content_escapes"], "Protocol-shaped target body still present: " + sid
            if expected["body_removed"]:
                assert record["target"]["content"] == "", "Removed target body reappeared: " + sid
                assert record["target_cleanup"]["policy"] == config["target_body_policy"]
        if observation_receipt is not None:
            expected = observed_records[sid]
            assert file_sha256(path) == expected["record_sha256"], "Formatted record changed: " + sid
            assert record["context_policy_version"] == expected_context
            assert record["observation_contract_sha256"] == contract_hash
            assert record["context_profile_sha256"] == policy["context_profile_sha256"]
            assert expected["target_ids_sha256"] == row["target_ids_sha256"]
        assert len(record["images"]) == row["images"] == len(row["image_hashes"])
        for ref, expected in zip(record["images"], row["image_hashes"]):
            image_path = str(Path(ref["path"]).resolve())
            assert Path(image_path).is_file(), image_path
            if verify_images:
                if image_path not in image_hashes:
                    image_hashes[image_path] = file_sha256(image_path)
                actual = image_hashes[image_path]
                assert actual == expected, "Image changed: " + image_path
        rows.append({**row, "record_path": str(path.resolve()), "record_sha256": file_sha256(path)})
    rows.sort(key=lambda row: row["sample_id"])
    assert rows, "No eligible decisions"
    updates_per_epoch = math.ceil(len(rows) / config["effective_batch_size"])
    total_updates = updates_per_epoch * config["epochs"]
    sources = {name: file_sha256(Path(__file__).with_name(name)) for name in (
        "train_raw_cot.py", "train_worker.py", "gpu_probe.py", "gpu_probe_worker.py", "pack_raw_cot.py")}
    manifest = {
        "config": config, "lengths_sha256": file_sha256(lengths),
        "policy_manifest_sha256": file_sha256(policy_path), "source_hashes": sources,
        "targets_sha256": file_sha256(config["targets_file"]),
        "eligible_decisions": len(rows), "excluded_decisions": len(excluded),
        "eligible_scenes": len({row["scene_id"] for row in rows}),
        "dataset_digest": digest([{key: row[key] for key in ("sample_id", "record_sha256", "total_tokens", "image_hashes")} for row in rows]),
        "updates_per_epoch": updates_per_epoch, "total_updates": total_updates,
        "warmup_steps": math.ceil(total_updates * config["warmup_ratio"]),
        "max_eligible_length": max(row["total_tokens"] for row in rows),
        "target_tokens_per_epoch": sum(row["target_tokens"] for row in rows),
        "tail_actual_batch": len(rows) % config["effective_batch_size"] or config["effective_batch_size"],
        "world_size": 2, "micro_batch_per_rank": 1, "images_verified": verify_images,
        "full_target_loss": True, "target_truncation": False,
    }
    if cleanup is not None:
        manifest["target_body_policy"] = config["target_body_policy"]
        manifest["target_cleanup_manifest_sha256"] = file_sha256(cleanup_path)
        manifest["body_cleaned_eligible_decisions"] = sum(cleaned_records[row["sample_id"]]["body_removed"] for row in rows)
    if observation_receipt is not None:
        manifest["context_policy_version"] = expected_context
        manifest["context_profile_sha256"] = policy["context_profile_sha256"]
        manifest["observation_contract_sha256"] = contract_hash
        manifest["observation_cleanup_manifest_sha256"] = file_sha256(observation_path)
    manifest["manifest_hash"] = digest(manifest)
    return rows, excluded, manifest




def execute(config, rows, manifest, run, resume=None, max_updates=None):
    os.environ["RAW_COT_REQUIRE_FAST_KERNELS"] = "1" if config.get("required_fast_kernels", False) else "0"
    if config.get("runtime_root"):
        os.environ["RAW_COT_CONTEXT_RUNTIME"] = config["runtime_root"]
    if config.get("cuda_allocator"):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = config["cuda_allocator"]
    deterministic = bool(config.get("deterministic_algorithms", False))
    os.environ["RAW_COT_DETERMINISTIC"] = "1" if deterministic else "0"
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = config.get("cublas_workspace_config", ":4096:8")
    bootstrap(Path(config["repo"]))
    import numpy as np
    import torch
    import ray
    from transformers import set_seed
    from pack_raw_cot import materialize_record, setup
    from roll.configs import ModelArguments, TrainingArguments
    from roll.configs.worker_config import WorkerConfig, StrategyArguments
    from roll.pipeline.sft.sft_config import SFTConfig
    from roll.distributed.executor.cluster import Cluster
    from roll.distributed.scheduler.resource_manager import ResourceManager
    from roll.distributed.scheduler.transfer_backend import init_transfer_backend
    from roll.distributed.scheduler.protocol import DataProto
    from roll.models.model_providers import get_extra_data_provider
    from roll.utils.constants import RAY_NAMESPACE
    from roll.utils.functionals import reduce_metrics

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(config["seed"])
    os.environ["roll_RPC_TIMEOUT"] = str(config.get("rpc_timeout", 3600))
    os.environ["RAY_DEDUP_LOGS"] = "1"
    cluster = None
    global_step, epoch, next_batch = 0, 0, 0
    progress = None
    if resume:
        progress = json.loads((resume / "progress.json").read_text())
        assert progress["manifest_hash"] == manifest["manifest_hash"], "Resume configuration/data/code changed"
        for entry in progress["rank_files"]:
            assert file_sha256(resume / Path(entry["path"]).name) == entry["sha256"], "Checkpoint shard changed"
        global_step, epoch, next_batch = progress["global_step"], progress["next_epoch"], progress["next_batch"]
    completed_this_run = 0
    try:
        ray.init(address="local", namespace=RAY_NAMESPACE, include_dashboard=False, log_to_driver=False,
                 num_cpus=8, _temp_dir="/tmp/rawcotsft_" + str(os.getpid()))
        processor = setup(config.get("runtime_root"), config.get("context_policy_version"))
        assert str(config["model"]) == str(__import__("pack_raw_cot").MODEL), "Packer/model mismatch"
        provider = get_extra_data_provider(config["model"], processor)
        from train_worker import RawCoTTrainWorker
        cls = RawCoTTrainWorker
        targets = json.loads(Path(config["targets_file"]).read_text())["targets"]
        worker = WorkerConfig(worker_cls="train_worker.RawCoTTrainWorker", device_mapping="[0, 1]", world_size=2,
            infer_batch_size=1, use_sequence_packing=False, apply_loss_scale=True,
            model_args=ModelArguments(model_name_or_path=config["model"], dtype="bf16", ulysses_size=1,
                attn_implementation="sdpa", lora_target=",".join(targets), lora_rank=64, lora_alpha=128, lora_dropout=.05,
                disable_gradient_checkpointing=False, gradient_checkpointing_use_reentrant=False),
            training_args=TrainingArguments(learning_rate=config["learning_rate"], per_device_train_batch_size=1,
                gradient_accumulation_steps=config["effective_batch_size"] // 2, max_steps=manifest["total_updates"],
                lr_scheduler_type="cosine", warmup_steps=manifest["warmup_steps"], weight_decay=config["weight_decay"]),
            strategy_args=StrategyArguments(strategy_name="fsdp2_train", strategy_config={"fsdp_size": 2,
                "param_dtype": "bf16", "reduce_dtype": "fp32", "offload_policy": False, "async_save_ckpt": False,
                "save_only_model": True, "apply_expert_patch": False,
                "wrap_policy": {"transformer_layer_cls_to_wrap": ["Qwen3_5DecoderLayer", "Qwen3_5VisionBlock"]}}),
            system_envs={key: os.environ[key] for key in ("PYTHONPATH", "LD_LIBRARY_PATH", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                "TOKENIZERS_PARALLELISM", "PYTHONDONTWRITEBYTECODE", "roll_RPC_TIMEOUT", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF",
                "RAW_COT_DETERMINISTIC", "CUBLAS_WORKSPACE_CONFIG", "RAW_COT_REQUIRE_FAST_KERNELS", "RAW_COT_CONTEXT_RUNTIME") if key in os.environ})
        pipeline = SFTConfig(pretrain=config["model"], sft_train=worker, seed=config["seed"], num_nodes=1,
            num_gpus_per_node=2, sequence_length=config["max_sequence_length"], output_dir=str(run), logging_dir=str(run / "logs"),
            track_with="stdout", max_steps=manifest["total_updates"], max_grad_norm=1.)
        dump(run / "resolved_roll_config.json", pipeline.to_dict())
        dump(run / "status.json", {"stage": "initializing", "pid": os.getpid()})
        init_transfer_backend(pipeline.transfer_backend)
        resources = ResourceManager(num_nodes=1, num_gpus_per_node=2)
        cluster = Cluster("sft_train", cls, resources, pipeline.sft_train)
        ray.get(cluster.initialize(pipeline_config=pipeline, blocking=False), timeout=config.get("rpc_timeout", 3600))
        dump(run / "gpu_audit.json", cluster.gpu_audit())
        if resume:
            restored = cluster.restore_training_state(str(resume), global_step)
            audit = cluster.parameter_audit("restored")
            assert all(row["sha256"] == progress["adapter_sha256"] for row in audit)
            dump(run / "resume_receipt.json", {"passed": True, "state": restored, "parameters": audit})
        dump(run / "before.json", cluster.parameter_audit("before"))

        def collate(selected):
            features = []
            for row in selected:
                assert file_sha256(row["record_path"]) == row["record_sha256"], "Record changed during training"
                feature = materialize_record(row["record_path"], processor)
                batch, mm = feature["batch"], feature["multi_modal_inputs"]
                assert batch["input_ids"].shape[1] == row["total_tokens"] <= config["max_sequence_length"]
                assert int((batch["labels"] != -100).sum()) == row["target_tokens"]
                batch = {key: value for key, value in batch.items() if key in ("input_ids", "attention_mask", "labels", "labels_for_loss")}
                batch["position_ids"] = provider(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                                image_grid_thw=mm["image_grid_thw"])["position_ids"]
                features.append((batch, mm))
            actual = len(features)
            if actual % 2:
                base, mm = features[-1]
                dummy = {key: value.clone() for key, value in base.items()}
                dummy["labels"].fill_(-100)
                dummy["labels_for_loss"].fill_(-100)
                features.append((dummy, mm))
            longest = max(batch["input_ids"].shape[1] for batch, _ in features)
            tensors = {}
            for key in features[0][0]:
                fill = processor.tokenizer.pad_token_id if key == "input_ids" else -100 if key.startswith("labels") else 0
                tensors[key] = torch.cat([torch.nn.functional.pad(batch[key], (0, longest - batch[key].shape[-1]), value=fill)
                                          for batch, _ in features], dim=0)
            objects = np.empty(len(features), dtype=object)
            for index, (_, mm) in enumerate(features):
                objects[index] = mm
            assert int((tensors["labels"] != -100).sum()) == sum(row["target_tokens"] for row in selected)
            return DataProto.from_dict(tensors=tensors, non_tensors={"multi_modal_inputs": objects}, meta_info={
                "loss_mask_keys": ["labels", "labels_for_loss"], "global_step": global_step + 1,
                "is_offload_states": False, "is_offload_optimizer_states_in_train_step": False}), actual, len(features) - actual, longest

        def checkpoint(next_epoch, batch_cursor):
            destination = run / ("checkpoint_step_" + f"{global_step:08d}")
            assert not destination.exists(), "Refusing checkpoint overwrite"
            audit = cluster.parameter_audit("after")
            assert len({row["sha256"] for row in audit}) == 1
            shards = cluster.save_training_state(str(destination), global_step)
            value = {"manifest_hash": manifest["manifest_hash"], "global_step": global_step,
                     "next_epoch": next_epoch, "next_batch": batch_cursor,
                     "adapter_sha256": audit[0]["sha256"], "rank_files": shards, "parameters": audit}
            dump(destination / "progress.json", value)
            dump(run / "latest_checkpoint.json", {"path": str(destination), **value})
            return destination

        while epoch < config["epochs"]:
            batches = epoch_batches(rows, config["effective_batch_size"], config["seed"], epoch)
            for index in range(next_batch, len(batches)):
                selected = batches[index]
                wall_started = time.monotonic()
                data, actual, dummy, longest = collate(selected)
                collate_seconds = time.monotonic() - wall_started
                started = time.monotonic()
                dump(run / "status.json", {"stage": "training", "epoch": epoch + 1,
                     "next_update": global_step + 1, "actual_batch": actual, "padded_length": longest, "pid": os.getpid()})
                result = DataProto.materialize_concat(data_refs=cluster.train_step(data, blocking=False), global_keys={"metrics"})
                metrics = {key: float(value) for key, value in reduce_metrics(result.meta_info["metrics"]).items()}
                assert all(math.isfinite(value) for value in metrics.values()), "Nonfinite train metrics"
                global_step += 1
                completed_this_run += 1
                rpc_seconds = time.monotonic() - started
                wall_seconds = time.monotonic() - wall_started
                actual_input_tokens = sum(row["total_tokens"] for row in selected)
                entry = {"global_step": global_step, "epoch": epoch + 1, "batch_index": index,
                         "actual_effective_batch": actual, "dummy_sync_slots": dummy, "ga_steps_per_rank": (actual + dummy) // 2,
                         "supervised_tokens": sum(row["target_tokens"] for row in selected), "padded_length": longest,
                         "sample_ids": [row["sample_id"] for row in selected], "seconds": rpc_seconds,
                         "rpc_seconds": rpc_seconds, "collate_seconds": collate_seconds, "wall_seconds": wall_seconds,
                         "actual_input_tokens": actual_input_tokens,
                         "actual_input_tokens_per_second": actual_input_tokens / wall_seconds, "metrics": metrics}
                with (run / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(entry) + "\n")
                end_epoch = index + 1 == len(batches)
                stop = max_updates is not None and completed_this_run >= max_updates
                if end_epoch or stop or global_step % int(config.get("checkpoint_interval", 25)) == 0:
                    next_epoch, cursor = (epoch + 1, 0) if end_epoch else (epoch, index + 1)
                    destination = checkpoint(next_epoch, cursor)
                    if stop:
                        dump(run / "status.json", {"stage": "bounded_run_complete", "global_step": global_step,
                             "checkpoint": str(destination), "formal_training_complete": global_step == manifest["total_updates"]})
                        print("RAW_COT_BOUNDED_TRAIN_COMPLETE", flush=True)
                        return
            epoch += 1
            next_batch = 0
        dump(run / "status.json", {"stage": "complete", "global_step": global_step, "formal_training_complete": True})
        print("RAW_COT_TRAIN_COMPLETE", flush=True)
    except BaseException as error:
        (run / "error_traceback.log").write_text(traceback.format_exc())
        dump(run / "status.json", {"stage": "failed", "global_step": global_step,
             "error_type": type(error).__name__, "message": str(error).splitlines()[0][:240]})
        raise
    finally:
        if ray.is_initialized():
            if cluster is not None:
                for actor in cluster.workers:
                    try:
                        ray.kill(actor, no_restart=True)
                    except Exception:
                        pass
            ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--max-updates", type=int, help="Bound updates in this invocation, save exact resumable cursor")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.max_sequence_length is not None:
        config["max_sequence_length"] = args.max_sequence_length
    if args.max_updates is not None and args.max_updates < 1:
        parser.error("--max-updates must be positive")
    run = args.root.resolve()
    if run.exists() and any(run.iterdir()):
        parser.error("Use a fresh --root; resume checkpoints may be in another run directory")
    rows, excluded, manifest = prepare(config)
    run.mkdir(parents=True, exist_ok=True)
    dump(run / "manifest.json", manifest)
    dump(run / "excluded_decisions.json", excluded)
    with (run / "eligible_index.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    if not args.execute:
        dump(run / "status.json", {"stage": "dry_run_complete", "gpu_started": False, "manifest_hash": manifest["manifest_hash"]})
        print(json.dumps({key: manifest[key] for key in ("eligible_decisions", "excluded_decisions", "eligible_scenes",
            "updates_per_epoch", "total_updates", "warmup_steps", "tail_actual_batch", "max_eligible_length", "manifest_hash")}))
        return
    execute(config, rows, manifest, run, args.resume.resolve() if args.resume else None, args.max_updates)


if __name__ == "__main__":
    main()
