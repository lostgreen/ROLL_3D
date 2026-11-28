#!/bin/sh
set -eu
TASK_ROOT=/data/zgw/short_cot_sft_20261124
TASK_REPO=/m2v_intern/xuboshen/zgw/ROLL
TASK_OLD=$TASK_REPO/artifacts/k_sft_native_thinking_v2_20261116
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ROLL_OTEL_ENABLED=false
export PYTHONPATH="$TASK_ROOT/code/training_source:$TASK_ROOT/vendor_fast:$TASK_OLD/runtime:$TASK_OLD/protocol_source:$TASK_OLD/data_source:$TASK_REPO"
export RAW_COT_REQUIRE_FAST_KERNELS=1
export RAW_COT_CONTEXT_RUNTIME="$TASK_REPO/artifacts/raw_cot_sft_preflight_20261121/runtime_context_v2"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="$TASK_ROOT/cache/train_triton"
export TORCH_EXTENSIONS_DIR="$TASK_ROOT/cache/train_extensions"
export CUDA_CACHE_PATH="$TASK_ROOT/cache/train_cuda"
exec /m2v_intern/xuboshen/zgw/venvs/roll3d_native_20261113/bin/python "$@"
