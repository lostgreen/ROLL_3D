#!/bin/sh
set -eu
TASK_ROOT=/data/zgw/short_cot_sft_20261124
unset PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export VLLM_CACHE_ROOT="$TASK_ROOT/cache/vllm"
export FLASHINFER_WORKSPACE_BASE="$TASK_ROOT/cache/flashinfer"
export TRITON_CACHE_DIR="$TASK_ROOT/cache/vllm_triton"
export TORCH_EXTENSIONS_DIR="$TASK_ROOT/cache/vllm_extensions"
export CUDA_CACHE_PATH="$TASK_ROOT/cache/vllm_cuda"
# Existing A800 toolchain; keep the old training Python out of this env.
TASK_COMPILER=/m2v_intern/xuboshen/zgw/venvs/roll3d_base_20261113
export CUDA_HOME="$TASK_COMPILER"
export CC="$TASK_COMPILER/bin/x86_64-conda-linux-gnu-cc"
export CXX="$TASK_COMPILER/bin/x86_64-conda-linux-gnu-c++"
export LD_PRELOAD="$TASK_COMPILER/lib/libstdc++.so.6"
export MAX_JOBS=4
export VLLM_WORKER_MULTIPROC_METHOD=spawn
exec "$TASK_ROOT/env0171/bin/python" "$@"
