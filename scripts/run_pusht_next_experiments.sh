#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable-wm}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$ROOT/.torch-extensions}"
export RWKV_JIT_ON="${RWKV_JIT_ON:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-rwkv}"
export PATH="$ROOT/.venv/bin:$PATH"

echo "===== Experiment 1/3: v17-nopos-prednone ====="
bash scripts/train_eval_pusht_rwkv_v17_norm.sh prednone all

echo "===== Experiment 2/3: v17-nopos-allln ====="
bash scripts/train_eval_pusht_rwkv_v17_norm.sh allln all

echo "===== Experiment 3/3: v20-learned-add ====="
bash scripts/train_eval_pusht_rwkv_v20.sh all
