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

if [[ ! -d outputs/pusht_rwkv_v7_stage2_ds/epoch_8 ]]; then
  echo "Missing base checkpoint outputs/pusht_rwkv_v7_stage2_ds/epoch_8"
  exit 1
fi

mkdir -p logs outputs
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG="logs/pusht_rwkv_v7_cont_${RUN_ID}.log"

echo "Logging to $LOG"
{
  echo "ROOT=$ROOT"
  echo "STABLEWM_HOME=$STABLEWM_HOME"
  echo "CUDA_HOME=$CUDA_HOME"
  echo "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
  echo "RWKV_JIT_ON=$RWKV_JIT_ON"
  .venv/bin/python --version
  .venv/bin/python scripts/check_rwkv_v7_interleave.py --config-name pusht_rwkv_v7_stage2_cont_ds
  nvidia-smi || true

  echo "===== DeepSpeed ZeRO-2 PushT RWKV v7 Stage 2 continuation ====="
  deepspeed --num_gpus=4 train_deepspeed.py \
    --config-name=pusht_rwkv_v7_stage2_cont_ds \
    --deepspeed_config=config/deepspeed/zero2.json

  echo "===== PushT MPC eval: RWKV v7 continuation ====="
  if [[ ! -d outputs/pusht_rwkv_v7_stage2_cont_ds/epoch_8 ]]; then
    echo "Expected continuation checkpoint outputs/pusht_rwkv_v7_stage2_cont_ds/epoch_8 was not found."
    exit 1
  fi
  bash eval_pusht_rwkv_v7_cont.sh outputs/pusht_rwkv_v7_stage2_cont_ds/epoch_8
} 2>&1 | tee "$LOG"
