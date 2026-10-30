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

STAGE="${1:-all}"
shift || true

if [[ ! -f "$STABLEWM_HOME/tworoom_train.h5" ]]; then
  echo "Missing $STABLEWM_HOME/tworoom_train.h5; run scripts/split_tworoom_h5.py first."
  exit 1
fi

launch_stage() {
  local config_name="$1"
  shift
  deepspeed --num_gpus=4 train_deepspeed.py \
    --config-name="$config_name" \
    --deepspeed_config=config/deepspeed/zero2.json \
    "$@"
}

if [[ "$STAGE" == "stage1" || "$STAGE" == "all" ]]; then
  echo "===== Split DeepSpeed ZeRO-2 Stage 1 ====="
  launch_stage tworoom_stage1_split_ds "$@"
fi

if [[ "$STAGE" == "stage2" || "$STAGE" == "all" ]]; then
  echo "===== Split DeepSpeed ZeRO-2 Stage 2 ====="
  launch_stage tworoom_stage2_split_ds "$@"
fi
