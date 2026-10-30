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

if [[ ! -f "$STABLEWM_HOME/pusht_expert_train.h5" ]]; then
  echo "Missing $STABLEWM_HOME/pusht_expert_train.h5"
  echo "Download it with: python scripts/download_lewm_data.py pusht --output $STABLEWM_HOME"
  exit 1
fi

mkdir -p logs outputs
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG="logs/pusht_deepspeed_zero2_${RUN_ID}.log"

launch_stage() {
  local config_name="$1"
  shift
  deepspeed --num_gpus=4 train_deepspeed.py \
    --config-name="$config_name" \
    --deepspeed_config=config/deepspeed/zero2.json \
    "$@"
}

echo "Logging to $LOG"
{
  echo "ROOT=$ROOT"
  echo "STABLEWM_HOME=$STABLEWM_HOME"
  echo "CUDA_HOME=$CUDA_HOME"
  echo "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
  echo "RWKV_JIT_ON=$RWKV_JIT_ON"
  .venv/bin/python --version
  .venv/bin/python - <<'PY'
import deepspeed, torch
print("deepspeed", deepspeed.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("num_gpus", torch.cuda.device_count())
PY
  nvidia-smi || true

  if [[ "$STAGE" == "stage1" || "$STAGE" == "all" ]]; then
    echo "===== DeepSpeed ZeRO-2 PushT Stage 1 ====="
    launch_stage pusht_stage1_ds "$@"
  fi

  if [[ "$STAGE" == "stage2" || "$STAGE" == "all" ]]; then
    echo "===== DeepSpeed ZeRO-2 PushT Stage 2 ====="
    if [[ "$STAGE" == "all" && ! -d outputs/pusht_lewmalign_stage1_ds/epoch_2 ]]; then
      echo "Expected stage1 DeepSpeed checkpoint outputs/pusht_lewmalign_stage1_ds/epoch_2 was not found."
      exit 1
    fi
    launch_stage pusht_stage2_ds "$@"
  fi
} 2>&1 | tee "$LOG"
