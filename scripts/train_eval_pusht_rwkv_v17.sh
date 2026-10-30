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

RUN_MODE="${1:-all}"
shift || true

STAGE1_CFG="pusht_rwkv_v17_add_action_stage1_ds"
STAGE2_CFG="pusht_rwkv_v17_add_action_stage2_ds"
STAGE1_DIR="outputs/pusht_rwkv_v17_add_action_stage1_ds"
STAGE2_DIR="outputs/pusht_rwkv_v17_add_action_stage2_ds"
EVAL_SCRIPT="eval_pusht_rwkv_v17_add_action.sh"

if [[ ! -f "$STABLEWM_HOME/pusht_expert_train.h5" ]]; then
  echo "Missing $STABLEWM_HOME/pusht_expert_train.h5"
  exit 1
fi

mkdir -p logs outputs
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG="logs/pusht_rwkv_v17_add_action_${RUN_ID}.log"

launch_stage() {
  local config_name="$1"
  shift
  deepspeed --num_gpus=4 train_deepspeed.py \
    --config-name="$config_name" \
    --deepspeed_config=config/deepspeed/zero2.json \
    "$@"
}

run_eval_pair() {
  local checkpoint="$1"
  bash "$EVAL_SCRIPT" "$checkpoint" "$checkpoint/mpc_eval_50_seed42_cem30.json"
  bash "$EVAL_SCRIPT" "$checkpoint" "$checkpoint/mpc_eval_50_seed42_cem30_history3.json" --world-history-size 3
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

  if [[ "$RUN_MODE" == "all" || "$RUN_MODE" == "stage1" ]]; then
    echo "===== PushT RWKV v17 add_action Stage 1 ====="
    launch_stage "$STAGE1_CFG" "$@"
  fi

  if [[ "$RUN_MODE" == "all" || "$RUN_MODE" == "stage2" ]]; then
    if [[ ! -d "$STAGE1_DIR/epoch_2" ]]; then
      echo "Expected stage1 checkpoint $STAGE1_DIR/epoch_2 was not found."
      exit 1
    fi
    echo "===== PushT RWKV v17 add_action Stage 2 ====="
    launch_stage "$STAGE2_CFG" "$@"
  fi

  if [[ "$RUN_MODE" == "all" || "$RUN_MODE" == "eval" ]]; then
    for epoch in 1 2 3 4 8 9 10; do
      ckpt="$STAGE2_DIR/epoch_${epoch}"
      if [[ ! -d "$ckpt" ]]; then
        echo "Expected stage2 checkpoint $ckpt was not found."
        exit 1
      fi
      echo "===== PushT MPC eval: v17 add_action epoch_${epoch} history=1/history=3 ====="
      run_eval_pair "$ckpt"
    done
  fi
} 2>&1 | tee "$LOG"
