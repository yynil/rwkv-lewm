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

CFG="pusht_rwkv_v17_nopos_continue_lr2e5_stage2_ds"
BASE_CKPT="outputs/pusht_rwkv_v17_nopos_stage2_ds/epoch_10"
OUT_DIR="outputs/pusht_rwkv_v17_nopos_continue_lr2e5_stage2_ds"
EVAL_SCRIPT="eval_pusht_rwkv_v17_nopos_continue.sh"

if [[ ! -f "$STABLEWM_HOME/pusht_expert_train.h5" ]]; then
  echo "Missing $STABLEWM_HOME/pusht_expert_train.h5"
  exit 1
fi
if [[ ! -d "$BASE_CKPT" ]]; then
  echo "Missing base checkpoint $BASE_CKPT"
  exit 1
fi

mkdir -p logs outputs
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG="logs/pusht_rwkv_v17_nopos_continue_lr2e5_${RUN_ID}.log"

launch_stage() {
  deepspeed --num_gpus=4 train_deepspeed.py \
    --config-name="$CFG" \
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
  echo "BASE_CKPT=$BASE_CKPT"
  .venv/bin/python --version
  .venv/bin/python - <<'PY'
import deepspeed, torch
print("deepspeed", deepspeed.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("num_gpus", torch.cuda.device_count())
PY
  nvidia-smi || true

  if [[ "$RUN_MODE" == "all" || "$RUN_MODE" == "train" ]]; then
    echo "===== PushT RWKV v17 no-pos continue lr=2e-5 Stage 2 ====="
    launch_stage "$@"
  fi

  if [[ "$RUN_MODE" == "all" || "$RUN_MODE" == "eval" ]]; then
    for epoch in 1 2 4 8 10; do
      ckpt="$OUT_DIR/epoch_${epoch}"
      if [[ ! -d "$ckpt" ]]; then
        echo "Expected checkpoint $ckpt was not found."
        exit 1
      fi
      echo "===== PushT MPC eval: v17 no-pos continue epoch_${epoch} history=1/history=3 ====="
      run_eval_pair "$ckpt"
    done
  fi
} 2>&1 | tee "$LOG"
