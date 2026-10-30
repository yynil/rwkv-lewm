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

VARIANT="${1:-scalar}"
RUN_MODE="${2:-all}"
shift 2 || true

case "$VARIANT" in
  scalar)
    LABEL="v21-scaled-scalar"
    STAGE1_CFG="pusht_rwkv_v21_scaled_scalar_stage1_ds"
    STAGE2_CFG="pusht_rwkv_v21_scaled_scalar_stage2_ds"
    STAGE1_DIR="outputs/pusht_rwkv_v21_scaled_scalar_stage1_ds"
    STAGE2_DIR="outputs/pusht_rwkv_v21_scaled_scalar_stage2_ds"
    ;;
  channel)
    LABEL="v21-scaled-channel"
    STAGE1_CFG="pusht_rwkv_v21_scaled_channel_stage1_ds"
    STAGE2_CFG="pusht_rwkv_v21_scaled_channel_stage2_ds"
    STAGE1_DIR="outputs/pusht_rwkv_v21_scaled_channel_stage1_ds"
    STAGE2_DIR="outputs/pusht_rwkv_v21_scaled_channel_stage2_ds"
    ;;
  *)
    echo "Usage: $0 {scalar|channel} {all|stage1|stage2|eval} [training overrides...]" >&2
    exit 2
    ;;
esac

if [[ ! -f "$STABLEWM_HOME/pusht_expert_train.h5" ]]; then
  echo "Missing $STABLEWM_HOME/pusht_expert_train.h5"
  exit 1
fi

mkdir -p logs outputs
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG="logs/pusht_rwkv_${LABEL}_${RUN_ID}.log"

launch_stage() {
  local config_name="$1"
  shift
  deepspeed --num_gpus=4 train_deepspeed.py \
    --config-name="$config_name" \
    --deepspeed_config=config/deepspeed/zero2.json \
    "$@"
}

run_eval() {
  local checkpoint="$1"
  bash eval_pusht_rwkv_v21.sh "$VARIANT" "$checkpoint" "$checkpoint/mpc_eval_50_seed42_cem30.json"
  if [[ "${EVAL_DIAGNOSTIC_H3:-0}" == "1" ]]; then
    bash eval_pusht_rwkv_v21.sh "$VARIANT" "$checkpoint" "$checkpoint/mpc_eval_50_seed42_cem30_history3.json" --world-history-size 3
  fi
}

echo "Logging to $LOG"
{
  echo "ROOT=$ROOT"
  echo "VARIANT=$VARIANT"
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
    echo "===== PushT RWKV ${LABEL} Stage 1 ====="
    launch_stage "$STAGE1_CFG" "$@"
  fi

  if [[ "$RUN_MODE" == "all" || "$RUN_MODE" == "stage2" ]]; then
    if [[ ! -d "$STAGE1_DIR/epoch_2" ]]; then
      echo "Expected stage1 checkpoint $STAGE1_DIR/epoch_2 was not found."
      exit 1
    fi
    echo "===== PushT RWKV ${LABEL} Stage 2 ====="
    launch_stage "$STAGE2_CFG" "$@"
  fi

  if [[ "$RUN_MODE" == "all" || "$RUN_MODE" == "eval" ]]; then
    for epoch in 1 2 3 4 8 9 10; do
      ckpt="$STAGE2_DIR/epoch_${epoch}"
      if [[ ! -d "$ckpt" ]]; then
        echo "Expected stage2 checkpoint $ckpt was not found."
        exit 1
      fi
      echo "===== PushT MPC eval: ${LABEL} epoch_${epoch} history=1 ====="
      run_eval "$ckpt"
    done
  fi
} 2>&1 | tee "$LOG"
