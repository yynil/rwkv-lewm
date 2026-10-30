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

if [[ ! -f "$STABLEWM_HOME/pusht_expert_train.h5" ]]; then
  echo "Missing $STABLEWM_HOME/pusht_expert_train.h5"
  exit 1
fi

mkdir -p logs outputs outputs/modulation_impact
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG="logs/pusht_rwkv_v11_v12_${RUN_ID}.log"

launch_stage() {
  local config_name="$1"
  shift
  deepspeed --num_gpus=4 train_deepspeed.py \
    --config-name="$config_name" \
    --deepspeed_config=config/deepspeed/zero2.json \
    "$@"
}

run_eval_pair() {
  local eval_script="$1"
  local checkpoint="$2"
  local output_dir="$3"
  bash "$eval_script" "$checkpoint" "$output_dir/mpc_eval_50_seed42_cem30.json"
  bash "$eval_script" "$checkpoint" "$output_dir/mpc_eval_50_seed42_cem30_history3.json" --world-history-size 3
}

run_diag() {
  local config_name="$1"
  local checkpoint="$2"
  local output="$3"
  .venv/bin/python scripts/diagnose_modulation_impact.py \
    --kind rwkv \
    --checkpoint "$checkpoint" \
    --config-name "$config_name" \
    --batches 8 \
    --batch-size 32 \
    --num-workers 2 \
    --output "$output"
}

run_variant() {
  local name="$1"
  local stage1_cfg="$2"
  local stage2_cfg="$3"
  local stage1_dir="$4"
  local stage2_dir="$5"
  local eval_script="$6"
  shift 6

  echo "===== PushT RWKV ${name} Stage 1 ====="
  launch_stage "$stage1_cfg" "$@"

  echo "===== PushT RWKV ${name} Stage 2 ====="
  if [[ ! -d "$stage1_dir/epoch_2" ]]; then
    echo "Expected stage1 checkpoint $stage1_dir/epoch_2 was not found."
    exit 1
  fi
  launch_stage "$stage2_cfg" "$@"

  for epoch in 1 8; do
    local ckpt="$stage2_dir/epoch_${epoch}"
    if [[ ! -d "$ckpt" ]]; then
      echo "Expected stage2 checkpoint $ckpt was not found."
      exit 1
    fi
    echo "===== PushT MPC eval: ${name} epoch_${epoch} history=1/history=3 ====="
    run_eval_pair "$eval_script" "$ckpt" "$ckpt"
    echo "===== Modulation impact: ${name} epoch_${epoch} ====="
    run_diag "$stage2_cfg" "$ckpt" "outputs/modulation_impact/${name}_epoch${epoch}.json"
  done
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

  if [[ "$RUN_MODE" == "all" || "$RUN_MODE" == "v11" ]]; then
    run_variant \
      "rwkv_v11_bounded_kv" \
      "pusht_rwkv_v11_bounded_kv_stage1_ds" \
      "pusht_rwkv_v11_bounded_kv_stage2_ds" \
      "outputs/pusht_rwkv_v11_bounded_kv_stage1_ds" \
      "outputs/pusht_rwkv_v11_bounded_kv_stage2_ds" \
      "eval_pusht_rwkv_v11_bounded_kv.sh" \
      "$@"
  fi

  if [[ "$RUN_MODE" == "all" || "$RUN_MODE" == "v12" ]]; then
    run_variant \
      "rwkv_v12_adapter" \
      "pusht_rwkv_v12_adapter_stage1_ds" \
      "pusht_rwkv_v12_adapter_stage2_ds" \
      "outputs/pusht_rwkv_v12_adapter_stage1_ds" \
      "outputs/pusht_rwkv_v12_adapter_stage2_ds" \
      "eval_pusht_rwkv_v12_adapter.sh" \
      "$@"
  fi
} 2>&1 | tee "$LOG"
