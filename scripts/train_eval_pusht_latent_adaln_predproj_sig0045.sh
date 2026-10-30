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

STAGE1_DIR="outputs/pusht_latent_adaln_predproj_stage1_ds"
STAGE2_DIR="outputs/pusht_latent_adaln_predproj_stage2_sig0045_ds"
EVAL_JSON="${STAGE2_DIR}/epoch_8/mpc_eval_50_seed42_cem30.json"

if [[ ! -f "$STABLEWM_HOME/pusht_expert_train.h5" ]]; then
  echo "Missing $STABLEWM_HOME/pusht_expert_train.h5"
  echo "Download it with: python scripts/download_lewm_data.py pusht --output $STABLEWM_HOME"
  exit 1
fi

if [[ ! -d "${STAGE1_DIR}/epoch_2" ]]; then
  echo "Expected stage1 checkpoint ${STAGE1_DIR}/epoch_2 was not found."
  exit 1
fi

mkdir -p logs outputs
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG="logs/pusht_latent_adaln_predproj_sig0045_${RUN_ID}.log"

launch_stage2() {
  deepspeed --num_gpus=4 train_deepspeed.py \
    --config-name=pusht_latent_adaln_predproj_stage2_sig0045_ds \
    --deepspeed_config=config/deepspeed/zero2.json \
    "$@"
}

run_eval() {
  local checkpoint="${1:-${STAGE2_DIR}/epoch_8}"
  local output="${2:-${checkpoint%/}/mpc_eval_50_seed42_cem30.json}"
  .venv/bin/python scripts/evaluate_pusht_mpc.py \
    --checkpoint "$checkpoint" \
    --train-config pusht_latent_adaln_predproj_stage2_sig0045_ds \
    --device cuda \
    --dataset-name pusht_expert_train \
    --img-size 224 \
    --num-eval 50 \
    --seed 42 \
    --goal-offset-steps 25 \
    --eval-budget 50 \
    --horizon 5 \
    --receding-horizon 5 \
    --action-block 5 \
    --cem-samples 300 \
    --cem-steps 30 \
    --cem-topk 30 \
    --cem-batch-size 1 \
    --var-scale 1.0 \
    --output "$output"
}

compare_results() {
  local current="$1"
  .venv/bin/python - "$current" <<'PY'
import json
import sys
from pathlib import Path

current = Path(sys.argv[1])
baselines = [
    ("previous_lewmalign", Path("outputs/pusht_lewmalign_stage2_ds/epoch_8/mpc_eval_50_seed42_cem30.json")),
    ("latent_adaln_original_sigreg", Path("outputs/pusht_latent_adaln_predproj_stage2_ds/epoch_8/mpc_eval_50_seed42_cem30.json")),
    ("bad_double_action_adaln_predproj", Path("outputs/pusht_adaln_predproj_stage2_ds/epoch_8/mpc_eval_50_seed42_cem30.json")),
    ("old_residual", Path("outputs/pusht_stage2_ds/epoch_10/mpc_eval_50_seed42_cem30.json")),
]

def read_rate(path: Path):
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return float(data["metrics"]["success_rate"])

cur_rate = read_rate(current)
print(f"current_latent_adaln_predproj_sig0045: {cur_rate:.1f}% ({current})")
for name, path in baselines:
    rate = read_rate(path)
    if rate is None:
        print(f"{name}: missing ({path})")
    else:
        print(f"{name}: {rate:.1f}% ({path}), delta={cur_rate - rate:+.1f}")
PY
}

echo "Logging to $LOG"
{
  echo "ROOT=$ROOT"
  echo "STABLEWM_HOME=$STABLEWM_HOME"
  echo "CUDA_HOME=$CUDA_HOME"
  echo "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
  echo "RWKV_JIT_ON=$RWKV_JIT_ON"
  echo "STAGE1_DIR=$STAGE1_DIR"
  echo "STAGE2_DIR=$STAGE2_DIR"
  .venv/bin/python --version
  .venv/bin/python - <<'PY'
import deepspeed, torch
print("deepspeed", deepspeed.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("num_gpus", torch.cuda.device_count())
PY
  nvidia-smi || true

  if [[ "$STAGE" == "stage2" || "$STAGE" == "all" ]]; then
    echo "===== DeepSpeed ZeRO-2 PushT Latent+AdaLN+PredProj Stage 2 sigreg=0.045 ====="
    launch_stage2 "$@"
  fi

  if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
    echo "===== PushT MPC eval: Latent+AdaLN+PredProj sigreg=0.045 ====="
    if [[ ! -d "${STAGE2_DIR}/epoch_8" ]]; then
      echo "Expected stage2 checkpoint ${STAGE2_DIR}/epoch_8 was not found."
      exit 1
    fi
    run_eval "${STAGE2_DIR}/epoch_8" "$EVAL_JSON"
    echo "Evaluation written to $EVAL_JSON"
    compare_results "$EVAL_JSON"
  fi
} 2>&1 | tee "$LOG"
