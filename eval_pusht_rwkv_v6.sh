#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export PATH="$ROOT/.venv/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
export RWKV_JIT_ON="${RWKV_JIT_ON:-0}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$ROOT/.torch-extensions}"
export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable-wm}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-rwkv}"

VARIANT="${1:?usage: bash eval_pusht_rwkv_v6.sh <a|b|c|d> [checkpoint] [output]}"
CHECKPOINT="${2:-outputs/pusht_rwkv_v6_${VARIANT}_stage2_ds/epoch_8}"
OUTPUT="${3:-${CHECKPOINT%/}/mpc_eval_50_seed42_cem30.json}"

.venv/bin/python scripts/evaluate_pusht_mpc.py \
  --checkpoint "$CHECKPOINT" \
  --train-config "pusht_rwkv_v6_${VARIANT}_stage2_ds" \
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
  --output "$OUTPUT"
