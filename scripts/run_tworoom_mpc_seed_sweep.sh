#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PATH="$ROOT/.venv/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
export RWKV_JIT_ON="${RWKV_JIT_ON:-0}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$ROOT/.torch-extensions}"
export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable-wm}"

CHECKPOINT="${1:-outputs/tworoom_stage2_ds/epoch_7}"
OUT_DIR="${2:-${CHECKPOINT%/}/mpc_seed_sweep}"
mkdir -p "$OUT_DIR"

for seed in 0 1 2 3 4; do
  for cem_steps in 10 30; do
    output="$OUT_DIR/mpc_eval_50_seed${seed}_cem${cem_steps}.json"
    if [[ -f "$output" ]]; then
      echo "Skipping existing $output"
      continue
    fi

    echo "===== seed=$seed cem_steps=$cem_steps ====="
    .venv/bin/python scripts/evaluate_tworoom_mpc.py \
      --checkpoint "$CHECKPOINT" \
      --train-config tworoom_stage2_ds \
      --device cuda \
      --dataset-name tworoom \
      --img-size 224 \
      --num-eval 50 \
      --seed "$seed" \
      --goal-offset-steps 25 \
      --eval-budget 50 \
      --horizon 5 \
      --receding-horizon 5 \
      --action-block 5 \
      --cem-samples 300 \
      --cem-steps "$cem_steps" \
      --cem-topk 30 \
      --cem-batch-size 1 \
      --var-scale 1.0 \
      --output "$output"
  done
done
