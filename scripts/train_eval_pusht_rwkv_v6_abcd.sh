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

VARIANTS=("${@:-a b c d}")
if [[ "$#" -eq 0 ]]; then
  VARIANTS=(a b c d)
fi

if [[ ! -f "$STABLEWM_HOME/pusht_expert_train.h5" ]]; then
  echo "Missing $STABLEWM_HOME/pusht_expert_train.h5"
  exit 1
fi

mkdir -p logs outputs
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG="logs/pusht_rwkv_v6_abcd_${RUN_ID}.log"
SUMMARY="outputs/pusht_rwkv_v6_abcd_summary_${RUN_ID}.jsonl"

launch_stage() {
  local config_name="$1"
  shift
  deepspeed --num_gpus=4 train_deepspeed.py \
    --config-name="$config_name" \
    --deepspeed_config=config/deepspeed/zero2.json \
    "$@"
}

summarize_variant() {
  local variant="$1"
  local result="outputs/pusht_rwkv_v6_${variant}_stage2_ds/epoch_8/mpc_eval_50_seed42_cem30.json"
  .venv/bin/python - "$variant" "$result" "$SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

variant = sys.argv[1]
result_path = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
if not result_path.exists():
    raise SystemExit(f"missing eval result: {result_path}")

data = json.loads(result_path.read_text())
success_rate = float(data["metrics"]["success_rate"])
baseline_v5 = 20.0
reference_lewm = 98.0
if success_rate > baseline_v5:
    verdict = "improved over v5"
elif success_rate == baseline_v5:
    verdict = "matched v5"
else:
    verdict = "worse than v5"
row = {
    "variant": f"v6-{variant}",
    "success_rate": success_rate,
    "baseline_v5_success_rate": baseline_v5,
    "reference_lewm_success_rate": reference_lewm,
    "verdict": verdict,
    "result": str(result_path),
}
summary_path.open("a").write(json.dumps(row, sort_keys=True) + "\n")
print(
    f"CONCLUSION v6-{variant}: PushT success_rate={success_rate:.1f}% "
    f"({verdict}; v5={baseline_v5:.1f}%, LeWM={reference_lewm:.1f}%). "
    f"result={result_path}",
    flush=True,
)
PY
}

echo "Logging to $LOG"
{
  echo "ROOT=$ROOT"
  echo "STABLEWM_HOME=$STABLEWM_HOME"
  echo "CUDA_HOME=$CUDA_HOME"
  echo "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
  echo "RWKV_JIT_ON=$RWKV_JIT_ON"
  echo "SUMMARY=$SUMMARY"
  .venv/bin/python --version
  nvidia-smi || true

  for variant in "${VARIANTS[@]}"; do
    case "$variant" in
      a|b|c|d) ;;
      *) echo "Unknown v6 variant: $variant"; exit 1 ;;
    esac

    echo "===== Check PushT RWKV v6-${variant} init ====="
    .venv/bin/python scripts/check_rwkv_v6_init.py --config-name "pusht_rwkv_v6_${variant}_stage2_ds"

    echo "===== DeepSpeed ZeRO-2 PushT RWKV v6-${variant} Stage 1 ====="
    launch_stage "pusht_rwkv_v6_${variant}_stage1_ds"

    echo "===== DeepSpeed ZeRO-2 PushT RWKV v6-${variant} Stage 2 ====="
    if [[ ! -d "outputs/pusht_rwkv_v6_${variant}_stage1_ds/epoch_2" ]]; then
      echo "Expected stage1 checkpoint outputs/pusht_rwkv_v6_${variant}_stage1_ds/epoch_2 was not found."
      exit 1
    fi
    launch_stage "pusht_rwkv_v6_${variant}_stage2_ds"

    echo "===== PushT MPC eval: RWKV v6-${variant} ====="
    if [[ ! -d "outputs/pusht_rwkv_v6_${variant}_stage2_ds/epoch_8" ]]; then
      echo "Expected stage2 checkpoint outputs/pusht_rwkv_v6_${variant}_stage2_ds/epoch_8 was not found."
      exit 1
    fi
    bash eval_pusht_rwkv_v6.sh "$variant" "outputs/pusht_rwkv_v6_${variant}_stage2_ds/epoch_8"
    summarize_variant "$variant"
  done
} 2>&1 | tee "$LOG"
