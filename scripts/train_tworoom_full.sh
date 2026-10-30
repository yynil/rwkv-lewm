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

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "Missing venv: $ROOT/.venv"
  echo "Create it with:"
  echo "  uv venv --python=3.12"
  echo "  uv pip install -e '.[swm]'"
  exit 1
fi

if [[ ! -f "$STABLEWM_HOME/tworoom.h5" ]]; then
  echo "Missing $STABLEWM_HOME/tworoom.h5"
  echo "Download first:"
  echo "  python scripts/download_lewm_data.py tworoom --output $STABLEWM_HOME"
  exit 1
fi

mkdir -p logs outputs
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG="logs/tworoom_${RUN_ID}.log"

echo "Logging to $LOG"
{
  echo "ROOT=$ROOT"
  echo "STABLEWM_HOME=$STABLEWM_HOME"
  echo "CUDA_HOME=$CUDA_HOME"
  echo "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
  echo "RWKV_JIT_ON=$RWKV_JIT_ON"
  "$ROOT/.venv/bin/python" --version
  "$ROOT/.venv/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
  nvidia-smi || true
  "$ROOT/.venv/bin/python" scripts/inspect_tworoom.py --data-root "$STABLEWM_HOME"

  if [[ "$STAGE" == "stage1" || "$STAGE" == "all" ]]; then
    echo
    echo "===== Stage 1: short sequence warmup ====="
    "$ROOT/.venv/bin/python" train.py --config-name=tworoom_stage1 "$@"
  fi

  if [[ "$STAGE" == "stage2" || "$STAGE" == "all" ]]; then
    echo
    echo "===== Stage 2: long sequence continuation ====="
    if [[ "$STAGE" == "all" && ! -f outputs/tworoom_stage1/rwkv_lewm_epoch_2.pt ]]; then
      echo "Expected stage1 checkpoint outputs/tworoom_stage1/rwkv_lewm_epoch_2.pt was not found."
      exit 1
    fi
    "$ROOT/.venv/bin/python" train.py --config-name=tworoom_stage2 "$@"
  fi
} 2>&1 | tee "$LOG"
