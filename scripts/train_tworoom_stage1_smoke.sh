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

python train.py --config-name=tworoom_stage1 \
  trainer.max_epochs=1 \
  trainer.max_steps=3 \
  loader.batch_size=2 \
  data.num_steps=8 \
  loss.sigreg.kwargs.num_proj=64 \
  output_dir=outputs/tworoom_smoke \
  "$@"
