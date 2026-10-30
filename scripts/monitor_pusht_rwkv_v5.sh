#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TRAIN_SESSION="${1:?usage: monitor_pusht_rwkv_v5.sh TRAIN_SESSION [TRAIN_LOG] [MONITOR_LOG]}"
TRAIN_LOG="${2:-logs/pusht_rwkv_v5_20260503_144040.log}"
MONITOR_LOG="${3:-logs/pusht_rwkv_v5_monitor.log}"

mkdir -p "$(dirname "$MONITOR_LOG")"

while tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; do
  {
    echo "===== monitor $(date '+%F %T') ====="
    for d in outputs/pusht_rwkv_v5_stage1_ds/epoch_* outputs/pusht_rwkv_v5_stage2_ds/epoch_*; do
      [[ -e "$d" ]] && echo "checkpoint: $d"
    done
    tail -n 120 "$TRAIN_LOG" 2>/dev/null \
      | grep -E "epoch=|success|score|=====|mpc|Expected|Traceback|Error" \
      | tail -n 30 || true
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits || true
  } >> "$MONITOR_LOG" 2>&1
  sleep 600
done

{
  echo "===== monitor final $(date '+%F %T') ====="
  tail -n 160 "$TRAIN_LOG" 2>/dev/null \
    | grep -E "epoch=|success|score|=====|mpc|Traceback|Error" \
    | tail -n 80 || true
} >> "$MONITOR_LOG" 2>&1
