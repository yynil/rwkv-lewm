#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SESSION="${SESSION:-rwkv_v13b_train_eval}"
INTERVAL="${INTERVAL:-1800}"
MONITOR_LOG="${MONITOR_LOG:-logs/pusht_rwkv_v13b_monitor_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p logs

while true; do
  {
    echo "===== RWKV v13b probe $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "tmux_session=$SESSION running"
      echo "--- train pane tail ---"
      tmux capture-pane -pt "$SESSION" -S -80 || true
    else
      echo "tmux_session=$SESSION stopped"
    fi
    echo "--- gpu ---"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits || true
    echo "--- parsed status ---"
    .venv/bin/python - <<'PY'
from pathlib import Path
import json
import re

root = Path.cwd()
logs = sorted(root.glob("logs/pusht_rwkv_v13b_*.log"), key=lambda p: p.stat().st_mtime)
logs = [p for p in logs if "monitor" not in p.name]
if not logs:
    print("run_log=none")
    raise SystemExit
log = logs[-1]
text = log.read_text(errors="replace")
print(f"run_log={log}")
print(f"log_size_mb={log.stat().st_size / 1024 / 1024:.2f}")
for marker in [
    "v13b final_transition Stage 1",
    "v13b final_transition Stage 2",
    "MPC eval",
    "Transition impact",
]:
    print(f"seen_{marker.replace(' ', '_')}={marker in text}")

metric_re = re.compile(
    r"epoch=(?P<epoch>\d+) step=(?P<step>\d+)\s+"
    r"loss=(?P<loss>[0-9.eE+-]+) pred=(?P<pred>[0-9.eE+-]+) "
    r"sigreg=(?P<sigreg>[0-9.eE+-]+).*?temporal=(?P<temporal>[0-9.eE+-]+).*?"
    r"delta=(?P<delta>[0-9.eE+-]+)"
)
records = []
for match in metric_re.finditer(text):
    records.append({
        k: int(v) if k in {"epoch", "step"} else float(v)
        for k, v in match.groupdict().items()
    })
print(f"metric_records={len(records)}")
if records:
    latest = records[-1]
    print(
        "latest_metric "
        f"epoch={latest['epoch']} step={latest['step']} "
        f"loss={latest['loss']:.8f} pred={latest['pred']:.8f} "
        f"sigreg={latest['sigreg']:.8f} temporal={latest['temporal']:.3e} "
        f"delta={latest['delta']:.6f}"
    )
    for epoch in sorted({r["epoch"] for r in records})[-10:]:
        epoch_records = [r for r in records if r["epoch"] == epoch]
        tail = epoch_records[-20:]
        print(
            f"epoch_trend epoch={epoch} records={len(epoch_records)} "
            f"first_pred={epoch_records[0]['pred']:.6f} "
            f"last_pred={epoch_records[-1]['pred']:.6f} "
            f"tail20_pred={sum(r['pred'] for r in tail)/len(tail):.6f} "
            f"tail20_loss={sum(r['loss'] for r in tail)/len(tail):.6f} "
            f"tail20_sigreg={sum(r['sigreg'] for r in tail)/len(tail):.6f}"
        )

for rel in [
    "outputs/pusht_rwkv_v13b_final_transition_stage1_ds/epoch_2",
    "outputs/pusht_rwkv_v13b_final_transition_stage2_ds/epoch_1",
    "outputs/pusht_rwkv_v13b_final_transition_stage2_ds/epoch_2",
    "outputs/pusht_rwkv_v13b_final_transition_stage2_ds/epoch_8",
]:
    print(f"checkpoint {rel}={(root / rel).is_dir()}")

for path in sorted(root.glob("outputs/pusht_rwkv_v13b_final_transition_stage2_ds/epoch_*/mpc_eval_50_seed42_cem30*.json")):
    try:
        data = json.loads(path.read_text())
        print(
            f"eval {path.relative_to(root)} "
            f"success={data.get('metrics', {}).get('success_rate')} "
            f"history={data.get('world_history_size')}"
        )
    except Exception as exc:
        print(f"eval_bad {path}: {exc}")

for path in sorted(root.glob("outputs/modulation_impact/rwkv_v13b_final_transition_epoch*.json")):
    print(f"diagnostic {path.relative_to(root)}=yes")
PY
    echo
  } | tee -a "$MONITOR_LOG"

  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    break
  fi
  sleep "$INTERVAL"
done
