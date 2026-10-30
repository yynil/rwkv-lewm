#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PATH="$ROOT/.venv/bin:$PATH"

SESSION="${SESSION:-rwkv_v11_v12_train_eval}"
INTERVAL="${INTERVAL:-1800}"
MONITOR_LOG="${MONITOR_LOG:-logs/pusht_rwkv_v11_v12_monitor_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p logs

echo "Monitoring $SESSION every ${INTERVAL}s"
echo "Writing snapshots to $MONITOR_LOG"

while true; do
  {
    echo
    echo "===== RWKV v11/v12 probe $(date '+%Y-%m-%d %H:%M:%S %Z') ====="

    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "tmux_session=$SESSION running"
      echo "--- train pane tail ---"
      tmux capture-pane -pt "$SESSION" -S -60 2>/dev/null || true
    else
      echo "tmux_session=$SESSION not_found"
    fi

    echo "--- gpu ---"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits || true

    echo "--- parsed status ---"
    .venv/bin/python - <<'PY'
import json
import re
from collections import defaultdict
from pathlib import Path

root = Path.cwd()
logs = sorted(root.glob("logs/pusht_rwkv_v11_v12_*.log"), key=lambda p: p.stat().st_mtime)
logs = [p for p in logs if "monitor" not in p.name]
if not logs:
    print("run_log=none")
    raise SystemExit

run_log = logs[-1]
text = run_log.read_text(errors="replace")
print(f"run_log={run_log}")
print(f"log_size_mb={run_log.stat().st_size / 1024 / 1024:.2f}")

markers = [
    "rwkv_v11_bounded_kv Stage 1",
    "rwkv_v11_bounded_kv Stage 2",
    "rwkv_v12_adapter Stage 1",
    "rwkv_v12_adapter Stage 2",
    "MPC eval",
    "Modulation impact",
]
seen = [m for m in markers if m in text]
print(f"seen_stages={seen}")

metric_re = re.compile(
    r"epoch=(?P<epoch>\d+) step=(?P<step>\d+) "
    r"loss=(?P<loss>[0-9.eE+-]+) pred=(?P<pred>[0-9.eE+-]+) "
    r"sigreg=(?P<sigreg>[0-9.eE+-]+).*?"
    r"temporal=(?P<temporal>[0-9.eE+-]+).*?"
    r"delta=(?P<delta>[0-9.eE+-]+)"
)
records = []
for line in text.splitlines():
    m = metric_re.search(line)
    if m:
        records.append({
            key: float(value) if key not in {"epoch", "step"} else int(value)
            for key, value in m.groupdict().items()
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
    grouped = defaultdict(list)
    for rec in records:
        grouped[rec["epoch"]].append(rec)
    for epoch in sorted(grouped)[-8:]:
        vals = grouped[epoch]
        tail = vals[-20:]
        print(
            f"epoch_trend epoch={epoch} records={len(vals)} "
            f"first_pred={vals[0]['pred']:.6f} last_pred={vals[-1]['pred']:.6f} "
            f"tail20_pred={sum(v['pred'] for v in tail)/len(tail):.6f} "
            f"tail20_loss={sum(v['loss'] for v in tail)/len(tail):.6f} "
            f"tail20_sigreg={sum(v['sigreg'] for v in tail)/len(tail):.6f}"
        )

checkpoints = [
    "outputs/pusht_rwkv_v11_bounded_kv_stage1_ds/epoch_2",
    "outputs/pusht_rwkv_v11_bounded_kv_stage2_ds/epoch_1",
    "outputs/pusht_rwkv_v11_bounded_kv_stage2_ds/epoch_8",
    "outputs/pusht_rwkv_v12_adapter_stage1_ds/epoch_2",
    "outputs/pusht_rwkv_v12_adapter_stage2_ds/epoch_1",
    "outputs/pusht_rwkv_v12_adapter_stage2_ds/epoch_8",
]
for rel in checkpoints:
    print(f"checkpoint {rel}={'yes' if (root / rel).is_dir() else 'no'}")

for path in sorted(root.glob("outputs/pusht_rwkv_v1[12]_*_stage2_ds/epoch_*/mpc_eval_50_seed42_cem30*.json")):
    try:
        data = json.loads(path.read_text())
        print(
            f"eval {path.relative_to(root)} "
            f"success={float(data['metrics']['success_rate']):.1f} "
            f"history={data.get('world_history_size')}"
        )
    except Exception as exc:
        print(f"eval_bad {path}: {exc}")

for path in sorted(root.glob("outputs/modulation_impact/rwkv_v1[12]_*epoch*.json")):
    print(f"diagnostic {path.relative_to(root)}=yes")
PY
  } 2>&1 | tee -a "$MONITOR_LOG"

  sleep "$INTERVAL"
done
