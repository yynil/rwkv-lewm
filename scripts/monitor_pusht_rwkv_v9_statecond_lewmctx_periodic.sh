#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PATH="$ROOT/.venv/bin:$PATH"

SESSION="${SESSION:-rwkv_v9_lewmctx_train}"
INTERVAL="${INTERVAL:-1800}"
MONITOR_LOG="${MONITOR_LOG:-logs/pusht_rwkv_v9_statecond_lewmctx_monitor_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p logs

echo "Monitoring ${SESSION} and latest logs/pusht_rwkv_v9_statecond_lewmctx_*.log every ${INTERVAL}s"
echo "Writing monitor snapshots to $MONITOR_LOG"

while true; do
  {
    echo
    echo "===== RWKV v9 LeWM-context periodic probe $(date '+%Y-%m-%d %H:%M:%S %Z') ====="

    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "tmux_session=$SESSION running"
      echo "--- train pane tail ---"
      tmux capture-pane -pt "$SESSION" -S -35 2>/dev/null || true
    else
      echo "tmux_session=$SESSION not_found"
    fi

    echo "--- gpu ---"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits || true

    echo "--- parsed training / eval status ---"
    .venv/bin/python - <<'PY'
import json
import re
from collections import defaultdict
from pathlib import Path

root = Path.cwd()
logs = sorted((root / "logs").glob("pusht_rwkv_v9_statecond_lewmctx_*.log"), key=lambda p: p.stat().st_mtime)
logs = [p for p in logs if "monitor" not in p.name]
if not logs:
    print("run_log=none")
    raise SystemExit

run_log = logs[-1]
text = run_log.read_text(errors="replace")
print(f"run_log={run_log}")

metric_re = re.compile(
    r"epoch=(?P<epoch>\d+) step=(?P<step>\d+) "
    r"loss=(?P<loss>[0-9.eE+-]+) pred=(?P<pred>[0-9.eE+-]+) "
    r"sigreg=(?P<sigreg>[0-9.eE+-]+).*?"
    r"temporal=(?P<temporal>[0-9.eE+-]+).*?"
    r"delta=(?P<delta>[0-9.eE+-]+)"
)

records = []
for line in text.splitlines():
    metric = metric_re.search(line)
    if metric:
        records.append({
            "epoch": int(metric.group("epoch")),
            "step": int(metric.group("step")),
            "loss": float(metric.group("loss")),
            "pred": float(metric.group("pred")),
            "sigreg": float(metric.group("sigreg")),
            "temporal": float(metric.group("temporal")),
            "delta": float(metric.group("delta")),
        })

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
            f"tail20_loss={sum(v['loss'] for v in tail)/len(tail):.6f}"
        )
else:
    print("latest_metric none")

stage2 = root / "outputs/pusht_rwkv_v9_statecond_lewmctx_stage2_ds/epoch_8"
eval_default = stage2 / "mpc_eval_50_seed42_cem30.json"
eval_h3 = stage2 / "mpc_eval_50_seed42_cem30_history3.json"
print("checkpoints_and_eval")
print(f"stage2_epoch8={'yes' if stage2.is_dir() else 'no'}")
for path in (eval_default, eval_h3):
    if path.exists():
        data = json.loads(path.read_text())
        print(f"{path.name}_success_rate={float(data['metrics']['success_rate']):.1f}")
    else:
        print(f"{path.name}_success_rate=pending")
PY
  } 2>&1 | tee -a "$MONITOR_LOG"

  sleep "$INTERVAL"
done
