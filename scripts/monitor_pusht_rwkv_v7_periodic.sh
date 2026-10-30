#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PATH="$ROOT/.venv/bin:$PATH"

INTERVAL="${INTERVAL:-1800}"
MONITOR_LOG="${MONITOR_LOG:-logs/pusht_rwkv_v7_periodic_monitor_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p logs

echo "Monitoring latest logs/pusht_rwkv_v7_*.log every ${INTERVAL}s"
echo "Writing monitor snapshots to $MONITOR_LOG"

while true; do
  {
    echo
    echo "===== RWKV v7 periodic probe $(date '+%Y-%m-%d %H:%M:%S %Z') ====="

    if tmux has-session -t rwkv_v7_train 2>/dev/null; then
      echo "tmux_session=rwkv_v7_train running"
      echo "--- train pane tail ---"
      tmux capture-pane -pt rwkv_v7_train -S -30 2>/dev/null || true
    else
      echo "tmux_session=rwkv_v7_train not_found"
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
logs = sorted((root / "logs").glob("pusht_rwkv_v7_*.log"), key=lambda p: p.stat().st_mtime)
logs = [p for p in logs if "periodic_monitor" not in p.name]
if not logs:
    print("run_log=none")
    raise SystemExit

run_log = logs[-1]
text = run_log.read_text(errors="replace")
print(f"run_log={run_log}")

marker_re = re.compile(r"===== .*RWKV v7 (Stage 1|Stage 2|eval)", re.I)
metric_re = re.compile(
    r"epoch=(?P<epoch>\d+) step=(?P<step>\d+) "
    r"loss=(?P<loss>[0-9.eE+-]+) pred=(?P<pred>[0-9.eE+-]+) "
    r"sigreg=(?P<sigreg>[0-9.eE+-]+).*?"
    r"temporal=(?P<temporal>[0-9.eE+-]+).*?"
    r"delta=(?P<delta>[0-9.eE+-]+)"
)

current_stage = None
records = []
for line in text.splitlines():
    marker = marker_re.search(line)
    if marker:
        current_stage = marker.group(1).lower().replace(" ", "")
    metric = metric_re.search(line)
    if metric:
        records.append(
            {
                "stage": current_stage or "?",
                "epoch": int(metric.group("epoch")),
                "step": int(metric.group("step")),
                "loss": float(metric.group("loss")),
                "pred": float(metric.group("pred")),
                "sigreg": float(metric.group("sigreg")),
                "temporal": float(metric.group("temporal")),
                "delta": float(metric.group("delta")),
            }
        )

if records:
    latest = records[-1]
    print(
        "latest_metric "
        f"{latest['stage']} epoch={latest['epoch']} step={latest['step']} "
        f"loss={latest['loss']:.8f} pred={latest['pred']:.8f} "
        f"sigreg={latest['sigreg']:.8f} temporal={latest['temporal']:.3e} "
        f"delta={latest['delta']:.6f}"
    )
    grouped = defaultdict(list)
    for rec in records:
        grouped[rec["stage"]].append(rec)
    for stage in sorted(grouped):
        vals = grouped[stage]
        tail = vals[-20:]
        mean_pred = sum(v["pred"] for v in tail) / len(tail)
        mean_loss = sum(v["loss"] for v in tail) / len(tail)
        print(
            f"trend {stage} records={len(vals)} "
            f"first_pred={vals[0]['pred']:.6f} last_pred={vals[-1]['pred']:.6f} "
            f"tail20_pred={mean_pred:.6f} tail20_loss={mean_loss:.6f}"
        )
else:
    print("latest_metric none")

stage1 = root / "outputs/pusht_rwkv_v7_stage1_ds/epoch_2"
stage2 = root / "outputs/pusht_rwkv_v7_stage2_ds/epoch_8"
stage2_cont = root / "outputs/pusht_rwkv_v7_stage2_cont_ds/epoch_8"
eval_json = stage2 / "mpc_eval_50_seed42_cem30.json"
eval_cont_json = stage2_cont / "mpc_eval_50_seed42_cem30.json"
print("checkpoints_and_eval")
print(f"stage1_epoch2={'yes' if stage1.is_dir() else 'no'}")
print(f"stage2_epoch8={'yes' if stage2.is_dir() else 'no'}")
print(f"stage2_cont_epoch8={'yes' if stage2_cont.is_dir() else 'no'}")
if eval_json.exists():
    data = json.loads(eval_json.read_text())
    print(f"success_rate={float(data['metrics']['success_rate']):.1f}")
    print(f"eval_json={eval_json}")
else:
    print("success_rate=pending")
if eval_cont_json.exists():
    data = json.loads(eval_cont_json.read_text())
    print(f"cont_success_rate={float(data['metrics']['success_rate']):.1f}")
    print(f"cont_eval_json={eval_cont_json}")
else:
    print("cont_success_rate=pending")
PY
  } 2>&1 | tee -a "$MONITOR_LOG"

  sleep "$INTERVAL"
done
