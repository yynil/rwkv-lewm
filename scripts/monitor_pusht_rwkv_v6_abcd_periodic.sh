#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PATH="$ROOT/.venv/bin:$PATH"

INTERVAL="${INTERVAL:-1800}"
RUN_LOG="${1:-}"
if [[ -z "$RUN_LOG" ]]; then
  RUN_LOG="$(ls -t logs/pusht_rwkv_v6_abcd_*.log 2>/dev/null | head -1 || true)"
fi
if [[ -z "$RUN_LOG" || ! -f "$RUN_LOG" ]]; then
  echo "Cannot find v6 abcd run log. Pass it explicitly as the first argument."
  exit 1
fi

MONITOR_LOG="${MONITOR_LOG:-logs/pusht_rwkv_v6_periodic_monitor_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p logs

echo "Monitoring $RUN_LOG every ${INTERVAL}s"
echo "Writing monitor snapshots to $MONITOR_LOG"

while true; do
  {
    echo
    echo "===== RWKV v6 periodic probe $(date '+%Y-%m-%d %H:%M:%S %Z') ====="

    if tmux has-session -t rwkv_v6_abcd 2>/dev/null; then
      echo "tmux_session=rwkv_v6_abcd running"
      echo "--- tmux pane tail ---"
      tmux capture-pane -pt rwkv_v6_abcd -S -25 2>/dev/null || true
    else
      echo "tmux_session=rwkv_v6_abcd not_found"
    fi

    echo "--- gpu ---"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits || true

    echo "--- parsed training / eval status ---"
    .venv/bin/python - "$RUN_LOG" <<'PY'
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

run_log = Path(sys.argv[1])
root = Path.cwd()
text = run_log.read_text(errors="replace") if run_log.exists() else ""
lines = text.splitlines()

marker_re = re.compile(r"===== .*RWKV v6-([abcd]).*(Stage 1|Stage 2|eval|init)", re.I)
metric_re = re.compile(
    r"epoch=(?P<epoch>\d+) step=(?P<step>\d+) "
    r"loss=(?P<loss>[0-9.eE+-]+) pred=(?P<pred>[0-9.eE+-]+) "
    r"sigreg=(?P<sigreg>[0-9.eE+-]+).*?"
    r"temporal=(?P<temporal>[0-9.eE+-]+).*?"
    r"delta=(?P<delta>[0-9.eE+-]+)"
)

current_variant = None
current_stage = None
records = []
for line in lines:
    marker = marker_re.search(line)
    if marker:
        current_variant = marker.group(1)
        current_stage = marker.group(2).lower().replace(" ", "")
    metric = metric_re.search(line)
    if metric:
        rec = {
            "variant": current_variant or "?",
            "stage": current_stage or "?",
            "epoch": int(metric.group("epoch")),
            "step": int(metric.group("step")),
            "loss": float(metric.group("loss")),
            "pred": float(metric.group("pred")),
            "sigreg": float(metric.group("sigreg")),
            "temporal": float(metric.group("temporal")),
            "delta": float(metric.group("delta")),
        }
        records.append(rec)

if records:
    latest = records[-1]
    print(
        "latest_metric "
        f"v6-{latest['variant']} {latest['stage']} "
        f"epoch={latest['epoch']} step={latest['step']} "
        f"loss={latest['loss']:.8f} pred={latest['pred']:.8f} "
        f"sigreg={latest['sigreg']:.8f} temporal={latest['temporal']:.3e} "
        f"delta={latest['delta']:.6f}"
    )

    grouped = defaultdict(list)
    for rec in records:
        grouped[(rec["variant"], rec["stage"])].append(rec)
    for key in sorted(grouped):
        vals = grouped[key]
        tail = vals[-20:]
        mean_pred = sum(v["pred"] for v in tail) / len(tail)
        mean_loss = sum(v["loss"] for v in tail) / len(tail)
        first = vals[0]
        last = vals[-1]
        print(
            f"trend v6-{key[0]} {key[1]} "
            f"records={len(vals)} first_pred={first['pred']:.6f} "
            f"last_pred={last['pred']:.6f} tail20_pred={mean_pred:.6f} "
            f"tail20_loss={mean_loss:.6f}"
        )
else:
    print("latest_metric none")

print("checkpoints_and_evals")
for variant in "abcd":
    stage1 = root / f"outputs/pusht_rwkv_v6_{variant}_stage1_ds/epoch_2"
    stage2 = root / f"outputs/pusht_rwkv_v6_{variant}_stage2_ds/epoch_8"
    eval_json = stage2 / "mpc_eval_50_seed42_cem30.json"
    parts = [
        f"v6-{variant}",
        f"stage1_epoch2={'yes' if stage1.is_dir() else 'no'}",
        f"stage2_epoch8={'yes' if stage2.is_dir() else 'no'}",
    ]
    if eval_json.exists():
        try:
            data = json.loads(eval_json.read_text())
            sr = float(data["metrics"]["success_rate"])
            parts.append(f"success_rate={sr:.1f}")
        except Exception as exc:
            parts.append(f"eval_parse_error={type(exc).__name__}")
    else:
        parts.append("success_rate=pending")
    print(" ".join(parts))

summary_files = sorted((root / "outputs").glob("pusht_rwkv_v6_abcd_summary_*.jsonl"), reverse=True)
if summary_files:
    print(f"summary_file={summary_files[0]}")
    rows = [line for line in summary_files[0].read_text(errors="replace").splitlines() if line.strip()]
    for row in rows[-8:]:
        print(f"summary {row}")
PY
  } 2>&1 | tee -a "$MONITOR_LOG"

  sleep "$INTERVAL"
done
