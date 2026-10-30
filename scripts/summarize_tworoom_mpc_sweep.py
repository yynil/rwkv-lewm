import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    return parser.parse_args()


def main():
    args = parse_args()
    groups = defaultdict(list)
    rows = []
    for pattern in args.paths:
        for path in sorted(Path().glob(pattern)):
            data = json.loads(path.read_text())
            cem_steps = int(data["cem_steps"])
            seed = int(data["seed"])
            success = float(data["metrics"]["success_rate"])
            time_s = float(data["evaluation_time_seconds"])
            failures = [
                i
                for i, ok in enumerate(data["metrics"]["episode_successes"])
                if not ok
            ]
            row = {
                "path": str(path),
                "cem_steps": cem_steps,
                "seed": seed,
                "success_rate": success,
                "failures": failures,
                "evaluation_time_seconds": time_s,
            }
            rows.append(row)
            groups[cem_steps].append(row)

    for row in sorted(rows, key=lambda r: (r["cem_steps"], r["seed"])):
        print(
            f"cem={row['cem_steps']:2d} seed={row['seed']:2d} "
            f"success={row['success_rate']:5.1f}% "
            f"failures={row['failures']} "
            f"time={row['evaluation_time_seconds']:.1f}s"
        )

    print()
    for cem_steps, values in sorted(groups.items()):
        rates = np.array([v["success_rate"] for v in values], dtype=np.float64)
        times = np.array([v["evaluation_time_seconds"] for v in values], dtype=np.float64)
        print(
            f"cem={cem_steps:2d}: n={len(rates)} "
            f"mean={rates.mean():.2f}% "
            f"std={rates.std(ddof=1) if len(rates) > 1 else 0.0:.2f}% "
            f"min={rates.min():.1f}% max={rates.max():.1f}% "
            f"mean_time={times.mean():.1f}s"
        )


if __name__ == "__main__":
    main()
