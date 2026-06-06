#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/.." && pwd)/src"

mkdir -p outputs/metrics

python scripts/evaluate_mcflow.py \
  --config configs/mcflow_h3.yaml \
  --checkpoint checkpoints/h3/checkpoint_best.pth \
  --split test \
  --samples-per-mode 10 \
  --seed 42 \
  --output outputs/metrics/mcflow_h3.json

python scripts/evaluate_mcflow.py \
  --config configs/mcflow_h5.yaml \
  --checkpoint checkpoints/h5/checkpoint_best.pth \
  --split test \
  --samples-per-mode 10 \
  --seed 42 \
  --output outputs/metrics/mcflow_h5.json

python scripts/evaluate_mcflow.py \
  --config configs/mcflow_h7.yaml \
  --checkpoint checkpoints/h7/checkpoint_best.pth \
  --split test \
  --samples-per-mode 10 \
  --seed 42 \
  --output outputs/metrics/mcflow_h7.json

python - <<'PY'
import csv
import json
from pathlib import Path

paths = [
    Path("outputs/metrics/mcflow_h3.json"),
    Path("outputs/metrics/mcflow_h5.json"),
    Path("outputs/metrics/mcflow_h7.json"),
]
rows = [json.loads(p.read_text()) for p in paths]
keys = [
    "split", "horizon", "past_len", "num_anchors", "samples_per_mode",
    "total_samples", "euler_steps", "rmse_mcflow", "mae_mcflow",
    "crps_mcflow", "rmse_stage1", "mae_stage1", "crps_stage1",
    "checkpoint", "seed",
]
out = Path("outputs/metrics/all_mcflow_results.csv")
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k) for k in keys})
print(f"Wrote {out}")
PY
