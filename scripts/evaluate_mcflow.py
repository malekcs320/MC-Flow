#!/usr/bin/env python
"""Evaluate an MC-Flow checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcflow.evaluate import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MC-Flow.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--samples-per-mode", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    metrics = evaluate_checkpoint(
        args.config,
        args.checkpoint,
        split=args.split,
        samples_per_mode=args.samples_per_mode,
        output=args.output,
        seed=args.seed,
        device=args.device,
    )
    print(f"Wrote metrics to {args.output}")
    print(
        f"{args.split} h={metrics['horizon']} "
        f"CRPS={metrics['crps_mcflow']:.4f} RMSE={metrics['rmse_mcflow']:.4f}"
    )


if __name__ == "__main__":
    main()
