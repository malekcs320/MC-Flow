#!/usr/bin/env python
"""Create rolling qualitative forecast figure for MC-Flow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcflow.plots import make_rolling_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Make rolling forecast figure.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-windows", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--samples-per-mode", type=int, default=50)
    parser.add_argument("--past-context-days", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    make_rolling_from_checkpoint(
        args.config,
        args.checkpoint,
        num_windows=args.num_windows,
        top_k=args.top_k,
        samples_per_mode=args.samples_per_mode,
        past_context_days=args.past_context_days,
        seed=args.seed,
        output=args.output,
        device=args.device,
    )
    print(f"Wrote rolling forecast figure to {args.output}")


if __name__ == "__main__":
    main()
