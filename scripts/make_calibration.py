#!/usr/bin/env python
"""Create calibration diagnostics for an MC-Flow checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcflow.plots import make_calibration_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Make calibration plot.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples-per-mode", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    result = make_calibration_from_checkpoint(
        args.config,
        args.checkpoint,
        samples_per_mode=args.samples_per_mode,
        seed=args.seed,
        output=args.output,
        split=args.split,
        device=args.device,
    )
    print(f"Wrote calibration plot to {args.output} (MACE={result['mace']:.4f})")


if __name__ == "__main__":
    main()
