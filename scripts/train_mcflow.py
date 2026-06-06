#!/usr/bin/env python
"""Train MC-Flow from a YAML config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcflow.config import load_config
from mcflow.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MC-Flow.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    args = parser.parse_args()
    train(load_config(args.config), seed=args.seed, output_dir=args.output_dir, use_wandb=args.wandb)


if __name__ == "__main__":
    main()
