"""Checkpoint loading and evaluation entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mcflow.config import load_config, save_json
from mcflow.data import prepare_loaders
from mcflow.metrics import evaluate_mcflow
from mcflow.model import HierarchicalFlowMatchedScenarios


def _resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_from_checkpoint(
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
) -> tuple[HierarchicalFlowMatchedScenarios, dict[str, Any], dict[int, dict[str, float]], dict[str, Any]]:
    cli_config = load_config(config_path)
    out_device = torch.device(device) if device is not None else _resolve_device(cli_config.get("device"))
    checkpoint = torch.load(checkpoint_path, map_location=out_device)
    ckpt_config = dict(checkpoint.get("config") or {})
    config = {**cli_config, **ckpt_config}

    # Local data paths and runtime device are intentionally allowed to differ from a saved run.
    for key in ["buildings_dir", "metadata_path", "weather_path", "num_workers", "pin_memory"]:
        if key in cli_config:
            config[key] = cli_config[key]
    config["device"] = str(out_device)

    _, _, _, stats_dict, meta_counts = prepare_loaders(config)
    model = HierarchicalFlowMatchedScenarios(config, meta_counts).to(out_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, stats_dict, checkpoint


@torch.no_grad()
def evaluate_checkpoint(
    config_path: str | Path,
    checkpoint_path: str | Path,
    split: str,
    samples_per_mode: int,
    output: str | Path | None = None,
    seed: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    if seed is not None:
        from mcflow.seed import set_seed

        set_seed(seed)
    model, config, stats_dict, checkpoint = load_model_from_checkpoint(config_path, checkpoint_path, device)
    train_loader, val_loader, test_loader, _, _ = prepare_loaders(config)
    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    if split not in loaders:
        raise ValueError(f"Unknown split {split!r}; expected one of {sorted(loaders)}")
    metrics = evaluate_mcflow(model, loaders[split], stats_dict, config, samples_per_mode, split)
    metrics.update(
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_best_crps": checkpoint.get("best_crps"),
            "seed": seed if seed is not None else checkpoint.get("seed"),
            "config": config,
        }
    )
    if output is not None:
        save_json(metrics, output)
    return metrics
