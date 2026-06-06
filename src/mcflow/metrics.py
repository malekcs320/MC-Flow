"""Canonical MC-Flow metric implementation."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from mcflow.data import to_physical
from mcflow.model import generate_mcflow_samples, generate_stage1_ablation


def weighted_crps(forecasts: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted energy score / multivariate CRPS surrogate.

    forecasts: (B, N, F, D), target: (B, F, D), weights: (B, N)
    """
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    diff_gt = torch.abs(forecasts - target.unsqueeze(1)).mean(dim=(2, 3))
    term1 = (weights * diff_gt).sum(dim=1)
    diff_scen = torch.abs(forecasts.unsqueeze(2) - forecasts.unsqueeze(1)).mean(dim=(3, 4))
    pair_weights = weights.unsqueeze(2) * weights.unsqueeze(1)
    term2 = 0.5 * (pair_weights * diff_scen).sum(dim=(1, 2))
    return term1 - term2


def top1_anchor_rmse_mae(
    mu: torch.Tensor, target: torch.Tensor, probs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz = mu.size(0)
    top_idx = probs.argmax(dim=1)
    pred = mu[torch.arange(bsz, device=mu.device), top_idx]
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2))
    mae = F.l1_loss(pred, target, reduction="none").mean(dim=(1, 2))
    return torch.sqrt(mse), mae


def top1_refined_rmse_mae(
    samples: torch.Tensor,
    target: torch.Tensor,
    probs: torch.Tensor,
    samples_per_mode: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, _, future_len, target_dim = samples.shape
    K = probs.size(1)
    sample_view = samples.view(bsz, K, samples_per_mode, future_len, target_dim)
    top_idx = probs.argmax(dim=1)
    pred = sample_view[torch.arange(bsz, device=samples.device), top_idx].mean(dim=1)
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2))
    mae = F.l1_loss(pred, target, reduction="none").mean(dim=(1, 2))
    return torch.sqrt(mse), mae


@torch.no_grad()
def evaluate_mcflow(
    model: torch.nn.Module,
    loader,
    stats_dict: dict[int, dict[str, float]],
    config: dict[str, Any],
    samples_per_mode: int,
    split: str,
) -> dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device
    total = {
        "rmse_mcflow": 0.0,
        "mae_mcflow": 0.0,
        "crps_mcflow": 0.0,
        "rmse_stage1": 0.0,
        "mae_stage1": 0.0,
        "crps_stage1": 0.0,
    }
    n = 0

    for batch in loader:
        y_f_scaled = batch["y_f"].to(device)
        if y_f_scaled.dim() == 2:
            y_f_scaled = y_f_scaled.unsqueeze(-1)
        raw_ids = batch["raw_b_id"]

        samples_scaled, weights, mu_scaled, probs = generate_mcflow_samples(
            model, batch, samples_per_mode=samples_per_mode, euler_steps=config.get("euler_steps")
        )
        mu_ablation_scaled, probs_stage1 = generate_stage1_ablation(model, batch)

        y_f_phys = to_physical(y_f_scaled, raw_ids, stats_dict, device)
        samples_phys = to_physical(samples_scaled, raw_ids, stats_dict, device)
        mu_phys = to_physical(mu_ablation_scaled, raw_ids, stats_dict, device)

        rmse_mc, mae_mc = top1_refined_rmse_mae(samples_phys, y_f_phys, probs, samples_per_mode)
        rmse_s1, mae_s1 = top1_anchor_rmse_mae(mu_phys, y_f_phys, probs_stage1)
        crps_mc = weighted_crps(samples_phys, y_f_phys, weights)
        crps_s1 = weighted_crps(mu_phys, y_f_phys, probs_stage1)

        bsz = y_f_phys.size(0)
        total["rmse_mcflow"] += rmse_mc.sum().item()
        total["mae_mcflow"] += mae_mc.sum().item()
        total["crps_mcflow"] += crps_mc.sum().item()
        total["rmse_stage1"] += rmse_s1.sum().item()
        total["mae_stage1"] += mae_s1.sum().item()
        total["crps_stage1"] += crps_s1.sum().item()
        n += bsz

    if n == 0:
        raise RuntimeError(f"No batches available for {split} evaluation.")

    out = {
        "split": split,
        "horizon": config["future_len"],
        "past_len": config["past_len"],
        "num_anchors": config["num_anchors"],
        "samples_per_mode": samples_per_mode,
        "total_samples": config["num_anchors"] * samples_per_mode,
        "euler_steps": config["euler_steps"],
    }
    out.update({k: v / n for k, v in total.items()})
    return out
