"""Training loop for MC-Flow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from mcflow.data import prepare_loaders
from mcflow.metrics import evaluate_mcflow
from mcflow.model import HierarchicalFlowMatchedScenarios
from mcflow.seed import set_seed


def _device_from_config(config: dict[str, Any]) -> torch.device:
    requested = config.get("device")
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_step(
    model: HierarchicalFlowMatchedScenarios,
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
) -> tuple[dict[str, float], torch.Tensor]:
    device = next(model.parameters()).device
    y_p = batch["y_p"].to(device)
    y_f = batch["y_f"].to(device)
    if y_f.dim() == 2:
        y_f = y_f.unsqueeze(-1)
    cal = batch["cal_full"].to(device)
    b_idx = batch["b_idx"].to(device)
    u_type = batch["unit_type"].to(device)
    c_year = batch["constr_year"].to(device)
    bsz, K = y_f.size(0), model.K

    optimizer.zero_grad(set_to_none=True)
    use_cuda = device.type == "cuda"

    with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
        ctx, _ = model.ctx_encoder(y_p, cal, b_idx, u_type, c_year)
        mu, sigma_sq = model.macro_head(ctx, cal[:, model.p :, :])
        logits = model.prob_head(ctx)
        y_f_exp = y_f.unsqueeze(1).expand(-1, K, -1, -1)

        mse_elementwise = (mu - y_f_exp) ** 2
        mse_all = mse_elementwise.mean(dim=(2, 3))
        var_loss_all = F.huber_loss(sigma_sq, mse_elementwise.detach(), reduction="none").mean(
            dim=(2, 3)
        )
        winners = mse_all.argmin(dim=1)
        winner_mask = F.one_hot(winners, num_classes=K).float()
        eps = float(config["rwta_epsilon"])
        rwta_weights = winner_mask * (1 - eps) + (1 - winner_mask) * (eps / (K - 1))
        loss_macro = (rwta_weights * mse_all).sum(dim=1).mean()
        loss_macro = loss_macro + (rwta_weights * var_loss_all).sum(dim=1).mean()

        target_temp = float(config["target_temp"])
        soft_targets = F.softmax(-mse_all / target_temp, dim=-1).detach()
        loss_prob = F.cross_entropy(logits / float(config["temp"]), soft_targets)

        mu_winner = mu[torch.arange(bsz, device=device), winners].detach()
        sigma_winner = torch.sqrt(sigma_sq[torch.arange(bsz, device=device), winners].detach())
        x_0_future = mu_winner + sigma_winner * torch.randn_like(mu_winner)
        x_1_future = y_f
        x_0 = torch.cat([y_p, x_0_future], dim=1)
        x_1 = torch.cat([y_p, x_1_future], dim=1)
        t = torch.rand(bsz, device=device).view(-1, 1, 1) * 0.98 + 0.01
        x_t = t * x_1 + (1 - t) * x_0
        full_skel = torch.cat([y_p, mu_winner], dim=1)
        u_pred = model.get_velocity(x_t, t.view(-1), cal, ctx.detach(), winners, full_skel)
        target_velocity = x_1 - x_0
        loss_cfm = F.mse_loss(u_pred[:, model.p :, :], target_velocity[:, model.p :, :])

        start = int(config.get("prob_warmup_start", 5))
        warmup = max(1, int(config.get("prob_warmup_epochs", 5)))
        prob_weight = min(1.0, max(0.0, (epoch - start) / warmup)) * float(config["lambda_prob"])
        loss = (
            float(config["lambda_macro"]) * loss_macro
            + prob_weight * loss_prob
            + float(config["lambda_cfm"]) * loss_cfm
        )

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
    scaler.step(optimizer)
    scaler.update()

    metrics = {
        "loss": float(loss.detach().cpu()),
        "macro_rwta": float(loss_macro.detach().cpu()),
        "prob_ce": float(loss_prob.detach().cpu()),
        "prob_weight": float(prob_weight),
        "cfm_mse": float(loss_cfm.detach().cpu()),
        "t1_acc": float((F.softmax(logits, dim=-1).argmax(dim=-1) == winners).float().mean().detach().cpu()),
    }
    return metrics, winners.detach().cpu()


def _save_checkpoint(
    model: HierarchicalFlowMatchedScenarios,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_crps: float,
    config: dict[str, Any],
    seed: int,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_crps": best_crps,
            "config": dict(config),
            "seed": seed,
        },
        path,
    )


def train(config: dict[str, Any], seed: int, output_dir: str | Path, use_wandb: bool = False) -> None:
    set_seed(seed)
    device = _device_from_config(config)
    config = dict(config)
    config["device"] = str(device)

    train_loader, val_loader, _, stats_dict, meta_counts = prepare_loaders(config)
    model = HierarchicalFlowMatchedScenarios(config, meta_counts).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["epochs"]), eta_min=float(config.get("eta_min", 1e-5))
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    wandb_run = None
    if use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=config.get("wandb_project", "mcflow"),
            name=Path(output_dir).name,
            config=config,
        )

    out_dir = Path(output_dir)
    latest_path = out_dir / "checkpoint_latest.pth"
    best_path = out_dir / "checkpoint_best.pth"
    best_crps = float("inf")

    for epoch in range(int(config["epochs"])):
        model.train()
        epoch_metrics = {"loss": 0.0, "macro_rwta": 0.0, "prob_ce": 0.0, "prob_weight": 0.0, "cfm_mse": 0.0, "t1_acc": 0.0}
        winners_all = []
        steps = 0
        for i, batch in enumerate(train_loader):
            if i >= int(config["num_batches_per_epoch"]):
                break
            batch_metrics, winners = train_step(model, batch, config, optimizer, scaler, epoch)
            for key, value in batch_metrics.items():
                epoch_metrics[key] += value
            winners_all.append(winners)
            steps += 1

        scheduler.step()
        for key in epoch_metrics:
            epoch_metrics[key] /= max(1, steps)
        winner_tensor = torch.cat(winners_all) if winners_all else torch.empty(0, dtype=torch.long)
        dead_anchors = int((torch.bincount(winner_tensor, minlength=int(config["num_anchors"])) == 0).sum())

        _save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_crps, config, seed, latest_path)
        if epoch % 5 == 0 or epoch == int(config["epochs"]) - 1:
            val_metrics = evaluate_mcflow(
                model, val_loader, stats_dict, config, samples_per_mode=1, split="val"
            )
            val_crps = float(val_metrics["crps_mcflow"])
            if val_crps < best_crps:
                best_crps = val_crps
                _save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_crps, config, seed, best_path)
            if wandb_run is not None:
                wandb_run.log({"epoch": epoch, "train/dead_anchors": dead_anchors, **{f"train/{k}": v for k, v in epoch_metrics.items()}, **{f"val/{k}": v for k, v in val_metrics.items()}})

        print(
            f"Epoch {epoch:03d} | Macro: {epoch_metrics['macro_rwta']:.4f} | "
            f"CFM: {epoch_metrics['cfm_mse']:.4f} | Dead: {dead_anchors}/{config['num_anchors']}"
        )

    if wandb_run is not None:
        wandb_run.finish()
