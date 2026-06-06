"""Paper figure generation for MC-Flow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from mcflow.data import prepare_loaders, to_physical
from mcflow.evaluate import load_model_from_checkpoint
from mcflow.model import generate_mcflow_samples
from mcflow.seed import set_seed


def _save_pdf_png(fig: plt.Figure, output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    png_path = output_path.with_suffix(".png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> np.ndarray:
    order = np.argsort(values, axis=0)
    sorted_values = np.take_along_axis(values, order, axis=0)
    tiled_weights = np.repeat(weights[:, None], values.shape[1], axis=1)
    sorted_weights = np.take_along_axis(tiled_weights, order, axis=0)
    cdf = np.cumsum(sorted_weights, axis=0)
    cdf = cdf / np.maximum(cdf[-1:, :], 1e-12)
    idx = (cdf >= quantile).argmax(axis=0)
    return sorted_values[idx, np.arange(values.shape[1])]


@torch.no_grad()
def plot_calibration(
    model: torch.nn.Module,
    loader,
    stats_dict: dict[int, dict[str, float]],
    config: dict[str, Any],
    samples_per_mode: int,
    output: str | Path,
    max_batches: int | None = None,
) -> dict[str, float]:
    device = next(model.parameters()).device
    model.eval()
    nominal_levels = np.arange(0.1, 1.0, 0.1)
    hits = np.zeros(len(nominal_levels), dtype=np.float64)
    total_observations = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        y_f = batch["y_f"].to(device)
        if y_f.dim() == 2:
            y_f = y_f.unsqueeze(-1)
        samples, weights, _, _ = generate_mcflow_samples(model, batch, samples_per_mode=samples_per_mode)
        y_phys = to_physical(y_f, batch["raw_b_id"], stats_dict, device)
        samples_phys = to_physical(samples, batch["raw_b_id"], stats_dict, device)

        for i in range(y_phys.size(0)):
            ensemble = samples_phys[i, :, :, 0].detach().cpu().numpy()
            w = weights[i].detach().cpu().numpy()
            gt = y_phys[i, :, 0].detach().cpu().numpy()
            for level_idx, nominal in enumerate(nominal_levels):
                low_q = (1.0 - nominal) / 2.0
                high_q = (1.0 + nominal) / 2.0
                lower = _weighted_quantile(ensemble, w, low_q)
                upper = _weighted_quantile(ensemble, w, high_q)
                hits[level_idx] += ((gt >= lower) & (gt <= upper)).sum()
            total_observations += gt.size

    empirical = hits / max(1, total_observations)
    mace = float(np.mean(np.abs(empirical - nominal_levels)))

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1.2, label="Ideal")
    ax.plot(nominal_levels, empirical, marker="o", color="#1A7F74", linewidth=2.0, label="MC-Flow")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Nominal interval coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(f"Calibration diagnostics (MACE={mace:.3f})")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    _save_pdf_png(fig, output)
    plt.close(fig)
    return {"mace": mace}


def _iter_base_datasets(concat_dataset):
    for ds in getattr(concat_dataset, "datasets", [concat_dataset]):
        if hasattr(ds, "dataset"):
            yield ds
        else:
            yield ds


@torch.no_grad()
def plot_rolling_forecast(
    model: torch.nn.Module,
    test_loader,
    stats_dict: dict[int, dict[str, float]],
    config: dict[str, Any],
    num_windows: int,
    top_k: int,
    samples_per_mode: int,
    past_context_days: int,
    output: str | Path,
    building_indices: list[int] | None = None,
) -> None:
    device = next(model.parameters()).device
    model.eval()
    datasets = list(_iter_base_datasets(test_loader.dataset))
    building_indices = building_indices or list(range(min(5, len(datasets))))
    nrows = len(building_indices)
    fig, axes = plt.subplots(nrows, 1, figsize=(14, max(3.2, 2.8 * nrows)), sharex=True)
    axes = np.atleast_1d(axes)
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, top_k)))

    for ax_idx, ds_idx in enumerate(building_indices):
        if ds_idx >= len(datasets):
            continue
        ax = axes[ax_idx]
        ds = datasets[ds_idx]
        actual_windows = min(num_windows, len(ds))
        all_true_t: list[int] = []
        all_true_y: list[float] = []

        for w in range(actual_windows):
            item = ds[w]
            batch = {
                key: value.unsqueeze(0).to(device) if torch.is_tensor(value) else value
                for key, value in item.items()
            }
            samples, _, _, probs = generate_mcflow_samples(
                model, batch, samples_per_mode=samples_per_mode, euler_steps=config.get("euler_steps")
            )
            raw_id = torch.tensor([int(item["raw_b_id"].item())])
            y_p_phys = to_physical(batch["y_p"], raw_id, stats_dict, device)[0, :, 0].cpu().numpy()
            y_f_phys = to_physical(batch["y_f"], raw_id, stats_dict, device)[0, :, 0].cpu().numpy()
            samples_phys = to_physical(samples, raw_id, stats_dict, device)[0, :, :, 0].cpu().numpy()

            start_day = w * int(config["future_len"])
            t_future = np.arange(start_day, start_day + int(config["future_len"]))
            all_true_t.extend(t_future.tolist())
            all_true_y.extend(y_f_phys.tolist())
            if w == 0:
                p_show = min(past_context_days, len(y_p_phys))
                ax.plot(
                    np.arange(-p_show, 0),
                    y_p_phys[-p_show:],
                    color="#222222",
                    linewidth=1.8,
                    label="Observed history",
                )

            K = int(config["num_anchors"])
            S = samples_per_mode
            sample_view = samples_phys.reshape(K, S, int(config["future_len"]))
            top_indices = torch.argsort(probs[0].detach().cpu(), descending=True)[:top_k].tolist()
            for rank, mode_idx in enumerate(top_indices):
                mode_samples = sample_view[mode_idx]
                mean_line = mode_samples.mean(axis=0)
                lower = np.percentile(mode_samples, 10, axis=0)
                upper = np.percentile(mode_samples, 90, axis=0)
                color = colors[rank]
                ax.plot(t_future, mean_line, color=color, linewidth=1.6, alpha=0.95)
                ax.fill_between(t_future, lower, upper, color=color, alpha=0.12, linewidth=0)
            ax.axvline(start_day, color="#BBBBBB", linestyle=":", linewidth=0.8)

        ax.plot(all_true_t, all_true_y, color="#111111", linewidth=1.8, label="Ground truth")
        ax.set_ylabel("kWh")
        ax.grid(True, axis="y", alpha=0.22)
        ax.set_title(f"Building {int(ds[0]['raw_b_id'].item())}", loc="left", fontsize=10)

    axes[-1].set_xlabel("Forecast day")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save_pdf_png(fig, output)
    plt.close(fig)


def make_calibration_from_checkpoint(
    config_path: str | Path,
    checkpoint_path: str | Path,
    samples_per_mode: int,
    seed: int,
    output: str | Path,
    split: str = "test",
    device: str | None = None,
) -> dict[str, float]:
    set_seed(seed)
    model, config, stats_dict, _ = load_model_from_checkpoint(config_path, checkpoint_path, device)
    train_loader, val_loader, test_loader, _, _ = prepare_loaders(config)
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[split]
    return plot_calibration(model, loader, stats_dict, config, samples_per_mode, output)


def make_rolling_from_checkpoint(
    config_path: str | Path,
    checkpoint_path: str | Path,
    num_windows: int,
    top_k: int,
    samples_per_mode: int,
    past_context_days: int,
    seed: int,
    output: str | Path,
    device: str | None = None,
) -> None:
    set_seed(seed)
    model, config, stats_dict, _ = load_model_from_checkpoint(config_path, checkpoint_path, device)
    _, _, test_loader, _, _ = prepare_loaders(config)
    plot_rolling_forecast(
        model,
        test_loader,
        stats_dict,
        config,
        num_windows=num_windows,
        top_k=top_k,
        samples_per_mode=samples_per_mode,
        past_context_days=past_context_days,
        output=output,
    )
