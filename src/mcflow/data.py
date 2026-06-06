"""Data loading and preprocessing for MC-Flow."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset


def load_building_metadata(csv_path: str | Path) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    df_meta = pd.read_csv(csv_path)
    df_meta["construction_year"] = pd.to_numeric(df_meta["construction_year"], errors="coerce")
    df_meta["building_id"] = pd.to_numeric(df_meta["building_id"], errors="coerce")
    df_meta = df_meta.dropna(subset=["building_id"]).copy()
    df_meta["building_id"] = df_meta["building_id"].astype(int)

    valid_years = df_meta[df_meta["construction_year"] > 1800]["construction_year"]
    median_year = valid_years.median()
    df_meta["construction_year"] = df_meta["construction_year"].apply(
        lambda x: x if x > 1800 else median_year
    )
    df_meta["construction_year"] = df_meta["construction_year"].fillna(median_year)
    df_meta["norm_year"] = (df_meta["construction_year"] - 1900) / 100.0
    df_meta["unit_code"], unit_categories = pd.factorize(df_meta["unit_type"])

    unique_bids = df_meta["building_id"].unique()
    bid_to_idx = {raw_bid: mapped_idx for mapped_idx, raw_bid in enumerate(unique_bids)}

    metadata_dict: dict[int, dict[str, Any]] = {}
    for _, row in df_meta.iterrows():
        raw_b_id = int(row["building_id"])
        metadata_dict[raw_b_id] = {
            "file_name": f"{raw_b_id}.csv",
            "unit_type": int(row["unit_code"]),
            "constr_year": float(row["norm_year"]),
            "mapped_idx": bid_to_idx[raw_b_id],
        }

    counts = {"num_buildings": len(unique_bids), "num_unit_types": len(unit_categories)}
    return metadata_dict, counts


class UrbanFlowDataset(Dataset):
    """Windowed per-building daily demand dataset."""

    def __init__(
        self,
        data: pd.DataFrame,
        config: dict[str, Any],
        raw_b_id: int,
        mapped_idx: int,
        unit_type: int,
        constr_year: float,
    ) -> None:
        self.vals = data[
            [
                "norm_demand",
                "dayofweek_sin",
                "dayofweek_cos",
                "dayofyear_sin",
                "dayofyear_cos",
                "norm_temp",
            ]
        ].values.astype(np.float32)
        self.config = config
        self.raw_b_id = raw_b_id
        self.mapped_idx = mapped_idx
        self.unit_type = unit_type
        self.constr_year = constr_year

    def __len__(self) -> int:
        return max(0, len(self.vals) - self.config["past_len"] - self.config["future_len"] + 1)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        p, f = self.config["past_len"], self.config["future_len"]
        y_p = torch.FloatTensor(self.vals[idx : idx + p, 0:1])
        y_f = torch.FloatTensor(self.vals[idx + p : idx + p + f, 0:1])
        cal_features = torch.FloatTensor(self.vals[idx : idx + p + f, 1:6])

        obs_mask = torch.zeros(p + f, 1)
        obs_mask[:p, 0] = 1.0
        cal_full = torch.cat([cal_features, obs_mask], dim=-1)

        return {
            "y_p": y_p,
            "y_f": y_f,
            "cal_full": cal_full,
            "b_idx": torch.tensor(self.mapped_idx, dtype=torch.long),
            "raw_b_id": torch.tensor(self.raw_b_id, dtype=torch.long),
            "unit_type": torch.tensor(self.unit_type, dtype=torch.long),
            "constr_year": torch.tensor([self.constr_year], dtype=torch.float32),
        }


def _empty_concat_guard(datasets: list[Dataset], split: str) -> ConcatDataset:
    if not datasets:
        raise RuntimeError(f"No {split} datasets were created. Check data paths and window lengths.")
    return ConcatDataset(datasets)


def prepare_loaders(
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader, DataLoader, dict[int, dict[str, float]], dict[str, int]]:
    metadata_dict, metadata_counts = load_building_metadata(config["metadata_path"])

    weather_df = pd.read_csv(config["weather_path"])
    weather_df["time"] = pd.to_datetime(weather_df["time"]).dt.tz_localize(None)
    weather_df = weather_df.sort_values("time").set_index("time").resample("1D").mean()

    n_weather_train = int(len(weather_df) * 0.70)
    train_weather = weather_df.iloc[:n_weather_train]["temperature"].values
    weather_std = max(float(train_weather.std()), 1e-8)
    weather_stats = {"mean": float(train_weather.mean()), "std": weather_std}
    weather_df["norm_temp"] = (weather_df["temperature"] - weather_stats["mean"]) / weather_stats["std"]

    train_datasets: list[Dataset] = []
    val_datasets: list[Dataset] = []
    test_datasets: list[Dataset] = []
    all_building_stats: dict[int, dict[str, float]] = {}

    for raw_b_id, meta in metadata_dict.items():
        file_path = os.path.join(config["buildings_dir"], meta["file_name"])
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        df["time"] = pd.to_datetime(df["time_rounded"]).dt.tz_localize(None)
        df = df.sort_values("time").set_index("time")
        df["demand_kwh"] = df["energy_heat_kwh"].diff().clip(lower=0).fillna(0)
        df = (
            df.resample("1D")
            .sum()
            .join(weather_df[["norm_temp"]], how="left")
            .interpolate()
            .bfill()
            .ffill()
        )
        df["dayofweek_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df["dayofweek_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        df["dayofyear_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
        df["dayofyear_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)

        n_train = int(len(df) * 0.7)
        n_val = int(len(df) * 0.1)
        p = config["past_len"]
        train_df = df.iloc[:n_train].copy()
        val_df = df.iloc[n_train - p : n_train + n_val].copy()
        test_df = df.iloc[n_train + n_val - p :].copy()

        log_train = np.log1p(train_df["demand_kwh"].values.astype(np.float64))
        b_stats = {"dem_mean": float(log_train.mean()), "dem_std": max(float(log_train.std()), 1e-8)}
        all_building_stats[raw_b_id] = b_stats

        for split_df in (train_df, val_df, test_df):
            split_df["norm_demand"] = (
                np.log1p(split_df["demand_kwh"]) - b_stats["dem_mean"]
            ) / b_stats["dem_std"]

        ds_args = (raw_b_id, meta["mapped_idx"], meta["unit_type"], meta["constr_year"])
        train_ds = UrbanFlowDataset(train_df, config, *ds_args)
        val_ds = UrbanFlowDataset(val_df, config, *ds_args)
        test_ds = UrbanFlowDataset(test_df, config, *ds_args)
        if len(train_ds) > 0:
            train_datasets.append(train_ds)
        if len(val_ds) > 0:
            val_datasets.append(Subset(val_ds, np.arange(0, len(val_ds), config["val_stride"])))
        if len(test_ds) > 0:
            test_datasets.append(Subset(test_ds, np.arange(0, len(test_ds), config["val_stride"])))

    kwargs = {
        "num_workers": int(config.get("num_workers", 0)),
        "pin_memory": bool(config.get("pin_memory", torch.cuda.is_available())),
    }
    train_loader = DataLoader(
        _empty_concat_guard(train_datasets, "train"),
        batch_size=config["batch_size"],
        shuffle=True,
        **kwargs,
    )
    val_loader = DataLoader(
        _empty_concat_guard(val_datasets, "validation"),
        batch_size=config["eval_batch_size"],
        shuffle=False,
        **kwargs,
    )
    test_loader = DataLoader(
        _empty_concat_guard(test_datasets, "test"),
        batch_size=config["eval_batch_size"],
        shuffle=False,
        **kwargs,
    )
    return train_loader, val_loader, test_loader, all_building_stats, metadata_counts


def to_physical(
    scaled_tensor: torch.Tensor,
    raw_b_ids: torch.Tensor | np.ndarray | list[int],
    stats_dict: dict[int, dict[str, float]],
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Invert normalized log-demand to physical kWh for tensors shaped B..."""
    out_device = device or scaled_tensor.device
    ids = raw_b_ids.detach().cpu().tolist() if torch.is_tensor(raw_b_ids) else list(raw_b_ids)
    b = scaled_tensor.size(0)
    view_shape = (b,) + (1,) * (scaled_tensor.dim() - 1)
    means = torch.tensor([stats_dict[int(i)]["dem_mean"] for i in ids], device=scaled_tensor.device).view(
        view_shape
    )
    stds = torch.tensor([stats_dict[int(i)]["dem_std"] for i in ids], device=scaled_tensor.device).view(
        view_shape
    )
    return torch.clamp(torch.expm1((scaled_tensor * stds) + means), min=0.0).to(out_device)
