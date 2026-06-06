"""MC-Flow model architecture and generation helpers."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextEncoder(nn.Module):
    def __init__(self, config: dict[str, Any], meta_counts: dict[str, int]) -> None:
        super().__init__()
        self.id_emb = nn.Embedding(meta_counts["num_buildings"], 16)
        self.unit_emb = nn.Embedding(meta_counts["num_unit_types"], 8)
        self.profile = nn.Sequential(nn.Linear(16 + 8 + 1, 32), nn.SiLU())

        self.temp_conv = nn.Sequential(
            nn.Conv1d(7, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.flat_dim = 128 + 32

    def forward(
        self,
        y_p: torch.Tensor,
        cal: torch.Tensor,
        b_idx: torch.Tensor,
        u_type: torch.Tensor,
        c_year: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hist_input = torch.cat([y_p, cal[:, : y_p.size(1), :]], dim=-1)
        ts_enc = self.temp_conv(hist_input.transpose(1, 2)).squeeze(-1)
        prof = self.profile(torch.cat([self.id_emb(b_idx), self.unit_emb(u_type), c_year], -1))
        return torch.cat([ts_enc, prof], -1), prof


class GatedResBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int, dilation: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, 2 * dim, 3, padding=dilation, dilation=dilation)
        self.cond = nn.Conv1d(cond_dim, 2 * dim, 1)
        self.res = nn.Conv1d(dim, dim, 1)
        self.skip = nn.Conv1d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = self.conv(x) + self.cond(c)
        a, b = gate.chunk(2, 1)
        act = self.dropout(torch.tanh(a) * torch.sigmoid(b))
        return x + self.res(act), self.skip(act)


class AutoregressiveMacroHead(nn.Module):
    def __init__(self, ctx_dim: int, K: int, future_len: int, target_dim: int) -> None:
        super().__init__()
        self.K = K
        self.f = future_len
        self.target_dim = target_dim
        self.init_h = nn.Linear(ctx_dim, 128)
        self.macro_mode_emb = nn.Embedding(K, 16)
        self.decoder_gru = nn.GRU(input_size=22, hidden_size=128, num_layers=1, batch_first=True)
        self.out_proj = nn.Linear(128, target_dim * 2)

    def forward(self, ctx: torch.Tensor, future_cal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = ctx.size(0)
        h_0 = self.init_h(ctx).repeat_interleave(self.K, dim=0).unsqueeze(0)
        mode_idx = torch.arange(self.K, device=ctx.device).repeat(bsz)
        mode_inputs = self.macro_mode_emb(mode_idx).unsqueeze(1).expand(-1, self.f, -1)
        future_cal_exp = future_cal.repeat_interleave(self.K, dim=0)
        gru_inputs = torch.cat([mode_inputs, future_cal_exp], dim=-1)
        gru_out, _ = self.decoder_gru(gru_inputs, h_0)
        out = self.out_proj(gru_out)
        mu_exp = out[..., : self.target_dim]
        sigma_pre = out[..., self.target_dim :]
        mu = mu_exp.view(bsz, self.K, self.f, self.target_dim)
        sigma_sq = F.softplus(sigma_pre.view(bsz, self.K, self.f, self.target_dim)) + 1e-4
        return mu, sigma_sq


class HierarchicalFlowMatchedScenarios(nn.Module):
    def __init__(self, config: dict[str, Any], meta_counts: dict[str, int], target_dim: int = 1) -> None:
        super().__init__()
        self.config = dict(config)
        self.p = int(config["past_len"])
        self.f = int(config["future_len"])
        self.K = int(config["num_anchors"])
        self.target_dim = target_dim

        self.ctx_encoder = ContextEncoder(config, meta_counts)
        self.macro_head = AutoregressiveMacroHead(self.ctx_encoder.flat_dim, self.K, self.f, target_dim)
        self.prob_head = nn.Sequential(
            nn.Linear(self.ctx_encoder.flat_dim, 128),
            nn.SiLU(),
            nn.Linear(128, self.K),
        )

        self.mode_emb = nn.Embedding(self.K, 16)
        self.ctx_proj = nn.Sequential(nn.Linear(self.ctx_encoder.flat_dim + 16, 64), nn.SiLU())
        self.t_mlp = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 32))
        self.cal_proj = nn.Conv1d(6, 32, 3, padding=1)
        self.in_proj = nn.Conv1d(target_dim, 64, 1)
        self.skel_proj = nn.Conv1d(target_dim, 32, 1)
        self.res_blocks = nn.ModuleList(
            [GatedResBlock(64, 160, d, dropout=0.05) for d in [1, 2, 4, 8, 16]]
        )
        self.final = nn.Sequential(nn.SiLU(), nn.Conv1d(64, target_dim, 1))

        half_dim = 16
        emb = math.log(10000.0) / (half_dim - 1)
        self.register_buffer("time_freqs", torch.exp(torch.arange(half_dim) * -emb))

    def forward_stage1(
        self,
        y_p: torch.Tensor,
        cal: torch.Tensor,
        b_idx: torch.Tensor,
        u_type: torch.Tensor,
        c_year: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx, _ = self.ctx_encoder(y_p, cal, b_idx, u_type, c_year)
        future_cal = cal[:, self.p :, :]
        mu, sigma_sq = self.macro_head(ctx, future_cal)
        probs = F.softmax(self.prob_head(ctx) / float(self.config["temp"]), dim=-1)
        return mu, sigma_sq, probs, ctx

    def get_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cal: torch.Tensor,
        ctx: torch.Tensor,
        mode_idx: torch.Tensor,
        full_skel: torch.Tensor,
    ) -> torch.Tensor:
        t_scaled = t.view(-1, 1) * 1000.0
        args = t_scaled * self.time_freqs.view(1, -1)
        t_enc = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        t_ctx = self.t_mlp(t_enc).unsqueeze(-1).expand(-1, -1, self.p + self.f)

        m_emb = self.mode_emb(mode_idx)
        c_proj = self.ctx_proj(torch.cat([ctx, m_emb], dim=-1)).unsqueeze(-1)
        c_proj = c_proj.expand(-1, -1, self.p + self.f)

        x = self.in_proj(x_t.transpose(1, 2))
        s_proj = self.skel_proj(full_skel.transpose(1, 2))
        c = torch.cat([self.cal_proj(cal.transpose(1, 2)), t_ctx, c_proj, s_proj], dim=1)

        skips = []
        for block in self.res_blocks:
            x, skip = block(x, c)
            skips.append(skip)
        return self.final(torch.stack(skips).sum(0)).transpose(1, 2)

    @torch.no_grad()
    def generate_scenarios(
        self,
        y_p: torch.Tensor,
        cal: torch.Tensor,
        b_idx: torch.Tensor,
        u_type: torch.Tensor,
        c_year: torch.Tensor,
        samples_per_mode: int = 1,
        euler_steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        samples, weights, mu, probs = generate_mcflow_samples(
            self,
            {
                "y_p": y_p,
                "cal_full": cal,
                "b_idx": b_idx,
                "unit_type": u_type,
                "constr_year": c_year,
            },
            samples_per_mode=samples_per_mode,
            euler_steps=euler_steps,
        )
        if samples_per_mode == 1:
            return samples.view(y_p.size(0), self.K, self.f, self.target_dim), probs, mu, weights
        return samples, weights, mu, probs


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def generate_stage1(
    model: HierarchicalFlowMatchedScenarios,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    batch = _batch_to_device(batch, device)
    mu, sigma_sq, probs, _ = model.forward_stage1(
        batch["y_p"],
        batch["cal_full"],
        batch["b_idx"],
        batch["unit_type"],
        batch["constr_year"],
    )
    return mu, sigma_sq, probs


@torch.no_grad()
def generate_stage1_ablation(
    model: HierarchicalFlowMatchedScenarios,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    mu, _, probs = generate_stage1(model, batch)
    return mu, probs


@torch.no_grad()
def generate_mcflow_samples(
    model: HierarchicalFlowMatchedScenarios,
    batch: dict[str, torch.Tensor],
    samples_per_mode: int,
    euler_steps: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate S refined samples for each mode with weights pi_k / S."""
    device = next(model.parameters()).device
    batch = _batch_to_device(batch, device)
    y_p = batch["y_p"]
    cal = batch["cal_full"]
    b_idx = batch["b_idx"]
    u_type = batch["unit_type"]
    c_year = batch["constr_year"]
    bsz = y_p.size(0)
    K = model.K
    S = int(samples_per_mode)

    mu, sigma_sq, probs, ctx = model.forward_stage1(y_p, cal, b_idx, u_type, c_year)
    sigma = torch.sqrt(sigma_sq)

    mu_exp = mu.unsqueeze(2).expand(-1, -1, S, -1, -1).reshape(bsz * K * S, model.f, model.target_dim)
    sigma_exp = sigma.unsqueeze(2).expand(-1, -1, S, -1, -1).reshape(
        bsz * K * S, model.f, model.target_dim
    )
    x_0_future = mu_exp + torch.randn_like(mu_exp) * sigma_exp

    y_p_exp = (
        y_p.unsqueeze(1)
        .unsqueeze(2)
        .expand(-1, K, S, -1, -1)
        .reshape(bsz * K * S, model.p, model.target_dim)
    )
    cal_exp = (
        cal.unsqueeze(1)
        .unsqueeze(2)
        .expand(-1, K, S, -1, -1)
        .reshape(bsz * K * S, model.p + model.f, 6)
    )
    ctx_exp = ctx.unsqueeze(1).unsqueeze(2).expand(-1, K, S, -1).reshape(bsz * K * S, -1)
    mode_idx_exp = torch.arange(K, device=device).view(1, K, 1).expand(bsz, K, S).reshape(-1)
    full_skel = torch.cat([y_p_exp, mu_exp], dim=1)
    x_t = torch.cat([y_p_exp, x_0_future], dim=1)

    steps = int(euler_steps if euler_steps is not None else model.config["euler_steps"])
    dt = 1.0 / steps
    for step in range(steps):
        t = torch.full((bsz * K * S,), step * dt, device=device)
        u = model.get_velocity(x_t, t, cal_exp, ctx_exp, mode_idx_exp, full_skel)
        x_t = x_t + u * dt
        x_t[:, : model.p, :] = y_p_exp

    samples = x_t[:, model.p :, :].view(bsz, K * S, model.f, model.target_dim)
    weights = probs.unsqueeze(-1).expand(-1, -1, S).reshape(bsz, K * S) / S
    return samples, weights, mu, probs
