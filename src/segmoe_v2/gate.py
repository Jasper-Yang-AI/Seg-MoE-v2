from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LesionGate3D(nn.Module):
    def __init__(
        self,
        *,
        num_experts: int,
        extra_channels: int = 0,
        hidden_channels: int = 16,
        coarse_factor: int = 4,
    ) -> None:
        super().__init__()
        in_channels = int(num_experts + extra_channels)
        self.num_experts = int(num_experts)
        self.coarse_factor = int(max(1, coarse_factor))
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(hidden_channels),
            nn.GELU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(hidden_channels),
            nn.GELU(),
        )
        self.head = nn.Conv3d(hidden_channels, self.num_experts, kernel_size=1)

    def forward(
        self,
        expert_probs: torch.Tensor,
        extra: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if expert_probs.ndim != 5:
            raise ValueError(f"Expected expert_probs [B,K,D,H,W], got {tuple(expert_probs.shape)}")
        x = expert_probs if extra is None else torch.cat([expert_probs, extra], dim=1)
        feat = self.encoder(x)
        weight_logits = self.head(feat)
        coarse_size = tuple(max(1, dim // self.coarse_factor) for dim in expert_probs.shape[-3:])
        coarse_logits = F.adaptive_avg_pool3d(weight_logits, coarse_size)
        coarse_weights = torch.softmax(coarse_logits, dim=1)
        weights = F.interpolate(coarse_weights, size=expert_probs.shape[-3:], mode="trilinear", align_corners=False)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        fused = (expert_probs * weights).sum(dim=1, keepdim=True)
        return fused, weights


def gate_regularization(
    weights: torch.Tensor,
    *,
    static_prior: torch.Tensor | None = None,
    static_weight: float = 0.1,
    smooth_weight: float = 0.05,
) -> torch.Tensor:
    loss = weights.new_zeros(())
    if static_prior is not None:
        while static_prior.ndim < weights.ndim:
            static_prior = static_prior.unsqueeze(-1)
        loss = loss + float(static_weight) * F.l1_loss(weights, static_prior.expand_as(weights))
    dz = torch.abs(weights[..., 1:, :, :] - weights[..., :-1, :, :]).mean()
    dy = torch.abs(weights[..., :, 1:, :] - weights[..., :, :-1, :]).mean()
    dx = torch.abs(weights[..., :, :, 1:] - weights[..., :, :, :-1]).mean()
    return loss + float(smooth_weight) * (dz + dy + dx)
