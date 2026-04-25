from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .labels import LAYER1_BACKGROUND_WEIGHT, LAYER1_CANDIDATE_POSITIVE_LABEL_VALUES, LAYER1_SOURCE_POSITIVE_WEIGHTS


LAYER1_PROBABILITY_CHANNELS: tuple[str, ...] = ("P_lesion",)


def _squeeze_segmentation_channel(target: torch.Tensor) -> torch.Tensor:
    if target.ndim >= 4 and target.shape[1] == 1:
        return target[:, 0]
    return target


def build_layer1_source_targets_torch(
    raw_target: torch.Tensor,
    *,
    positive_label_values: Sequence[int] = LAYER1_CANDIDATE_POSITIVE_LABEL_VALUES,
    background_weight: float = LAYER1_BACKGROUND_WEIGHT,
    source_positive_weights: dict[int, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = _squeeze_segmentation_channel(raw_target).long()
    positive_values = torch.as_tensor(tuple(int(v) for v in positive_label_values), device=target.device)
    positive = torch.isin(target, positive_values).float()
    weights = torch.full_like(positive, float(background_weight), dtype=torch.float32)
    for label_value, weight in (source_positive_weights or LAYER1_SOURCE_POSITIVE_WEIGHTS).items():
        weights = torch.where(target == int(label_value), torch.as_tensor(float(weight), device=target.device), weights)
    return positive.unsqueeze(1), weights.unsqueeze(1)


def weighted_binary_dice_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    *,
    smooth: float = 1e-5,
) -> torch.Tensor:
    spatial_axes = tuple(range(2, probs.ndim))
    weighted_probs = probs * weights
    weighted_targets = targets * weights
    intersection = (weighted_probs * targets).sum(dim=spatial_axes)
    denominator = weighted_probs.sum(dim=spatial_axes) + weighted_targets.sum(dim=spatial_axes)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return (1.0 - dice).mean()


def layer1_source_aware_loss(
    logits: torch.Tensor,
    raw_target: torch.Tensor,
    *,
    source_positive_weights: dict[int, float] | None = None,
) -> torch.Tensor:
    targets, weights = build_layer1_source_targets_torch(
        raw_target,
        source_positive_weights=source_positive_weights,
    )
    if logits.shape[1] != 1:
        raise ValueError(f"Layer1 source-aware loss expects one sigmoid logit channel, got {tuple(logits.shape)}")
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    bce = (bce * weights).sum() / weights.sum().clamp_min(1.0)
    probs = torch.sigmoid(logits)
    return bce + weighted_binary_dice_loss(probs, targets, weights)


def layer1_tp_fp_fn_tn(
    logits: torch.Tensor,
    raw_target: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    targets, _weights = build_layer1_source_targets_torch(raw_target)
    predicted = torch.sigmoid(logits) > 0.5
    targets_bool = targets > 0.5
    spatial_axes = tuple(range(2, logits.ndim))
    tp = (predicted & targets_bool).sum(dim=spatial_axes)
    fp = (predicted & (~targets_bool)).sum(dim=spatial_axes)
    fn = ((~predicted) & targets_bool).sum(dim=spatial_axes)
    tn = ((~predicted) & (~targets_bool)).sum(dim=spatial_axes)
    return (
        tp.sum(dim=0).detach().cpu().numpy().astype(np.float64),
        fp.sum(dim=0).detach().cpu().numpy().astype(np.float64),
        fn.sum(dim=0).detach().cpu().numpy().astype(np.float64),
        tn.sum(dim=0).detach().cpu().numpy().astype(np.float64),
    )
