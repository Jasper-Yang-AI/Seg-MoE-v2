from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


LAYER2_PROBABILITY_CHANNELS: tuple[str, ...] = ("P_lesion",)
LAYER2_SOURCE_LABEL_VALUES = {"background": 0, "pca_lesion": 1, "hard_negative": 2}
LAYER2_BACKGROUND_WEIGHT = 1.0
LAYER2_SOURCE_WEIGHTS = {1: 1.25, 2: 2.5}


def build_layer2_source_labels(
    label: np.ndarray,
    cohort_type: str,
    fp_risk_map: np.ndarray | None = None,
) -> np.ndarray:
    label = np.asarray(label)
    source = np.zeros(label.shape, dtype=np.uint8)
    if fp_risk_map is not None:
        fp_risk_map = np.asarray(fp_risk_map)
        if fp_risk_map.shape != label.shape:
            raise ValueError(f"fp_risk_map shape {fp_risk_map.shape} does not match label shape {label.shape}")
        source[fp_risk_map > 0] = LAYER2_SOURCE_LABEL_VALUES["hard_negative"]

    lesion_mask = label == 3
    if str(cohort_type).lower() == "nca":
        source[lesion_mask] = LAYER2_SOURCE_LABEL_VALUES["hard_negative"]
    else:
        source[lesion_mask] = LAYER2_SOURCE_LABEL_VALUES["pca_lesion"]
    return source


def build_layer2_source_weight_map(
    source_labels: np.ndarray,
    *,
    background_weight: float = LAYER2_BACKGROUND_WEIGHT,
    source_weights: dict[int, float] | None = None,
) -> np.ndarray:
    source_labels = np.asarray(source_labels)
    weights = np.full(source_labels.shape, float(background_weight), dtype=np.float32)
    for label_value, weight in (source_weights or LAYER2_SOURCE_WEIGHTS).items():
        weights[source_labels == int(label_value)] = float(weight)
    return weights


def _squeeze_segmentation_channel(target: torch.Tensor) -> torch.Tensor:
    if target.ndim >= 4 and target.shape[1] == 1:
        return target[:, 0]
    return target


def _require_torch() -> None:
    if torch is None or F is None:
        raise ModuleNotFoundError("torch is required for Layer2 torch targets, loss, and metrics.")


def _weighted_binary_dice_loss(
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


def build_layer2_targets_torch(
    raw_target: torch.Tensor,
    *,
    background_weight: float = LAYER2_BACKGROUND_WEIGHT,
    source_weights: dict[int, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_torch()
    target = _squeeze_segmentation_channel(raw_target).long()
    positive = (target == LAYER2_SOURCE_LABEL_VALUES["pca_lesion"]).float()
    weights = torch.full_like(positive, float(background_weight), dtype=torch.float32)
    for label_value, weight in (source_weights or LAYER2_SOURCE_WEIGHTS).items():
        weights = torch.where(target == int(label_value), torch.as_tensor(float(weight), device=target.device), weights)
    return positive.unsqueeze(1), weights.unsqueeze(1)


def layer2_hard_negative_loss(
    logits: torch.Tensor,
    raw_target: torch.Tensor,
    *,
    source_weights: dict[int, float] | None = None,
) -> torch.Tensor:
    _require_torch()
    targets, weights = build_layer2_targets_torch(raw_target, source_weights=source_weights)
    if logits.shape[1] != 1:
        raise ValueError(f"Layer2 hard-negative loss expects one sigmoid logit channel, got {tuple(logits.shape)}")
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    bce = (bce * weights).sum() / weights.sum().clamp_min(1.0)
    probs = torch.sigmoid(logits)
    return bce + _weighted_binary_dice_loss(probs, targets, weights)


def layer2_tp_fp_fn_tn(
    logits: torch.Tensor,
    raw_target: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _require_torch()
    targets, _weights = build_layer2_targets_torch(raw_target)
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
