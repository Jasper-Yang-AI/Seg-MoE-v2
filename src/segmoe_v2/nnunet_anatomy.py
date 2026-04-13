from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


ANATOMY_HEADS: tuple[str, str, str] = ("WG", "PZ", "TZ")
ANATOMY_PROBABILITY_CHANNELS: tuple[str, str, str] = ("P_WG", "P_PZ", "P_TZ")


def _insert_crop_into_image(
    image: np.ndarray,
    crop: np.ndarray,
    bbox: Sequence[Sequence[int]],
) -> np.ndarray:
    result = np.asarray(image).copy()
    slices = tuple(slice(int(bounds[0]), int(bounds[1])) for bounds in bbox)
    result[slices] = crop
    return result


def _squeeze_segmentation_channel(target: torch.Tensor) -> torch.Tensor:
    if target.ndim >= 4 and target.shape[1] == 1:
        return target[:, 0]
    return target


def build_anatomy_head_targets_torch(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target = _squeeze_segmentation_channel(target).long()
    wg_target = torch.isin(target, torch.tensor((1, 2, 3), device=target.device)).float()
    pz_target = (target == 1).float()
    tz_target = (target == 2).float()

    wg_valid = torch.ones_like(wg_target, dtype=torch.bool)
    pz_valid = target != 3
    tz_valid = target != 3

    stacked_targets = torch.stack((wg_target, pz_target, tz_target), dim=1)
    stacked_valid = torch.stack((wg_valid, pz_valid, tz_valid), dim=1)
    return stacked_targets, stacked_valid


def masked_binary_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    smooth: float = 1e-5,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    valid = valid_mask.float()
    probs = probs * valid
    targets = targets * valid

    spatial_axes = tuple(range(2, logits.ndim))
    intersection = (probs * targets).sum(dim=spatial_axes)
    probs_sum = probs.sum(dim=spatial_axes)
    targets_sum = targets.sum(dim=spatial_axes)
    valid_any = (valid.sum(dim=spatial_axes) > 0).float()

    dice = (2.0 * intersection + smooth) / (probs_sum + targets_sum + smooth)
    per_head_loss = 1.0 - dice
    weighted = per_head_loss * valid_any
    denominator = valid_any.sum().clamp_min(1.0)
    return weighted.sum() / denominator


def masked_binary_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    valid = valid_mask.float()
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * valid
    denominator = valid.sum().clamp_min(1.0)
    return losses.sum() / denominator


class MaskedAnatomySegLoss(torch.nn.Module):
    def __init__(
        self,
        *,
        head_weights: Sequence[float] = (1.0, 1.0, 1.0),
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        weights = torch.tensor(tuple(float(v) for v in head_weights), dtype=torch.float32)
        self.register_buffer("head_weights", weights)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, raw_target: torch.Tensor) -> torch.Tensor:
        targets, valid_mask = build_anatomy_head_targets_torch(raw_target)
        if logits.shape[1] != len(ANATOMY_HEADS):
            raise ValueError(f"Expected {len(ANATOMY_HEADS)} anatomy heads, got shape {tuple(logits.shape)}")

        total = logits.new_tensor(0.0)
        for head_index in range(len(ANATOMY_HEADS)):
            head_logits = logits[:, head_index : head_index + 1]
            head_targets = targets[:, head_index : head_index + 1]
            head_valid = valid_mask[:, head_index : head_index + 1]
            head_loss = masked_binary_bce_loss(head_logits, head_targets, head_valid)
            head_loss = head_loss + masked_binary_dice_loss(
                head_logits,
                head_targets,
                head_valid,
                smooth=self.smooth,
            )
            total = total + self.head_weights[head_index] * head_loss
        return total


def anatomy_tp_fp_fn(
    logits: torch.Tensor,
    raw_target: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets, valid_mask = build_anatomy_head_targets_torch(raw_target)
    probs = torch.sigmoid(logits)
    predicted = probs > 0.5
    valid = valid_mask.bool()
    targets_bool = targets > 0.5

    spatial_axes = tuple(range(2, logits.ndim))
    tp = ((predicted & targets_bool) & valid).sum(dim=spatial_axes)
    fp = ((predicted & (~targets_bool)) & valid).sum(dim=spatial_axes)
    fn = (((~predicted) & targets_bool) & valid).sum(dim=spatial_axes)

    tp_sum = tp.sum(dim=0).detach().cpu().numpy().astype(np.float64)
    fp_sum = fp.sum(dim=0).detach().cpu().numpy().astype(np.float64)
    fn_sum = fn.sum(dim=0).detach().cpu().numpy().astype(np.float64)
    return tp_sum, fp_sum, fn_sum


def deep_supervision_weights(
    *,
    num_outputs: int,
    is_ddp: bool,
    do_compile: bool,
) -> np.ndarray:
    if num_outputs <= 1:
        return np.asarray([1.0], dtype=np.float32)
    weights = np.asarray([1 / (2**i) for i in range(num_outputs)], dtype=np.float32)
    if is_ddp and not do_compile:
        weights[-1] = 1e-6
    else:
        weights[-1] = 0.0
    weights = weights / weights.sum()
    return weights


def _resample_logits_to_cropped_shape(
    predicted_logits: torch.Tensor | np.ndarray,
    *,
    plans_manager: Any,
    configuration_manager: Any,
    properties_dict: dict[str, Any],
) -> torch.Tensor:
    spacing_transposed = [properties_dict["spacing"][i] for i in plans_manager.transpose_forward]
    current_spacing = (
        configuration_manager.spacing
        if len(configuration_manager.spacing) == len(properties_dict["shape_after_cropping_and_before_resampling"])
        else [spacing_transposed[0], *configuration_manager.spacing]
    )
    resampled = configuration_manager.resampling_fn_probabilities(
        predicted_logits,
        properties_dict["shape_after_cropping_and_before_resampling"],
        current_spacing,
        [properties_dict["spacing"][i] for i in plans_manager.transpose_forward],
    )
    if isinstance(resampled, torch.Tensor):
        return resampled
    return torch.as_tensor(resampled)


def convert_anatomy_logits_to_probabilities_with_correct_shape(
    predicted_logits: torch.Tensor | np.ndarray,
    *,
    plans_manager: Any,
    configuration_manager: Any,
    properties_dict: dict[str, Any],
) -> np.ndarray:
    resampled_logits = _resample_logits_to_cropped_shape(
        predicted_logits,
        plans_manager=plans_manager,
        configuration_manager=configuration_manager,
        properties_dict=properties_dict,
    )
    probabilities = torch.sigmoid(resampled_logits).cpu().numpy().astype(np.float32)

    reverted = np.zeros((probabilities.shape[0], *properties_dict["shape_before_cropping"]), dtype=np.float32)
    for channel_idx in range(probabilities.shape[0]):
        reverted[channel_idx] = _insert_crop_into_image(
            np.zeros(properties_dict["shape_before_cropping"], dtype=np.float32),
            probabilities[channel_idx],
            properties_dict["bbox_used_for_cropping"],
        )
    reverted = reverted.transpose([0] + [axis + 1 for axis in plans_manager.transpose_backward])
    return reverted


def anatomy_hard_masks_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return (np.asarray(probabilities, dtype=np.float32) > 0.5).astype(np.uint8)


def write_anatomy_probability_bundle(
    *,
    probabilities: np.ndarray,
    output_file_truncated: str | Path,
    channel_names: Sequence[str] = ANATOMY_PROBABILITY_CHANNELS,
    properties_dict: dict[str, Any] | None = None,
    save_quality_masks: bool = False,
) -> Path:
    output_file_truncated = Path(output_file_truncated)
    output_file_truncated.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "probabilities": np.asarray(probabilities, dtype=np.float32),
        "channel_names": np.asarray(tuple(str(name) for name in channel_names)),
    }
    np.savez_compressed(output_file_truncated.with_suffix(".npz"), **payload)
    if properties_dict is not None:
        output_file_truncated.with_suffix(".pkl").write_bytes(__import__("pickle").dumps(properties_dict))
    if save_quality_masks:
        quality_masks = anatomy_hard_masks_from_probabilities(probabilities)
        np.savez_compressed(output_file_truncated.with_name(output_file_truncated.name + "_qc").with_suffix(".npz"), masks=quality_masks)
    return output_file_truncated.with_suffix(".npz")


def write_anatomy_prediction_manifest(
    records: Iterable[dict[str, Any]],
    *,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path
