from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, label

from .contracts import FPComponentRecord
from .features import binary_entropy, expert_consensus, expert_disagreement


_CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int, int, int]:
    coords = np.argwhere(mask)
    z0, y0, x0 = coords.min(axis=0).tolist()
    z1, y1, x1 = (coords.max(axis=0) + 1).tolist()
    return int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)


def _centroid_from_mask(mask: np.ndarray) -> tuple[float, float, float]:
    coords = np.argwhere(mask)
    center = coords.mean(axis=0)
    return float(center[0]), float(center[1]), float(center[2])


def _max_iou(component_mask: np.ndarray, gt_components: np.ndarray, n_gt: int) -> float:
    if n_gt == 0:
        return 0.0
    comp_sum = int(component_mask.sum())
    best = 0.0
    for gt_id in range(1, n_gt + 1):
        gt_mask = gt_components == gt_id
        inter = int(np.logical_and(component_mask, gt_mask).sum())
        if inter == 0:
            continue
        union = comp_sum + int(gt_mask.sum()) - inter
        best = max(best, inter / max(union, 1))
    return float(best)


def _dominant_zone(component_mask: np.ndarray, anatomy_priors: Mapping[str, np.ndarray] | None) -> str:
    if not anatomy_priors:
        return "outside_WG"
    wg = anatomy_priors.get("P_WG") if "P_WG" in anatomy_priors else anatomy_priors.get("WG")
    pz = anatomy_priors.get("P_PZ") if "P_PZ" in anatomy_priors else anatomy_priors.get("PZ")
    tz = anatomy_priors.get("P_TZ") if "P_TZ" in anatomy_priors else anatomy_priors.get("TZ")
    if wg is None or pz is None or tz is None:
        return "outside_WG"
    wg_mean = float(np.asarray(wg)[component_mask].mean())
    if wg_mean < 0.5:
        return "outside_WG"
    pz_mean = float(np.asarray(pz)[component_mask].mean())
    tz_mean = float(np.asarray(tz)[component_mask].mean())
    if max(pz_mean, tz_mean) < 0.5:
        return "WG"
    return "PZ" if pz_mean >= tz_mean else "TZ"


def build_fp_bank(
    *,
    case_id: str,
    source_layer: str,
    predictor_fold: int,
    expert_probs: np.ndarray,
    gt_lesion: np.ndarray,
    image_channels: Mapping[str, np.ndarray],
    anatomy_priors: Mapping[str, np.ndarray] | None = None,
    wg_distance_map: np.ndarray | None = None,
    threshold: float = 0.5,
    min_component_size: int = 8,
) -> tuple[list[FPComponentRecord], np.ndarray]:
    expert_probs = np.asarray(expert_probs, dtype=np.float32)
    if expert_probs.ndim != 4:
        raise ValueError(f"Expected expert_probs with shape [K,D,H,W], got {expert_probs.shape}")

    gt_lesion = np.asarray(gt_lesion, dtype=bool)
    mean_prob = expert_consensus(expert_probs)
    entropy = binary_entropy(mean_prob)
    disagreement = expert_disagreement(expert_probs)
    candidate_map = mean_prob >= float(threshold)

    comp_labels, n_comp = label(candidate_map, structure=_CONNECTIVITY_26)
    gt_labels, n_gt = label(gt_lesion, structure=_CONNECTIVITY_26)

    records: list[FPComponentRecord] = []
    risk_map = np.zeros_like(mean_prob, dtype=np.float32)

    for component_id in range(1, n_comp + 1):
        component_mask = comp_labels == component_id
        volume_voxels = int(component_mask.sum())
        if volume_voxels < int(min_component_size):
            continue

        overlap_voxels = int(np.logical_and(component_mask, gt_lesion).sum())
        max_iou = _max_iou(component_mask, gt_labels, n_gt)
        if overlap_voxels > 0 and max_iou >= 0.10:
            continue

        dominant_zone = _dominant_zone(component_mask, anatomy_priors)
        distance_mm = float("nan")
        if wg_distance_map is not None:
            distance_mm = float(np.asarray(wg_distance_map, dtype=np.float32)[component_mask].mean())

        expert_prob_mean = tuple(float(expert_probs[i][component_mask].mean()) for i in range(expert_probs.shape[0]))
        expert_prob_max = tuple(float(expert_probs[i][component_mask].max()) for i in range(expert_probs.shape[0]))
        entropy_vals = entropy[component_mask]
        disagreement_vals = disagreement[component_mask]

        intensity_mean = {name: float(np.asarray(arr)[component_mask].mean()) for name, arr in image_channels.items()}
        intensity_std = {name: float(np.asarray(arr)[component_mask].std()) for name, arr in image_channels.items()}
        intensity_p10 = {name: float(np.percentile(np.asarray(arr)[component_mask], 10)) for name, arr in image_channels.items()}
        intensity_p90 = {name: float(np.percentile(np.asarray(arr)[component_mask], 90)) for name, arr in image_channels.items()}

        record = FPComponentRecord(
            case_id=str(case_id),
            source_layer=source_layer,  # type: ignore[arg-type]
            predictor_fold=int(predictor_fold),
            component_id=int(component_id),
            bbox_zyx=_bbox_from_mask(component_mask),
            centroid_zyx=_centroid_from_mask(component_mask),
            volume_voxels=volume_voxels,
            dominant_zone=dominant_zone,
            signed_distance_to_wg_boundary_mm=distance_mm,
            expert_prob_mean=expert_prob_mean,
            expert_prob_max=expert_prob_max,
            entropy_mean=float(entropy_vals.mean()),
            entropy_max=float(entropy_vals.max()),
            disagreement_mean=float(disagreement_vals.mean()),
            disagreement_max=float(disagreement_vals.max()),
            intensity_mean=intensity_mean,
            intensity_std=intensity_std,
            intensity_p10=intensity_p10,
            intensity_p90=intensity_p90,
            overlap_voxels=overlap_voxels,
            max_iou=max_iou,
            fp_kind="pure" if overlap_voxels == 0 else "low_iou",
        )
        records.append(record)

        component_score = float(mean_prob[component_mask].mean() * (1.0 + record.entropy_mean + record.disagreement_mean) / 3.0)
        if dominant_zone == "outside_WG" or (not np.isnan(distance_mm) and abs(distance_mm) < 3.0):
            component_score *= 1.25
        risk_map[component_mask] = np.maximum(risk_map[component_mask], component_score)

    if records:
        risk_map = gaussian_filter(risk_map, sigma=1.0)
    risk_map = np.clip(risk_map, 0.0, 1.0).astype(np.float32)
    return records, risk_map


def write_fp_bank(
    out_dir: str | Path,
    *,
    components: list[FPComponentRecord],
    risk_map: np.ndarray,
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    components_path = out_dir / "components.parquet"
    risk_map_path = out_dir / "fp_risk_map.npz"
    pd.DataFrame([record.to_dict() for record in components]).to_parquet(components_path, index=False)
    np.savez_compressed(risk_map_path, fp_risk_map=np.asarray(risk_map, dtype=np.float32))
    return components_path, risk_map_path
