from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_dilation, distance_transform_edt
from torch.utils.data import Dataset

from .contracts import CaseManifestRow
from .features import lesion_stats_from_experts
from .labels import build_anatomy_targets, build_lesion_target
from .sampling import choose_layer1_crop_mode


def load_nifti_zyx(path: str | Path, *, dtype: np.dtype | None = np.float32) -> np.ndarray:
    arr = np.asanyarray(nib.load(str(path)).dataobj)
    arr = np.transpose(arr, (2, 1, 0))
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def _choose_center(candidate_mask: np.ndarray, rng: random.Random) -> tuple[int, int, int]:
    coords = np.argwhere(candidate_mask)
    if len(coords) == 0:
        shape = candidate_mask.shape
        return tuple(rng.randrange(int(s)) for s in shape)
    idx = int(rng.randrange(len(coords)))
    z, y, x = coords[idx].tolist()
    return int(z), int(y), int(x)


def _extract_patch(arr: np.ndarray, center: tuple[int, int, int], patch_size: tuple[int, int, int]) -> np.ndarray:
    if arr.ndim < 3:
        raise ValueError(f"Expected array with trailing [D,H,W], got shape={arr.shape}")
    d, h, w = arr.shape[-3:]
    pd, ph, pw = patch_size
    cz, cy, cx = center

    starts = [cz - pd // 2, cy - ph // 2, cx - pw // 2]
    limits = [d, h, w]
    sizes = [pd, ph, pw]
    slices = []
    pad_width = [(0, 0)] * arr.ndim
    for axis, (start, size, limit) in enumerate(zip(starts, sizes, limits), start=arr.ndim - 3):
        start = max(0, min(start, limit - size))
        end = start + size
        if start < 0:
            pad_width[axis] = (-start, pad_width[axis][1])
        if end > limit:
            pad_width[axis] = (pad_width[axis][0], end - limit)
        slices.append(slice(max(0, start), min(end, limit)))

    patch = arr[(..., slices[0], slices[1], slices[2])]
    if patch.shape[-3:] != patch_size:
        trailing = patch.shape[-3:]
        full_pad = []
        for got, need in zip(trailing, patch_size):
            short = max(0, need - got)
            left = short // 2
            right = short - left
            full_pad.append((left, right))
        if arr.ndim == 3:
            pad = full_pad
        elif arr.ndim == 4:
            pad = [(0, 0)] + full_pad
        else:
            pad = [(0, 0)] * (arr.ndim - 3) + full_pad
        patch = np.pad(patch, pad, mode="edge")
    return patch


def _signed_distance(mask: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    mask = mask.astype(bool)
    inside = distance_transform_edt(mask, sampling=spacing)
    outside = distance_transform_edt(~mask, sampling=spacing)
    return (inside - outside).astype(np.float32)


@dataclass(frozen=True, slots=True)
class CaseArrays:
    modalities: dict[str, np.ndarray]
    label: np.ndarray
    anatomy_priors: dict[str, np.ndarray] | None = None
    expert_probs: np.ndarray | None = None
    fp_risk_map: np.ndarray | None = None


class Layer1LesionDataset(Dataset):
    def __init__(
        self,
        cases: Sequence[CaseManifestRow],
        *,
        patch_size: tuple[int, int, int],
        anatomy_prior_map: Mapping[str, Mapping[str, np.ndarray]] | None = None,
        case_cache: Mapping[str, CaseArrays] | None = None,
        seed: int = 42,
    ) -> None:
        self.cases = list(cases)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.anatomy_prior_map = anatomy_prior_map or {}
        self.case_cache = case_cache or {}
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.cases)

    def _load_case(self, row: CaseManifestRow) -> CaseArrays:
        cached = self.case_cache.get(row.case_id)
        if cached is not None:
            return cached
        modalities = {
            "T2W": load_nifti_zyx(row.t2w_path),
            "ADC": load_nifti_zyx(row.adc_path),
            "DWI": load_nifti_zyx(row.dwi_path),
        }
        label = load_nifti_zyx(row.label_path, dtype=np.int16)
        priors = self.anatomy_prior_map.get(row.case_id)
        return CaseArrays(modalities=modalities, label=label, anatomy_priors=dict(priors) if priors else None)

    def __getitem__(self, index: int):
        row = self.cases[index]
        rng = random.Random(self.seed + index)
        bundle = self._load_case(row)

        lesion_target = build_lesion_target(bundle.label, row.cohort_type)
        anatomy = build_anatomy_targets(bundle.label)
        wg_mask = anatomy["WG"]["target"].astype(bool)
        choice = choose_layer1_crop_mode(row.cohort_type, rng=rng)

        if choice.mode == "lesion_positive" and lesion_target.any():
            center = _choose_center(lesion_target > 0, rng)
        elif choice.mode == "wg_or_boundary_background":
            signed = _signed_distance(wg_mask, row.spacing)
            candidate = np.logical_or(wg_mask, np.abs(signed) <= 5.0)
            center = _choose_center(candidate, rng)
        else:
            center = _choose_center(lesion_target == 0, rng)

        channels = [bundle.modalities[name] for name in ("T2W", "ADC", "DWI")]
        if bundle.anatomy_priors:
            channels.extend(bundle.anatomy_priors[name] for name in ("P_WG", "P_PZ", "P_TZ"))
        image = np.stack([_extract_patch(arr, center, self.patch_size) for arr in channels], axis=0).astype(np.float32)
        target = _extract_patch(lesion_target, center, self.patch_size).astype(np.uint8)

        return (
            torch.from_numpy(image).float(),
            torch.from_numpy(target).float(),
            {"case_id": row.case_id, "cohort_type": row.cohort_type, "crop_mode": choice.mode},
        )


class Layer2PatchDataset(Dataset):
    def __init__(
        self,
        cases: Sequence[CaseManifestRow],
        *,
        patch_size: tuple[int, int, int],
        expert_prob_map: Mapping[str, np.ndarray],
        anatomy_prior_map: Mapping[str, Mapping[str, np.ndarray]],
        fp_risk_map: Mapping[str, np.ndarray],
        case_cache: Mapping[str, CaseArrays] | None = None,
        seed: int = 42,
    ) -> None:
        self.cases = list(cases)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.expert_prob_map = expert_prob_map
        self.anatomy_prior_map = anatomy_prior_map
        self.fp_risk_map = fp_risk_map
        self.case_cache = case_cache or {}
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.cases)

    def _load_case(self, row: CaseManifestRow) -> CaseArrays:
        cached = self.case_cache.get(row.case_id)
        if cached is not None:
            return cached
        modalities = {
            "T2W": load_nifti_zyx(row.t2w_path),
            "ADC": load_nifti_zyx(row.adc_path),
            "DWI": load_nifti_zyx(row.dwi_path),
        }
        label = load_nifti_zyx(row.label_path, dtype=np.int16)
        return CaseArrays(
            modalities=modalities,
            label=label,
            anatomy_priors=dict(self.anatomy_prior_map[row.case_id]),
            expert_probs=np.asarray(self.expert_prob_map[row.case_id], dtype=np.float32),
            fp_risk_map=np.asarray(self.fp_risk_map[row.case_id], dtype=np.float32),
        )

    def __getitem__(self, index: int):
        row = self.cases[index]
        rng = random.Random(self.seed + index)
        bundle = self._load_case(row)
        assert bundle.expert_probs is not None
        assert bundle.anatomy_priors is not None
        assert bundle.fp_risk_map is not None

        lesion_target = build_lesion_target(bundle.label, row.cohort_type)
        fp_mask = bundle.fp_risk_map > 0
        regular_bg = np.logical_not(np.logical_or(lesion_target > 0, fp_mask))

        token = rng.random()
        if token < 0.4 and lesion_target.any():
            mode = "lesion_positive"
            center = _choose_center(lesion_target > 0, rng)
        elif token < 0.7 and fp_mask.any():
            mode = "fp_hard_negative"
            center = _choose_center(fp_mask, rng)
        else:
            mode = "regular_background"
            center = _choose_center(regular_bg, rng)

        stats = lesion_stats_from_experts(bundle.expert_probs)
        channels = [bundle.modalities[name] for name in ("T2W", "ADC", "DWI")]
        channels.extend(bundle.expert_probs[i] for i in range(bundle.expert_probs.shape[0]))
        channels.extend([stats["entropy"], stats["disagreement"]])
        channels.extend(bundle.anatomy_priors[name] for name in ("P_WG", "P_PZ", "P_TZ"))
        channels.append(bundle.fp_risk_map)

        image = np.stack([_extract_patch(arr, center, self.patch_size) for arr in channels], axis=0).astype(np.float32)
        target = _extract_patch(lesion_target, center, self.patch_size).astype(np.uint8)
        fp_weight = _extract_patch(bundle.fp_risk_map, center, self.patch_size)
        fp_weight = np.where(fp_weight > 0, 2.5, 1.0).astype(np.float32)

        return (
            torch.from_numpy(image).float(),
            torch.from_numpy(target).float(),
            torch.from_numpy(fp_weight).float(),
            {"case_id": row.case_id, "sample_mode": mode},
        )


class GatePatchDataset(Dataset):
    def __init__(
        self,
        cases: Sequence[CaseManifestRow],
        *,
        patch_size: tuple[int, int, int],
        expert_prob_map: Mapping[str, np.ndarray],
        anatomy_prior_map: Mapping[str, Mapping[str, np.ndarray]],
        fp_risk_map: Mapping[str, np.ndarray],
        label_map: Mapping[str, np.ndarray] | None = None,
        seed: int = 42,
    ) -> None:
        self.cases = list(cases)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.expert_prob_map = expert_prob_map
        self.anatomy_prior_map = anatomy_prior_map
        self.fp_risk_map = fp_risk_map
        self.label_map = label_map or {}
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int):
        row = self.cases[index]
        rng = random.Random(self.seed + index)
        expert_probs = np.asarray(self.expert_prob_map[row.case_id], dtype=np.float32)
        priors = self.anatomy_prior_map[row.case_id]
        fp_risk = np.asarray(self.fp_risk_map[row.case_id], dtype=np.float32)
        label_arr = self.label_map.get(row.case_id)
        if label_arr is None:
            label_arr = build_lesion_target(load_nifti_zyx(row.label_path, dtype=np.int16), row.cohort_type)
        else:
            label_arr = np.asarray(label_arr, dtype=np.uint8)

        lesion_mask = label_arr > 0
        fp_mask = fp_risk > 0
        focus_mask = np.logical_or(lesion_mask, fp_mask)
        center = _choose_center(focus_mask if focus_mask.any() else np.ones_like(label_arr, dtype=bool), rng)

        stats = lesion_stats_from_experts(expert_probs)
        extra_channels = [
            stats["consensus"],
            stats["entropy"],
            stats["disagreement"],
            priors["P_WG"],
            priors["P_PZ"],
            priors["P_TZ"],
            fp_risk,
        ]
        expert_patch = np.stack([_extract_patch(expert_probs[i], center, self.patch_size) for i in range(expert_probs.shape[0])], axis=0)
        extra_patch = np.stack([_extract_patch(arr, center, self.patch_size) for arr in extra_channels], axis=0)
        target_patch = _extract_patch(label_arr, center, self.patch_size).astype(np.uint8)
        return (
            torch.from_numpy(expert_patch).float(),
            torch.from_numpy(extra_patch).float(),
            torch.from_numpy(target_patch).float(),
            {"case_id": row.case_id},
        )


def fp_weight_map_from_components(components_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(components_mask, dtype=bool)
    dilated = binary_dilation(mask, iterations=2)
    return np.where(dilated, 2.5, 1.0).astype(np.float32)
