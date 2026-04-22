from __future__ import annotations

from typing import TypedDict

import numpy as np


IGNORE_INDEX = -1
LAYER1_CANDIDATE_POSITIVE_LABEL_VALUES = (1, 2)
LAYER1_BACKGROUND_WEIGHT = 1.0
LAYER1_SOURCE_POSITIVE_WEIGHTS = {1: 1.25, 2: 0.75}


class HeadTarget(TypedDict):
    target: np.ndarray
    valid_mask: np.ndarray


def build_anatomy_targets(mask: np.ndarray) -> dict[str, HeadTarget]:
    mask = np.asarray(mask)
    wg_target = np.isin(mask, (1, 2, 3)).astype(np.uint8)
    wg_valid = np.ones(mask.shape, dtype=bool)

    pz_valid = mask != 3
    pz_target = (mask == 1).astype(np.uint8)

    tz_valid = mask != 3
    tz_target = (mask == 2).astype(np.uint8)

    return {
        "WG": {"target": wg_target, "valid_mask": wg_valid},
        "PZ": {"target": pz_target, "valid_mask": pz_valid},
        "TZ": {"target": tz_target, "valid_mask": tz_valid},
    }


def build_masked_head_target(mask: np.ndarray, head_name: str) -> np.ndarray:
    head_name = head_name.upper()
    bundle = build_anatomy_targets(mask)[head_name]
    out = np.full(mask.shape, IGNORE_INDEX, dtype=np.int8)
    out[bundle["valid_mask"]] = bundle["target"][bundle["valid_mask"]].astype(np.int8)
    return out


def build_lesion_target(mask: np.ndarray, cohort_type: str) -> np.ndarray:
    mask = np.asarray(mask)
    if str(cohort_type).lower() == "nca":
        return np.zeros(mask.shape, dtype=np.uint8)
    return (mask == 3).astype(np.uint8)


def build_layer1_lesion_mimic_source(mask: np.ndarray, cohort_type: str) -> np.ndarray:
    """Tri-state Layer1 source labels: 1=PCA lesion, 2=NCA mimic."""
    mask = np.asarray(mask)
    source = np.zeros(mask.shape, dtype=np.uint8)
    if str(cohort_type).lower() == "nca":
        source[mask == 3] = 2
    else:
        source[mask == 3] = 1
    return source


def build_layer1_high_recall_target(
    source_labels: np.ndarray,
    *,
    positive_label_values: tuple[int, ...] = LAYER1_CANDIDATE_POSITIVE_LABEL_VALUES,
) -> np.ndarray:
    source_labels = np.asarray(source_labels)
    return np.isin(source_labels, positive_label_values).astype(np.uint8)


def build_layer1_source_weight_map(
    source_labels: np.ndarray,
    *,
    background_weight: float = LAYER1_BACKGROUND_WEIGHT,
    source_positive_weights: dict[int, float] | None = None,
) -> np.ndarray:
    """Voxel weights for candidate-first Layer1 BCE from tri-state source labels."""
    source_labels = np.asarray(source_labels)
    weights = np.full(source_labels.shape, float(background_weight), dtype=np.float32)
    for label_value, weight in (source_positive_weights or LAYER1_SOURCE_POSITIVE_WEIGHTS).items():
        weights[source_labels == int(label_value)] = float(weight)
    return weights


def masked_binary_targets(mask: np.ndarray) -> dict[str, np.ndarray]:
    return {name: build_masked_head_target(mask, name) for name in ("WG", "PZ", "TZ")}
