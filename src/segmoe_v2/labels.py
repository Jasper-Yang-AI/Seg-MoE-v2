from __future__ import annotations

from typing import TypedDict

import numpy as np


IGNORE_INDEX = -1


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


def masked_binary_targets(mask: np.ndarray) -> dict[str, np.ndarray]:
    return {name: build_masked_head_target(mask, name) for name in ("WG", "PZ", "TZ")}
