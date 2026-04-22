from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np


BBoxZYX = tuple[int, int, int, int, int, int]


def _coerce_bbox_zyx(bbox_zyx: Sequence[int]) -> BBoxZYX:
    values = tuple(int(v) for v in bbox_zyx)
    if len(values) != 6:
        raise ValueError(f"Expected bbox_zyx with 6 values, got {values}")
    z0, z1, y0, y1, x0, x1 = values
    if z0 < 0 or y0 < 0 or x0 < 0 or z1 < z0 or y1 < y0 or x1 < x0:
        raise ValueError(f"Invalid bbox_zyx={values}")
    return z0, z1, y0, y1, x0, x1


def crop_zyx(arr: np.ndarray, bbox_zyx: Sequence[int] | None) -> np.ndarray:
    """Crop an array by trailing Z/Y/X axes."""
    arr = np.asarray(arr)
    if bbox_zyx is None:
        return arr
    z0, z1, y0, y1, x0, x1 = _coerce_bbox_zyx(bbox_zyx)
    return arr[..., z0:z1, y0:y1, x0:x1]


def _expand_axis_to_min_size(start: int, stop: int, limit: int, min_size: int) -> tuple[int, int]:
    start = int(start)
    stop = int(stop)
    limit = int(limit)
    min_size = int(min_size)
    if limit <= 0:
        return 0, 0
    if min_size <= 0 or stop - start >= min_size:
        return max(0, start), min(limit, stop)
    if limit <= min_size:
        return 0, limit
    center = (start + stop) / 2.0
    new_start = int(np.floor(center - min_size / 2.0))
    new_stop = new_start + min_size
    if new_start < 0:
        new_stop -= new_start
        new_start = 0
    if new_stop > limit:
        shift = new_stop - limit
        new_start = max(0, new_start - shift)
        new_stop = limit
    return int(new_start), int(new_stop)


def expand_bbox_to_min_size(
    bbox_zyx: Sequence[int],
    *,
    min_size_zyx: Sequence[int],
    shape_zyx: Sequence[int],
) -> BBoxZYX:
    """Expand a Z/Y/X bbox around its center until it reaches a fixed minimum size."""
    z0, z1, y0, y1, x0, x1 = _coerce_bbox_zyx(bbox_zyx)
    nz, ny, nx = (int(v) for v in shape_zyx)
    mz, my, mx = (int(v) for v in min_size_zyx)
    ez0, ez1 = _expand_axis_to_min_size(z0, z1, nz, mz)
    ey0, ey1 = _expand_axis_to_min_size(y0, y1, ny, my)
    ex0, ex1 = _expand_axis_to_min_size(x0, x1, nx, mx)
    return ez0, ez1, ey0, ey1, ex0, ex1


def reinflate_crop(
    crop: np.ndarray,
    bbox_zyx: Sequence[int],
    native_shape_zyx: Sequence[int],
    *,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Place a crop with trailing Z/Y/X axes back into full native image space."""
    crop = np.asarray(crop)
    z0, z1, y0, y1, x0, x1 = _coerce_bbox_zyx(bbox_zyx)
    native_shape = tuple(int(v) for v in native_shape_zyx)
    if len(native_shape) != 3:
        raise ValueError(f"Expected native_shape_zyx with 3 values, got {native_shape}")
    bbox_shape = (z1 - z0, y1 - y0, x1 - x0)
    if tuple(crop.shape[-3:]) != bbox_shape:
        raise ValueError(f"Crop trailing shape {crop.shape[-3:]} does not match bbox shape {bbox_shape}")
    output_shape = tuple(crop.shape[:-3]) + native_shape
    full = np.full(output_shape, fill_value, dtype=crop.dtype)
    full[..., z0:z1, y0:y1, x0:x1] = crop
    return full


def load_reinflated_prediction_npz(
    path: str | Path,
    *,
    field: str | None = None,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Load a crop-space prediction npz and reinflate it using stored bbox metadata."""
    payload: Any = np.load(str(path), allow_pickle=True)
    selected = field
    if selected is None:
        for candidate in ("probabilities", "probs", "logits", "data"):
            if candidate in payload:
                selected = candidate
                break
    if selected is None or selected not in payload:
        raise KeyError(f"{path} does not contain a prediction field")
    if "bbox_zyx" not in payload or "native_shape_zyx" not in payload:
        raise KeyError(f"{path} must contain bbox_zyx and native_shape_zyx")
    return reinflate_crop(
        np.asarray(payload[selected]),
        tuple(int(v) for v in np.asarray(payload["bbox_zyx"]).tolist()),
        tuple(int(v) for v in np.asarray(payload["native_shape_zyx"]).tolist()),
        fill_value=fill_value,
    )
