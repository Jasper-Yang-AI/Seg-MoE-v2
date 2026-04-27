from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.ndimage import label as connected_components

from .contracts import CaseManifestRow
from .io_utils import load_jsonl, resolve_local_path, save_jsonl, stable_hash
from .roi import expand_bbox_to_min_size


DEFAULT_WG_THRESHOLD = 0.35
DEFAULT_MARGIN_MM = 12.0
DEFAULT_MIN_CROP_SIZE_ZYX = (24, 192, 192)
_CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


@dataclass(frozen=True, slots=True)
class GlandCropRecord:
    case_id: str
    bbox_zyx: tuple[int, int, int, int, int, int]
    crop_shape_zyx: tuple[int, int, int]
    native_shape_zyx: tuple[int, int, int]
    source_prob_path: str
    source_manifest_hash: str
    wg_threshold: float = DEFAULT_WG_THRESHOLD
    margin_mm: float = DEFAULT_MARGIN_MM
    min_crop_size_zyx: tuple[int, int, int] = DEFAULT_MIN_CROP_SIZE_ZYX
    spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0)
    warning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bbox_zyx"] = list(self.bbox_zyx)
        payload["crop_shape_zyx"] = list(self.crop_shape_zyx)
        payload["native_shape_zyx"] = list(self.native_shape_zyx)
        payload["min_crop_size_zyx"] = list(self.min_crop_size_zyx)
        payload["spacing_zyx"] = list(self.spacing_zyx)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GlandCropRecord":
        return cls(
            case_id=str(payload["case_id"]),
            bbox_zyx=tuple(int(v) for v in payload["bbox_zyx"]),  # type: ignore[arg-type]
            crop_shape_zyx=tuple(int(v) for v in payload["crop_shape_zyx"]),  # type: ignore[arg-type]
            native_shape_zyx=tuple(int(v) for v in payload["native_shape_zyx"]),  # type: ignore[arg-type]
            source_prob_path=str(payload.get("source_prob_path", "")),
            source_manifest_hash=str(payload.get("source_manifest_hash", "")),
            wg_threshold=float(payload.get("wg_threshold", DEFAULT_WG_THRESHOLD)),
            margin_mm=float(payload.get("margin_mm", DEFAULT_MARGIN_MM)),
            min_crop_size_zyx=tuple(
                int(v) for v in payload.get("min_crop_size_zyx", DEFAULT_MIN_CROP_SIZE_ZYX)
            ),  # type: ignore[arg-type]
            spacing_zyx=tuple(float(v) for v in payload.get("spacing_zyx", (1.0, 1.0, 1.0))),  # type: ignore[arg-type]
            warning=str(payload.get("warning", "")),
            metadata=dict(payload.get("metadata", {})),
        )


def _normalise_channel_names(raw: Any) -> list[str]:
    if raw is None:
        return []
    arr = np.asarray(raw)
    return [str(item.decode("utf-8") if isinstance(item, bytes) else item) for item in arr.tolist()]


def _load_probabilities(path: str | Path) -> tuple[np.ndarray, list[str]]:
    path = resolve_local_path(path)
    payload = np.load(str(path), allow_pickle=True)
    if "probabilities" in payload:
        probabilities = np.asarray(payload["probabilities"], dtype=np.float32)
    elif "probs" in payload:
        probabilities = np.asarray(payload["probs"], dtype=np.float32)
    else:
        raise KeyError(f"{path} must contain `probabilities` or `probs`.")
    channel_names = _normalise_channel_names(payload["channel_names"] if "channel_names" in payload else None)
    return probabilities, channel_names


def _extract_wg_probability(prob_path: str | Path, record: Mapping[str, Any]) -> np.ndarray:
    probabilities, payload_channel_names = _load_probabilities(prob_path)
    channel_names = [str(v) for v in record.get("channel_names", ())] or payload_channel_names
    if probabilities.ndim != 4:
        raise ValueError(f"Expected anatomy probability bundle [C,Z,Y,X], got {probabilities.shape}")
    if channel_names and "P_WG" in channel_names:
        return probabilities[channel_names.index("P_WG")]
    return probabilities[0]


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int, int, int]:
    coords = np.argwhere(mask)
    z0, y0, x0 = coords.min(axis=0).tolist()
    z1, y1, x1 = (coords.max(axis=0) + 1).tolist()
    return int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    components, n_components = connected_components(mask.astype(bool), structure=_CONNECTIVITY_26)
    if n_components == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = np.bincount(components.reshape(-1))
    sizes[0] = 0
    return components == int(sizes.argmax())


def _expand_bbox(
    bbox_zyx: Sequence[int],
    *,
    spacing_zyx: Sequence[float],
    margin_mm: float,
    shape_zyx: Sequence[int],
) -> tuple[int, int, int, int, int, int]:
    z0, z1, y0, y1, x0, x1 = (int(v) for v in bbox_zyx)
    dz, dy, dx = (int(math.ceil(float(margin_mm) / max(float(spacing), 1e-6))) for spacing in spacing_zyx)
    nz, ny, nx = (int(v) for v in shape_zyx)
    return (
        max(0, z0 - dz),
        min(nz, z1 + dz),
        max(0, y0 - dy),
        min(ny, y1 + dy),
        max(0, x0 - dx),
        min(nx, x1 + dx),
    )


def build_gland_crop_records(
    rows: Iterable[CaseManifestRow],
    prediction_manifest: Iterable[Mapping[str, Any]],
    *,
    wg_threshold: float = DEFAULT_WG_THRESHOLD,
    margin_mm: float = DEFAULT_MARGIN_MM,
    min_crop_size_zyx: Sequence[int] = DEFAULT_MIN_CROP_SIZE_ZYX,
) -> list[GlandCropRecord]:
    materialized = list(rows)
    prediction_by_case = {str(record["case_id"]): dict(record) for record in prediction_manifest}
    source_manifest_hash = stable_hash([row.to_dict() for row in materialized])
    records: list[GlandCropRecord] = []

    for row in materialized:
        prediction = prediction_by_case.get(row.case_id)
        if prediction is None:
            raise KeyError(f"No anatomy prediction record found for case_id={row.case_id}")
        prob_path = str(resolve_local_path(prediction.get("prob_path") or prediction.get("probabilities_path")))
        try:
            wg_probability = _extract_wg_probability(prob_path, prediction)
        except Exception as exc:
            raise RuntimeError(f"Failed to load anatomy probability for case_id={row.case_id}: {prob_path}") from exc
        if wg_probability.ndim != 3:
            raise ValueError(f"Expected P_WG shape [Z,Y,X], got {wg_probability.shape}")

        native_shape = tuple(int(v) for v in wg_probability.shape)
        threshold_mask = wg_probability >= float(wg_threshold)
        largest = _largest_component(threshold_mask)
        warning = ""
        raw_wg_bbox: tuple[int, int, int, int, int, int] | None = None
        margin_bbox: tuple[int, int, int, int, int, int] | None = None
        if not largest.any():
            bbox = (0, native_shape[0], 0, native_shape[1], 0, native_shape[2])
            warning = "empty_wg_mask_used_full_image"
        else:
            raw_wg_bbox = _bbox_from_mask(largest)
            margin_bbox = _expand_bbox(
                raw_wg_bbox,
                spacing_zyx=row.spacing,
                margin_mm=float(margin_mm),
                shape_zyx=native_shape,
            )
            bbox = expand_bbox_to_min_size(
                margin_bbox,
                min_size_zyx=min_crop_size_zyx,
                shape_zyx=native_shape,
            )
        z0, z1, y0, y1, x0, x1 = bbox
        crop_shape = (z1 - z0, y1 - y0, x1 - x0)
        records.append(
            GlandCropRecord(
                case_id=row.case_id,
                bbox_zyx=bbox,
                crop_shape_zyx=tuple(int(v) for v in crop_shape),
                native_shape_zyx=native_shape,
                source_prob_path=prob_path,
                source_manifest_hash=str(prediction.get("source_manifest_hash") or source_manifest_hash),
                wg_threshold=float(wg_threshold),
                margin_mm=float(margin_mm),
                min_crop_size_zyx=tuple(int(v) for v in min_crop_size_zyx),
                spacing_zyx=tuple(float(v) for v in row.spacing),
                warning=warning,
                metadata={
                    "prediction_fold": prediction.get("fold"),
                    "prediction_split": prediction.get("split"),
                    "raw_wg_bbox_zyx": list(raw_wg_bbox) if raw_wg_bbox is not None else [],
                    "margin_bbox_zyx": list(margin_bbox) if margin_bbox is not None else [],
                    "expanded_bbox_zyx": list(bbox),
                },
            )
        )
    return records


def write_gland_crop_manifest(records: Iterable[GlandCropRecord], path: str | Path) -> Path:
    return save_jsonl((record.to_dict() for record in records), path)


def load_gland_crop_manifest(path: str | Path) -> list[GlandCropRecord]:
    return [GlandCropRecord.from_dict(row) for row in load_jsonl(path)]


def index_gland_crop_manifest(records: Iterable[GlandCropRecord]) -> dict[str, GlandCropRecord]:
    return {record.case_id: record for record in records}
