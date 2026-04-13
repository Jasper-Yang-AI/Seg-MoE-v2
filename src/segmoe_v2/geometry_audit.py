from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import nibabel as nib
import numpy as np

from .contracts import CaseManifestRow
from .io_utils import save_csv_rows, save_json


@dataclass(frozen=True, slots=True)
class GeometryAuditThresholds:
    soft_spacing_mm: float = 1e-4
    hard_spacing_mm: float = 1e-2
    soft_origin_mm: float = 1e-2
    hard_origin_mm: float = 1.0
    soft_direction: float = 1e-4
    hard_direction: float = 1e-3

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class GeometryAuditCaseResult:
    case_id: str
    patient_id: str
    cohort_type: str
    era_bin: str
    fixed_split: str
    val_fold: int | None
    image_shape_match: bool
    label_shape_match: bool
    image_axcodes_match: bool
    label_axcodes_match: bool
    image_spacing_max_diff_mm: float
    label_spacing_max_diff_mm: float
    image_origin_max_diff_mm: float
    label_origin_max_diff_mm: float
    image_direction_max_diff: float
    label_direction_max_diff: float
    image_affine_max_diff: float
    label_affine_max_diff: float
    t2w_axcodes: str
    adc_axcodes: str
    dwi_axcodes: str
    label_axcodes: str
    recommendation: str
    needs_preprocessing: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in tuple(payload.items()):
            if isinstance(value, float):
                payload[key] = round(value, 8)
        return payload


def default_geometry_csv_path(manifest_path: str | Path) -> Path:
    manifest_path = Path(manifest_path)
    return manifest_path.parent / "geometry_audit.csv"


def default_geometry_summary_path(manifest_path: str | Path) -> Path:
    manifest_path = Path(manifest_path)
    return manifest_path.parent / "geometry_audit_summary.json"


def _canonical_axcodes(affine: np.ndarray) -> str:
    matrix = np.asarray(affine, dtype=np.float64)[:3, :3]
    labels = {
        0: ("L", "R"),
        1: ("P", "A"),
        2: ("I", "S"),
    }
    used_world_axes: set[int] = set()
    codes: list[str] = []
    for voxel_axis in range(3):
        column = matrix[:, voxel_axis]
        world_axis = int(np.argmax(np.abs(column)))
        if world_axis in used_world_axes:
            codes.append("?")
            continue
        used_world_axes.add(world_axis)
        negative, positive = labels[world_axis]
        codes.append(positive if column[world_axis] >= 0 else negative)
    return "".join(codes)


def _direction_matrix(affine: np.ndarray) -> np.ndarray:
    matrix = np.asarray(affine, dtype=np.float64)[:3, :3]
    zooms = np.linalg.norm(matrix, axis=0)
    zooms = np.where(zooms == 0.0, 1.0, zooms)
    return matrix / zooms


def _geometry_payload(path: str | Path) -> dict[str, Any]:
    image = nib.load(str(path))
    affine = np.asarray(image.affine, dtype=np.float64)
    return {
        "path": str(path),
        "shape": tuple(int(v) for v in image.shape[:3]),
        "spacing_xyz": tuple(float(v) for v in image.header.get_zooms()[:3]),
        "origin_xyz": tuple(float(v) for v in affine[:3, 3]),
        "direction": _direction_matrix(affine),
        "affine": affine,
        "axcodes": _canonical_axcodes(affine),
    }


def _max_abs_diff(left: Sequence[float] | np.ndarray, right: Sequence[float] | np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))))


def _classify_case(
    *,
    image_shape_match: bool,
    label_shape_match: bool,
    image_axcodes_match: bool,
    label_axcodes_match: bool,
    image_spacing_max_diff_mm: float,
    label_spacing_max_diff_mm: float,
    image_origin_max_diff_mm: float,
    label_origin_max_diff_mm: float,
    image_direction_max_diff: float,
    label_direction_max_diff: float,
    thresholds: GeometryAuditThresholds,
) -> tuple[str, bool, str]:
    hard_reasons: list[str] = []
    soft_reasons: list[str] = []

    if not image_shape_match:
        hard_reasons.append("image_shape_mismatch")
    if not label_shape_match:
        hard_reasons.append("label_shape_mismatch")
    if not image_axcodes_match:
        hard_reasons.append("image_axcodes_mismatch")
    if not label_axcodes_match:
        hard_reasons.append("label_axcodes_mismatch")

    for name, value, soft_limit, hard_limit in (
        ("image_spacing", image_spacing_max_diff_mm, thresholds.soft_spacing_mm, thresholds.hard_spacing_mm),
        ("label_spacing", label_spacing_max_diff_mm, thresholds.soft_spacing_mm, thresholds.hard_spacing_mm),
        ("image_origin", image_origin_max_diff_mm, thresholds.soft_origin_mm, thresholds.hard_origin_mm),
        ("label_origin", label_origin_max_diff_mm, thresholds.soft_origin_mm, thresholds.hard_origin_mm),
        ("image_direction", image_direction_max_diff, thresholds.soft_direction, thresholds.hard_direction),
        ("label_direction", label_direction_max_diff, thresholds.soft_direction, thresholds.hard_direction),
    ):
        if value > hard_limit:
            hard_reasons.append(f"{name}>{hard_limit}")
        elif value > soft_limit:
            soft_reasons.append(f"{name}>{soft_limit}")

    if hard_reasons:
        return "resample_required", True, ";".join(hard_reasons)
    if soft_reasons:
        return "header_harmonize_recommended", True, ";".join(soft_reasons)
    return "no_action", False, ""


def audit_case_geometry(
    row: CaseManifestRow,
    *,
    thresholds: GeometryAuditThresholds | None = None,
) -> GeometryAuditCaseResult:
    thresholds = thresholds or GeometryAuditThresholds()
    t2 = _geometry_payload(row.t2w_path)
    adc = _geometry_payload(row.adc_path)
    dwi = _geometry_payload(row.dwi_path)
    label = _geometry_payload(row.label_path)

    image_geometries = [adc, dwi]
    image_shape_match = all(g["shape"] == t2["shape"] for g in image_geometries)
    label_shape_match = label["shape"] == t2["shape"]
    image_axcodes_match = all(g["axcodes"] == t2["axcodes"] for g in image_geometries)
    label_axcodes_match = label["axcodes"] == t2["axcodes"]

    image_spacing_max_diff_mm = max(_max_abs_diff(t2["spacing_xyz"], g["spacing_xyz"]) for g in image_geometries) if image_geometries else 0.0
    label_spacing_max_diff_mm = _max_abs_diff(t2["spacing_xyz"], label["spacing_xyz"])
    image_origin_max_diff_mm = max(_max_abs_diff(t2["origin_xyz"], g["origin_xyz"]) for g in image_geometries) if image_geometries else 0.0
    label_origin_max_diff_mm = _max_abs_diff(t2["origin_xyz"], label["origin_xyz"])
    image_direction_max_diff = max(_max_abs_diff(t2["direction"], g["direction"]) for g in image_geometries) if image_geometries else 0.0
    label_direction_max_diff = _max_abs_diff(t2["direction"], label["direction"])
    image_affine_max_diff = max(_max_abs_diff(t2["affine"], g["affine"]) for g in image_geometries) if image_geometries else 0.0
    label_affine_max_diff = _max_abs_diff(t2["affine"], label["affine"])

    recommendation, needs_preprocessing, note = _classify_case(
        image_shape_match=image_shape_match,
        label_shape_match=label_shape_match,
        image_axcodes_match=image_axcodes_match,
        label_axcodes_match=label_axcodes_match,
        image_spacing_max_diff_mm=image_spacing_max_diff_mm,
        label_spacing_max_diff_mm=label_spacing_max_diff_mm,
        image_origin_max_diff_mm=image_origin_max_diff_mm,
        label_origin_max_diff_mm=label_origin_max_diff_mm,
        image_direction_max_diff=image_direction_max_diff,
        label_direction_max_diff=label_direction_max_diff,
        thresholds=thresholds,
    )

    return GeometryAuditCaseResult(
        case_id=row.case_id,
        patient_id=row.patient_id,
        cohort_type=row.cohort_type,
        era_bin=row.era_bin,
        fixed_split=row.fixed_split,
        val_fold=row.val_fold,
        image_shape_match=image_shape_match,
        label_shape_match=label_shape_match,
        image_axcodes_match=image_axcodes_match,
        label_axcodes_match=label_axcodes_match,
        image_spacing_max_diff_mm=image_spacing_max_diff_mm,
        label_spacing_max_diff_mm=label_spacing_max_diff_mm,
        image_origin_max_diff_mm=image_origin_max_diff_mm,
        label_origin_max_diff_mm=label_origin_max_diff_mm,
        image_direction_max_diff=image_direction_max_diff,
        label_direction_max_diff=label_direction_max_diff,
        image_affine_max_diff=image_affine_max_diff,
        label_affine_max_diff=label_affine_max_diff,
        t2w_axcodes=t2["axcodes"],
        adc_axcodes=adc["axcodes"],
        dwi_axcodes=dwi["axcodes"],
        label_axcodes=label["axcodes"],
        recommendation=recommendation,
        needs_preprocessing=needs_preprocessing,
        note=note,
    )


def audit_geometry(
    rows: Iterable[CaseManifestRow],
    *,
    thresholds: GeometryAuditThresholds | None = None,
) -> list[GeometryAuditCaseResult]:
    thresholds = thresholds or GeometryAuditThresholds()
    return [audit_case_geometry(row, thresholds=thresholds) for row in rows]


def build_geometry_summary(
    results: Sequence[GeometryAuditCaseResult],
    *,
    thresholds: GeometryAuditThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or GeometryAuditThresholds()
    recommendation_counts = Counter(result.recommendation for result in results)
    split_counts = Counter(result.fixed_split for result in results)
    cohort_counts = Counter(result.cohort_type for result in results)

    def _top_cases(key: str, limit: int = 10) -> list[dict[str, Any]]:
        ranked = sorted(results, key=lambda item: getattr(item, key), reverse=True)[:limit]
        return [
            {
                "case_id": item.case_id,
                "recommendation": item.recommendation,
                key: round(float(getattr(item, key)), 8),
            }
            for item in ranked
        ]

    flagged = [result for result in results if result.needs_preprocessing]
    return {
        "total_cases": len(results),
        "recommendation_counts": dict(sorted(recommendation_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "cohort_counts": dict(sorted(cohort_counts.items())),
        "thresholds": thresholds.to_dict(),
        "needs_preprocessing_count": len(flagged),
        "needs_preprocessing_case_ids": [result.case_id for result in flagged],
        "top_image_origin_diff_cases": _top_cases("image_origin_max_diff_mm"),
        "top_image_direction_diff_cases": _top_cases("image_direction_max_diff"),
        "top_label_origin_diff_cases": _top_cases("label_origin_max_diff_mm"),
        "top_label_direction_diff_cases": _top_cases("label_direction_max_diff"),
    }


def write_geometry_audit_artifacts(
    results: Sequence[GeometryAuditCaseResult],
    *,
    csv_path: str | Path,
    summary_path: str | Path,
    thresholds: GeometryAuditThresholds | None = None,
) -> dict[str, Path]:
    csv_rows = [result.to_dict() for result in results]
    summary = build_geometry_summary(results, thresholds=thresholds)
    csv_output = save_csv_rows(csv_rows, csv_path)
    summary_output = save_json(summary, summary_path)
    return {"csv": csv_output, "summary": summary_output}


def format_geometry_summary(summary: dict[str, Any]) -> str:
    lines = [
        "Geometry audit",
        f"  total_cases: {summary.get('total_cases', 0)}",
        f"  recommendation_counts: {summary.get('recommendation_counts', {})}",
        f"  needs_preprocessing_count: {summary.get('needs_preprocessing_count', 0)}",
        f"  split_counts: {summary.get('split_counts', {})}",
        f"  cohort_counts: {summary.get('cohort_counts', {})}",
    ]
    top_direction = summary.get("top_image_direction_diff_cases", [])
    if top_direction:
        lines.append("  worst_image_direction_case: "
                     f"{top_direction[0]['case_id']} ({top_direction[0]['image_direction_max_diff']})")
    top_origin = summary.get("top_image_origin_diff_cases", [])
    if top_origin:
        lines.append("  worst_image_origin_case: "
                     f"{top_origin[0]['case_id']} ({top_origin[0]['image_origin_max_diff_mm']} mm)")
    return "\n".join(lines)
