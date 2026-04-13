from __future__ import annotations

import csv
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from .contracts import CaseManifestRow
from .geometry_audit import GeometryAuditThresholds
from .io_utils import ensure_parent, save_csv_rows, save_json, save_jsonl


GEOMETRY_FIX_RECOMMENDATIONS: tuple[str, ...] = (
    "header_harmonize_recommended",
    "resample_required",
)


@dataclass(frozen=True, slots=True)
class GeometryFixCaseResult:
    case_id: str
    recommendation: str
    t2w_action: str
    adc_action: str
    dwi_action: str
    label_action: str
    output_dir: str
    source_root: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_geometry_fix_root(manifest_path: str | Path) -> Path:
    manifest_path = Path(manifest_path)
    return manifest_path.parent.parent / "geometry_fixed"


def default_geometry_fixed_manifest_path(manifest_path: str | Path) -> Path:
    manifest_path = Path(manifest_path)
    return manifest_path.parent / f"{manifest_path.stem}.geometry_fixed{manifest_path.suffix}"


def default_geometry_fix_report_csv(output_root: str | Path) -> Path:
    return Path(output_root) / "geometry_fix_report.csv"


def default_geometry_fix_report_json(output_root: str | Path) -> Path:
    return Path(output_root) / "geometry_fix_report.json"


def load_geometry_audit_csv(path: str | Path) -> dict[str, dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {str(row["case_id"]): row for row in rows}


def _copy_file(source: str | Path, destination: str | Path, *, overwrite: bool = False) -> Path:
    source = Path(source)
    destination = Path(destination)
    ensure_parent(destination)
    if destination.exists():
        if not overwrite:
            return destination
        destination.unlink()
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _source_dtype(image: nib.spatialimages.SpatialImage) -> np.dtype[Any]:
    dtype = image.get_data_dtype()
    return np.dtype(dtype if dtype is not None else np.float32)


def _write_nifti_like_reference(
    data: np.ndarray,
    *,
    reference: nib.spatialimages.SpatialImage,
    dtype: np.dtype[Any],
    destination: str | Path,
) -> Path:
    destination = Path(destination)
    ensure_parent(destination)
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    image = nib.Nifti1Image(np.asarray(data, dtype=dtype), affine=reference.affine, header=header)
    nib.save(image, str(destination))
    return destination


def _header_harmonize_to_reference(
    *,
    source_path: str | Path,
    reference_path: str | Path,
    destination: str | Path,
) -> Path:
    source = nib.load(str(source_path))
    reference = nib.load(str(reference_path))
    data = np.asanyarray(source.dataobj)
    return _write_nifti_like_reference(
        data,
        reference=reference,
        dtype=_source_dtype(source),
        destination=destination,
    )


def _resample_to_reference(
    *,
    source_path: str | Path,
    reference_path: str | Path,
    destination: str | Path,
    is_label: bool,
) -> Path:
    source = nib.load(str(source_path))
    reference = nib.load(str(reference_path))
    order = 0 if is_label else 1
    resampled = resample_from_to(source, reference, order=order, mode="nearest", cval=0.0)
    if is_label:
        dtype = _source_dtype(source)
        data = np.rint(np.asanyarray(resampled.dataobj)).astype(dtype, copy=False)
    else:
        dtype = np.float32
        data = np.asanyarray(resampled.dataobj, dtype=np.float32)
    return _write_nifti_like_reference(
        data,
        reference=reference,
        dtype=dtype,
        destination=destination,
    )


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
        "shape": tuple(int(v) for v in image.shape[:3]),
        "spacing_xyz": tuple(float(v) for v in image.header.get_zooms()[:3]),
        "origin_xyz": tuple(float(v) for v in affine[:3, 3]),
        "direction": _direction_matrix(affine),
        "affine": affine,
        "axcodes": _canonical_axcodes(affine),
    }


def _max_abs_diff(left: Sequence[float] | np.ndarray, right: Sequence[float] | np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))))


def _decide_action_against_reference(
    *,
    source_path: str | Path,
    reference_path: str | Path,
    is_label: bool,
    thresholds: GeometryAuditThresholds,
) -> str:
    source = _geometry_payload(source_path)
    reference = _geometry_payload(reference_path)

    shape_match = source["shape"] == reference["shape"]
    axcodes_match = source["axcodes"] == reference["axcodes"]
    spacing_diff = _max_abs_diff(source["spacing_xyz"], reference["spacing_xyz"])
    origin_diff = _max_abs_diff(source["origin_xyz"], reference["origin_xyz"])
    direction_diff = _max_abs_diff(source["direction"], reference["direction"])

    if not shape_match or not axcodes_match:
        return "resample"
    if spacing_diff > thresholds.hard_spacing_mm:
        return "resample"
    if direction_diff > thresholds.hard_direction:
        return "resample"
    if origin_diff > thresholds.hard_origin_mm:
        return "resample"

    if spacing_diff > thresholds.soft_spacing_mm:
        return "header_harmonize"
    if direction_diff > thresholds.soft_direction:
        return "header_harmonize"
    if origin_diff > thresholds.soft_origin_mm:
        return "header_harmonize"

    return "copy"


def _apply_action(
    *,
    action: str,
    source_path: str | Path,
    reference_path: str | Path,
    destination: str | Path,
    is_label: bool,
    overwrite: bool,
) -> Path:
    destination = Path(destination)
    if destination.exists() and not overwrite:
        return destination
    if action == "copy":
        return _copy_file(source_path, destination, overwrite=overwrite)
    if action == "header_harmonize":
        return _header_harmonize_to_reference(
            source_path=source_path,
            reference_path=reference_path,
            destination=destination,
        )
    if action == "resample":
        return _resample_to_reference(
            source_path=source_path,
            reference_path=reference_path,
            destination=destination,
            is_label=is_label,
        )
    raise ValueError(f"Unsupported geometry action: {action}")


def _case_output_dir(row: CaseManifestRow, output_root: str | Path) -> Path:
    root_name = Path(str(row.metadata.get("root", "unknown_root"))).name
    return Path(output_root) / root_name


def fix_geometry_to_t2(
    rows: Iterable[CaseManifestRow],
    *,
    geometry_audit_rows: Mapping[str, Mapping[str, str]],
    output_root: str | Path,
    include_recommendations: Sequence[str] = GEOMETRY_FIX_RECOMMENDATIONS,
    thresholds: GeometryAuditThresholds | None = None,
    overwrite: bool = False,
) -> tuple[list[CaseManifestRow], list[GeometryFixCaseResult]]:
    thresholds = thresholds or GeometryAuditThresholds()
    include = {str(item) for item in include_recommendations}
    patched_rows: list[CaseManifestRow] = []
    reports: list[GeometryFixCaseResult] = []

    for row in rows:
        audit_row = geometry_audit_rows.get(row.case_id)
        recommendation = str(audit_row.get("recommendation", "no_action")) if audit_row else "no_action"
        if recommendation not in include:
            patched_rows.append(row)
            continue

        case_dir = _case_output_dir(row, output_root)
        case_dir.mkdir(parents=True, exist_ok=True)
        t2_out = case_dir / f"{row.case_id}_0000.nii.gz"
        adc_out = case_dir / f"{row.case_id}_0001.nii.gz"
        dwi_out = case_dir / f"{row.case_id}_0002.nii.gz"
        label_out = case_dir / f"{row.case_id}.nii.gz"

        t2_action = "copy"
        adc_action = _decide_action_against_reference(
            source_path=row.adc_path,
            reference_path=row.t2w_path,
            is_label=False,
            thresholds=thresholds,
        )
        dwi_action = _decide_action_against_reference(
            source_path=row.dwi_path,
            reference_path=row.t2w_path,
            is_label=False,
            thresholds=thresholds,
        )
        label_action = _decide_action_against_reference(
            source_path=row.label_path,
            reference_path=row.t2w_path,
            is_label=True,
            thresholds=thresholds,
        )

        _apply_action(
            action=t2_action,
            source_path=row.t2w_path,
            reference_path=row.t2w_path,
            destination=t2_out,
            is_label=False,
            overwrite=overwrite,
        )
        _apply_action(
            action=adc_action,
            source_path=row.adc_path,
            reference_path=row.t2w_path,
            destination=adc_out,
            is_label=False,
            overwrite=overwrite,
        )
        _apply_action(
            action=dwi_action,
            source_path=row.dwi_path,
            reference_path=row.t2w_path,
            destination=dwi_out,
            is_label=False,
            overwrite=overwrite,
        )
        _apply_action(
            action=label_action,
            source_path=row.label_path,
            reference_path=row.t2w_path,
            destination=label_out,
            is_label=True,
            overwrite=overwrite,
        )

        metadata = dict(row.metadata)
        metadata.update(
            {
                "geometry_fix_applied": True,
                "geometry_fix_recommendation": recommendation,
                "geometry_fix_output_root": str(case_dir),
                "geometry_fix_actions": {
                    "t2w": t2_action,
                    "adc": adc_action,
                    "dwi": dwi_action,
                    "label": label_action,
                },
            }
        )
        patched_rows.append(
            replace(
                row,
                t2w_path=t2_out,
                adc_path=adc_out,
                dwi_path=dwi_out,
                label_path=label_out,
                metadata=metadata,
            )
        )
        reports.append(
            GeometryFixCaseResult(
                case_id=row.case_id,
                recommendation=recommendation,
                t2w_action=t2_action,
                adc_action=adc_action,
                dwi_action=dwi_action,
                label_action=label_action,
                output_dir=str(case_dir),
                source_root=str(row.metadata.get("root", "")),
            )
        )

    return patched_rows, reports


def build_geometry_fix_summary(
    reports: Sequence[GeometryFixCaseResult],
    *,
    include_recommendations: Sequence[str],
    output_root: str | Path,
    manifest_out: str | Path,
) -> dict[str, Any]:
    recommendation_counts: dict[str, int] = {}
    action_counts = {
        "t2w": {},
        "adc": {},
        "dwi": {},
        "label": {},
    }
    for report in reports:
        recommendation_counts[report.recommendation] = recommendation_counts.get(report.recommendation, 0) + 1
        for key, value in (
            ("t2w", report.t2w_action),
            ("adc", report.adc_action),
            ("dwi", report.dwi_action),
            ("label", report.label_action),
        ):
            counter = action_counts[key]
            counter[value] = counter.get(value, 0) + 1

    return {
        "fixed_case_count": len(reports),
        "fixed_case_ids": [report.case_id for report in reports],
        "recommendations_included": list(include_recommendations),
        "recommendation_counts": dict(sorted(recommendation_counts.items())),
        "action_counts": {key: dict(sorted(value.items())) for key, value in action_counts.items()},
        "output_root": str(output_root),
        "manifest_out": str(manifest_out),
    }


def write_geometry_fix_artifacts(
    patched_rows: Sequence[CaseManifestRow],
    reports: Sequence[GeometryFixCaseResult],
    *,
    manifest_out: str | Path,
    report_csv_out: str | Path,
    report_json_out: str | Path,
    include_recommendations: Sequence[str],
    output_root: str | Path,
) -> dict[str, Path]:
    manifest_path = save_jsonl((row.to_dict() for row in patched_rows), manifest_out)
    report_csv = save_csv_rows((report.to_dict() for report in reports), report_csv_out)
    report_json = save_json(
        build_geometry_fix_summary(
            reports,
            include_recommendations=include_recommendations,
            output_root=output_root,
            manifest_out=manifest_out,
        ),
        report_json_out,
    )
    return {
        "manifest": manifest_path,
        "report_csv": report_csv,
        "report_json": report_json,
    }
