from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    import nibabel as nib
except ModuleNotFoundError:
    class _MissingNibabel:
        def _raise(self, *_args: Any, **_kwargs: Any) -> None:
            raise ModuleNotFoundError(
                "nibabel is required for scanning image roots. Install the project runtime dependencies first."
            )

        def load(self, *_args: Any, **_kwargs: Any) -> None:
            self._raise()

    nib = _MissingNibabel()

from .contracts import CaseManifestRow
from .io_utils import load_json, load_jsonl, load_pickle, save_csv_rows, save_json, save_jsonl, save_pickle


_ROOT_RE = re.compile(r"(?P<era>\d{4}_\d{4}).*(?P<cohort>pca|nca)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ManifestAuditReport:
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)


def _strip_nii_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def _parse_root_metadata(root: Path) -> tuple[str, str]:
    match = _ROOT_RE.search(root.name)
    if not match:
        raise ValueError(f"Could not infer era/cohort from root name: {root}")
    return match.group("era"), match.group("cohort").lower()


def _normalise_label_values(values: np.ndarray) -> tuple[int, ...]:
    unique = np.unique(np.asarray(values))
    return tuple(sorted({int(round(float(v))) for v in unique.tolist()}))


def _extract_label_values(image: Any) -> tuple[int, ...]:
    if hasattr(image, "dataobj"):
        label_array = np.asarray(image.dataobj)
    elif hasattr(image, "get_fdata"):
        label_array = np.asarray(image.get_fdata())
    else:
        raise TypeError("Loaded image does not expose `dataobj` or `get_fdata` for label statistics.")
    return _normalise_label_values(label_array)


def load_patient_map_csv(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    mapping: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = {str(name).strip() for name in reader.fieldnames or ()}
        if not {"case_id", "patient_id"}.issubset(fieldnames):
            raise ValueError("Patient map CSV must contain `case_id` and `patient_id` columns.")
        for row in reader:
            case_id = str(row.get("case_id", "")).strip()
            patient_id = str(row.get("patient_id", "")).strip()
            if not case_id or not patient_id:
                continue
            mapping[case_id] = patient_id
    return mapping


def _scan_single_root(
    root: Path,
    *,
    patient_map: Mapping[str, str] | None = None,
    patient_map_source: str | Path | None = None,
) -> list[CaseManifestRow]:
    era_bin, cohort_type = _parse_root_metadata(root)
    patient_map = patient_map or {}
    rows: list[CaseManifestRow] = []

    label_files = sorted(
        p for p in root.glob("*.nii*")
        if "_0000" not in p.name and "_0001" not in p.name and "_0002" not in p.name
    )
    for label_path in label_files:
        case_id = _strip_nii_suffix(label_path.name)
        t2w_path = root / f"{case_id}_0000.nii.gz"
        adc_path = root / f"{case_id}_0001.nii.gz"
        dwi_path = root / f"{case_id}_0002.nii.gz"
        if not (t2w_path.exists() and adc_path.exists() and dwi_path.exists()):
            continue

        image = nib.load(str(label_path))
        spacing = tuple(float(v) for v in image.header.get_zooms()[:3][::-1])
        image_shape = tuple(int(v) for v in image.shape[:3][::-1])
        affine_hash = np.asarray(image.affine, dtype=np.float32).tobytes().hex()[:24]
        label_unique_values = _extract_label_values(image)
        metadata: dict[str, Any] = {"root": str(root)}
        if patient_map_source is not None:
            metadata["patient_map_source"] = str(patient_map_source)
        rows.append(
            CaseManifestRow(
                case_id=case_id,
                patient_id=str(patient_map.get(case_id, case_id)),
                era_bin=era_bin,
                cohort_type=cohort_type,  # type: ignore[arg-type]
                has_lesion_label3=3 in label_unique_values,
                label_unique_values=label_unique_values,
                t2w_path=t2w_path,
                adc_path=adc_path,
                dwi_path=dwi_path,
                label_path=label_path,
                spacing=spacing,
                image_shape=image_shape,
                affine_hash=affine_hash,
                metadata=metadata,
            )
        )
    return rows


def scan_case_roots(
    roots: Iterable[str | Path],
    *,
    patient_map_path: str | Path | None = None,
    patient_map: Mapping[str, str] | None = None,
) -> list[CaseManifestRow]:
    resolved_patient_map = dict(load_patient_map_csv(patient_map_path))
    if patient_map:
        resolved_patient_map.update({str(k): str(v) for k, v in patient_map.items()})

    rows: list[CaseManifestRow] = []
    for root in roots:
        rows.extend(
            _scan_single_root(
                Path(root),
                patient_map=resolved_patient_map,
                patient_map_source=patient_map_path,
            )
        )
    return sorted(rows, key=lambda row: (row.era_bin, row.cohort_type, row.patient_id, row.case_id))


def build_case_manifest(
    discovered_rows: Iterable[CaseManifestRow],
    *,
    test_ratio: float = 0.15,
    n_folds: int = 5,
    seed: int = 42,
) -> list[CaseManifestRow]:
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError(f"test_ratio must be in [0,1), got {test_ratio}")
    if n_folds <= 0:
        raise ValueError(f"n_folds must be positive, got {n_folds}")

    rows = list(discovered_rows)
    by_stratum: dict[tuple[str, str], dict[str, list[CaseManifestRow]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_stratum[(row.era_bin, row.cohort_type)][row.patient_id].append(row)

    rng = np.random.default_rng(seed)
    assigned: list[CaseManifestRow] = []

    for patient_groups in by_stratum.values():
        groups = list(patient_groups.values())
        rng.shuffle(groups)
        if len(groups) >= max(5, n_folds) and test_ratio > 0:
            n_test = max(1, int(round(len(groups) * test_ratio)))
        else:
            n_test = 0

        for group in groups[:n_test]:
            for row in group:
                assigned.append(replace(row, fixed_split="test", val_fold=None))
        for index, group in enumerate(groups[n_test:]):
            fold = index % n_folds
            for row in group:
                assigned.append(replace(row, fixed_split="trainval", val_fold=fold))

    return sorted(assigned, key=lambda row: row.case_id)


def _export_fold_splits(rows: Iterable[CaseManifestRow]) -> list[dict[str, list[str]]]:
    rows = list(rows)
    folds = sorted({row.val_fold for row in rows if row.fixed_split == "trainval" and row.val_fold is not None})
    split_dicts: list[dict[str, list[str]]] = []
    for fold in folds:
        train = sorted(row.case_id for row in rows if row.fixed_split == "trainval" and row.val_fold != fold)
        val = sorted(row.case_id for row in rows if row.fixed_split == "trainval" and row.val_fold == fold)
        split_dicts.append({"train": train, "val": val})
    return split_dicts


def export_nnunet_splits(rows: Iterable[CaseManifestRow]) -> list[dict[str, list[str]]]:
    return _export_fold_splits(rows)


def export_mednext_splits(rows: Iterable[CaseManifestRow]) -> list[dict[str, list[str]]]:
    return _export_fold_splits(rows)


def export_segmamba_splits(rows: Iterable[CaseManifestRow]) -> list[dict[str, list[str]]]:
    return _export_fold_splits(rows)


def _count_by(rows: Sequence[CaseManifestRow], attr: str) -> dict[str, int]:
    counts = Counter(str(getattr(row, attr)) for row in rows)
    return dict(sorted(counts.items()))


def build_manifest_summary_rows(rows: Iterable[CaseManifestRow]) -> list[dict[str, Any]]:
    materialized = list(rows)
    total_cases = len(materialized)
    output: list[dict[str, Any]] = []

    def add(section: str, group: str, name: str, count: int, ratio: float | None = None) -> None:
        output.append(
            {
                "section": section,
                "group": group,
                "name": name,
                "count": int(count),
                "ratio": "" if ratio is None else round(float(ratio), 6),
            }
        )

    add("overall", "all", "total_cases", total_cases, 1.0 if total_cases else None)
    for cohort, count in _count_by(materialized, "cohort_type").items():
        add("cohort", "all", cohort, count, count / total_cases if total_cases else None)
    for era, count in _count_by(materialized, "era_bin").items():
        add("era", "all", era, count, count / total_cases if total_cases else None)
    for split, count in _count_by(materialized, "fixed_split").items():
        add("split", "all", split, count, count / total_cases if total_cases else None)

    folds = sorted({row.val_fold for row in materialized if row.fixed_split == "trainval" and row.val_fold is not None})
    for fold in folds:
        fold_rows = [row for row in materialized if row.fixed_split == "trainval" and row.val_fold == fold]
        add("fold_val", f"fold_{fold}", "cases", len(fold_rows), len(fold_rows) / total_cases if total_cases else None)

    split_groups: dict[str, list[CaseManifestRow]] = {
        "test": [row for row in materialized if row.fixed_split == "test"],
    }
    for fold in folds:
        split_groups[f"fold_{fold}_val"] = [
            row for row in materialized if row.fixed_split == "trainval" and row.val_fold == fold
        ]
    for split_name, split_rows in split_groups.items():
        split_total = len(split_rows)
        for cohort, count in _count_by(split_rows, "cohort_type").items():
            add(
                "split_cohort",
                split_name,
                cohort,
                count,
                count / split_total if split_total else None,
            )

    return output


def default_summary_path(manifest_path: str | Path) -> Path:
    manifest_path = Path(manifest_path)
    return manifest_path.parent / "manifest_summary.csv"


def write_manifest_summary(rows: Iterable[CaseManifestRow], path: str | Path) -> Path:
    summary_rows = build_manifest_summary_rows(rows)
    return save_csv_rows(
        summary_rows,
        path,
        fieldnames=("section", "group", "name", "count", "ratio"),
    )


def load_case_manifest(path: str | Path) -> list[CaseManifestRow]:
    return [CaseManifestRow.from_dict(row) for row in load_jsonl(path)]


def load_nnunet_splits(path: str | Path) -> list[dict[str, list[str]]]:
    payload = load_json(path)
    return [{"train": list(item.get("train", [])), "val": list(item.get("val", []))} for item in payload]


def load_mednext_splits(path: str | Path) -> list[dict[str, list[str]]]:
    payload = load_pickle(path)
    return [{"train": list(item.get("train", [])), "val": list(item.get("val", []))} for item in payload]


def load_segmamba_splits(path: str | Path) -> list[dict[str, list[str]]]:
    payload = load_json(path)
    return [{"train": list(item.get("train", [])), "val": list(item.get("val", []))} for item in payload]


def _compare_split_payloads(
    *,
    left_name: str,
    left_splits: Sequence[Mapping[str, Sequence[str]]],
    right_name: str,
    right_splits: Sequence[Mapping[str, Sequence[str]]],
    errors: list[str],
) -> None:
    if len(left_splits) != len(right_splits):
        errors.append(
            f"{left_name} has {len(left_splits)} folds while {right_name} has {len(right_splits)} folds."
        )
        return

    for fold_index, (left, right) in enumerate(zip(left_splits, right_splits, strict=True)):
        for split_name in ("train", "val"):
            left_members = sorted(str(v) for v in left.get(split_name, ()))
            right_members = sorted(str(v) for v in right.get(split_name, ()))
            if left_members != right_members:
                errors.append(
                    f"{left_name} and {right_name} mismatch on fold {fold_index} {split_name} members."
                )


def _audit_cohort_balance(
    rows: Sequence[CaseManifestRow],
    *,
    n_folds: int,
    errors: list[str],
    warnings: list[str],
) -> None:
    global_counts = Counter(row.cohort_type for row in rows)
    total = len(rows)
    if not total or len(global_counts) < 2:
        return

    global_ratios = {cohort: count / total for cohort, count in global_counts.items()}
    split_groups: dict[str, list[CaseManifestRow]] = {
        "test": [row for row in rows if row.fixed_split == "test"],
    }
    for fold in range(n_folds):
        split_groups[f"fold_{fold}_val"] = [
            row for row in rows if row.fixed_split == "trainval" and row.val_fold == fold
        ]

    for split_name, split_rows in split_groups.items():
        if not split_rows:
            errors.append(f"{split_name} contains zero cases.")
            continue
        split_counts = Counter(row.cohort_type for row in split_rows)
        if any(split_counts.get(cohort, 0) == 0 for cohort in global_counts):
            errors.append(f"{split_name} does not contain both PCA and NCA cases.")
        split_total = len(split_rows)
        for cohort, global_count in global_counts.items():
            local_count = split_counts.get(cohort, 0)
            if local_count == 0 and global_count >= n_folds + 1:
                errors.append(
                    f"{split_name} is missing cohort {cohort} even though the full manifest has {global_count} cases."
                )
            local_ratio = local_count / split_total
            if abs(local_ratio - global_ratios[cohort]) > 0.25:
                warnings.append(
                    f"{split_name} cohort ratio for {cohort} deviates from global ratio by more than 0.25."
                )


def audit_manifest(
    rows: Iterable[CaseManifestRow],
    *,
    nnunet_splits: Sequence[Mapping[str, Sequence[str]]] | None = None,
    mednext_splits: Sequence[Mapping[str, Sequence[str]]] | None = None,
    segmamba_splits: Sequence[Mapping[str, Sequence[str]]] | None = None,
) -> ManifestAuditReport:
    materialized = list(rows)
    errors: list[str] = []
    warnings: list[str] = []
    case_ids = [row.case_id for row in materialized]
    patient_ids = [row.patient_id for row in materialized]
    duplicate_cases = sorted({case_id for case_id, count in Counter(case_ids).items() if count > 1})
    duplicate_patients = sorted({patient_id for patient_id, count in Counter(patient_ids).items() if count > 1})

    if duplicate_cases:
        errors.append(f"Duplicate case_id values found: {', '.join(duplicate_cases)}")

    for row in materialized:
        for path_name in ("t2w_path", "adc_path", "dwi_path", "label_path"):
            if not Path(getattr(row, path_name)).exists():
                errors.append(f"{row.case_id} is missing required file `{path_name}`.")

    for row in materialized:
        if row.fixed_split == "trainval" and row.val_fold is None:
            errors.append(f"{row.case_id} is trainval but has no val_fold.")
        if row.fixed_split == "test" and row.val_fold is not None:
            errors.append(f"{row.case_id} is test but still has val_fold={row.val_fold}.")

    test_patients = {row.patient_id for row in materialized if row.fixed_split == "test"}
    trainval_patients = {row.patient_id for row in materialized if row.fixed_split == "trainval"}
    overlap_with_test = sorted(test_patients & trainval_patients)
    if overlap_with_test:
        errors.append(f"Patient leakage between test and trainval: {', '.join(overlap_with_test)}")

    folds = sorted({row.val_fold for row in materialized if row.fixed_split == "trainval" and row.val_fold is not None})
    for fold in folds:
        train_patients = {
            row.patient_id for row in materialized if row.fixed_split == "trainval" and row.val_fold != fold
        }
        val_patients = {
            row.patient_id for row in materialized if row.fixed_split == "trainval" and row.val_fold == fold
        }
        leakage = sorted(train_patients & val_patients)
        if leakage:
            errors.append(f"Patient leakage detected between train and val for fold {fold}: {', '.join(leakage)}")

    _audit_cohort_balance(materialized, n_folds=len(folds), errors=errors, warnings=warnings)

    expected_nnunet = export_nnunet_splits(materialized)
    expected_mednext = export_mednext_splits(materialized)
    expected_segmamba = export_segmamba_splits(materialized)
    if nnunet_splits is not None:
        _compare_split_payloads(
            left_name="manifest-derived nnUNet",
            left_splits=expected_nnunet,
            right_name="provided nnUNet",
            right_splits=nnunet_splits,
            errors=errors,
        )
    if mednext_splits is not None:
        _compare_split_payloads(
            left_name="manifest-derived MedNeXt",
            left_splits=expected_mednext,
            right_name="provided MedNeXt",
            right_splits=mednext_splits,
            errors=errors,
        )
    if segmamba_splits is not None:
        _compare_split_payloads(
            left_name="manifest-derived SegMamba",
            left_splits=expected_segmamba,
            right_name="provided SegMamba",
            right_splits=segmamba_splits,
            errors=errors,
        )
    if nnunet_splits is not None and mednext_splits is not None:
        _compare_split_payloads(
            left_name="nnUNet",
            left_splits=nnunet_splits,
            right_name="MedNeXt",
            right_splits=mednext_splits,
            errors=errors,
        )
    if nnunet_splits is not None and segmamba_splits is not None:
        _compare_split_payloads(
            left_name="nnUNet",
            left_splits=nnunet_splits,
            right_name="SegMamba",
            right_splits=segmamba_splits,
            errors=errors,
        )
    if nnunet_splits is None and mednext_splits is not None and segmamba_splits is not None:
        _compare_split_payloads(
            left_name="MedNeXt",
            left_splits=mednext_splits,
            right_name="SegMamba",
            right_splits=segmamba_splits,
            errors=errors,
        )
    stats = {
        "total_cases": len(materialized),
        "total_patients": len(set(patient_ids)),
        "duplicate_case_ids": duplicate_cases,
        "duplicate_patient_ids": duplicate_patients,
        "fold_count": len(folds),
        "split_counts": _count_by(materialized, "fixed_split"),
        "cohort_counts": _count_by(materialized, "cohort_type"),
        "era_counts": _count_by(materialized, "era_bin"),
    }
    return ManifestAuditReport(errors=tuple(errors), warnings=tuple(warnings), stats=stats)


def format_audit_report(report: ManifestAuditReport) -> str:
    lines = [
        "Manifest audit",
        f"  total_cases: {report.stats.get('total_cases', 0)}",
        f"  total_patients: {report.stats.get('total_patients', 0)}",
        f"  fold_count: {report.stats.get('fold_count', 0)}",
    ]
    split_counts = report.stats.get("split_counts", {})
    cohort_counts = report.stats.get("cohort_counts", {})
    era_counts = report.stats.get("era_counts", {})
    lines.append(f"  split_counts: {split_counts}")
    lines.append(f"  cohort_counts: {cohort_counts}")
    lines.append(f"  era_counts: {era_counts}")
    if report.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in report.warnings)
    if report.errors:
        lines.append("Errors:")
        lines.extend(f"  - {error}" for error in report.errors)
    else:
        lines.append("No audit errors.")
    return "\n".join(lines)


def audit_manifest_artifacts(
    *,
    manifest_path: str | Path,
    nnunet_splits_path: str | Path | None = None,
    mednext_splits_path: str | Path | None = None,
    segmamba_splits_path: str | Path | None = None,
) -> ManifestAuditReport:
    rows = load_case_manifest(manifest_path)
    nnunet_splits = load_nnunet_splits(nnunet_splits_path) if nnunet_splits_path is not None else None
    mednext_splits = load_mednext_splits(mednext_splits_path) if mednext_splits_path is not None else None
    segmamba_splits = load_segmamba_splits(segmamba_splits_path) if segmamba_splits_path is not None else None
    return audit_manifest(
        rows,
        nnunet_splits=nnunet_splits,
        mednext_splits=mednext_splits,
        segmamba_splits=segmamba_splits,
    )


def write_manifest_artifacts(
    rows: Iterable[CaseManifestRow],
    *,
    manifest_path: str | Path,
    summary_path: str | Path | None = None,
    nnunet_splits_path: str | Path | None = None,
    mednext_splits_path: str | Path | None = None,
    segmamba_splits_path: str | Path | None = None,
) -> dict[str, Path]:
    rows = list(rows)
    if summary_path is None:
        summary_path = default_summary_path(manifest_path)

    outputs: dict[str, Path] = {
        "manifest": save_jsonl((row.to_dict() for row in rows), manifest_path),
        "summary": write_manifest_summary(rows, summary_path),
    }
    if nnunet_splits_path is not None:
        outputs["nnunet_splits"] = save_json(export_nnunet_splits(rows), nnunet_splits_path)
    if mednext_splits_path is not None:
        outputs["mednext_splits"] = save_pickle(export_mednext_splits(rows), mednext_splits_path)
    if segmamba_splits_path is not None:
        outputs["segmamba_splits"] = save_json(export_segmamba_splits(rows), segmamba_splits_path)
    return outputs
