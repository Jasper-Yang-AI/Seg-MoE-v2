from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

import numpy as np

from segmoe_v2.contracts import CaseManifestRow
from segmoe_v2.manifest import (
    audit_manifest,
    audit_manifest_artifacts,
    build_case_manifest,
    build_manifest_summary_rows,
    export_nnformer_splits,
    export_nnunet_splits,
    scan_case_roots,
    write_manifest_artifacts,
)


def _write_case(root: Path, case_id: str) -> None:
    (root / f"{case_id}.nii.gz").touch()
    for idx in range(3):
        (root / f"{case_id}_000{idx}.nii.gz").touch()


class _FakeHeader:
    def get_zooms(self):
        return (1.0, 1.0, 1.0)


class _FakeImage:
    def __init__(self, label_values: list[int]):
        self.header = _FakeHeader()
        self.shape = (8, 9, 10)
        self.affine = np.eye(4, dtype=np.float32)
        self.dataobj = np.asarray(label_values, dtype=np.int16)


def _fake_load(path: str) -> _FakeImage:
    return _FakeImage([0, 1, 2, 3] if "pca" in path else [0, 1, 2])


def test_scan_build_summary_and_artifacts(tmp_path: Path) -> None:
    pca_root = tmp_path / "njmu_2012_2019_pca_zscore"
    nca_root = tmp_path / "njmu_2020_2023_nca_zscore"
    pca_root.mkdir()
    nca_root.mkdir()
    for idx in range(8):
        _write_case(pca_root, f"pca_{idx:03d}")
    for idx in range(8):
        _write_case(nca_root, f"nca_{idx:03d}")

    patient_map = {"pca_000": "patient_alpha"}
    with patch("segmoe_v2.manifest.nib.load", side_effect=_fake_load):
        scanned = scan_case_roots([pca_root, nca_root], patient_map=patient_map)
    assert len(scanned) == 16
    assert scanned[0].spacing == (1.0, 1.0, 1.0)
    assert scanned[0].image_shape == (10, 9, 8)
    assert any(row.patient_id == "patient_alpha" for row in scanned)
    assert any(row.has_lesion_label3 for row in scanned if row.cohort_type == "pca")
    assert all(row.label_unique_values == (0, 1, 2) for row in scanned if row.cohort_type == "nca")

    manifest = build_case_manifest(scanned, test_ratio=0.25, n_folds=5, seed=42)
    summary_rows = build_manifest_summary_rows(manifest)
    assert any(row["name"] == "total_cases" and row["count"] == 16 for row in summary_rows)
    assert any(row["section"] == "split_cohort" and row["group"] == "test" for row in summary_rows)

    manifest_path = tmp_path / "artifacts" / "manifest.jsonl"
    summary_path = tmp_path / "artifacts" / "manifest_summary.csv"
    nnunet_path = tmp_path / "artifacts" / "splits_final.json"
    nnformer_path = tmp_path / "artifacts" / "splits_final.pkl"
    write_manifest_artifacts(
        manifest,
        manifest_path=manifest_path,
        summary_path=summary_path,
        nnunet_splits_path=nnunet_path,
        nnformer_splits_path=nnformer_path,
    )

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        summary_csv = list(csv.DictReader(handle))
    assert any(row["name"] == "total_cases" and row["count"] == "16" for row in summary_csv)

    report = audit_manifest_artifacts(
        manifest_path=manifest_path,
        nnunet_splits_path=nnunet_path,
        nnformer_splits_path=nnformer_path,
    )
    assert not report.has_errors


def test_audit_detects_fold_mismatch() -> None:
    rows = [
        row
        for row in build_case_manifest(
            [
                *[
                    _row(f"pca_{idx:03d}", cohort_type="pca", val_fold=idx % 5)
                    for idx in range(10)
                ],
                *[
                    _row(f"nca_{idx:03d}", cohort_type="nca", val_fold=idx % 5)
                    for idx in range(10)
                ],
            ],
            test_ratio=0.0,
            n_folds=5,
            seed=42,
        )
    ]
    nnunet_splits = export_nnunet_splits(rows)
    nnformer_splits = export_nnformer_splits(rows)
    nnformer_splits[0]["val"] = list(nnformer_splits[1]["val"])
    report = audit_manifest(rows, nnunet_splits=nnunet_splits, nnformer_splits=nnformer_splits)
    assert report.has_errors
    assert any("mismatch on fold 0 val members" in message for message in report.errors)


def test_audit_detects_cohort_imbalance() -> None:
    rows = [
        _row("pca_001", cohort_type="pca", fixed_split="test", val_fold=None),
        _row("nca_001", cohort_type="nca", fixed_split="trainval", val_fold=0),
        _row("pca_002", cohort_type="pca", fixed_split="trainval", val_fold=0),
        _row("nca_002", cohort_type="nca", fixed_split="trainval", val_fold=1),
        _row("pca_003", cohort_type="pca", fixed_split="trainval", val_fold=1),
        _row("nca_003", cohort_type="nca", fixed_split="trainval", val_fold=2),
        _row("pca_004", cohort_type="pca", fixed_split="trainval", val_fold=2),
        _row("nca_004", cohort_type="nca", fixed_split="trainval", val_fold=3),
        _row("pca_005", cohort_type="pca", fixed_split="trainval", val_fold=3),
        _row("nca_005", cohort_type="nca", fixed_split="trainval", val_fold=4),
        _row("pca_006", cohort_type="pca", fixed_split="trainval", val_fold=4),
    ]
    report = audit_manifest(rows)
    assert report.has_errors
    assert any("does not contain both PCA and NCA" in message for message in report.errors)


def _row(
    case_id: str,
    *,
    cohort_type: str,
    fixed_split: str = "trainval",
    val_fold: int | None = 0,
):
    return CaseManifestRow(
        case_id=case_id,
        patient_id=case_id,
        era_bin="2012_2019",
        cohort_type=cohort_type,  # type: ignore[arg-type]
        has_lesion_label3=(cohort_type == "pca"),
        label_unique_values=(0, 1, 2, 3) if cohort_type == "pca" else (0, 1, 2),
        fixed_split=fixed_split,
        val_fold=val_fold,
        t2w_path=Path(f"{case_id}_0000.nii.gz"),
        adc_path=Path(f"{case_id}_0001.nii.gz"),
        dwi_path=Path(f"{case_id}_0002.nii.gz"),
        label_path=Path(f"{case_id}.nii.gz"),
        spacing=(1.0, 1.0, 1.0),
        image_shape=(8, 8, 8),
        affine_hash="abc",
    )
