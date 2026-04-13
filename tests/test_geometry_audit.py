from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from segmoe_v2.cli.main import main
from segmoe_v2.contracts import CaseManifestRow
from segmoe_v2.geometry_audit import GeometryAuditThresholds, audit_geometry
from segmoe_v2.io_utils import save_jsonl


class _FakeHeader:
    def __init__(self, zooms: tuple[float, float, float]) -> None:
        self._zooms = zooms

    def get_zooms(self) -> tuple[float, float, float]:
        return self._zooms


class _FakeImage:
    def __init__(self, *, shape: tuple[int, int, int], affine: np.ndarray, zooms: tuple[float, float, float]) -> None:
        self.shape = shape
        self.affine = affine
        self.header = _FakeHeader(zooms)


def _row(case_id: str) -> CaseManifestRow:
    return CaseManifestRow(
        case_id=case_id,
        patient_id=case_id,
        era_bin="2012_2019",
        cohort_type="pca",
        has_lesion_label3=True,
        label_unique_values=(0, 1, 2, 3),
        fixed_split="trainval",
        val_fold=0,
        t2w_path=Path(f"{case_id}_0000.nii.gz"),
        adc_path=Path(f"{case_id}_0001.nii.gz"),
        dwi_path=Path(f"{case_id}_0002.nii.gz"),
        label_path=Path(f"{case_id}.nii.gz"),
        spacing=(1.0, 1.0, 1.0),
        image_shape=(4, 4, 4),
        affine_hash="abc",
    )


def _make_image(*, affine: np.ndarray, shape: tuple[int, int, int] = (4, 4, 4), zooms: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> _FakeImage:
    return _FakeImage(shape=shape, affine=np.asarray(affine, dtype=np.float64), zooms=zooms)


def test_audit_geometry_classifies_cases() -> None:
    clean = _row("clean_case")
    header = _row("header_case")
    resample = _row("resample_case")

    base_affine = np.eye(4, dtype=np.float64)
    header_adc_affine = base_affine.copy()
    header_adc_affine[:3, 3] = (0.05, 0.0, 0.0)
    flipped_affine = np.diag([-1.0, 1.0, 1.0, 1.0]).astype(np.float64)

    payloads = {
        str(clean.t2w_path): _make_image(affine=base_affine),
        str(clean.adc_path): _make_image(affine=base_affine),
        str(clean.dwi_path): _make_image(affine=base_affine),
        str(clean.label_path): _make_image(affine=base_affine),
        str(header.t2w_path): _make_image(affine=base_affine),
        str(header.adc_path): _make_image(affine=header_adc_affine),
        str(header.dwi_path): _make_image(affine=base_affine),
        str(header.label_path): _make_image(affine=base_affine),
        str(resample.t2w_path): _make_image(affine=base_affine),
        str(resample.adc_path): _make_image(affine=flipped_affine),
        str(resample.dwi_path): _make_image(affine=base_affine),
        str(resample.label_path): _make_image(affine=base_affine),
    }

    with patch("segmoe_v2.geometry_audit.nib.load", side_effect=lambda path: payloads[str(path)]):
        results = audit_geometry([clean, header, resample], thresholds=GeometryAuditThresholds())

    by_case = {result.case_id: result for result in results}
    assert by_case["clean_case"].recommendation == "no_action"
    assert not by_case["clean_case"].needs_preprocessing
    assert by_case["header_case"].recommendation == "header_harmonize_recommended"
    assert by_case["header_case"].needs_preprocessing
    assert by_case["resample_case"].recommendation == "resample_required"
    assert by_case["resample_case"].needs_preprocessing
    assert not by_case["resample_case"].image_axcodes_match


def test_audit_geometry_cli_writes_outputs(tmp_path: Path) -> None:
    case = _row("case_001")
    manifest_path = tmp_path / "manifest" / "cases.jsonl"
    save_jsonl([case.to_dict()], manifest_path)

    base_affine = np.eye(4, dtype=np.float64)
    payloads = {
        str(case.t2w_path): _make_image(affine=base_affine),
        str(case.adc_path): _make_image(affine=base_affine),
        str(case.dwi_path): _make_image(affine=base_affine),
        str(case.label_path): _make_image(affine=base_affine),
    }

    csv_path = tmp_path / "manifest" / "geometry_audit.csv"
    summary_path = tmp_path / "manifest" / "geometry_audit_summary.json"
    with patch("segmoe_v2.geometry_audit.nib.load", side_effect=lambda path: payloads[str(path)]):
        main(
            [
                "audit-geometry",
                "--manifest",
                str(manifest_path),
                "--csv-out",
                str(csv_path),
                "--summary-out",
                str(summary_path),
            ]
        )

    assert csv_path.exists()
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total_cases"] == 1
    assert summary["recommendation_counts"]["no_action"] == 1
