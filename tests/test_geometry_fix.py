from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch

from segmoe_v2.cli.main import main
from segmoe_v2.contracts import CaseManifestRow
from segmoe_v2.geometry_audit import GeometryAuditThresholds
from segmoe_v2.geometry_fix import _decide_action_against_reference, fix_geometry_to_t2
from segmoe_v2.io_utils import save_jsonl


def _row(case_id: str, root: Path) -> CaseManifestRow:
    return CaseManifestRow(
        case_id=case_id,
        patient_id=case_id,
        era_bin="2012_2019",
        cohort_type="pca",
        has_lesion_label3=True,
        label_unique_values=(0, 1, 2, 3),
        fixed_split="trainval",
        val_fold=0,
        t2w_path=root / f"{case_id}_0000.nii.gz",
        adc_path=root / f"{case_id}_0001.nii.gz",
        dwi_path=root / f"{case_id}_0002.nii.gz",
        label_path=root / f"{case_id}.nii.gz",
        spacing=(1.0, 1.0, 1.0),
        image_shape=(4, 4, 4),
        affine_hash="abc",
        metadata={"root": str(root)},
    )


def _touch_case_files(row: CaseManifestRow) -> None:
    for path in (row.t2w_path, row.adc_path, row.dwi_path, row.label_path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")


def test_decide_action_against_reference_distinguishes_copy_header_and_resample() -> None:
    payloads = {
        "t2": {
            "shape": (4, 4, 4),
            "axcodes": "RAI",
            "spacing_xyz": (1.0, 1.0, 1.0),
            "origin_xyz": (0.0, 0.0, 0.0),
            "direction": (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
        },
        "copy": {
            "shape": (4, 4, 4),
            "axcodes": "RAI",
            "spacing_xyz": (1.0, 1.0, 1.0),
            "origin_xyz": (0.0, 0.0, 0.0),
            "direction": (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
        },
        "header": {
            "shape": (4, 4, 4),
            "axcodes": "RAI",
            "spacing_xyz": (1.0, 1.0, 1.0),
            "origin_xyz": (0.05, 0.0, 0.0),
            "direction": (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
        },
        "resample": {
            "shape": (4, 4, 4),
            "axcodes": "RAI",
            "spacing_xyz": (1.0, 1.0, 1.0),
            "origin_xyz": (0.2, 0.0, 0.0),
            "direction": (
                (1.0, 0.002, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
        },
    }

    with patch("segmoe_v2.geometry_fix._geometry_payload", side_effect=lambda path: payloads[str(path)]):
        assert _decide_action_against_reference(
            source_path="copy",
            reference_path="t2",
            is_label=False,
            thresholds=GeometryAuditThresholds(),
        ) == "copy"
        assert _decide_action_against_reference(
            source_path="header",
            reference_path="t2",
            is_label=False,
            thresholds=GeometryAuditThresholds(),
        ) == "header_harmonize"
        assert _decide_action_against_reference(
            source_path="resample",
            reference_path="t2",
            is_label=False,
            thresholds=GeometryAuditThresholds(),
        ) == "resample"


def test_fix_geometry_to_t2_repairs_flagged_cases(tmp_path: Path) -> None:
    source_root = tmp_path / "source_root"
    clean = _row("clean_case", source_root)
    header = _row("header_case", source_root)
    resample = _row("resample_case", source_root)
    for row in (clean, header, resample):
        _touch_case_files(row)

    geometry_rows = {
        header.case_id: {"recommendation": "header_harmonize_recommended"},
        resample.case_id: {"recommendation": "resample_required"},
    }

    def fake_decide(*, source_path: str | Path, reference_path: str | Path, is_label: bool, thresholds: GeometryAuditThresholds) -> str:
        name = Path(str(source_path)).name
        if name.startswith("header_case_0001"):
            return "header_harmonize"
        if name.startswith("resample_case_0001"):
            return "resample"
        return "copy"

    def fake_apply(
        *,
        action: str,
        source_path: str | Path,
        reference_path: str | Path,
        destination: str | Path,
        is_label: bool,
        overwrite: bool,
    ) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(f"{action}|{Path(str(source_path)).name}", encoding="utf-8")
        return destination

    with patch("segmoe_v2.geometry_fix._decide_action_against_reference", side_effect=fake_decide):
        with patch("segmoe_v2.geometry_fix._apply_action", side_effect=fake_apply):
            patched_rows, reports = fix_geometry_to_t2(
                [clean, header, resample],
                geometry_audit_rows=geometry_rows,
                output_root=tmp_path / "geometry_fixed",
                thresholds=GeometryAuditThresholds(),
            )

    by_case = {row.case_id: row for row in patched_rows}
    report_by_case = {report.case_id: report for report in reports}

    assert by_case["clean_case"].adc_path == clean.adc_path
    assert report_by_case["header_case"].adc_action == "header_harmonize"
    assert report_by_case["resample_case"].adc_action == "resample"
    assert Path(by_case["header_case"].adc_path).exists()
    assert Path(by_case["resample_case"].adc_path).exists()
    assert by_case["header_case"].metadata["geometry_fix_applied"] is True
    assert by_case["header_case"].metadata["geometry_fix_actions"]["adc"] == "header_harmonize"
    assert by_case["resample_case"].metadata["geometry_fix_actions"]["adc"] == "resample"


def test_fix_geometry_cli_writes_manifest_and_reports(tmp_path: Path) -> None:
    source_root = tmp_path / "source_root"
    case = _row("header_case", source_root)
    _touch_case_files(case)

    manifest_path = tmp_path / "manifest" / "cases.jsonl"
    save_jsonl([case.to_dict()], manifest_path)

    audit_csv = tmp_path / "manifest" / "geometry_audit.csv"
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    with audit_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "recommendation"])
        writer.writeheader()
        writer.writerow({"case_id": case.case_id, "recommendation": "header_harmonize_recommended"})

    output_root = tmp_path / "geometry_fixed"
    manifest_out = tmp_path / "manifest" / "cases.geometry_fixed.jsonl"
    report_csv_out = output_root / "geometry_fix_report.csv"
    report_json_out = output_root / "geometry_fix_report.json"

    def fake_decide(*, source_path: str | Path, reference_path: str | Path, is_label: bool, thresholds: GeometryAuditThresholds) -> str:
        if Path(str(source_path)).name.startswith("header_case_0001"):
            return "header_harmonize"
        return "copy"

    def fake_apply(
        *,
        action: str,
        source_path: str | Path,
        reference_path: str | Path,
        destination: str | Path,
        is_label: bool,
        overwrite: bool,
    ) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(f"{action}|{Path(str(source_path)).name}", encoding="utf-8")
        return destination

    with patch("segmoe_v2.geometry_fix._decide_action_against_reference", side_effect=fake_decide):
        with patch("segmoe_v2.geometry_fix._apply_action", side_effect=fake_apply):
            main(
                [
                    "fix-geometry-to-t2",
                    "--manifest",
                    str(manifest_path),
                    "--audit-csv",
                    str(audit_csv),
                    "--output-root",
                    str(output_root),
                    "--manifest-out",
                    str(manifest_out),
                    "--report-csv-out",
                    str(report_csv_out),
                    "--report-json-out",
                    str(report_json_out),
                ]
            )

    assert manifest_out.exists()
    assert report_csv_out.exists()
    assert report_json_out.exists()

    report_json = json.loads(report_json_out.read_text(encoding="utf-8"))
    assert report_json["fixed_case_count"] == 1
    assert report_json["recommendation_counts"]["header_harmonize_recommended"] == 1
