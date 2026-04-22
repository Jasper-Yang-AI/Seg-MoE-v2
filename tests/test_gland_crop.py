from __future__ import annotations

from pathlib import Path

import numpy as np

from segmoe_v2.contracts import CaseManifestRow
from segmoe_v2.gland_crop import build_gland_crop_records


def _row(case_id: str, *, spacing=(1.0, 1.0, 1.0)) -> CaseManifestRow:
    return CaseManifestRow(
        case_id=case_id,
        patient_id=case_id,
        era_bin="2012_2019",
        cohort_type="pca",
        fixed_split="trainval",
        val_fold=0,
        t2w_path=f"{case_id}_0000.nii.gz",
        adc_path=f"{case_id}_0001.nii.gz",
        dwi_path=f"{case_id}_0002.nii.gz",
        label_path=f"{case_id}.nii.gz",
        spacing=spacing,
        image_shape=(20, 20, 20),
    )


def test_gland_crop_uses_threshold_largest_component_and_margin(tmp_path: Path) -> None:
    probabilities = np.zeros((3, 20, 20, 20), dtype=np.float32)
    probabilities[0, 5:10, 5:10, 5:10] = 0.8
    probabilities[0, 7:8, 7:8, 7:8] = 0.0
    probabilities[0, 15:17, 15:17, 15:17] = 0.9
    prob_path = tmp_path / "case_a.npz"
    np.savez_compressed(prob_path, probabilities=probabilities, channel_names=np.asarray(["P_WG", "P_PZ", "P_TZ"]))

    records = build_gland_crop_records(
        [_row("case_a")],
        [
            {
                "case_id": "case_a",
                "prob_path": str(prob_path),
                "channel_names": ["P_WG", "P_PZ", "P_TZ"],
                "fold": 0,
                "split": "val_0",
            }
        ],
        wg_threshold=0.35,
        margin_mm=2.0,
        min_crop_size_zyx=(0, 0, 0),
    )

    record = records[0]
    assert record.bbox_zyx == (3, 12, 3, 12, 3, 12)
    assert record.crop_shape_zyx == (9, 9, 9)
    assert record.warning == ""
    assert record.metadata["raw_wg_bbox_zyx"] == [5, 10, 5, 10, 5, 10]
    assert record.metadata["margin_bbox_zyx"] == [3, 12, 3, 12, 3, 12]


def test_gland_crop_expands_to_minimum_size_when_possible(tmp_path: Path) -> None:
    probabilities = np.zeros((3, 32, 256, 256), dtype=np.float32)
    probabilities[0, 12:16, 100:120, 100:120] = 0.9
    prob_path = tmp_path / "case_min.npz"
    np.savez_compressed(prob_path, probabilities=probabilities, channel_names=np.asarray(["P_WG", "P_PZ", "P_TZ"]))

    record = build_gland_crop_records(
        [_row("case_min")],
        [{"case_id": "case_min", "prob_path": str(prob_path), "channel_names": ["P_WG", "P_PZ", "P_TZ"]}],
        margin_mm=0.0,
        min_crop_size_zyx=(24, 192, 192),
    )[0]

    assert record.crop_shape_zyx == (24, 192, 192)
    assert record.bbox_zyx[0] >= 0
    assert record.bbox_zyx[1] <= 32


def test_gland_crop_falls_back_to_full_image_when_wg_empty(tmp_path: Path) -> None:
    prob_path = tmp_path / "case_b.npz"
    np.savez_compressed(
        prob_path,
        probabilities=np.zeros((3, 4, 5, 6), dtype=np.float32),
        channel_names=np.asarray(["P_WG", "P_PZ", "P_TZ"]),
    )

    record = build_gland_crop_records(
        [_row("case_b")],
        [{"case_id": "case_b", "prob_path": str(prob_path), "channel_names": ["P_WG", "P_PZ", "P_TZ"]}],
    )[0]

    assert record.bbox_zyx == (0, 4, 0, 5, 0, 6)
    assert record.warning == "empty_wg_mask_used_full_image"
