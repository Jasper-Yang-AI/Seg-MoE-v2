from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from segmoe_v2.cli.main import main
from segmoe_v2.contracts import CaseManifestRow
from segmoe_v2.io_utils import save_jsonl


def _write_nifti(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(array), affine=np.eye(4, dtype=np.float32)), str(path))


def test_visualize_anatomy_qc_writes_png_and_summary(tmp_path: Path) -> None:
    case_id = "case_normal_001"
    data_root = tmp_path / "data_root"
    t2_path = data_root / f"{case_id}_0000.nii.gz"
    adc_path = data_root / f"{case_id}_0001.nii.gz"
    dwi_path = data_root / f"{case_id}_0002.nii.gz"
    label_path = data_root / f"{case_id}.nii.gz"

    image = np.zeros((8, 8, 4), dtype=np.float32)
    image[2:6, 2:6, 2] = 5.0
    label = np.zeros((8, 8, 4), dtype=np.uint8)
    label[3:5, 3:5, 2] = 1
    for path in (t2_path, adc_path, dwi_path):
        _write_nifti(path, image)
    _write_nifti(label_path, label)

    manifest_path = tmp_path / "manifest" / "cases.jsonl"
    row = CaseManifestRow(
        case_id=case_id,
        patient_id=case_id,
        era_bin="2012_2019",
        cohort_type="nca",
        has_lesion_label3=False,
        label_unique_values=(0, 1, 2),
        fixed_split="trainval",
        val_fold=0,
        t2w_path=t2_path,
        adc_path=adc_path,
        dwi_path=dwi_path,
        label_path=label_path,
        spacing=(1.0, 1.0, 1.0),
        image_shape=(8, 8, 4),
        affine_hash="abc",
        metadata={"root": str(data_root)},
    )
    save_jsonl([row.to_dict()], manifest_path)

    prob_path = tmp_path / "predictions" / f"{case_id}.npz"
    prob_path.parent.mkdir(parents=True, exist_ok=True)
    probabilities = np.zeros((3, 8, 8, 4), dtype=np.float32)
    probabilities[0, 2:6, 2:6, 2] = 0.9
    probabilities[1, 3:5, 3:5, 2] = 0.3
    probabilities[2, 2:4, 2:4, 2] = 0.2
    np.savez_compressed(prob_path, probabilities=probabilities, channel_names=np.asarray(("P_WG", "P_PZ", "P_TZ")))

    prediction_manifest = tmp_path / "predictions" / "prediction_manifest.jsonl"
    save_jsonl(
        [
            {
                "case_id": case_id,
                "fold": 0,
                "split": "val",
                "channel_names": ["P_WG", "P_PZ", "P_TZ"],
                "prob_path": str(prob_path),
                "source_manifest_hash": "xyz",
                "hierarchy_consistency_applied": True,
            }
        ],
        prediction_manifest,
    )

    output_dir = tmp_path / "qc"
    main(
        [
            "visualize-anatomy-qc",
            "--manifest",
            str(manifest_path),
            "--prediction-manifest",
            str(prediction_manifest),
            "--output-dir",
            str(output_dir),
            "--normal-count",
            "1",
            "--lesion-count",
            "0",
            "--geometry-fix-count",
            "0",
        ]
    )

    summary_path = output_dir / "selection_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["actual_counts"]["normal_case"] == 1
    pngs = list(output_dir.glob("*.png"))
    assert len(pngs) == 1
