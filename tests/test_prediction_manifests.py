from __future__ import annotations

from pathlib import Path

import numpy as np

from segmoe_v2.io_utils import load_jsonl, save_jsonl
from segmoe_v2.prediction_manifests import (
    audit_prediction_manifest,
    build_layer1_prediction_manifest,
    merge_prediction_manifest_files,
)


def test_build_layer1_prediction_manifest_infers_mednext_softmax(tmp_path: Path) -> None:
    dataset_index = tmp_path / "dataset_index.jsonl"
    save_jsonl(
        [
            {
                "case_id": "case_a",
                "fixed_split": "trainval",
                "val_fold": 0,
                "source_manifest_hash": "abc",
                "metadata": {"labels_available": True, "bbox_zyx": [0, 4, 0, 4, 0, 4]},
            }
        ],
        dataset_index,
    )
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    np.savez_compressed(prediction_dir / "case_a.npz", softmax=np.ones((2, 4, 4, 4), dtype=np.float32))

    output = build_layer1_prediction_manifest(
        prediction_dir=prediction_dir,
        dataset_index=dataset_index,
        output=tmp_path / "manifest.jsonl",
        model_name="MedNeXt",
        fold=0,
        split="val",
    )

    rows = load_jsonl(output)
    assert rows[0]["model_name"] == "MedNeXt"
    assert rows[0]["prob_path"].endswith("case_a.npz")
    assert rows[0]["channel_names"] == ["background", "P_lesion"]
    assert rows[0]["metadata"]["prediction_key"] == "softmax"


def test_merge_prediction_manifest_files_deduplicates_exact_records(tmp_path: Path) -> None:
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    row = {"case_id": "case_a", "model_name": "nnUNet", "split": "val_0", "prob_path": "case_a.npz"}
    save_jsonl([row], first)
    save_jsonl([row, {**row, "case_id": "case_b", "prob_path": "case_b.npz"}], second)

    output = merge_prediction_manifest_files([first, second], tmp_path / "merged.jsonl")

    rows = load_jsonl(output)
    assert [row["case_id"] for row in rows] == ["case_a", "case_b"]


def test_audit_prediction_manifest_checks_npz_integrity(tmp_path: Path) -> None:
    prediction = tmp_path / "predictions" / "case_a.npz"
    prediction.parent.mkdir()
    np.savez_compressed(prediction, probabilities=np.ones((1, 2, 2, 2), dtype=np.float32))
    manifest = tmp_path / "manifest.jsonl"
    save_jsonl([{"case_id": "case_a", "prob_path": str(prediction)}], manifest)

    summary = audit_prediction_manifest(manifest, bad_out=tmp_path / "bad.jsonl")

    assert summary["ok"] == 1
    assert summary["bad"] == 0
    assert summary["missing"] == 0
    assert load_jsonl(tmp_path / "bad.jsonl") == []
