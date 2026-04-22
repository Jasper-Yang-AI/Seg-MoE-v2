from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from segmoe_v2.backend_data import export_nnunet_task, prepare_layer1_moe_data, prepare_segmamba_data
from segmoe_v2.contracts import CaseManifestRow
from segmoe_v2.gland_crop import GlandCropRecord
from segmoe_v2.io_utils import save_jsonl


class _Header:
    @staticmethod
    def copy():
        return _Header()


class _Image:
    def __init__(self, data: np.ndarray):
        self.dataobj = data
        self.affine = np.eye(4, dtype=np.float32)
        self.header = _Header()
        self.shape = data.shape


class _ExportedImage(_Image):
    def __init__(self, dataobj, affine=None, header=None):
        super().__init__(np.asarray(dataobj))
        self.affine = affine
        self.header = header


def _case(case_id: str, cohort: str) -> CaseManifestRow:
    return CaseManifestRow(
        case_id=case_id,
        patient_id=case_id,
        era_bin="2012_2019",
        cohort_type=cohort,  # type: ignore[arg-type]
        has_lesion_label3=True,
        label_unique_values=(0, 1, 2, 3),
        fixed_split="trainval",
        val_fold=0,
        t2w_path=f"{case_id}_0000.nii.gz",
        adc_path=f"{case_id}_0001.nii.gz",
        dwi_path=f"{case_id}_0002.nii.gz",
        label_path=f"{case_id}.nii.gz",
        spacing=(1.0, 1.0, 1.0),
        image_shape=(6, 6, 6),
        affine_hash="abc",
    )


def test_layer1_backend_exports_crop_six_channels_and_tristate_labels(tmp_path: Path) -> None:
    rows = [_case("pca_a", "pca"), _case("nca_a", "nca")]
    prob_manifest = tmp_path / "anatomy_predictions.jsonl"
    crop_records: list[GlandCropRecord] = []
    pred_rows = []
    for row in rows:
        prob_path = tmp_path / f"{row.case_id}_anatomy.npz"
        probs = np.ones((3, 6, 6, 6), dtype=np.float32) * 0.8
        np.savez_compressed(prob_path, probabilities=probs, channel_names=np.asarray(["P_WG", "P_PZ", "P_TZ"]))
        pred_rows.append({"case_id": row.case_id, "prob_path": str(prob_path), "channel_names": ["P_WG", "P_PZ", "P_TZ"]})
        crop_records.append(
            GlandCropRecord(
                case_id=row.case_id,
                bbox_zyx=(1, 5, 1, 5, 1, 5),
                crop_shape_zyx=(4, 4, 4),
                native_shape_zyx=(6, 6, 6),
                source_prob_path=str(prob_path),
                source_manifest_hash="hash",
            )
        )
    save_jsonl(pred_rows, prob_manifest)

    def fake_load(path: str) -> _Image:
        data = np.zeros((6, 6, 6), dtype=np.float32)
        if path.endswith(".nii.gz") and "_000" not in Path(path).name:
            data = data.astype(np.int16)
            data[2:4, 2:4, 2:4] = 3
        elif "_0001" in path:
            data += 2
        elif "_0002" in path:
            data += 3
        else:
            data += 1
        return _Image(data)

    saved: dict[str, np.ndarray] = {}

    def fake_save(image: _ExportedImage, path: str) -> None:
        saved[str(path)] = np.asarray(image.dataobj)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).touch()

    with patch("segmoe_v2.backend_data.nib.load", side_effect=fake_load), patch(
        "segmoe_v2.backend_data.nib.save", side_effect=fake_save
    ), patch("segmoe_v2.backend_data.nib.Nifti1Image", _ExportedImage):
        outputs = export_nnunet_task(
            rows,
            task_root=tmp_path / "nnunet",
            dataset_id=701,
            dataset_name="Layer1",
            task="lesion",
            anatomy_prediction_manifest=prob_manifest,
            crop_manifest=crop_records,
        )
        seg_outputs = prepare_segmamba_data(
            rows,
            output_dir=tmp_path / "segmamba",
            task="lesion",
            anatomy_prediction_manifest=prob_manifest,
            crop_manifest=crop_records,
        )

    dataset_json = __import__("json").loads(Path(outputs["dataset_json"]).read_text(encoding="utf-8"))
    assert dataset_json["channel_names"]["5"] == "P_TZ"
    assert dataset_json["labels"] == {"background": 0, "candidate": 1}
    assert dataset_json["segmoe_source_positive_weights"] == {"1": 1.25, "2": 0.75}
    assert any(path.endswith("pca_a_0005.nii.gz") for path in saved)
    pca_label = next(array for path, array in saved.items() if path.endswith("labelsTr/pca_a.nii.gz"))
    nca_label = next(array for path, array in saved.items() if path.endswith("labelsTr/nca_a.nii.gz"))
    nca_source = next(array for path, array in saved.items() if path.endswith("sourceLabelsTr/nca_a.nii.gz"))
    pca_weight = next(array for path, array in saved.items() if path.endswith("weightsTr/pca_a.nii.gz"))
    nca_weight = next(array for path, array in saved.items() if path.endswith("weightsTr/nca_a.nii.gz"))
    assert set(np.unique(pca_label).tolist()) == {0, 1}
    assert set(np.unique(nca_label).tolist()) == {0, 1}
    assert set(np.unique(nca_source).tolist()) == {0, 2}
    assert np.isclose(float(pca_weight.max()), 1.25)
    assert np.isclose(float(nca_weight.min()), 0.75)
    assert pca_label.shape == (4, 4, 4)

    seg_config = __import__("json").loads(Path(seg_outputs["segmamba_config"]).read_text(encoding="utf-8"))
    assert seg_config["input_channels"] == 6
    assert seg_config["positive_label_values"] == [1, 2]
    assert seg_config["source_positive_weights"] == {"1": 1.25, "2": 0.75}
    seg_npz = np.load(tmp_path / "segmamba" / "arrays" / "nca_a.npz")
    assert seg_npz["data"].shape == (6, 4, 4, 4)
    assert set(np.unique(seg_npz["seg_source"]).tolist()) == {0, 2}
    assert set(np.unique(seg_npz["seg_target"]).tolist()) == {0, 1}
    assert np.isclose(float(seg_npz["voxel_weight"].min()), 0.75)


def test_prepare_layer1_moe_writes_shared_three_expert_config(tmp_path: Path) -> None:
    rows = [_case("pca_a", "pca")]
    prob_path = tmp_path / "pca_a_anatomy.npz"
    np.savez_compressed(
        prob_path,
        probabilities=np.ones((3, 6, 6, 6), dtype=np.float32),
        channel_names=np.asarray(["P_WG", "P_PZ", "P_TZ"]),
    )
    prob_manifest = tmp_path / "anatomy_predictions.jsonl"
    save_jsonl([{"case_id": "pca_a", "prob_path": str(prob_path), "channel_names": ["P_WG", "P_PZ", "P_TZ"]}], prob_manifest)
    crop_records = [
        GlandCropRecord(
            case_id="pca_a",
            bbox_zyx=(0, 6, 0, 6, 0, 6),
            crop_shape_zyx=(6, 6, 6),
            native_shape_zyx=(6, 6, 6),
            source_prob_path=str(prob_path),
            source_manifest_hash="hash",
        )
    ]

    def fake_load(path: str) -> _Image:
        data = np.ones((6, 6, 6), dtype=np.float32)
        if path.endswith(".nii.gz") and "_000" not in Path(path).name:
            data = np.zeros((6, 6, 6), dtype=np.int16)
            data[2:4, 2:4, 2:4] = 3
        return _Image(data)

    with patch("segmoe_v2.backend_data.nib.load", side_effect=fake_load), patch(
        "segmoe_v2.backend_data.nib.save", side_effect=lambda image, path: Path(path).touch()
    ), patch("segmoe_v2.backend_data.nib.Nifti1Image", _ExportedImage):
        outputs = prepare_layer1_moe_data(
            rows,
            anatomy_prediction_manifest=prob_manifest,
            crop_manifest=crop_records,
            config_out=tmp_path / "layer1_moe_config.json",
            nnunet_task_root=tmp_path / "nnunet",
            nnunet_dataset_id=702,
            mednext_task_root=tmp_path / "mednext",
            mednext_dataset_id=703,
            segmamba_output_dir=tmp_path / "segmamba",
        )

    config = __import__("json").loads(Path(outputs["layer1_moe_config"]).read_text(encoding="utf-8"))
    assert config["expert_names"] == ["nnunet", "mednext", "segmamba"]
    assert config["source_positive_weights"] == {"1": 1.25, "2": 0.75}
    assert config["experts"]["nnunet"]["role"] == "local_boundary_expert"
    assert config["experts"]["mednext"]["role"] == "large_kernel_multiscale_context_expert"
    assert config["experts"]["segmamba"]["role"] == "long_range_roi_context_expert"
