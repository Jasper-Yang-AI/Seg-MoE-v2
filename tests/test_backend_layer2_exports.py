from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from segmoe_v2.backend_data import prepare_layer2_moe_data
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


def _fake_load(path: str) -> _Image:
    data = np.ones((6, 6, 6), dtype=np.float32)
    if "_0001" in path:
        data *= 2
    elif "_0002" in path:
        data *= 3
    elif path.endswith(".nii.gz") and "_000" not in Path(path).name:
        data = np.zeros((6, 6, 6), dtype=np.int16)
        data[2:4, 2:4, 2:4] = 3
    return _Image(data)


def _write_layer2_inputs(
    tmp_path: Path,
    rows: list[CaseManifestRow],
    *,
    include_segmamba: bool = True,
    bbox_override: tuple[int, int, int, int, int, int] | None = None,
) -> tuple[Path, list[GlandCropRecord], list[Path]]:
    anatomy_rows = []
    crop_records: list[GlandCropRecord] = []
    layer1_rows = []
    bbox = (1, 5, 1, 5, 1, 5)
    for row in rows:
        anatomy_path = tmp_path / f"{row.case_id}_anatomy.npz"
        probs = np.ones((3, 6, 6, 6), dtype=np.float32) * 0.8
        np.savez_compressed(anatomy_path, probabilities=probs, channel_names=np.asarray(["P_WG", "P_PZ", "P_TZ"]))
        anatomy_rows.append({"case_id": row.case_id, "prob_path": str(anatomy_path), "channel_names": ["P_WG", "P_PZ", "P_TZ"]})
        crop_records.append(
            GlandCropRecord(
                case_id=row.case_id,
                bbox_zyx=bbox,
                crop_shape_zyx=(4, 4, 4),
                native_shape_zyx=(6, 6, 6),
                source_prob_path=str(anatomy_path),
                source_manifest_hash="hash",
            )
        )
        nnunet_path = tmp_path / f"{row.case_id}_nnunet.npz"
        mednext_path = tmp_path / f"{row.case_id}_mednext.npz"
        segmamba_path = tmp_path / f"{row.case_id}_segmamba.npz"
        np.savez_compressed(nnunet_path, probabilities=np.ones((1, 4, 4, 4), dtype=np.float32) * 0.7, channel_names=np.asarray(["P_lesion"]))
        np.savez_compressed(
            mednext_path,
            softmax=np.stack(
                [np.ones((4, 4, 4), dtype=np.float32) * 0.2, np.ones((4, 4, 4), dtype=np.float32) * 0.8],
                axis=0,
            ),
            channel_names=np.asarray(["background", "P_lesion"]),
        )
        np.savez_compressed(segmamba_path, logits=np.ones((1, 4, 4, 4), dtype=np.float32), channel_names=np.asarray(["P_lesion_logit"]))
        metadata = {
            "bbox_zyx": list(bbox_override or bbox),
            "native_shape_zyx": [6, 6, 6],
        }
        layer1_rows.extend(
            [
                {
                    "task": "lesion",
                    "stage": "layer1",
                    "model_name": "nnUNet",
                    "fold": 0,
                    "split": "val_0",
                    "case_id": row.case_id,
                    "predictor_fold": 0,
                    "prob_path": str(nnunet_path),
                    "channel_names": ["P_lesion"],
                    "metadata": metadata,
                },
                {
                    "task": "lesion",
                    "stage": "layer1",
                    "model_name": "MedNeXt",
                    "fold": 0,
                    "split": "val_0",
                    "case_id": row.case_id,
                    "predictor_fold": 0,
                    "prob_path": str(mednext_path),
                    "channel_names": ["background", "P_lesion"],
                    "metadata": metadata,
                },
            ]
        )
        if include_segmamba:
            layer1_rows.append(
                {
                    "task": "lesion",
                    "stage": "layer1",
                    "model_name": "SegMamba",
                    "fold": 0,
                    "split": "val_0",
                    "case_id": row.case_id,
                    "predictor_fold": 0,
                    "logit_path": str(segmamba_path),
                    "channel_names": ["P_lesion_logit"],
                    "metadata": metadata,
                }
            )

    anatomy_manifest = tmp_path / "anatomy_predictions.jsonl"
    layer1_manifest = tmp_path / "layer1_predictions.jsonl"
    save_jsonl(anatomy_rows, anatomy_manifest)
    save_jsonl(layer1_rows, layer1_manifest)
    return anatomy_manifest, crop_records, [layer1_manifest]


def test_prepare_layer2_moe_writes_three_backend_contract(tmp_path: Path) -> None:
    rows = [_case("pca_a", "pca"), _case("nca_a", "nca")]
    anatomy_manifest, crop_records, layer1_manifests = _write_layer2_inputs(tmp_path, rows)
    saved: dict[str, np.ndarray] = {}

    def fake_save(image: _ExportedImage, path: str) -> None:
        saved[str(path)] = np.asarray(image.dataobj)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).touch()

    with patch("segmoe_v2.backend_data.nib.load", side_effect=_fake_load), patch(
        "segmoe_v2.backend_data.nib.save", side_effect=fake_save
    ), patch("segmoe_v2.backend_data.nib.Nifti1Image", _ExportedImage):
        outputs = prepare_layer2_moe_data(
            rows,
            anatomy_prediction_manifest=anatomy_manifest,
            crop_manifest=crop_records,
            layer1_prediction_manifests=layer1_manifests,
            config_out=tmp_path / "layer2_moe_config.json",
            nnunet_task_root=tmp_path / "nnunet",
            mednext_task_root=tmp_path / "mednext",
            segmamba_output_dir=tmp_path / "segmamba",
        )

    config = json.loads(Path(outputs["layer2_moe_config"]).read_text(encoding="utf-8"))
    assert config["layer"] == "layer2"
    assert config["channel_names"][11] == "fp_risk"
    assert config["experts"]["nnunet"]["trainer"] == "nnUNetTrainerSegMoELayer2"
    assert config["experts"]["mednext"]["trainer"] == "nnUNetTrainerV2_MedNeXt_S_kernel3_SegMoELayer2"
    assert config["experts"]["mednext"]["initial_lr"] == 5e-4
    assert config["experts"]["segmamba"]["training_defaults"]["learning_rate"] == 0.005

    nnunet_json = json.loads((tmp_path / "nnunet" / "Dataset503_ProstateLayer2" / "dataset.json").read_text(encoding="utf-8"))
    assert nnunet_json["channel_names"]["11"] == "fp_risk"
    assert nnunet_json["segmoe_source_weights"] == {"1": 1.25, "2": 2.5}
    nnunet_index_row = json.loads(
        (tmp_path / "nnunet" / "Dataset503_ProstateLayer2" / "dataset_index.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert nnunet_index_row["metadata"]["bbox_zyx"] == [1, 5, 1, 5, 1, 5]
    assert nnunet_index_row["metadata"]["native_shape_zyx"] == [6, 6, 6]
    assert any(path.endswith("imagesTr/pca_a_0011.nii.gz") for path in saved)
    pca_label = next(array for path, array in saved.items() if path.endswith("labelsTr/pca_a.nii.gz"))
    nca_label = next(array for path, array in saved.items() if path.endswith("labelsTr/nca_a.nii.gz"))
    assert 1 in set(np.unique(pca_label).tolist())
    assert 2 in set(np.unique(nca_label).tolist())

    seg_config = json.loads((tmp_path / "segmamba" / "segmamba_config.json").read_text(encoding="utf-8"))
    assert seg_config["stage"] == "layer2"
    assert seg_config["input_channels"] == 12
    assert seg_config["patch_size"] == [32, 224, 224]
    assert seg_config["learning_rate"] == 0.005
    assert seg_config["val_every"] == 5
    assert seg_config["val_batches"] == 320
    seg_npz = np.load(tmp_path / "segmamba" / "arrays" / "nca_a.npz")
    assert seg_npz["data"].shape == (12, 4, 4, 4)
    assert set(np.unique(seg_npz["seg_source"]).tolist()) <= {0, 2}
    assert float(seg_npz["voxel_weight"].max()) == 2.5


def test_layer2_export_requires_all_three_layer1_experts(tmp_path: Path) -> None:
    rows = [_case("pca_a", "pca")]
    anatomy_manifest, crop_records, layer1_manifests = _write_layer2_inputs(tmp_path, rows, include_segmamba=False)

    with patch("segmoe_v2.backend_data.nib.load", side_effect=_fake_load), pytest.raises(KeyError, match="segmamba"):
        prepare_layer2_moe_data(
            rows,
            anatomy_prediction_manifest=anatomy_manifest,
            crop_manifest=crop_records,
            layer1_prediction_manifests=layer1_manifests,
            config_out=tmp_path / "layer2_moe_config.json",
            nnunet_task_root=tmp_path / "nnunet",
            mednext_task_root=tmp_path / "mednext",
            segmamba_output_dir=tmp_path / "segmamba",
        )


def test_layer2_export_rejects_layer1_bbox_mismatch(tmp_path: Path) -> None:
    rows = [_case("pca_a", "pca")]
    anatomy_manifest, crop_records, layer1_manifests = _write_layer2_inputs(
        tmp_path,
        rows,
        bbox_override=(0, 4, 0, 4, 0, 4),
    )

    with patch("segmoe_v2.backend_data.nib.load", side_effect=_fake_load), pytest.raises(ValueError, match="bbox mismatch"):
        prepare_layer2_moe_data(
            rows,
            anatomy_prediction_manifest=anatomy_manifest,
            crop_manifest=crop_records,
            layer1_prediction_manifests=layer1_manifests,
            config_out=tmp_path / "layer2_moe_config.json",
            nnunet_task_root=tmp_path / "nnunet",
            mednext_task_root=tmp_path / "mednext",
            segmamba_output_dir=tmp_path / "segmamba",
        )
