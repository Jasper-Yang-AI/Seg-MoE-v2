from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from segmoe_v2.cli.main import main


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


class _BackendFakeImage:
    class _Header:
        @staticmethod
        def copy():
            return object()

    def __init__(self, payload: np.ndarray):
        self.header = self._Header()
        self.shape = payload.shape
        self.affine = np.eye(4, dtype=np.float32)
        self.dataobj = payload


def _fake_backend_load(path: str) -> _BackendFakeImage:
    payload = np.asarray([0, 1, 2, 3] if "pca" in path else [0, 1, 2], dtype=np.int16).reshape(1, 1, 1, -1)
    return _BackendFakeImage(payload)


def test_cli_manifest_and_backend_exports(tmp_path: Path) -> None:
    pca_root = tmp_path / "njmu_2012_2019_pca_zscore"
    nca_root = tmp_path / "njmu_2020_2023_nca_zscore"
    pca_root.mkdir()
    nca_root.mkdir()
    for idx in range(8):
        _write_case(pca_root, f"pca_{idx:03d}")
    for idx in range(8):
        _write_case(nca_root, f"nca_{idx:03d}")

    manifest_path = tmp_path / "manifest" / "cases.jsonl"
    summary_path = tmp_path / "manifest" / "manifest_summary.csv"
    nnunet_splits = tmp_path / "manifest" / "nnunet_splits_final.json"
    mednext_splits = tmp_path / "manifest" / "mednext_splits_final.pkl"
    segmamba_splits = tmp_path / "manifest" / "segmamba_splits_final.json"
    nnunet_root = tmp_path / "exports" / "nnunet_anatomy"
    mednext_root = tmp_path / "exports" / "mednext_lesion"
    segmamba_output = tmp_path / "exports" / "segmamba_anatomy"

    with patch("segmoe_v2.manifest.nib.load", side_effect=_fake_load):
        main(
            [
                "build-manifest",
                "--roots",
                str(pca_root),
                str(nca_root),
                "--manifest-out",
                str(manifest_path),
                "--summary-out",
                str(summary_path),
                "--nnunet-splits-out",
                str(nnunet_splits),
                "--mednext-splits-out",
                str(mednext_splits),
                "--segmamba-splits-out",
                str(segmamba_splits),
            ]
        )

    assert manifest_path.exists()
    assert summary_path.exists()
    assert nnunet_splits.exists()
    assert mednext_splits.exists()
    assert segmamba_splits.exists()

    main(
        [
            "audit-manifest",
            "--manifest",
            str(manifest_path),
            "--nnunet-splits",
            str(nnunet_splits),
            "--mednext-splits",
            str(mednext_splits),
            "--segmamba-splits",
            str(segmamba_splits),
        ]
    )

    main(
        [
            "export-nnunet-task",
            "--manifest",
            str(manifest_path),
            "--task-root",
            str(nnunet_root),
            "--dataset-id",
            "501",
            "--dataset-name",
            "ProstateCanonical",
            "--task",
            "anatomy",
        ]
    )
    nnunet_dataset_dir = nnunet_root / "Dataset501_ProstateCanonical"
    assert (nnunet_dataset_dir / "dataset.json").exists()
    assert (nnunet_dataset_dir / "splits_final.json").exists()
    anatomy_dataset_json = __import__("json").loads((nnunet_dataset_dir / "dataset.json").read_text(encoding="utf-8"))
    assert anatomy_dataset_json["segmoe_task"] == "anatomy"
    assert anatomy_dataset_json["segmoe_target_adapter"] == "nnUNetTrainerSegMoEAnatomy"

    saved_labels: dict[str, np.ndarray] = {}

    def _fake_backend_save(image, path):
        saved_labels[str(path)] = np.asanyarray(image.dataobj)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).touch()

    class _FakeExportedImage:
        def __init__(self, dataobj, affine=None, header=None):
            self.dataobj = np.asarray(dataobj)
            self.affine = affine
            self.header = header

    with patch("segmoe_v2.backend_data.nib.load", side_effect=_fake_backend_load), patch(
        "segmoe_v2.backend_data.nib.save", side_effect=_fake_backend_save
    ), patch(
        "segmoe_v2.backend_data.nib.Nifti1Image", _FakeExportedImage
    ):
        main(
            [
                "export-mednext-task",
                "--manifest",
                str(manifest_path),
                "--task-root",
                str(mednext_root),
                "--dataset-id",
                "502",
                "--dataset-name",
                "ProstateCanonical",
                "--task",
                "lesion",
            ]
        )
    mednext_dataset_dir = mednext_root / "Task502_ProstateCanonical"
    assert (mednext_dataset_dir / "dataset.json").exists()
    assert (mednext_dataset_dir / "splits_final.pkl").exists()
    mednext_dataset_json = __import__("json").loads((mednext_dataset_dir / "dataset.json").read_text(encoding="utf-8"))
    assert mednext_dataset_json["segmoe_main_label_mode"] == "source"
    assert mednext_dataset_json["segmoe_include_test_labels"] is False
    assert any(path.endswith("pca_000.nii.gz") for path in saved_labels)
    exported_label = next(array for path, array in saved_labels.items() if path.endswith("labelsTr/pca_000.nii.gz"))
    exported_source = next(array for path, array in saved_labels.items() if path.endswith("sourceLabelsTr/pca_000.nii.gz"))
    exported_weight = next(array for path, array in saved_labels.items() if path.endswith("weightsTr/pca_000.nii.gz"))
    assert set(np.unique(exported_label).tolist()) <= {0, 1}
    assert set(np.unique(exported_source).tolist()) <= {0, 1}
    assert float(exported_weight.max()) == 1.25

    main(
        [
            "prepare-segmamba-data",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(segmamba_output),
            "--task",
            "anatomy",
        ]
    )
    assert (segmamba_output / "dataset_index.jsonl").exists()
    assert (segmamba_output / "split_metadata.json").exists()
    assert (segmamba_output / "segmamba_config.json").exists()
    assert (segmamba_output / "fold_0_train.jsonl").exists()
    split_metadata = __import__("json").loads((segmamba_output / "split_metadata.json").read_text(encoding="utf-8"))
    assert split_metadata["task"] == "anatomy"
    assert split_metadata["model"] == "SegMamba"
