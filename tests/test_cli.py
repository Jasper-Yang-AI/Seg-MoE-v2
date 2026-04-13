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
    nnformer_splits = tmp_path / "manifest" / "nnformer_splits_final.pkl"
    nnunet_root = tmp_path / "exports" / "nnunet_anatomy"
    nnformer_root = tmp_path / "exports" / "nnformer_lesion"
    swin_output = tmp_path / "exports" / "swin_anatomy"

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
                "--nnformer-splits-out",
                str(nnformer_splits),
            ]
        )

    assert manifest_path.exists()
    assert summary_path.exists()
    assert nnunet_splits.exists()
    assert nnformer_splits.exists()

    main(
        [
            "audit-manifest",
            "--manifest",
            str(manifest_path),
            "--nnunet-splits",
            str(nnunet_splits),
            "--nnformer-splits",
            str(nnformer_splits),
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
                "export-nnformer-task",
                "--manifest",
                str(manifest_path),
                "--task-root",
                str(nnformer_root),
                "--dataset-id",
                "502",
                "--dataset-name",
                "ProstateCanonical",
                "--task",
                "lesion",
            ]
        )
    nnformer_dataset_dir = nnformer_root / "Task502_ProstateCanonical"
    assert (nnformer_dataset_dir / "dataset.json").exists()
    assert (nnformer_dataset_dir / "splits_final.pkl").exists()
    assert any(path.endswith("pca_000.nii.gz") for path in saved_labels)
    exported_label = next(array for path, array in saved_labels.items() if path.endswith("pca_000.nii.gz"))
    assert set(np.unique(exported_label).tolist()) <= {0, 1}

    main(
        [
            "prepare-swinunetr-data",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(swin_output),
            "--task",
            "anatomy",
        ]
    )
    assert (swin_output / "dataset_index.jsonl").exists()
    assert (swin_output / "split_metadata.json").exists()
    assert (swin_output / "fold_0_train.jsonl").exists()
    split_metadata = __import__("json").loads((swin_output / "split_metadata.json").read_text(encoding="utf-8"))
    assert split_metadata["task"] == "anatomy"
