from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from segmoe_v2.io_utils import save_json, save_jsonl
from segmoe_v2.segmamba_adapter import (
    SegMambaLayer1Dataset,
    _train_original_style,
    export_original_style_data,
    layer1_high_recall_loss,
    layer1_target_from_source_labels,
    predict,
    train,
)


def test_layer1_target_from_source_labels_treats_lesion_and_mimic_as_positive() -> None:
    target = layer1_target_from_source_labels(np.array([0, 1, 2, 3]))
    assert target.tolist() == [0.0, 1.0, 1.0, 0.0]


def test_segmamba_adapter_train_dry_run_requires_original_style_config(tmp_path: Path) -> None:
    save_jsonl([{"case_id": "case_a"}], tmp_path / "fold_0_train.jsonl")
    save_jsonl([{"case_id": "case_b"}], tmp_path / "fold_0_val.jsonl")
    save_jsonl([{"case_id": "case_c"}], tmp_path / "test.jsonl")
    config = tmp_path / "segmamba_config.json"
    save_json(
        {
            "train_list_pattern": str(tmp_path / "fold_{fold}_train.jsonl"),
            "val_list_pattern": str(tmp_path / "fold_{fold}_val.jsonl"),
            "test_list": str(tmp_path / "test.jsonl"),
            "input_channels": 6,
            "positive_label_values": [1, 2],
            "sampling_policy": {"pca_lesion": 0.5, "nca_mimic": 0.25, "random_gland": 0.25},
        },
        config,
    )

    train_summary = train(config, fold=0, dry_run=True)
    predict_summary = predict(config, fold=0, split="val", dry_run=True)

    assert train_summary["ready"] is False
    assert train_summary["data_format"] == "segmamba_npz"
    assert train_summary["required_data_format"] == "segmamba_original"
    assert "export-original" in train_summary["export_command"]
    assert predict_summary["logit_field"] == "logits"


def test_segmamba_dataset_samples_layer1_patch_with_mimic_positive(tmp_path: Path) -> None:
    data = np.zeros((6, 8, 8, 8), dtype=np.float32)
    data[3] = 1.0
    seg = np.zeros((1, 8, 8, 8), dtype=np.uint8)
    seg[:, 3:5, 3:5, 3:5] = 2
    array_path = tmp_path / "nca_case.npz"
    np.savez_compressed(
        array_path,
        data=data,
        seg=seg,
        bbox_zyx=np.asarray([0, 8, 0, 8, 0, 8]),
        native_shape_zyx=np.asarray([8, 8, 8]),
    )
    dataset = SegMambaLayer1Dataset(
        [{"case_id": "nca_case", "segmamba_npz": str(array_path)}],
        positive_label_values=[1, 2],
        patch_size=[4, 4, 4],
        seed=0,
    )

    sample = dataset[0]

    assert sample["data"].shape == (6, 4, 4, 4)
    assert sample["target"].shape == (1, 4, 4, 4)
    assert sample["voxel_weight"].shape == (1, 4, 4, 4)
    assert float(sample["target"].max()) == 1.0
    assert float(sample["voxel_weight"].min()) == 0.75


def test_export_original_style_data_writes_class_locations(tmp_path: Path) -> None:
    data = np.zeros((6, 8, 8, 8), dtype=np.float32)
    seg = np.zeros((1, 8, 8, 8), dtype=np.uint8)
    seg[:, 2:4, 2:4, 2:4] = 2
    array_path = tmp_path / "arrays" / "case_a.npz"
    array_path.parent.mkdir()
    np.savez_compressed(
        array_path,
        data=data,
        seg=seg,
        seg_source=seg,
        bbox_zyx=np.asarray([0, 8, 0, 8, 0, 8]),
        native_shape_zyx=np.asarray([8, 8, 8]),
    )
    record = {"case_id": "case_a", "segmamba_npz": str(array_path), "image": str(array_path)}
    save_jsonl([record], tmp_path / "dataset_index.jsonl")
    save_jsonl([record], tmp_path / "fold_0_train.jsonl")
    save_jsonl([record], tmp_path / "fold_0_val.jsonl")
    save_jsonl([], tmp_path / "test.jsonl")
    save_json({"folds": {"0": {"train_count": 1, "val_count": 1}}}, tmp_path / "split_metadata.json")
    config = tmp_path / "segmamba_config.json"
    save_json(
        {
            "dataset_index": str(tmp_path / "dataset_index.jsonl"),
            "split_metadata": str(tmp_path / "split_metadata.json"),
            "train_list_pattern": str(tmp_path / "fold_{fold}_train.jsonl"),
            "val_list_pattern": str(tmp_path / "fold_{fold}_val.jsonl"),
            "test_list": str(tmp_path / "test.jsonl"),
            "positive_label_values": [1, 2],
        },
        config,
    )

    summary = export_original_style_data(config, output_dir=tmp_path / "original", folds=[0], link_mode="copy")

    original_config = tmp_path / "original" / "segmamba_original_config.json"
    pkl_path = tmp_path / "original" / "fullres" / "case_a.pkl"
    with pkl_path.open("rb") as handle:
        properties = pickle.load(handle)
    train_summary = train(original_config, fold=0, dry_run=True)

    assert summary["cases"] == 1
    assert original_config.exists()
    assert train_summary["data_format"] == "segmamba_original"
    assert train_summary["trainer"] == "external/SegMamba/light_training/trainer.py"
    assert train_summary["train_cases"] == 1
    assert train_summary["batch_size"] == 2
    assert train_summary["optimizer"] == "sgd"
    assert train_summary["learning_rate"] == 1e-2
    assert train_summary["pin_memory"] is False
    assert properties["class_locations"][2].shape[1] == 4
    assert (tmp_path / "original" / "splits" / "fold_0_split.json").exists()


def test_original_style_train_uses_uppercase_ddp_for_vendored_trainer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    captured: dict[str, str] = {}

    class FakeMedicalDataset:
        def __init__(self, paths: list[str]) -> None:
            self.paths = paths

    class FakeTrainer:
        def __init__(self, env_type: str, *args: object, **kwargs: object) -> None:
            captured["env_type"] = env_type
            self.writer = None

        def train(self, train_dataset: FakeMedicalDataset, val_dataset: FakeMedicalDataset) -> None:
            captured["train_cases"] = str(len(train_dataset.paths))
            captured["val_cases"] = str(len(val_dataset.paths))
            captured["pin_memory"] = str(self.pin_memory)

    light_training = types.ModuleType("light_training")
    dataloading = types.ModuleType("light_training.dataloading")
    dataset = types.ModuleType("light_training.dataloading.dataset")
    trainer = types.ModuleType("light_training.trainer")
    dataset.MedicalDataset = FakeMedicalDataset
    trainer.Trainer = FakeTrainer
    monkeypatch.setitem(sys.modules, "light_training", light_training)
    monkeypatch.setitem(sys.modules, "light_training.dataloading", dataloading)
    monkeypatch.setitem(sys.modules, "light_training.dataloading.dataset", dataset)
    monkeypatch.setitem(sys.modules, "light_training.trainer", trainer)
    monkeypatch.setattr("segmoe_v2.segmamba_adapter.build_segmamba_model", lambda **_kwargs: object())
    monkeypatch.setattr("segmoe_v2.segmamba_adapter._build_original_optimizer", lambda _model, _config: object())
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("RANK", "1")
    save_json({"train": ["case_a.npz"], "validation": ["case_b.npz"]}, tmp_path / "fold_0_split.json")

    _train_original_style(
        {
            "repo_root": str(tmp_path),
            "original_data_dir": str(tmp_path),
            "original_split_pattern": str(tmp_path / "fold_{fold}_split.json"),
        },
        fold=0,
        max_epochs=1,
        summary={},
    )

    assert captured == {"env_type": "DDP", "train_cases": "1", "val_cases": "1", "pin_memory": "False"}


def test_layer1_high_recall_loss_uses_source_weights_only_for_bce() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.zeros((1, 1, 2, 1, 1), dtype=torch.float32)
    target = torch.ones_like(logits)
    low_weight = torch.ones_like(logits) * 0.75
    high_weight = torch.ones_like(logits) * 1.25

    low_loss = layer1_high_recall_loss(logits, target, low_weight)
    high_loss = layer1_high_recall_loss(logits, target, high_weight)

    assert float(high_loss) > float(low_loss)
