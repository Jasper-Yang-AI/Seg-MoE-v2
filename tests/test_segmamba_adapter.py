from __future__ import annotations

from pathlib import Path

import numpy as np

from segmoe_v2.io_utils import save_json, save_jsonl
from segmoe_v2.segmamba_adapter import SegMambaLayer1Dataset, layer1_high_recall_loss, layer1_target_from_source_labels, predict, train


def test_layer1_target_from_source_labels_treats_lesion_and_mimic_as_positive() -> None:
    target = layer1_target_from_source_labels(np.array([0, 1, 2, 3]))
    assert target.tolist() == [0.0, 1.0, 1.0, 0.0]


def test_segmamba_adapter_dry_run_reads_fold_lists(tmp_path: Path) -> None:
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

    assert train_summary["input_channels"] == 6
    assert train_summary["positive_label_values"] == [1, 2]
    assert train_summary["source_positive_weights"] == {"1": 1.25, "2": 0.75}
    assert train_summary["train_cases"] == 1
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


def test_layer1_high_recall_loss_uses_source_weights_only_for_bce() -> None:
    torch = __import__("torch")
    logits = torch.zeros((1, 1, 2, 1, 1), dtype=torch.float32)
    target = torch.ones_like(logits)
    low_weight = torch.ones_like(logits) * 0.75
    high_weight = torch.ones_like(logits) * 1.25

    low_loss = layer1_high_recall_loss(logits, target, low_weight)
    high_loss = layer1_high_recall_loss(logits, target, high_weight)

    assert float(high_loss) > float(low_loss)
