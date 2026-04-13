from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from segmoe_v2.nnunet_anatomy import (
    ANATOMY_REGION_KEYS,
    ANATOMY_PROBABILITY_CHANNELS,
    MaskedAnatomySegLoss,
    anatomy_consistency_loss,
    anatomy_consistency_weight,
    anatomy_hard_masks_from_probabilities,
    anatomy_tp_fp_fn_tn,
    apply_anatomy_modality_dropout,
    build_anatomy_validation_summary,
    build_anatomy_head_targets_torch,
    build_t2_only_input,
    convert_anatomy_logits_to_probabilities_with_correct_shape,
    enforce_anatomy_probability_hierarchy,
    write_anatomy_probability_bundle,
)


def test_anatomy_torch_targets_follow_ignore_policy() -> None:
    raw = torch.tensor([[[[0, 1], [2, 3]]]], dtype=torch.int64)
    targets, valid = build_anatomy_head_targets_torch(raw)

    assert targets.shape == (1, 3, 2, 2)
    assert valid.shape == (1, 3, 2, 2)
    assert torch.equal(targets[0, 0], torch.tensor([[0.0, 1.0], [1.0, 1.0]]))
    assert torch.equal(targets[0, 1], torch.tensor([[0.0, 1.0], [0.0, 0.0]]))
    assert torch.equal(targets[0, 2], torch.tensor([[0.0, 0.0], [1.0, 0.0]]))
    assert valid[0, 1, 1, 1].item() == 0
    assert valid[0, 2, 1, 1].item() == 0


def test_masked_anatomy_loss_ignores_lesion_for_pz_tz() -> None:
    raw = torch.tensor([[[[3, 3], [3, 3]]]], dtype=torch.int64)
    logits = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    loss = MaskedAnatomySegLoss()(logits, raw)
    assert torch.isfinite(loss)
    assert float(loss) > 0


def test_anatomy_modality_dropout_keeps_t2_and_zeros_adc_dwi_when_requested() -> None:
    data = torch.arange(1, 1 + 3 * 2 * 2 * 2, dtype=torch.float32).reshape(1, 3, 2, 2, 2)
    dropped = apply_anatomy_modality_dropout(data, adc_dropout_p=1.0, dwi_dropout_p=1.0, training=True)
    assert torch.equal(dropped[:, 0], data[:, 0])
    assert torch.count_nonzero(dropped[:, 1]) == 0
    assert torch.count_nonzero(dropped[:, 2]) == 0

    t2_only = build_t2_only_input(data)
    assert torch.equal(t2_only[:, 0], data[:, 0])
    assert torch.count_nonzero(t2_only[:, 1]) == 0
    assert torch.count_nonzero(t2_only[:, 2]) == 0


def test_anatomy_consistency_weight_uses_linear_warmup() -> None:
    assert np.isclose(anatomy_consistency_weight(current_epoch=0, base_lambda=0.2, warmup_epochs=10), 0.0)
    assert np.isclose(anatomy_consistency_weight(current_epoch=5, base_lambda=0.2, warmup_epochs=10), 0.1)
    assert np.isclose(anatomy_consistency_weight(current_epoch=10, base_lambda=0.2, warmup_epochs=10), 0.2)
    assert np.isclose(anatomy_consistency_weight(current_epoch=25, base_lambda=0.2, warmup_epochs=10), 0.2)


def test_anatomy_consistency_loss_masks_lesion_voxels() -> None:
    full_logits = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    t2_logits = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    t2_logits[:, :, 0, 1] = 4.0
    raw_target = torch.tensor([[[[0, 3], [0, 0]]]], dtype=torch.int64)
    masked_loss = anatomy_consistency_loss(full_logits, t2_logits, raw_target, mask_lesion=True)
    unmasked_loss = anatomy_consistency_loss(full_logits, t2_logits, raw_target, mask_lesion=False)
    assert torch.isfinite(masked_loss)
    assert torch.isfinite(unmasked_loss)
    assert np.isclose(float(masked_loss), 0.0)
    assert float(unmasked_loss) > float(masked_loss)


def test_anatomy_tp_fp_fn_tn_respects_lesion_masking() -> None:
    logits = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    logits[:, 0, 0, 1] = 10.0
    logits[:, 1, 0, 1] = 10.0
    raw_target = torch.tensor([[[[0, 3], [1, 2]]]], dtype=torch.int64)

    tp, fp, fn, tn = anatomy_tp_fp_fn_tn(logits, raw_target)

    assert tp.tolist() == [1.0, 0.0, 0.0]
    assert fp.tolist() == [0.0, 0.0, 0.0]
    assert fn.tolist() == [2.0, 1.0, 1.0]
    assert tn.tolist() == [1.0, 2.0, 2.0]


def test_build_anatomy_validation_summary_aggregates_per_case_metrics() -> None:
    metric_per_case = [
        {
            "reference_file": "case_a",
            "prediction_file": "case_a.npz",
            "metrics": {
                ANATOMY_REGION_KEYS[0]: {"Dice": 0.8, "IoU": 2 / 3, "FP": 1.0, "TP": 4.0, "FN": 1.0, "TN": 10.0, "n_pred": 5.0, "n_ref": 5.0},
                ANATOMY_REGION_KEYS[1]: {"Dice": 0.6, "IoU": 3 / 7, "FP": 2.0, "TP": 3.0, "FN": 2.0, "TN": 9.0, "n_pred": 5.0, "n_ref": 5.0},
                ANATOMY_REGION_KEYS[2]: {"Dice": 0.4, "IoU": 0.25, "FP": 3.0, "TP": 2.0, "FN": 3.0, "TN": 8.0, "n_pred": 5.0, "n_ref": 5.0},
            },
        },
        {
            "reference_file": "case_b",
            "prediction_file": "case_b.npz",
            "metrics": {
                ANATOMY_REGION_KEYS[0]: {"Dice": 1.0, "IoU": 1.0, "FP": 0.0, "TP": 5.0, "FN": 0.0, "TN": 11.0, "n_pred": 5.0, "n_ref": 5.0},
                ANATOMY_REGION_KEYS[1]: {"Dice": 0.8, "IoU": 2 / 3, "FP": 1.0, "TP": 4.0, "FN": 1.0, "TN": 10.0, "n_pred": 5.0, "n_ref": 5.0},
                ANATOMY_REGION_KEYS[2]: {"Dice": 0.6, "IoU": 3 / 7, "FP": 2.0, "TP": 3.0, "FN": 2.0, "TN": 9.0, "n_pred": 5.0, "n_ref": 5.0},
            },
        },
    ]

    summary = build_anatomy_validation_summary(metric_per_case)

    assert np.isclose(summary["mean"][ANATOMY_REGION_KEYS[0]]["Dice"], 0.9)
    assert np.isclose(summary["mean"][ANATOMY_REGION_KEYS[1]]["Dice"], 0.7)
    assert np.isclose(summary["mean"][ANATOMY_REGION_KEYS[2]]["Dice"], 0.5)
    assert np.isclose(summary["foreground_mean"]["Dice"], 0.7)
    assert summary["channel_names"] == list(ANATOMY_PROBABILITY_CHANNELS)


class _FakePlansManager:
    transpose_forward = (0, 1, 2)
    transpose_backward = (0, 1, 2)


class _FakeConfigurationManager:
    spacing = (1.0, 1.0, 1.0)

    @staticmethod
    def resampling_fn_probabilities(predicted_logits, target_shape, current_spacing, target_spacing):
        return predicted_logits


def test_anatomy_probability_export_bundle_uses_three_channels(tmp_path: Path) -> None:
    logits = torch.tensor(
        [
            [[[-2.0, 2.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]],
            [[[1.0, -1.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 0.0], [2.0, -2.0]], [[0.0, 0.0], [0.0, 0.0]]],
        ],
        dtype=torch.float32,
    )
    properties = {
        "spacing": (1.0, 1.0, 1.0),
        "shape_after_cropping_and_before_resampling": (2, 2, 2),
        "shape_before_cropping": (2, 2, 2),
        "bbox_used_for_cropping": [[0, 2], [0, 2], [0, 2]],
    }
    probabilities = convert_anatomy_logits_to_probabilities_with_correct_shape(
        logits,
        plans_manager=_FakePlansManager(),
        configuration_manager=_FakeConfigurationManager(),
        properties_dict=properties,
    )
    out_path = write_anatomy_probability_bundle(
        probabilities=probabilities,
        output_file_truncated=tmp_path / "case_a",
        channel_names=ANATOMY_PROBABILITY_CHANNELS,
        properties_dict=properties,
        save_quality_masks=True,
    )
    payload = np.load(out_path, allow_pickle=True)
    assert payload["probabilities"].shape == (3, 2, 2, 2)
    assert payload["channel_names"].tolist() == list(ANATOMY_PROBABILITY_CHANNELS)
    assert np.all(payload["probabilities"][1] <= payload["probabilities"][0] + 1e-6)
    assert np.all(payload["probabilities"][2] <= payload["probabilities"][0] + 1e-6)
    assert np.all(payload["probabilities"][1] + payload["probabilities"][2] <= payload["probabilities"][0] + 1e-6)
    qc = np.load(tmp_path / "case_a_qc.npz")
    assert qc["masks"].shape == (3, 2, 2, 2)
    assert np.array_equal(qc["masks"], anatomy_hard_masks_from_probabilities(payload["probabilities"]))


def test_hierarchy_postprocess_preserves_wg_and_clamps_subregions() -> None:
    probabilities = np.asarray(
        [
            [[[0.4, 0.8]]],
            [[[0.7, 0.6]]],
            [[[0.5, 0.7]]],
        ],
        dtype=np.float32,
    )
    corrected = enforce_anatomy_probability_hierarchy(probabilities)
    assert np.allclose(corrected[0], probabilities[0])
    assert np.all(corrected[1] <= corrected[0] + 1e-6)
    assert np.all(corrected[2] <= corrected[0] + 1e-6)
    assert np.all(corrected[1] + corrected[2] <= corrected[0] + 1e-6)
