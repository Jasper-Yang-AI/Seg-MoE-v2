from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from segmoe_v2.nnunet_anatomy import (
    ANATOMY_PROBABILITY_CHANNELS,
    MaskedAnatomySegLoss,
    anatomy_hard_masks_from_probabilities,
    build_anatomy_head_targets_torch,
    convert_anatomy_logits_to_probabilities_with_correct_shape,
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
    qc = np.load(tmp_path / "case_a_qc.npz")
    assert qc["masks"].shape == (3, 2, 2, 2)
    assert np.array_equal(qc["masks"], anatomy_hard_masks_from_probabilities(payload["probabilities"]))
