from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from segmoe_v2.layer2 import (
    build_layer2_source_labels,
    build_layer2_targets_torch,
    layer2_hard_negative_loss,
    layer2_tp_fp_fn_tn,
)


def test_layer2_source_labels_prioritize_pca_positive_over_fp() -> None:
    label = np.zeros((4, 4, 4), dtype=np.int16)
    label[1, 1, 1] = 3
    fp_risk = np.zeros_like(label, dtype=np.float32)
    fp_risk[1, 1, 1] = 0.8
    fp_risk[2, 2, 2] = 0.7

    source = build_layer2_source_labels(label, "pca", fp_risk)

    assert int(source[1, 1, 1]) == 1
    assert int(source[2, 2, 2]) == 2
    assert int(source[0, 0, 0]) == 0


def test_layer2_source_labels_treat_nca_mimic_as_hard_negative() -> None:
    label = np.zeros((4, 4, 4), dtype=np.int16)
    label[1, 1, 1] = 3

    source = build_layer2_source_labels(label, "nca")

    assert int(source[1, 1, 1]) == 2
    assert set(np.unique(source).tolist()) == {0, 2}


def test_layer2_targets_make_label2_negative_with_hard_weight() -> None:
    raw = torch.tensor([[[[[0, 1, 2]]]]], dtype=torch.long)

    target, weight = build_layer2_targets_torch(raw)

    assert target.flatten().tolist() == [0.0, 1.0, 0.0]
    assert weight.flatten().tolist() == [1.0, 1.25, 2.5]


def test_layer2_loss_and_metrics_use_only_label1_as_positive() -> None:
    raw = torch.tensor([[[[[1, 2, 0]]]]], dtype=torch.long)
    logits = torch.tensor([[[[[6.0, -6.0, -6.0]]]]], dtype=torch.float32)

    loss = layer2_hard_negative_loss(logits, raw)
    tp, fp, fn, tn = layer2_tp_fp_fn_tn(logits, raw)

    assert torch.isfinite(loss)
    assert tp.tolist() == [1.0]
    assert fp.tolist() == [0.0]
    assert fn.tolist() == [0.0]
    assert tn.tolist() == [2.0]
