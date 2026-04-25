from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from segmoe_v2.layer1 import build_layer1_source_targets_torch, layer1_source_aware_loss, layer1_tp_fp_fn_tn


def test_layer1_source_targets_preserve_candidate_positive_and_source_weights() -> None:
    raw_target = torch.tensor([[[[0, 1], [2, 0]]]], dtype=torch.long)

    target, weights = build_layer1_source_targets_torch(raw_target)

    assert target.tolist() == [[[[0.0, 1.0], [1.0, 0.0]]]]
    assert weights.tolist() == [[[[1.0, 1.25], [0.75, 1.0]]]]


def test_layer1_source_aware_loss_and_metrics_treat_mimic_as_positive() -> None:
    raw_target = torch.tensor([[[[0, 1], [2, 0]]]], dtype=torch.long)
    logits = torch.tensor([[[[-6.0, 6.0], [6.0, -6.0]]]], dtype=torch.float32)

    loss = layer1_source_aware_loss(logits, raw_target)
    tp, fp, fn, tn = layer1_tp_fp_fn_tn(logits, raw_target)

    assert float(loss) < 0.1
    assert tp.tolist() == [2.0]
    assert fp.tolist() == [0.0]
    assert fn.tolist() == [0.0]
    assert tn.tolist() == [2.0]
