from __future__ import annotations

import numpy as np

from segmoe_v2.calibration import fit_temperature_scaler


def _mean_nll(logits: np.ndarray, target: np.ndarray) -> float:
    return float((np.logaddexp(0.0, logits) - target * logits).mean())


def test_temperature_scaling_fits_without_fallback() -> None:
    logits = np.array([6.0, -6.0, 6.0, -6.0, 6.0, -6.0], dtype=np.float32)
    target = np.array([1, 0, 0, 1, 1, 0], dtype=np.uint8)

    scaler, record = fit_temperature_scaler(
        logits,
        target,
        stage="layer1_to_layer2",
        fold=0,
        expert="expert_a",
        source_oof_manifest_hash="abc",
    )

    assert not record.fallback_used
    assert 0.5 <= scaler.temperature <= 5.0
    assert _mean_nll(scaler.apply_logits(logits), target) <= _mean_nll(logits, target) + 1e-6


def test_temperature_scaling_falls_back_when_no_positives() -> None:
    probs = np.full((8,), 0.2, dtype=np.float32)
    target = np.zeros((8,), dtype=np.uint8)

    scaler, record = fit_temperature_scaler(
        probs,
        target,
        input_domain="probs",
        stage="layer2_to_gate",
        fold=1,
        expert="expert_b",
        source_oof_manifest_hash="xyz",
    )

    assert record.fallback_used
    assert scaler.temperature == 1.0
