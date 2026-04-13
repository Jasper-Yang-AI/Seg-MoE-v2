from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize

from .contracts import CalibrationRecord
from .features import clip_probs, logit, sigmoid


def _binary_nll(logits: np.ndarray, target: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return np.logaddexp(0.0, logits) - target * logits


@dataclass(frozen=True, slots=True)
class TemperatureScaler:
    temperature: float = 1.0

    def apply_logits(self, logits: np.ndarray) -> np.ndarray:
        return np.asarray(logits, dtype=np.float32) / float(self.temperature)

    def apply_probs(self, probs: np.ndarray) -> np.ndarray:
        logits = logit(clip_probs(np.asarray(probs, dtype=np.float32)))
        return sigmoid(logits / float(self.temperature)).astype(np.float32)

    def to_dict(self) -> dict[str, float]:
        return {"temperature": float(self.temperature)}


def fit_temperature_scaler(
    logits_or_probs: np.ndarray,
    target: np.ndarray,
    *,
    input_domain: str = "logits",
    valid_mask: np.ndarray | None = None,
    bounds: tuple[float, float] = (0.5, 5.0),
    max_neg_to_pos_ratio: int = 10,
    stage: str,
    fold: int,
    expert: str,
    source_oof_manifest_hash: str,
    fit_case_count: int = 1,
    random_seed: int = 42,
) -> tuple[TemperatureScaler, CalibrationRecord]:
    raw = np.asarray(logits_or_probs, dtype=np.float32)
    target = np.asarray(target, dtype=np.uint8)
    if valid_mask is None:
        valid_mask = np.ones(target.shape, dtype=bool)

    if input_domain == "logits":
        logits = raw
    elif input_domain == "probs":
        logits = logit(raw)
    else:
        raise ValueError(f"Unsupported input_domain={input_domain!r}")

    flat_logits = logits[valid_mask].reshape(-1)
    flat_target = target[valid_mask].reshape(-1)

    pos_idx = np.flatnonzero(flat_target == 1)
    neg_idx = np.flatnonzero(flat_target == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        scaler = TemperatureScaler(1.0)
        record = CalibrationRecord(
            stage=stage,  # type: ignore[arg-type]
            fold=int(fold),
            expert=str(expert),
            temperature=1.0,
            fit_case_count=int(fit_case_count),
            n_pos_voxels=int(len(pos_idx)),
            n_neg_voxels=int(len(neg_idx)),
            source_oof_manifest_hash=str(source_oof_manifest_hash),
            fallback_used=True,
        )
        return scaler, record

    rng = np.random.default_rng(random_seed)
    keep_neg = min(len(neg_idx), max_neg_to_pos_ratio * len(pos_idx))
    sampled_neg = rng.choice(neg_idx, size=keep_neg, replace=False) if keep_neg < len(neg_idx) else neg_idx
    keep_idx = np.concatenate([pos_idx, sampled_neg])
    flat_logits = flat_logits[keep_idx].astype(np.float64)
    flat_target = flat_target[keep_idx].astype(np.float64)

    log_low = float(np.log(bounds[0]))
    log_high = float(np.log(bounds[1]))

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        return float(_binary_nll(flat_logits / temperature, flat_target).mean())

    result = optimize.minimize_scalar(objective, bounds=(log_low, log_high), method="bounded")
    temperature = float(np.exp(result.x)) if result.success else 1.0
    scaler = TemperatureScaler(temperature=temperature)
    record = CalibrationRecord(
        stage=stage,  # type: ignore[arg-type]
        fold=int(fold),
        expert=str(expert),
        temperature=temperature,
        fit_case_count=int(fit_case_count),
        n_pos_voxels=int(len(pos_idx)),
        n_neg_voxels=int(len(sampled_neg)),
        source_oof_manifest_hash=str(source_oof_manifest_hash),
        fallback_used=not result.success,
    )
    return scaler, record
