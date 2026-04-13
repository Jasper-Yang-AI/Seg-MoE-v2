from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize

from .features import clip_probs


@dataclass(frozen=True, slots=True)
class StaticConvexFusion:
    weights: np.ndarray
    bias: float = 0.0

    def apply(self, expert_probs: np.ndarray) -> np.ndarray:
        expert_probs = np.asarray(expert_probs, dtype=np.float32)
        if expert_probs.shape[0] != len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} experts, got {expert_probs.shape[0]}")
        fused = np.tensordot(self.weights.astype(np.float32), expert_probs, axes=(0, 0))
        if self.bias != 0.0:
            fused = np.clip(fused + self.bias, 0.0, 1.0)
        return fused.astype(np.float32)


def fit_static_convex_fusion(
    expert_probs: np.ndarray,
    target: np.ndarray,
    *,
    fit_bias: bool = False,
    l2_weight: float = 1e-4,
) -> StaticConvexFusion:
    expert_probs = clip_probs(np.asarray(expert_probs, dtype=np.float32))
    target = np.asarray(target, dtype=np.float32)
    if expert_probs.ndim < 2:
        raise ValueError(f"Expected expert_probs [K,...], got {expert_probs.shape}")

    k = expert_probs.shape[0]
    flat_probs = expert_probs.reshape(k, -1)
    flat_target = target.reshape(-1)

    def unpack(params: np.ndarray) -> tuple[np.ndarray, float]:
        if fit_bias:
            raw_w = params[:-1]
            bias = float(params[-1])
        else:
            raw_w = params
            bias = 0.0
        weights = np.exp(raw_w - raw_w.max())
        weights = weights / weights.sum()
        return weights, bias

    def objective(params: np.ndarray) -> float:
        weights, bias = unpack(params)
        fused = weights @ flat_probs
        fused = np.clip(fused + bias, 1e-6, 1.0 - 1e-6)
        nll = -(flat_target * np.log(fused) + (1.0 - flat_target) * np.log(1.0 - fused)).mean()
        reg = float(l2_weight) * float(np.square(weights).sum())
        return float(nll + reg)

    x0 = np.zeros(k + (1 if fit_bias else 0), dtype=np.float64)
    result = optimize.minimize(objective, x0=x0, method="L-BFGS-B")
    if not result.success:
        weights = np.full((k,), 1.0 / k, dtype=np.float32)
        bias = 0.0
    else:
        weights, bias = unpack(result.x)
        weights = weights.astype(np.float32)
    return StaticConvexFusion(weights=weights, bias=float(bias))
