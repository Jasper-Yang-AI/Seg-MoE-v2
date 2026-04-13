from __future__ import annotations

import numpy as np


def clip_probs(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(probs, dtype=np.float32), eps, 1.0 - eps)


def logit(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    probs = clip_probs(probs, eps=eps)
    return np.log(probs / (1.0 - probs))


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-logits))


def binary_entropy(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    probs = clip_probs(probs, eps=eps)
    entropy = -(probs * np.log(probs) + (1.0 - probs) * np.log(1.0 - probs))
    return (entropy / np.log(2.0)).astype(np.float32)


def expert_consensus(expert_probs: np.ndarray) -> np.ndarray:
    expert_probs = np.asarray(expert_probs, dtype=np.float32)
    if expert_probs.ndim < 2:
        raise ValueError(f"Expected expert_probs with leading expert axis, got shape={expert_probs.shape}")
    return expert_probs.mean(axis=0).astype(np.float32)


def expert_disagreement(expert_probs: np.ndarray) -> np.ndarray:
    expert_probs = np.asarray(expert_probs, dtype=np.float32)
    if expert_probs.ndim < 2:
        raise ValueError(f"Expected expert_probs with leading expert axis, got shape={expert_probs.shape}")
    return expert_probs.std(axis=0).astype(np.float32)


def lesion_stats_from_experts(expert_probs: np.ndarray) -> dict[str, np.ndarray]:
    mean_prob = expert_consensus(expert_probs)
    disagreement = expert_disagreement(expert_probs)
    entropy = binary_entropy(mean_prob)
    return {"consensus": mean_prob, "entropy": entropy, "disagreement": disagreement}
