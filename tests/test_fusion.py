from __future__ import annotations

import numpy as np

from segmoe_v2.fusion import fit_static_convex_fusion


def test_static_convex_fusion_prefers_better_expert() -> None:
    target = np.array([1, 1, 0, 0], dtype=np.float32)
    expert_a = np.array([0.9, 0.8, 0.1, 0.2], dtype=np.float32)
    expert_b = np.array([0.6, 0.6, 0.4, 0.4], dtype=np.float32)
    expert_c = np.array([0.2, 0.3, 0.8, 0.7], dtype=np.float32)

    fusion = fit_static_convex_fusion(np.stack([expert_a, expert_b, expert_c], axis=0), target)

    assert np.isclose(float(fusion.weights.sum()), 1.0, atol=1e-5)
    assert fusion.weights[0] > fusion.weights[1]
    assert fusion.weights[0] > fusion.weights[2]
