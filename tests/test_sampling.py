from __future__ import annotations

import random

from segmoe_v2.sampling import Layer1BalancedBatchSampler, choose_layer1_crop_mode


def test_layer1_balanced_sampler_tracks_target_ratio() -> None:
    cohort_types = ["pca"] * 60 + ["nca"] * 30
    sampler = Layer1BalancedBatchSampler(cohort_types, batch_size=3, steps_per_epoch=500, seed=7)

    pca_count = 0
    nca_count = 0
    for batch in sampler:
        for idx in batch:
            if cohort_types[idx] == "pca":
                pca_count += 1
            else:
                nca_count += 1

    ratio = pca_count / max(nca_count, 1)
    assert 1.7 <= ratio <= 2.3


def test_layer1_crop_modes_match_expected_bias() -> None:
    rng = random.Random(123)
    pca_modes = [choose_layer1_crop_mode("pca", rng).mode for _ in range(600)]
    nca_modes = [choose_layer1_crop_mode("nca", rng).mode for _ in range(600)]

    pca_pos_ratio = pca_modes.count("lesion_positive") / len(pca_modes)
    nca_boundary_ratio = nca_modes.count("wg_or_boundary_background") / len(nca_modes)

    assert 0.58 <= pca_pos_ratio <= 0.75
    assert 0.42 <= nca_boundary_ratio <= 0.58
