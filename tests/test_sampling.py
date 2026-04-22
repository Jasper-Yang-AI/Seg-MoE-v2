from __future__ import annotations

from segmoe_v2.sampling import Layer1HighRecallBatchSampler


def test_layer1_high_recall_sampler_tracks_candidate_policy() -> None:
    cohort_types = ["pca"] * 60 + ["nca"] * 30
    sampler = Layer1HighRecallBatchSampler(cohort_types, batch_size=4, steps_per_epoch=500, seed=7)

    counts = {"pca_lesion": 0, "nca_mimic": 0, "random_gland": 0}
    for batch in sampler:
        for index, mode in batch:
            assert cohort_types[index] in {"pca", "nca"}
            counts[mode] += 1

    total = sum(counts.values())
    assert abs(counts["pca_lesion"] / total - 0.50) < 0.05
    assert abs(counts["nca_mimic"] / total - 0.25) < 0.05
    assert abs(counts["random_gland"] / total - 0.25) < 0.05
