from __future__ import annotations

import numpy as np

from segmoe_v2.labels import (
    IGNORE_INDEX,
    build_anatomy_targets,
    build_layer1_high_recall_target,
    build_layer1_lesion_mimic_source,
    build_layer1_source_weight_map,
    build_lesion_target,
    build_masked_head_target,
)


def test_anatomy_targets_follow_wg_pz_tz_rules() -> None:
    mask = np.array(
        [
            [[0, 1], [2, 3]],
            [[3, 2], [1, 0]],
        ],
        dtype=np.int16,
    )
    bundle = build_anatomy_targets(mask)

    assert np.array_equal(bundle["WG"]["target"], np.isin(mask, (1, 2, 3)).astype(np.uint8))
    assert np.array_equal(bundle["PZ"]["target"], (mask == 1).astype(np.uint8))
    assert np.array_equal(bundle["TZ"]["target"], (mask == 2).astype(np.uint8))
    assert np.array_equal(bundle["PZ"]["valid_mask"], mask != 3)
    assert np.array_equal(bundle["TZ"]["valid_mask"], mask != 3)


def test_masked_targets_ignore_lesion_for_pz_tz() -> None:
    mask = np.array([0, 1, 2, 3], dtype=np.int16)
    pz = build_masked_head_target(mask, "PZ")
    tz = build_masked_head_target(mask, "TZ")

    assert pz.tolist() == [0, 1, 0, IGNORE_INDEX]
    assert tz.tolist() == [0, 0, 1, IGNORE_INDEX]


def test_lesion_target_respects_pca_vs_nca() -> None:
    mask = np.array([0, 1, 2, 3, 3], dtype=np.int16)
    assert build_lesion_target(mask, "pca").tolist() == [0, 0, 0, 1, 1]
    assert build_lesion_target(mask, "nca").tolist() == [0, 0, 0, 0, 0]


def test_layer1_source_labels_keep_lesion_and_mimic_positive_sources() -> None:
    mask = np.array([0, 1, 2, 3, 3], dtype=np.int16)
    pca_source = build_layer1_lesion_mimic_source(mask, "pca")
    nca_source = build_layer1_lesion_mimic_source(mask, "nca")

    assert pca_source.tolist() == [0, 0, 0, 1, 1]
    assert nca_source.tolist() == [0, 0, 0, 2, 2]
    assert build_layer1_high_recall_target(pca_source).tolist() == [0, 0, 0, 1, 1]
    assert build_layer1_high_recall_target(nca_source).tolist() == [0, 0, 0, 1, 1]


def test_layer1_source_weight_map_is_source_aware_but_candidate_first() -> None:
    source = np.array([0, 1, 2, 1, 0], dtype=np.uint8)
    weights = build_layer1_source_weight_map(source)

    assert weights.tolist() == [1.0, 1.25, 0.75, 1.25, 1.0]
    assert build_layer1_high_recall_target(source).tolist() == [0, 1, 1, 1, 0]
