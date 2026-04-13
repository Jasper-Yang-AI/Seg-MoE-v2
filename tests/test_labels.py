from __future__ import annotations

import numpy as np

from segmoe_v2.labels import IGNORE_INDEX, build_anatomy_targets, build_lesion_target, build_masked_head_target


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
