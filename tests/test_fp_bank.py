from __future__ import annotations

import numpy as np

from segmoe_v2.fp_bank import build_fp_bank


def test_fp_bank_extracts_false_positive_component_and_risk_map() -> None:
    expert_probs = np.zeros((3, 12, 12, 12), dtype=np.float32)
    expert_probs[:, 1:4, 1:4, 1:4] = 0.9
    gt_lesion = np.zeros((12, 12, 12), dtype=np.uint8)
    image_channels = {
        "T2W": np.ones((12, 12, 12), dtype=np.float32),
        "ADC": np.ones((12, 12, 12), dtype=np.float32) * 2,
        "DWI": np.ones((12, 12, 12), dtype=np.float32) * 3,
    }
    anatomy_priors = {
        "P_WG": np.zeros((12, 12, 12), dtype=np.float32),
        "P_PZ": np.zeros((12, 12, 12), dtype=np.float32),
        "P_TZ": np.zeros((12, 12, 12), dtype=np.float32),
    }

    records, risk_map = build_fp_bank(
        case_id="case_001",
        source_layer="layer1",
        predictor_fold=0,
        expert_probs=expert_probs,
        gt_lesion=gt_lesion,
        image_channels=image_channels,
        anatomy_priors=anatomy_priors,
        threshold=0.5,
        min_component_size=4,
    )

    assert len(records) == 1
    assert records[0].dominant_zone == "outside_WG"
    assert records[0].fp_kind == "pure"
    assert risk_map[2, 2, 2] > 0
