from __future__ import annotations

import numpy as np

from segmoe_v2.contracts import CaseManifestRow
from segmoe_v2.datasets import CaseArrays, Layer1LesionDataset, Layer2PatchDataset


def _make_case(case_id: str) -> CaseManifestRow:
    return CaseManifestRow(
        case_id=case_id,
        patient_id=case_id,
        era_bin="2012_2019",
        cohort_type="pca",
        fixed_split="trainval",
        val_fold=0,
        t2w_path=f"{case_id}_0000.nii.gz",
        adc_path=f"{case_id}_0001.nii.gz",
        dwi_path=f"{case_id}_0002.nii.gz",
        label_path=f"{case_id}.nii.gz",
        spacing=(1.0, 1.0, 1.0),
        image_shape=(8, 8, 8),
        affine_hash="abc",
    )


def _make_nca_case(case_id: str) -> CaseManifestRow:
    row = _make_case(case_id)
    return CaseManifestRow(**{**row.to_dict(), "cohort_type": "nca"})


def test_layer1_dataset_stacks_modalities_and_priors() -> None:
    row = _make_case("case_a")
    priors = {
        row.case_id: {
            "P_WG": np.ones((8, 8, 8), dtype=np.float32) * 0.8,
            "P_PZ": np.ones((8, 8, 8), dtype=np.float32) * 0.3,
            "P_TZ": np.ones((8, 8, 8), dtype=np.float32) * 0.7,
        }
    }
    label = np.zeros((8, 8, 8), dtype=np.int16)
    label[2:5, 2:5, 2:5] = 3
    case_cache = {
        row.case_id: CaseArrays(
            modalities={
                "T2W": np.ones((8, 8, 8), dtype=np.float32),
                "ADC": np.ones((8, 8, 8), dtype=np.float32) * 2,
                "DWI": np.ones((8, 8, 8), dtype=np.float32) * 3,
            },
            label=label,
            anatomy_priors=priors[row.case_id],
        )
    }
    dataset = Layer1LesionDataset(
        [row],
        patch_size=(4, 4, 4),
        anatomy_prior_map=priors,
        case_cache=case_cache,
        seed=1,
    )
    image, target, voxel_weight, meta = dataset[0]

    assert image.shape == (6, 4, 4, 4)
    assert target.shape == (4, 4, 4)
    assert voxel_weight.shape == (4, 4, 4)
    assert float(voxel_weight.max()) == 1.25
    assert meta["case_id"] == row.case_id


def test_layer1_dataset_treats_nca_mimic_as_positive_for_high_recall() -> None:
    row = _make_nca_case("case_nca")
    priors = {
        row.case_id: {
            "P_WG": np.ones((8, 8, 8), dtype=np.float32) * 0.8,
            "P_PZ": np.ones((8, 8, 8), dtype=np.float32) * 0.3,
            "P_TZ": np.ones((8, 8, 8), dtype=np.float32) * 0.7,
        }
    }
    label = np.zeros((8, 8, 8), dtype=np.int16)
    label[2:5, 2:5, 2:5] = 3
    case_cache = {
        row.case_id: CaseArrays(
            modalities={
                "T2W": np.ones((8, 8, 8), dtype=np.float32),
                "ADC": np.ones((8, 8, 8), dtype=np.float32) * 2,
                "DWI": np.ones((8, 8, 8), dtype=np.float32) * 3,
            },
            label=label,
            anatomy_priors=priors[row.case_id],
        )
    }
    dataset = Layer1LesionDataset(
        [row],
        patch_size=(4, 4, 4),
        anatomy_prior_map=priors,
        case_cache=case_cache,
        seed=1,
    )
    _image, target, voxel_weight, meta = dataset[(0, "nca_mimic")]

    assert int(target.max()) == 1
    assert float(voxel_weight.min()) == 0.75
    assert meta["requested_crop_mode"] == "nca_mimic"


def test_layer2_dataset_outputs_expected_channel_count_and_fp_weight() -> None:
    row = _make_case("case_b")
    expert_probs = {row.case_id: np.stack([np.full((8, 8, 8), 0.2, dtype=np.float32), np.full((8, 8, 8), 0.5, dtype=np.float32), np.full((8, 8, 8), 0.8, dtype=np.float32)], axis=0)}
    priors = {
        row.case_id: {
            "P_WG": np.ones((8, 8, 8), dtype=np.float32) * 0.9,
            "P_PZ": np.ones((8, 8, 8), dtype=np.float32) * 0.4,
            "P_TZ": np.ones((8, 8, 8), dtype=np.float32) * 0.6,
        }
    }
    fp_risk = {row.case_id: np.pad(np.ones((2, 2, 2), dtype=np.float32), ((3, 3), (3, 3), (3, 3)))}
    label = np.zeros((8, 8, 8), dtype=np.int16)
    label[2:5, 2:5, 2:5] = 3
    case_cache = {
        row.case_id: CaseArrays(
            modalities={
                "T2W": np.ones((8, 8, 8), dtype=np.float32),
                "ADC": np.ones((8, 8, 8), dtype=np.float32) * 2,
                "DWI": np.ones((8, 8, 8), dtype=np.float32) * 3,
            },
            label=label,
            anatomy_priors=priors[row.case_id],
            expert_probs=expert_probs[row.case_id],
            fp_risk_map=fp_risk[row.case_id],
        )
    }

    dataset = Layer2PatchDataset(
        [row],
        patch_size=(4, 4, 4),
        expert_prob_map=expert_probs,
        anatomy_prior_map=priors,
        fp_risk_map=fp_risk,
        case_cache=case_cache,
        seed=5,
    )
    image, target, fp_weight, meta = dataset[0]

    assert image.shape[0] == 12
    assert target.shape == (4, 4, 4)
    assert fp_weight.shape == (4, 4, 4)
    assert float(fp_weight.max()) >= 1.0
    assert meta["case_id"] == row.case_id
