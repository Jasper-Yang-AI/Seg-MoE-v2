from __future__ import annotations

import sys
from pathlib import Path

from segmoe_v2.contracts import CaseManifestRow, TaskSpec
from segmoe_v2.runners import MedNeXtRunner, NnUNetResEncRunner, SegMambaRunner


def _case(case_id: str) -> CaseManifestRow:
    return CaseManifestRow(
        case_id=case_id,
        patient_id=case_id,
        era_bin="2012_2019",
        cohort_type="pca",
        has_lesion_label3=True,
        label_unique_values=(0, 1, 2, 3),
        fixed_split="trainval",
        val_fold=0,
        t2w_path=Path(f"{case_id}_0000.nii.gz"),
        adc_path=Path(f"{case_id}_0001.nii.gz"),
        dwi_path=Path(f"{case_id}_0002.nii.gz"),
        label_path=Path(f"{case_id}.nii.gz"),
        spacing=(1.0, 1.0, 1.0),
        image_shape=(8, 8, 8),
        affine_hash="abc",
    )


def test_vendored_runners_resolve_default_roots_and_commands(tmp_path: Path) -> None:
    case = _case("case_001")

    nnunet = NnUNetResEncRunner(workspace=tmp_path)
    anatomy_train = nnunet.train_fold(
        0,
        TaskSpec.anatomy(),
        [case],
        {"dataset_id": 500, "dry_run": True},
    )
    assert "nnUNetTrainerSegMoEAnatomy" in anatomy_train["command"]
    assert "--npz" in anatomy_train["command"]
    nnunet_train = nnunet.train_fold(
        0,
        TaskSpec.lesion(),
        [case],
        {"dataset_id": 501, "dry_run": True},
    )
    assert nnunet.repo_root.name == "nnU-Net"
    assert nnunet_train["command"][:3] == [sys.executable, "-m", "nnunetv2.run.run_training"]
    assert "nnUNetTrainerSegMoELayer1" in nnunet_train["command"]
    assert "--npz" in nnunet_train["command"]
    nnunet_predict = nnunet.predict_fold(
        0,
        "val_0",
        TaskSpec.anatomy(),
        [case],
        {},
        {
            "dataset_id": 501,
            "predict_input_dir": tmp_path / "predict_input",
            "predict_output_dir": tmp_path / "predict_output",
            "dry_run": True,
        },
    )[0]
    assert nnunet_predict["command"][:3] == [sys.executable, "-m", "segmoe_v2.nnunet_anatomy_predict"]
    assert nnunet_predict["split"] == "val_0"
    nnunet_env = nnunet._env({})
    assert "nnUNet_raw" in nnunet_env
    repo_root = Path(__file__).resolve().parents[1]
    assert nnunet_env["nnUNet_raw"] == str(repo_root / "nnUNet_raw")
    assert nnunet_env["nnUNet_preprocessed"] == str(repo_root / "nnUNet_preprocessed")
    assert nnunet_env["nnUNet_results"] == str(repo_root / "nnUNet_results")
    assert str(nnunet.repo_root) in nnunet_env["PYTHONPATH"]
    assert str(repo_root / "src") in nnunet_env["PYTHONPATH"]

    mednext = MedNeXtRunner(workspace=tmp_path)
    mednext_train = mednext.train_fold(
        0,
        TaskSpec.lesion(),
        [case],
        {"task_id": 502, "dry_run": True},
    )
    assert mednext.repo_root.name == "MedNeXt-main"
    assert mednext_train["command"][:6] == [
        sys.executable,
        "-m",
        "nnunet_mednext.run.run_training",
        "3d_fullres",
        "nnUNetTrainerV2_MedNeXt_S_kernel3_SegMoELayer1",
        "502",
    ]
    mednext_env = mednext._env({})
    assert "nnUNet_raw_data_base" in mednext_env
    assert str(mednext.repo_root) in mednext_env["PYTHONPATH"]

    segmamba = SegMambaRunner(workspace=tmp_path)
    assert segmamba.repo_root.name == "SegMamba"
    config_path = tmp_path / "segmamba_config.json"
    config_path.write_text("{}", encoding="utf-8")
    segmamba_train = segmamba.train_fold(
        0,
        TaskSpec.lesion(),
        [case],
        {"dry_run": True, "config": config_path},
    )
    assert segmamba_train["command"][:4] == [sys.executable, "-m", "segmoe_v2.segmamba_adapter", "train"]
    assert "--config" in segmamba_train["command"]
    segmamba_env = segmamba._env(0, "train", {})
    assert str(segmamba.repo_root / "mamba") in segmamba_env["PYTHONPATH"]
    assert str(segmamba.repo_root / "causal-conv1d") in segmamba_env["PYTHONPATH"]
