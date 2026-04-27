from pathlib import Path

from segmoe_v2.io_utils import resolve_local_path


def test_resolve_local_path_maps_windows_drive_to_wsl_mount() -> None:
    resolved = resolve_local_path(r"E:\NJMU_prostate_data\njmu_2012_2019_nca_zscore\case_0000.nii.gz")

    assert resolved == Path("/mnt/e/NJMU_prostate_data/njmu_2012_2019_nca_zscore/case_0000.nii.gz")


def test_resolve_local_path_relocates_repo_windows_paths() -> None:
    resolved = resolve_local_path(r"E:\Seg-MoE-v2\data\exports\anatomy_all\prediction_manifest.jsonl", root="/tmp/repo")

    assert resolved == Path("/tmp/repo/data/exports/anatomy_all/prediction_manifest.jsonl")


def test_resolve_local_path_maps_relative_windows_path_to_project_root() -> None:
    resolved = resolve_local_path(r"data\geometry_fixed\case_0000.nii.gz", root="/tmp/repo")

    assert resolved == Path("/tmp/repo/data/geometry_fixed/case_0000.nii.gz")
