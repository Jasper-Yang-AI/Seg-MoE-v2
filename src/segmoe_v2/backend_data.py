from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

try:
    import nibabel as nib
except ModuleNotFoundError:
    class _MissingNibabel:
        def _raise(self, *_args: Any, **_kwargs: Any) -> None:
            raise ModuleNotFoundError(
                "nibabel is required for label conversion during backend data export. "
                "Install the project runtime dependencies first."
            )

        def load(self, *_args: Any, **_kwargs: Any) -> None:
            self._raise()

        def save(self, *_args: Any, **_kwargs: Any) -> None:
            self._raise()

        def Nifti1Image(self, *_args: Any, **_kwargs: Any) -> None:
            self._raise()

    nib = _MissingNibabel()

from .contracts import CaseManifestRow
from .io_utils import save_json, save_jsonl, save_pickle, stable_hash


TaskName = Literal["anatomy", "lesion"]
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9]+")


def _sanitize_dataset_name(name: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", str(name).strip()).strip("_")
    return cleaned or "SegMoE"


def resolve_vendored_backend_root(backend: str, explicit_root: str | Path | None = None) -> Path:
    if explicit_root is not None:
        return Path(explicit_root)
    package_root = Path(__file__).resolve().parent
    candidates = {
        "nnunet": ("nnU-Net",),
        "mednext": ("MedNeXt-main", "MedNeXt", "mednext"),
        "segmamba": ("SegMamba-main", "SegMamba", "segmamba"),
    }
    key = str(backend).lower()
    if key not in candidates:
        raise ValueError(f"Unsupported backend: {backend}")
    roots = [package_root / name for name in candidates[key]]
    return next((root for root in roots if root.exists()), roots[0])


def _link_or_copy_file(source: str | Path, destination: str | Path) -> Path:
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _task_probability_channels(task: TaskName) -> list[str]:
    if task == "anatomy":
        return ["P_WG", "P_PZ", "P_TZ"]
    return ["P_lesion"]


def _task_output_heads(task: TaskName) -> list[str]:
    if task == "anatomy":
        return ["WG", "PZ", "TZ"]
    return ["lesion"]


def _build_dataset_records(rows: Iterable[CaseManifestRow], *, task: TaskName) -> list[dict[str, Any]]:
    materialized = list(rows)
    manifest_hash = stable_hash([row.to_dict() for row in materialized])
    records: list[dict[str, Any]] = []
    for row in materialized:
        records.append(
            {
                "case_id": row.case_id,
                "patient_id": row.patient_id,
                "cohort_type": row.cohort_type,
                "era_bin": row.era_bin,
                "fixed_split": row.fixed_split,
                "val_fold": row.val_fold,
                "task": task,
                "image": [str(row.t2w_path), str(row.adc_path), str(row.dwi_path)],
                "label": str(row.label_path),
                "has_lesion_label3": row.has_lesion_label3,
                "label_unique_values": list(row.label_unique_values),
                "output_heads": _task_output_heads(task),
                "probability_channels": _task_probability_channels(task),
                "source_manifest_hash": manifest_hash,
                "metadata": dict(row.metadata),
            }
        )
    return records


def _build_dataset_json(rows: list[CaseManifestRow], *, dataset_name: str, task: TaskName) -> dict[str, Any]:
    manifest_hash = stable_hash([row.to_dict() for row in rows])
    base = {
        "name": str(dataset_name),
        "description": "SegMoE v2 task-aware canonical dataset export",
        "channel_names": {"0": "T2W", "1": "ADC", "2": "DWI"},
        "numTraining": sum(1 for row in rows if row.fixed_split == "trainval"),
        "file_ending": ".nii.gz",
        "segmoe_task": task,
        "segmoe_source_manifest_hash": manifest_hash,
        "segmoe_probability_channels": _task_probability_channels(task),
    }
    if task == "anatomy":
        base["labels"] = {"background": 0, "PZ": 1, "TZ": 2, "lesion": 3}
        base["segmoe_target_adapter"] = "nnUNetTrainerSegMoEAnatomy"
    else:
        base["labels"] = {"background": 0, "lesion": 1}
        base["segmoe_target_adapter"] = "binary_lesion_from_label3_with_nca_zero_mask"
    return base


def _export_label_for_task(
    row: CaseManifestRow,
    *,
    task: TaskName,
    destination: str | Path,
) -> Path:
    destination = Path(destination)
    if task == "anatomy":
        return _link_or_copy_file(row.label_path, destination)

    image = nib.load(str(row.label_path))
    label_array = np.asanyarray(image.dataobj)
    if str(row.cohort_type).lower() == "nca":
        lesion = np.zeros_like(label_array, dtype=np.uint8)
    else:
        lesion = (label_array == 3).astype(np.uint8)
    header = image.header.copy() if hasattr(image.header, "get_data_dtype") else None
    exported = nib.Nifti1Image(lesion, affine=image.affine, header=header)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(exported, str(destination))
    return destination


def export_nnunet_task(
    rows: Iterable[CaseManifestRow],
    *,
    task_root: str | Path,
    dataset_id: int,
    dataset_name: str,
    task: TaskName = "lesion",
) -> dict[str, Path]:
    materialized = list(rows)
    dataset_dir = Path(task_root) / f"Dataset{int(dataset_id):03d}_{_sanitize_dataset_name(dataset_name)}"
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    labels_ts = dataset_dir / "labelsTs"
    for directory in (images_tr, labels_tr, images_ts, labels_ts):
        directory.mkdir(parents=True, exist_ok=True)

    for row in materialized:
        target_images = images_tr if row.fixed_split == "trainval" else images_ts
        target_labels = labels_tr if row.fixed_split == "trainval" else labels_ts
        _link_or_copy_file(row.t2w_path, target_images / f"{row.case_id}_0000.nii.gz")
        _link_or_copy_file(row.adc_path, target_images / f"{row.case_id}_0001.nii.gz")
        _link_or_copy_file(row.dwi_path, target_images / f"{row.case_id}_0002.nii.gz")
        _export_label_for_task(row, task=task, destination=target_labels / f"{row.case_id}.nii.gz")

    outputs = {
        "dataset_dir": dataset_dir,
        "dataset_json": save_json(_build_dataset_json(materialized, dataset_name=dataset_name, task=task), dataset_dir / "dataset.json"),
        "splits": save_json(
            [
                {
                    "train": sorted(r.case_id for r in materialized if r.fixed_split == "trainval" and r.val_fold != fold),
                    "val": sorted(r.case_id for r in materialized if r.fixed_split == "trainval" and r.val_fold == fold),
                }
                for fold in sorted({row.val_fold for row in materialized if row.fixed_split == "trainval" and row.val_fold is not None})
            ],
            dataset_dir / "splits_final.json",
        ),
        "dataset_index": save_jsonl(_build_dataset_records(materialized, task=task), dataset_dir / "dataset_index.jsonl"),
    }
    return outputs


def export_mednext_task(
    rows: Iterable[CaseManifestRow],
    *,
    task_root: str | Path,
    dataset_id: int,
    dataset_name: str,
    task: TaskName = "lesion",
) -> dict[str, Path]:
    materialized = list(rows)
    dataset_dir = Path(task_root) / f"Task{int(dataset_id):03d}_{_sanitize_dataset_name(dataset_name)}"
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    labels_ts = dataset_dir / "labelsTs"
    for directory in (images_tr, labels_tr, images_ts, labels_ts):
        directory.mkdir(parents=True, exist_ok=True)

    for row in materialized:
        target_images = images_tr if row.fixed_split == "trainval" else images_ts
        target_labels = labels_tr if row.fixed_split == "trainval" else labels_ts
        _link_or_copy_file(row.t2w_path, target_images / f"{row.case_id}_0000.nii.gz")
        _link_or_copy_file(row.adc_path, target_images / f"{row.case_id}_0001.nii.gz")
        _link_or_copy_file(row.dwi_path, target_images / f"{row.case_id}_0002.nii.gz")
        _export_label_for_task(row, task=task, destination=target_labels / f"{row.case_id}.nii.gz")

    train_ids = sorted(row.case_id for row in materialized if row.fixed_split == "trainval")
    test_ids = sorted(row.case_id for row in materialized if row.fixed_split == "test")
    manifest_hash = stable_hash([row.to_dict() for row in materialized])
    dataset_json = {
        "name": str(dataset_name),
        "description": "SegMoE v2 canonical dataset export for MedNeXt",
        "tensorImageSize": "4D",
        "reference": "",
        "licence": "",
        "release": "0.1",
        "modality": {"0": "T2W", "1": "ADC", "2": "DWI"},
        "numTraining": len(train_ids),
        "numTest": len(test_ids),
        "training": [{"image": f"./imagesTr/{case_id}.nii.gz", "label": f"./labelsTr/{case_id}.nii.gz"} for case_id in train_ids],
        "test": [f"./imagesTs/{case_id}.nii.gz" for case_id in test_ids],
        "segmoe_task": task,
        "segmoe_source_manifest_hash": manifest_hash,
        "segmoe_probability_channels": _task_probability_channels(task),
    }
    if task == "anatomy":
        dataset_json["labels"] = {"0": "background", "1": "PZ", "2": "TZ", "3": "lesion"}
        dataset_json["segmoe_target_adapter"] = "multiclass_anatomy_labels"
    else:
        dataset_json["labels"] = {"0": "background", "1": "lesion"}
        dataset_json["segmoe_target_adapter"] = "binary_lesion_from_label3_with_nca_zero_mask"

    outputs = {
        "dataset_dir": dataset_dir,
        "dataset_json": save_json(dataset_json, dataset_dir / "dataset.json"),
        "splits": save_pickle(
            [
                {
                    "train": sorted(r.case_id for r in materialized if r.fixed_split == "trainval" and r.val_fold != fold),
                    "val": sorted(r.case_id for r in materialized if r.fixed_split == "trainval" and r.val_fold == fold),
                }
                for fold in sorted({row.val_fold for row in materialized if row.fixed_split == "trainval" and row.val_fold is not None})
            ],
            dataset_dir / "splits_final.pkl",
        ),
        "dataset_index": save_jsonl(_build_dataset_records(materialized, task=task), dataset_dir / "dataset_index.jsonl"),
    }
    return outputs


def prepare_segmamba_data(
    rows: Iterable[CaseManifestRow],
    *,
    output_dir: str | Path,
    task: TaskName = "lesion",
) -> dict[str, Path]:
    materialized = list(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_index = _build_dataset_records(materialized, task=task)
    split_metadata: dict[str, Any] = {
        "model": "SegMamba",
        "modalities": ["T2W", "ADC", "DWI"],
        "task": task,
        "probability_channels": _task_probability_channels(task),
        "input_channels": 3,
        "output_heads": _task_output_heads(task),
        "total_cases": len(materialized),
        "test_cases": sorted(row.case_id for row in materialized if row.fixed_split == "test"),
        "folds": {},
    }
    outputs: dict[str, Path] = {
        "dataset_index": save_jsonl(dataset_index, output_dir / "dataset_index.jsonl"),
        "test_list": save_jsonl(
            [record for record in dataset_index if record["fixed_split"] == "test"],
            output_dir / "test.jsonl",
        ),
    }

    folds = sorted({row.val_fold for row in materialized if row.fixed_split == "trainval" and row.val_fold is not None})
    for fold in folds:
        train_rows = [
            record for record in dataset_index if record["fixed_split"] == "trainval" and record["val_fold"] != fold
        ]
        val_rows = [
            record for record in dataset_index if record["fixed_split"] == "trainval" and record["val_fold"] == fold
        ]
        outputs[f"fold_{fold}_train"] = save_jsonl(train_rows, output_dir / f"fold_{fold}_train.jsonl")
        outputs[f"fold_{fold}_val"] = save_jsonl(val_rows, output_dir / f"fold_{fold}_val.jsonl")
        split_metadata["folds"][str(fold)] = {
            "train_count": len(train_rows),
            "val_count": len(val_rows),
            "train_cases": [row["case_id"] for row in train_rows],
            "val_cases": [row["case_id"] for row in val_rows],
        }

    outputs["split_metadata"] = save_json(split_metadata, output_dir / "split_metadata.json")
    outputs["segmamba_config"] = save_json(
        {
            "model": "SegMamba",
            "dataset_index": str(outputs["dataset_index"]),
            "split_metadata": str(outputs["split_metadata"]),
            "test_list": str(outputs["test_list"]),
            "train_list_pattern": str(output_dir / "fold_{fold}_train.jsonl"),
            "val_list_pattern": str(output_dir / "fold_{fold}_val.jsonl"),
            "task": task,
            "modalities": ["T2W", "ADC", "DWI"],
            "input_channels": 3,
            "output_heads": _task_output_heads(task),
            "probability_channels": _task_probability_channels(task),
        },
        output_dir / "segmamba_config.json",
    )
    return outputs
