from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

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
from .gland_crop import DEFAULT_MIN_CROP_SIZE_ZYX, GlandCropRecord, index_gland_crop_manifest, load_gland_crop_manifest
from .io_utils import load_jsonl, resolve_local_path, save_json, save_jsonl, save_pickle, stable_hash
from .labels import (
    LAYER1_BACKGROUND_WEIGHT,
    LAYER1_CANDIDATE_POSITIVE_LABEL_VALUES,
    LAYER1_SOURCE_POSITIVE_WEIGHTS,
    build_layer1_high_recall_target,
    build_layer1_source_weight_map,
)


TaskName = Literal["anatomy", "lesion"]
Layer1MainLabelMode = Literal["binary", "source"]
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9]+")
LAYER1_INPUT_CHANNELS = ("T2W", "ADC", "DWI", "P_WG", "P_PZ", "P_TZ")
LAYER1_POSITIVE_LABEL_VALUES = LAYER1_CANDIDATE_POSITIVE_LABEL_VALUES
LAYER1_SAMPLING_POLICY = {"pca_lesion": 0.50, "nca_mimic": 0.25, "random_gland": 0.25}
LAYER1_SOURCE_AWARE_WEIGHTS = dict(LAYER1_SOURCE_POSITIVE_WEIGHTS)


def _sanitize_dataset_name(name: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", str(name).strip()).strip("_")
    return cleaned or "SegMoE"


def resolve_vendored_backend_root(backend: str, explicit_root: str | Path | None = None) -> Path:
    if explicit_root is not None:
        return Path(explicit_root)
    package_root = Path(__file__).resolve().parent
    project_root = package_root.parents[1]
    env_vars = {
        "nnunet": "SEGMOE_NNUNET_ROOT",
        "mednext": "SEGMOE_MEDNEXT_ROOT",
        "segmamba": "SEGMOE_SEGMAMBA_ROOT",
    }
    candidates = {
        "nnunet": ("nnU-Net",),
        "mednext": ("MedNeXt-main", "MedNeXt", "mednext"),
        "segmamba": ("SegMamba", "segmamba"),
    }
    key = str(backend).lower()
    if key not in candidates:
        raise ValueError(f"Unsupported backend: {backend}")
    env_root = os.environ.get(env_vars[key])
    if env_root:
        return Path(env_root)
    roots = [project_root / "external" / name for name in candidates[key]]
    roots.extend(package_root / name for name in candidates[key])
    return next((root for root in roots if root.exists()), roots[0])


def _link_or_copy_file(source: str | Path, destination: str | Path) -> Path:
    source = resolve_local_path(source)
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


def _coerce_crop_index(crop_manifest: str | Path | Iterable[GlandCropRecord] | None) -> dict[str, GlandCropRecord]:
    if crop_manifest is None:
        return {}
    if isinstance(crop_manifest, (str, Path)):
        return index_gland_crop_manifest(load_gland_crop_manifest(crop_manifest))
    return index_gland_crop_manifest(crop_manifest)


def _load_prediction_index(prediction_manifest: str | Path | None) -> dict[str, dict[str, Any]]:
    if prediction_manifest is None:
        return {}
    return {str(row["case_id"]): dict(row) for row in load_jsonl(prediction_manifest)}


def _normalise_channel_names(raw: Any) -> list[str]:
    if raw is None:
        return []
    arr = np.asarray(raw)
    return [str(item.decode("utf-8") if isinstance(item, bytes) else item) for item in arr.tolist()]


def _load_anatomy_prior_bundle(record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    prob_path = record.get("prob_path") or record.get("probabilities_path")
    if not prob_path:
        raise KeyError(f"Anatomy prediction record for {record.get('case_id')} has no prob_path.")
    prob_path = resolve_local_path(prob_path)
    payload = np.load(str(prob_path), allow_pickle=True)
    if "probabilities" in payload:
        probabilities = np.asarray(payload["probabilities"], dtype=np.float32)
    elif "probs" in payload:
        probabilities = np.asarray(payload["probs"], dtype=np.float32)
    else:
        raise KeyError(f"{prob_path} must contain `probabilities` or `probs`.")
    channel_names = [str(v) for v in record.get("channel_names", ())] or _normalise_channel_names(
        payload["channel_names"] if "channel_names" in payload else None
    )
    if not channel_names:
        channel_names = ["P_WG", "P_PZ", "P_TZ"][: probabilities.shape[0]]
    return {name: probabilities[idx].astype(np.float32) for idx, name in enumerate(channel_names)}


def _crop_zyx(arr: np.ndarray, bbox_zyx: tuple[int, int, int, int, int, int] | None) -> np.ndarray:
    if bbox_zyx is None:
        return arr
    z0, z1, y0, y1, x0, x1 = bbox_zyx
    return arr[..., z0:z1, y0:y1, x0:x1]


def _crop_xyz(arr: np.ndarray, bbox_zyx: tuple[int, int, int, int, int, int] | None) -> np.ndarray:
    if bbox_zyx is None:
        return arr
    z0, z1, y0, y1, x0, x1 = bbox_zyx
    return arr[x0:x1, y0:y1, z0:z1]


def _load_nifti_zyx(path: str | Path, *, dtype: np.dtype | None = np.float32) -> np.ndarray:
    path = resolve_local_path(path)
    arr = np.asanyarray(nib.load(str(path)).dataobj)
    arr = np.transpose(arr, (2, 1, 0))
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def _cropped_affine(affine: np.ndarray, bbox_zyx: tuple[int, int, int, int, int, int] | None) -> np.ndarray:
    out = np.asarray(affine, dtype=np.float64).copy()
    if bbox_zyx is None:
        return out
    z0, _z1, y0, _y1, x0, _x1 = bbox_zyx
    origin = np.asarray(affine, dtype=np.float64) @ np.asarray([x0, y0, z0, 1.0], dtype=np.float64)
    out[:3, 3] = origin[:3]
    return out


def _save_nifti_xyz(
    data_xyz: np.ndarray,
    *,
    reference_image: Any,
    destination: str | Path,
    bbox_zyx: tuple[int, int, int, int, int, int] | None,
) -> Path:
    destination = Path(destination)
    header = reference_image.header.copy() if hasattr(reference_image.header, "copy") else None
    exported = nib.Nifti1Image(data_xyz, affine=_cropped_affine(reference_image.affine, bbox_zyx), header=header)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(exported, str(destination))
    return destination


def _write_cropped_source_nifti(
    source: str | Path,
    destination: str | Path,
    *,
    bbox_zyx: tuple[int, int, int, int, int, int] | None,
) -> Path:
    source = resolve_local_path(source)
    if bbox_zyx is None:
        return _link_or_copy_file(source, destination)
    image = nib.load(str(source))
    data = _crop_xyz(np.asanyarray(image.dataobj), bbox_zyx)
    return _save_nifti_xyz(data, reference_image=image, destination=destination, bbox_zyx=bbox_zyx)


def _write_prior_nifti(
    prior_zyx: np.ndarray,
    *,
    reference_path: str | Path,
    destination: str | Path,
    bbox_zyx: tuple[int, int, int, int, int, int] | None,
) -> Path:
    reference_path = resolve_local_path(reference_path)
    reference = nib.load(str(reference_path))
    cropped = _crop_zyx(np.asarray(prior_zyx, dtype=np.float32), bbox_zyx)
    return _save_nifti_xyz(
        np.transpose(cropped, (2, 1, 0)),
        reference_image=reference,
        destination=destination,
        bbox_zyx=bbox_zyx,
    )


def _layer1_source_label_from_image(label_image: Any, cohort_type: str) -> np.ndarray:
    label_array = np.asanyarray(label_image.dataobj)
    source = np.zeros(label_array.shape, dtype=np.uint8)
    if str(cohort_type).lower() == "nca":
        source[label_array == 3] = 2
    else:
        source[label_array == 3] = 1
    return source


def _layer1_source_target_weight_from_image(
    label_image: Any,
    cohort_type: str,
    *,
    bbox_zyx: tuple[int, int, int, int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = _layer1_source_label_from_image(label_image, cohort_type)
    source = _crop_xyz(source, bbox_zyx)
    target = build_layer1_high_recall_target(source)
    weights = build_layer1_source_weight_map(source)
    return source.astype(np.uint8), target.astype(np.uint8), weights.astype(np.float32)


def _export_label_bundle_for_task(
    row: CaseManifestRow,
    *,
    task: TaskName,
    destination: str | Path,
    source_destination: str | Path | None = None,
    weight_destination: str | Path | None = None,
    bbox_zyx: tuple[int, int, int, int, int, int] | None = None,
    layer1_main_label_mode: Layer1MainLabelMode = "source",
) -> dict[str, Path]:
    destination = Path(destination)
    label_path = resolve_local_path(row.label_path)
    if task == "anatomy":
        if bbox_zyx is not None:
            raise ValueError("Anatomy export does not support gland ROI cropping.")
        return {"label": _link_or_copy_file(label_path, destination)}

    image = nib.load(str(label_path))
    source, target, weights = _layer1_source_target_weight_from_image(image, row.cohort_type, bbox_zyx=bbox_zyx)
    main_label = source if layer1_main_label_mode == "source" else target
    outputs = {
        "label": _save_nifti_xyz(main_label, reference_image=image, destination=destination, bbox_zyx=bbox_zyx)
    }
    if source_destination is not None:
        outputs["source_label"] = _save_nifti_xyz(
            source,
            reference_image=image,
            destination=source_destination,
            bbox_zyx=bbox_zyx,
        )
    if weight_destination is not None:
        outputs["voxel_weight"] = _save_nifti_xyz(
            weights,
            reference_image=image,
            destination=weight_destination,
            bbox_zyx=bbox_zyx,
        )
    return outputs


def _build_dataset_records(
    rows: Iterable[CaseManifestRow],
    *,
    task: TaskName,
    prediction_index: Mapping[str, Mapping[str, Any]] | None = None,
    crop_index: Mapping[str, GlandCropRecord] | None = None,
    include_test_labels: bool = False,
) -> list[dict[str, Any]]:
    materialized = list(rows)
    manifest_hash = stable_hash([row.to_dict() for row in materialized])
    prediction_index = prediction_index or {}
    crop_index = crop_index or {}
    records: list[dict[str, Any]] = []
    for row in materialized:
        crop = crop_index.get(row.case_id)
        metadata = dict(row.metadata)
        if crop is not None:
            metadata.update(
                {
                    "bbox_zyx": list(crop.bbox_zyx),
                    "crop_shape_zyx": list(crop.crop_shape_zyx),
                    "native_shape_zyx": list(crop.native_shape_zyx),
                    "gland_crop_source_prob_path": crop.source_prob_path,
                    "gland_crop_warning": crop.warning,
                }
            )
        if task == "lesion":
            metadata["positive_label_values"] = list(LAYER1_POSITIVE_LABEL_VALUES)
            metadata["sampling_policy"] = dict(LAYER1_SAMPLING_POLICY)
            metadata["background_weight"] = float(LAYER1_BACKGROUND_WEIGHT)
            metadata["source_positive_weights"] = {str(k): float(v) for k, v in LAYER1_SOURCE_AWARE_WEIGHTS.items()}
        labels_available = row.fixed_split == "trainval" or include_test_labels
        metadata["labels_available"] = labels_available
        image_channels = [
            str(resolve_local_path(row.t2w_path)),
            str(resolve_local_path(row.adc_path)),
            str(resolve_local_path(row.dwi_path)),
        ]
        prior_record = prediction_index.get(row.case_id)
        anatomy_priors: dict[str, str] = {}
        if prior_record is not None:
            anatomy_priors = {
                "prob_path": str(resolve_local_path(prior_record.get("prob_path") or prior_record.get("probabilities_path")))
            }
            image_channels.extend([f"{row.case_id}:P_WG", f"{row.case_id}:P_PZ", f"{row.case_id}:P_TZ"])
        records.append(
            {
                "case_id": row.case_id,
                "patient_id": row.patient_id,
                "cohort_type": row.cohort_type,
                "era_bin": row.era_bin,
                "fixed_split": row.fixed_split,
                "val_fold": row.val_fold,
                "task": task,
                "image": image_channels,
                "label": str(resolve_local_path(row.label_path)) if labels_available else "",
                "anatomy_priors": anatomy_priors,
                "has_lesion_label3": row.has_lesion_label3 if labels_available else False,
                "label_unique_values": list(row.label_unique_values) if labels_available else [],
                "output_heads": _task_output_heads(task),
                "probability_channels": _task_probability_channels(task),
                "source_manifest_hash": manifest_hash,
                "metadata": metadata,
            }
        )
    return records


def _build_dataset_json(
    rows: list[CaseManifestRow],
    *,
    dataset_name: str,
    task: TaskName,
    with_anatomy_priors: bool = False,
    layer1_main_label_mode: Layer1MainLabelMode = "source",
    include_test_labels: bool = False,
) -> dict[str, Any]:
    manifest_hash = stable_hash([row.to_dict() for row in rows])
    channel_names = {"0": "T2W", "1": "ADC", "2": "DWI"}
    if task == "lesion" and with_anatomy_priors:
        channel_names.update({"3": "P_WG", "4": "P_PZ", "5": "P_TZ"})
    base = {
        "name": str(dataset_name),
        "description": "SegMoE v2 task-aware canonical dataset export",
        "channel_names": channel_names,
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
        if layer1_main_label_mode == "source":
            base["labels"] = {"background": 0, "pca_lesion": 1, "nca_mimic": 2}
            base["segmoe_target_adapter"] = "layer1_source_aware_label1_or_label2_positive"
            base["segmoe_main_label_mode"] = "source"
        else:
            base["labels"] = {"background": 0, "candidate": 1}
            base["segmoe_target_adapter"] = "layer1_high_recall_label1_or_label2_positive"
            base["segmoe_main_label_mode"] = "binary"
        base["segmoe_positive_label_values"] = list(LAYER1_POSITIVE_LABEL_VALUES)
        base["segmoe_source_label_values"] = {"background": 0, "pca_lesion": 1, "nca_mimic": 2}
        base["segmoe_background_weight"] = float(LAYER1_BACKGROUND_WEIGHT)
        base["segmoe_source_positive_weights"] = {str(k): float(v) for k, v in LAYER1_SOURCE_AWARE_WEIGHTS.items()}
        base["segmoe_sampling_policy"] = dict(LAYER1_SAMPLING_POLICY)
        base["segmoe_include_test_labels"] = bool(include_test_labels)
        base["segmoe_sidecars"] = {
            "source_labels": {"train": "sourceLabelsTr", "test": "sourceLabelsTs"},
            "voxel_weights": {"train": "weightsTr", "test": "weightsTs"},
        }
    return base


def export_nnunet_task(
    rows: Iterable[CaseManifestRow],
    *,
    task_root: str | Path,
    dataset_id: int,
    dataset_name: str,
    task: TaskName = "lesion",
    anatomy_prediction_manifest: str | Path | None = None,
    crop_manifest: str | Path | Iterable[GlandCropRecord] | None = None,
    layer1_main_label_mode: Layer1MainLabelMode = "source",
    include_test_labels: bool = False,
) -> dict[str, Path]:
    materialized = list(rows)
    prediction_index = _load_prediction_index(anatomy_prediction_manifest)
    crop_index = _coerce_crop_index(crop_manifest)
    with_anatomy_priors = task == "lesion" and bool(prediction_index)
    dataset_dir = Path(task_root) / f"Dataset{int(dataset_id):03d}_{_sanitize_dataset_name(dataset_name)}"
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    source_labels_tr = dataset_dir / "sourceLabelsTr"
    weights_tr = dataset_dir / "weightsTr"
    images_ts = dataset_dir / "imagesTs"
    labels_ts = dataset_dir / "labelsTs"
    source_labels_ts = dataset_dir / "sourceLabelsTs"
    weights_ts = dataset_dir / "weightsTs"
    train_sidecar_dirs = (source_labels_tr, weights_tr) if task == "lesion" else ()
    test_label_dirs = (labels_ts, source_labels_ts, weights_ts) if include_test_labels and task == "lesion" else (
        (labels_ts,) if include_test_labels else ()
    )
    for directory in (images_tr, labels_tr, images_ts, *train_sidecar_dirs, *test_label_dirs):
        directory.mkdir(parents=True, exist_ok=True)

    for row in materialized:
        labels_available = row.fixed_split == "trainval" or include_test_labels
        target_images = images_tr if row.fixed_split == "trainval" else images_ts
        target_labels = labels_tr if row.fixed_split == "trainval" else labels_ts
        target_source_labels = source_labels_tr if row.fixed_split == "trainval" else source_labels_ts
        target_weights = weights_tr if row.fixed_split == "trainval" else weights_ts
        crop = crop_index.get(row.case_id)
        bbox = crop.bbox_zyx if task == "lesion" and crop is not None else None
        _write_cropped_source_nifti(row.t2w_path, target_images / f"{row.case_id}_0000.nii.gz", bbox_zyx=bbox)
        _write_cropped_source_nifti(row.adc_path, target_images / f"{row.case_id}_0001.nii.gz", bbox_zyx=bbox)
        _write_cropped_source_nifti(row.dwi_path, target_images / f"{row.case_id}_0002.nii.gz", bbox_zyx=bbox)
        if with_anatomy_priors:
            prior_record = prediction_index.get(row.case_id)
            if prior_record is None:
                raise KeyError(f"Missing anatomy prediction for case_id={row.case_id}")
            priors = _load_anatomy_prior_bundle(prior_record)
            for channel_idx, channel_name in enumerate(("P_WG", "P_PZ", "P_TZ"), start=3):
                _write_prior_nifti(
                    priors[channel_name],
                    reference_path=row.t2w_path,
                    destination=target_images / f"{row.case_id}_{channel_idx:04d}.nii.gz",
                    bbox_zyx=bbox,
                )
        if labels_available:
            _export_label_bundle_for_task(
                row,
                task=task,
                destination=target_labels / f"{row.case_id}.nii.gz",
                source_destination=target_source_labels / f"{row.case_id}.nii.gz" if task == "lesion" else None,
                weight_destination=target_weights / f"{row.case_id}.nii.gz" if task == "lesion" else None,
                bbox_zyx=bbox,
                layer1_main_label_mode=layer1_main_label_mode,
            )

    outputs = {
        "dataset_dir": dataset_dir,
        "dataset_json": save_json(
            _build_dataset_json(
                materialized,
                dataset_name=dataset_name,
                task=task,
                with_anatomy_priors=with_anatomy_priors,
                layer1_main_label_mode=layer1_main_label_mode,
                include_test_labels=include_test_labels,
            ),
            dataset_dir / "dataset.json",
        ),
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
        "dataset_index": save_jsonl(
            _build_dataset_records(
                materialized,
                task=task,
                prediction_index=prediction_index,
                crop_index=crop_index,
                include_test_labels=include_test_labels,
            ),
            dataset_dir / "dataset_index.jsonl",
        ),
    }
    return outputs


def export_mednext_task(
    rows: Iterable[CaseManifestRow],
    *,
    task_root: str | Path,
    dataset_id: int,
    dataset_name: str,
    task: TaskName = "lesion",
    anatomy_prediction_manifest: str | Path | None = None,
    crop_manifest: str | Path | Iterable[GlandCropRecord] | None = None,
    layer1_main_label_mode: Layer1MainLabelMode = "source",
    include_test_labels: bool = False,
) -> dict[str, Path]:
    materialized = list(rows)
    prediction_index = _load_prediction_index(anatomy_prediction_manifest)
    crop_index = _coerce_crop_index(crop_manifest)
    with_anatomy_priors = task == "lesion" and bool(prediction_index)
    dataset_dir = Path(task_root) / f"Task{int(dataset_id):03d}_{_sanitize_dataset_name(dataset_name)}"
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    source_labels_tr = dataset_dir / "sourceLabelsTr"
    weights_tr = dataset_dir / "weightsTr"
    images_ts = dataset_dir / "imagesTs"
    labels_ts = dataset_dir / "labelsTs"
    source_labels_ts = dataset_dir / "sourceLabelsTs"
    weights_ts = dataset_dir / "weightsTs"
    train_sidecar_dirs = (source_labels_tr, weights_tr) if task == "lesion" else ()
    test_label_dirs = (labels_ts, source_labels_ts, weights_ts) if include_test_labels and task == "lesion" else (
        (labels_ts,) if include_test_labels else ()
    )
    for directory in (images_tr, labels_tr, images_ts, *train_sidecar_dirs, *test_label_dirs):
        directory.mkdir(parents=True, exist_ok=True)

    for row in materialized:
        labels_available = row.fixed_split == "trainval" or include_test_labels
        target_images = images_tr if row.fixed_split == "trainval" else images_ts
        target_labels = labels_tr if row.fixed_split == "trainval" else labels_ts
        target_source_labels = source_labels_tr if row.fixed_split == "trainval" else source_labels_ts
        target_weights = weights_tr if row.fixed_split == "trainval" else weights_ts
        crop = crop_index.get(row.case_id)
        bbox = crop.bbox_zyx if task == "lesion" and crop is not None else None
        _write_cropped_source_nifti(row.t2w_path, target_images / f"{row.case_id}_0000.nii.gz", bbox_zyx=bbox)
        _write_cropped_source_nifti(row.adc_path, target_images / f"{row.case_id}_0001.nii.gz", bbox_zyx=bbox)
        _write_cropped_source_nifti(row.dwi_path, target_images / f"{row.case_id}_0002.nii.gz", bbox_zyx=bbox)
        if with_anatomy_priors:
            prior_record = prediction_index.get(row.case_id)
            if prior_record is None:
                raise KeyError(f"Missing anatomy prediction for case_id={row.case_id}")
            priors = _load_anatomy_prior_bundle(prior_record)
            for channel_idx, channel_name in enumerate(("P_WG", "P_PZ", "P_TZ"), start=3):
                _write_prior_nifti(
                    priors[channel_name],
                    reference_path=row.t2w_path,
                    destination=target_images / f"{row.case_id}_{channel_idx:04d}.nii.gz",
                    bbox_zyx=bbox,
                )
        if labels_available:
            _export_label_bundle_for_task(
                row,
                task=task,
                destination=target_labels / f"{row.case_id}.nii.gz",
                source_destination=target_source_labels / f"{row.case_id}.nii.gz" if task == "lesion" else None,
                weight_destination=target_weights / f"{row.case_id}.nii.gz" if task == "lesion" else None,
                bbox_zyx=bbox,
                layer1_main_label_mode=layer1_main_label_mode,
            )

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
        "modality": {
            str(idx): name
            for idx, name in enumerate(LAYER1_INPUT_CHANNELS if with_anatomy_priors else ("T2W", "ADC", "DWI"))
        },
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
        if layer1_main_label_mode == "source":
            dataset_json["labels"] = {"0": "background", "1": "pca_lesion", "2": "nca_mimic"}
            dataset_json["segmoe_target_adapter"] = "layer1_source_aware_label1_or_label2_positive"
            dataset_json["segmoe_main_label_mode"] = "source"
        else:
            dataset_json["labels"] = {"0": "background", "1": "candidate"}
            dataset_json["segmoe_target_adapter"] = "layer1_high_recall_label1_or_label2_positive"
            dataset_json["segmoe_main_label_mode"] = "binary"
        dataset_json["segmoe_positive_label_values"] = list(LAYER1_POSITIVE_LABEL_VALUES)
        dataset_json["segmoe_source_label_values"] = {"0": "background", "1": "pca_lesion", "2": "nca_mimic"}
        dataset_json["segmoe_background_weight"] = float(LAYER1_BACKGROUND_WEIGHT)
        dataset_json["segmoe_source_positive_weights"] = {
            str(k): float(v) for k, v in LAYER1_SOURCE_AWARE_WEIGHTS.items()
        }
        dataset_json["segmoe_sampling_policy"] = dict(LAYER1_SAMPLING_POLICY)
        dataset_json["segmoe_include_test_labels"] = bool(include_test_labels)
        dataset_json["segmoe_sidecars"] = {
            "source_labels": {"train": "sourceLabelsTr", "test": "sourceLabelsTs"},
            "voxel_weights": {"train": "weightsTr", "test": "weightsTs"},
        }

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
        "dataset_index": save_jsonl(
            _build_dataset_records(
                materialized,
                task=task,
                prediction_index=prediction_index,
                crop_index=crop_index,
                include_test_labels=include_test_labels,
            ),
            dataset_dir / "dataset_index.jsonl",
        ),
    }
    return outputs


def prepare_segmamba_data(
    rows: Iterable[CaseManifestRow],
    *,
    output_dir: str | Path,
    task: TaskName = "lesion",
    anatomy_prediction_manifest: str | Path | None = None,
    crop_manifest: str | Path | Iterable[GlandCropRecord] | None = None,
    include_test_labels: bool = False,
) -> dict[str, Path]:
    materialized = list(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_index = _load_prediction_index(anatomy_prediction_manifest)
    crop_index = _coerce_crop_index(crop_manifest)
    with_anatomy_priors = task == "lesion" and bool(prediction_index)

    dataset_index = _build_dataset_records(
        materialized,
        task=task,
        prediction_index=prediction_index,
        crop_index=crop_index,
        include_test_labels=include_test_labels,
    )
    if with_anatomy_priors:
        arrays_dir = output_dir / "arrays"
        arrays_dir.mkdir(parents=True, exist_ok=True)
        for row, record in zip(materialized, dataset_index, strict=True):
            crop = crop_index.get(row.case_id)
            bbox = crop.bbox_zyx if crop is not None else None
            prior_record = prediction_index.get(row.case_id)
            if prior_record is None:
                raise KeyError(f"Missing anatomy prediction for case_id={row.case_id}")
            priors = _load_anatomy_prior_bundle(prior_record)
            channels = [
                _crop_zyx(_load_nifti_zyx(row.t2w_path), bbox),
                _crop_zyx(_load_nifti_zyx(row.adc_path), bbox),
                _crop_zyx(_load_nifti_zyx(row.dwi_path), bbox),
                _crop_zyx(priors["P_WG"], bbox),
                _crop_zyx(priors["P_PZ"], bbox),
                _crop_zyx(priors["P_TZ"], bbox),
            ]
            data = np.stack(channels, axis=0).astype(np.float32)
            labels_available = row.fixed_split == "trainval" or include_test_labels
            if labels_available:
                label_image = nib.load(str(resolve_local_path(row.label_path)))
                source_xyz = _layer1_source_label_from_image(label_image, row.cohort_type)
                source_zyx = np.transpose(source_xyz, (2, 1, 0))
                seg_source = _crop_zyx(source_zyx, bbox).astype(np.uint8)
                seg_target = build_layer1_high_recall_target(seg_source).astype(np.uint8)
                voxel_weight = build_layer1_source_weight_map(seg_source).astype(np.float32)
            else:
                seg_source = np.zeros(data.shape[1:], dtype=np.uint8)
                seg_target = np.zeros(data.shape[1:], dtype=np.uint8)
                voxel_weight = np.ones(data.shape[1:], dtype=np.float32)
            array_path = arrays_dir / f"{row.case_id}.npz"
            np.savez_compressed(
                array_path,
                data=data,
                seg=seg_source[None],
                seg_source=seg_source[None],
                seg_target=seg_target[None],
                voxel_weight=voxel_weight[None],
                channel_names=np.asarray(LAYER1_INPUT_CHANNELS),
                positive_label_values=np.asarray(LAYER1_POSITIVE_LABEL_VALUES, dtype=np.uint8),
                source_positive_weights=np.asarray(
                    [LAYER1_SOURCE_AWARE_WEIGHTS[1], LAYER1_SOURCE_AWARE_WEIGHTS[2]],
                    dtype=np.float32,
                ),
                background_weight=np.asarray(LAYER1_BACKGROUND_WEIGHT, dtype=np.float32),
                bbox_zyx=np.asarray(bbox if bbox is not None else (0, data.shape[1], 0, data.shape[2], 0, data.shape[3])),
                native_shape_zyx=np.asarray(crop.native_shape_zyx if crop is not None else row.image_shape),
            )
            record["segmamba_npz"] = str(array_path)
            record["image"] = str(array_path)
            record["label"] = str(array_path) if labels_available else ""
            record.setdefault("metadata", {})["labels_available"] = labels_available

    channel_names = list(LAYER1_INPUT_CHANNELS if with_anatomy_priors else ("T2W", "ADC", "DWI"))
    split_metadata: dict[str, Any] = {
        "model": "SegMamba",
        "modalities": channel_names,
        "task": task,
        "probability_channels": _task_probability_channels(task),
        "input_channels": len(channel_names),
        "channel_names": channel_names,
        "output_heads": _task_output_heads(task),
        "positive_label_values": list(LAYER1_POSITIVE_LABEL_VALUES) if task == "lesion" else [],
        "source_positive_weights": {str(k): float(v) for k, v in LAYER1_SOURCE_AWARE_WEIGHTS.items()}
        if task == "lesion"
        else {},
        "background_weight": float(LAYER1_BACKGROUND_WEIGHT) if task == "lesion" else 1.0,
        "sampling_policy": dict(LAYER1_SAMPLING_POLICY) if task == "lesion" else {},
        "patch_size": [128, 128, 128] if task == "lesion" else [],
        "crop_manifest": str(crop_manifest) if isinstance(crop_manifest, (str, Path)) else "",
        "anatomy_prediction_manifest": str(anatomy_prediction_manifest) if anatomy_prediction_manifest else "",
        "logit_output": {"primary": True, "field": "logits", "space": "roi_crop"},
        "include_test_labels": bool(include_test_labels),
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
            "modalities": channel_names,
            "channel_names": channel_names,
            "input_channels": len(channel_names),
            "output_heads": _task_output_heads(task),
            "probability_channels": _task_probability_channels(task),
            "positive_label_values": list(LAYER1_POSITIVE_LABEL_VALUES) if task == "lesion" else [],
            "source_positive_weights": {str(k): float(v) for k, v in LAYER1_SOURCE_AWARE_WEIGHTS.items()}
            if task == "lesion"
            else {},
            "background_weight": float(LAYER1_BACKGROUND_WEIGHT) if task == "lesion" else 1.0,
            "sampling_policy": dict(LAYER1_SAMPLING_POLICY) if task == "lesion" else {},
            "patch_size": [128, 128, 128] if task == "lesion" else [],
            "crop_manifest": str(crop_manifest) if isinstance(crop_manifest, (str, Path)) else "",
            "anatomy_prediction_manifest": str(anatomy_prediction_manifest) if anatomy_prediction_manifest else "",
            "logit_output": {"primary": True, "field": "logits", "space": "roi_crop"},
            "include_test_labels": bool(include_test_labels),
        },
        output_dir / "segmamba_config.json",
    )
    return outputs


def prepare_layer1_moe_data(
    rows: Iterable[CaseManifestRow],
    *,
    anatomy_prediction_manifest: str | Path,
    crop_manifest: str | Path | Iterable[GlandCropRecord],
    config_out: str | Path,
    nnunet_task_root: str | Path,
    mednext_task_root: str | Path,
    segmamba_output_dir: str | Path,
    nnunet_dataset_id: int = 502,
    nnunet_dataset_name: str = "ProstateLayer1",
    mednext_dataset_id: int = 502,
    mednext_dataset_name: str = "ProstateLayer1",
) -> dict[str, Path]:
    materialized = list(rows)
    manifest_hash = stable_hash([row.to_dict() for row in materialized])
    nnunet_outputs = export_nnunet_task(
        materialized,
        task_root=nnunet_task_root,
        dataset_id=int(nnunet_dataset_id),
        dataset_name=nnunet_dataset_name,
        task="lesion",
        anatomy_prediction_manifest=anatomy_prediction_manifest,
        crop_manifest=crop_manifest,
        layer1_main_label_mode="source",
    )
    mednext_outputs = export_mednext_task(
        materialized,
        task_root=mednext_task_root,
        dataset_id=int(mednext_dataset_id),
        dataset_name=mednext_dataset_name,
        task="lesion",
        anatomy_prediction_manifest=anatomy_prediction_manifest,
        crop_manifest=crop_manifest,
        layer1_main_label_mode="source",
    )
    segmamba_outputs = prepare_segmamba_data(
        materialized,
        output_dir=segmamba_output_dir,
        task="lesion",
        anatomy_prediction_manifest=anatomy_prediction_manifest,
        crop_manifest=crop_manifest,
    )
    config_path = save_json(
        {
            "layer": "layer1",
            "task": "candidate_first_lesion_mimic",
            "source_manifest_hash": manifest_hash,
            "anatomy_prediction_manifest": str(anatomy_prediction_manifest),
            "crop_manifest": str(crop_manifest) if isinstance(crop_manifest, (str, Path)) else "",
            "channel_names": list(LAYER1_INPUT_CHANNELS),
            "candidate_positive_label_values": list(LAYER1_POSITIVE_LABEL_VALUES),
            "background_weight": float(LAYER1_BACKGROUND_WEIGHT),
            "source_positive_weights": {str(k): float(v) for k, v in LAYER1_SOURCE_AWARE_WEIGHTS.items()},
            "min_crop_size_zyx": list(DEFAULT_MIN_CROP_SIZE_ZYX),
            "expert_names": ["nnunet", "mednext", "segmamba"],
            "experts": {
                "nnunet": {
                    "role": "local_boundary_expert",
                    "preference": "standard nnU-Net local patch recipe",
                    "trainer": "nnUNetTrainerSegMoELayer1",
                    "main_label_mode": "source",
                    "dataset_id": int(nnunet_dataset_id),
                    "dataset_name": str(nnunet_dataset_name),
                    "dataset_dir": str(nnunet_outputs["dataset_dir"]),
                    "dataset_json": str(nnunet_outputs["dataset_json"]),
                    "splits": str(nnunet_outputs["splits"]),
                    "dataset_index": str(nnunet_outputs["dataset_index"]),
                },
                "mednext": {
                    "role": "large_kernel_multiscale_context_expert",
                    "preference": "larger effective receptive field and stronger multiscale augmentation",
                    "trainer": "nnUNetTrainerV2_MedNeXt_S_kernel3_SegMoELayer1",
                    "main_label_mode": "source",
                    "dataset_id": int(mednext_dataset_id),
                    "dataset_name": str(mednext_dataset_name),
                    "dataset_dir": str(mednext_outputs["dataset_dir"]),
                    "dataset_json": str(mednext_outputs["dataset_json"]),
                    "splits": str(mednext_outputs["splits"]),
                    "dataset_index": str(mednext_outputs["dataset_index"]),
                },
                "segmamba": {
                    "role": "long_range_roi_context_expert",
                    "preference": "full ROI prediction with longest feasible z-context",
                    "trainer": "segmoe_v2.segmamba_adapter",
                    "main_label_mode": "source_npz_seg_source",
                    "output_dir": str(Path(segmamba_output_dir)),
                    "dataset_index": str(segmamba_outputs["dataset_index"]),
                    "split_metadata": str(segmamba_outputs["split_metadata"]),
                    "config": str(segmamba_outputs["segmamba_config"]),
                },
            },
            "fusion_protocol": {
                "calibration": "temperature_scaling_per_expert_on_oof_trainval",
                "baselines": ["mean_ensemble", "static_convex_fusion"],
                "primary": "learned_spatial_moe_gate",
                "gate_inputs": [
                    "P_nnunet",
                    "P_mednext",
                    "P_segmamba",
                    "expert_mean",
                    "expert_entropy",
                    "expert_disagreement",
                    "P_WG",
                    "P_PZ",
                    "P_TZ",
                ],
            },
        },
        config_out,
    )
    return {
        "layer1_moe_config": config_path,
        "nnunet_dataset_dir": nnunet_outputs["dataset_dir"],
        "mednext_dataset_dir": mednext_outputs["dataset_dir"],
        "segmamba_output_dir": Path(segmamba_output_dir),
    }
