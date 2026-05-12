from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .backend_data import (
    LAYER1_INPUT_CHANNELS,
    LAYER1_POSITIVE_LABEL_VALUES,
    LAYER1_SAMPLING_POLICY,
    LAYER1_SOURCE_AWARE_WEIGHTS,
    resolve_vendored_backend_root,
)
from .contracts import PredictionRecord
from .io_utils import load_json, load_jsonl, save_json, save_jsonl, save_pickle
from .labels import LAYER1_BACKGROUND_WEIGHT, build_layer1_high_recall_target, build_layer1_source_weight_map


def layer1_target_from_source_labels(
    source_labels: np.ndarray,
    *,
    positive_label_values: Sequence[int] = LAYER1_POSITIVE_LABEL_VALUES,
) -> np.ndarray:
    return build_layer1_high_recall_target(
        np.asarray(source_labels),
        positive_label_values=tuple(int(v) for v in positive_label_values),
    ).astype(np.float32)


def layer1_weight_from_source_labels(source_labels: np.ndarray) -> np.ndarray:
    return build_layer1_source_weight_map(np.asarray(source_labels)).astype(np.float32)


def _insert_import_paths(repo_root: str | Path) -> None:
    repo_root = Path(repo_root)
    for path in (repo_root, repo_root / "mamba", repo_root / "causal-conv1d"):
        resolved = str(path)
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def build_segmamba_model(
    *,
    repo_root: str | Path | None = None,
    in_channels: int = len(LAYER1_INPUT_CHANNELS),
    out_channels: int = 1,
    **kwargs: Any,
) -> Any:
    _insert_import_paths(repo_root or resolve_vendored_backend_root("segmamba"))
    from model_segmamba.segmamba import SegMamba

    return SegMamba(in_chans=int(in_channels), out_chans=int(out_channels), **kwargs)


def _load_records(config: Mapping[str, Any], *, fold: int, split: str) -> list[dict[str, Any]]:
    if split == "train":
        path = str(config["train_list_pattern"]).format(fold=int(fold))
    elif split in {"val", "validation"}:
        path = str(config["val_list_pattern"]).format(fold=int(fold))
    elif split == "test":
        path = str(config["test_list"])
    else:
        raise ValueError(f"Unsupported split: {split}")
    return load_jsonl(path)


class SegMambaLayer1Dataset:
    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        positive_label_values: Sequence[int],
        patch_size: Sequence[int] | None = None,
        seed: int = 42,
    ) -> None:
        self.records = [dict(record) for record in records]
        self.positive_label_values = tuple(int(v) for v in positive_label_values)
        self.patch_size = tuple(int(v) for v in patch_size) if patch_size is not None else None
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[int(index)]
        npz_path = record.get("segmamba_npz") or record.get("image")
        if not npz_path or not str(npz_path).endswith(".npz"):
            raise ValueError("SegMambaLayer1Dataset requires prepared .npz records from prepare-segmamba-data.")
        payload = np.load(str(npz_path), allow_pickle=True)
        data = np.asarray(payload["data"], dtype=np.float32)
        source = np.asarray(payload["seg_source"] if "seg_source" in payload else payload["seg"], dtype=np.uint8)
        if "seg_target" in payload:
            target = np.asarray(payload["seg_target"], dtype=np.float32)
        else:
            target = layer1_target_from_source_labels(source, positive_label_values=self.positive_label_values)
        if "voxel_weight" in payload:
            voxel_weight = np.asarray(payload["voxel_weight"], dtype=np.float32)
        else:
            voxel_weight = layer1_weight_from_source_labels(source)
        requested_mode = "full_roi"
        fallback = False
        if self.patch_size is not None:
            requested_mode, fallback, center = _choose_layer1_center(
                source[0],
                data[3] if data.shape[0] > 3 else None,
                seed=self.seed + int(index),
            )
            data = _extract_patch(data, center, self.patch_size)
            source = _extract_patch(source, center, self.patch_size)
            target = _extract_patch(target, center, self.patch_size)
            voxel_weight = _extract_patch(voxel_weight, center, self.patch_size)
        return {
            "data": data,
            "target": target,
            "voxel_weight": voxel_weight,
            "source": source,
            "record": record,
            "requested_mode": requested_mode,
            "fallback": fallback,
            "bbox_zyx": np.asarray(payload["bbox_zyx"], dtype=np.int64) if "bbox_zyx" in payload else None,
            "native_shape_zyx": np.asarray(payload["native_shape_zyx"], dtype=np.int64) if "native_shape_zyx" in payload else None,
        }


def _sample_mode(rng: np.random.Generator) -> str:
    token = float(rng.random())
    if token < LAYER1_SAMPLING_POLICY["pca_lesion"]:
        return "pca_lesion"
    if token < LAYER1_SAMPLING_POLICY["pca_lesion"] + LAYER1_SAMPLING_POLICY["nca_mimic"]:
        return "nca_mimic"
    return "random_gland"


def _choose_center(mask: np.ndarray, rng: np.random.Generator) -> tuple[int, int, int]:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return tuple(int(rng.integers(0, max(int(size), 1))) for size in mask.shape)
    return tuple(int(v) for v in coords[int(rng.integers(0, len(coords)))].tolist())


def _choose_layer1_center(
    source: np.ndarray,
    gland_probability: np.ndarray | None,
    *,
    seed: int,
) -> tuple[str, bool, tuple[int, int, int]]:
    rng = np.random.default_rng(int(seed))
    mode = _sample_mode(rng)
    fallback = False
    if mode == "pca_lesion" and np.any(source == 1):
        return mode, fallback, _choose_center(source == 1, rng)
    if mode == "nca_mimic" and np.any(source == 2):
        return mode, fallback, _choose_center(source == 2, rng)
    gland_mask = gland_probability >= 0.35 if gland_probability is not None else np.ones_like(source, dtype=bool)
    if not gland_mask.any():
        gland_mask = np.ones_like(source, dtype=bool)
    fallback = mode != "random_gland"
    return "random_gland", fallback, _choose_center(gland_mask, rng)


def _extract_patch(arr: np.ndarray, center: tuple[int, int, int], patch_size: tuple[int, int, int]) -> np.ndarray:
    d, h, w = arr.shape[-3:]
    pd, ph, pw = patch_size
    starts = [center[0] - pd // 2, center[1] - ph // 2, center[2] - pw // 2]
    sizes = [pd, ph, pw]
    limits = [d, h, w]
    slices = []
    for start, size, limit in zip(starts, sizes, limits):
        start = max(0, min(int(start), max(int(limit) - int(size), 0)))
        slices.append(slice(start, min(start + int(size), int(limit))))
    patch = arr[(..., slices[0], slices[1], slices[2])]
    if patch.shape[-3:] == patch_size:
        return patch
    pad_spec = [(0, 0)] * patch.ndim
    for axis, (got, need) in enumerate(zip(patch.shape[-3:], patch_size), start=patch.ndim - 3):
        short = max(0, int(need) - int(got))
        pad_spec[axis] = (short // 2, short - short // 2)
    return np.pad(patch, pad_spec, mode="edge")


def _binary_dice_loss_with_logits(logits: Any, target: Any, eps: float = 1e-6) -> Any:
    import torch

    probs = torch.sigmoid(logits)
    dims = tuple(range(2, logits.ndim))
    intersection = (probs * target).sum(dim=dims)
    denominator = probs.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def layer1_high_recall_loss(logits: Any, target: Any, voxel_weight: Any | None = None) -> Any:
    import torch.nn.functional as F

    bce = F.binary_cross_entropy_with_logits(logits, target, weight=voxel_weight)
    dice = _binary_dice_loss_with_logits(logits, target)
    return bce + dice


def _collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import torch

    return {
        "data": torch.from_numpy(np.stack([item["data"] for item in batch], axis=0)).float(),
        "target": torch.from_numpy(np.stack([item["target"] for item in batch], axis=0)).float(),
        "voxel_weight": torch.from_numpy(np.stack([item["voxel_weight"] for item in batch], axis=0)).float(),
        "record": [item["record"] for item in batch],
        "bbox_zyx": [item["bbox_zyx"] for item in batch],
        "native_shape_zyx": [item["native_shape_zyx"] for item in batch],
    }


def _save_state_and_delete_last(model: Any, save_path: str | Path, *, delete_symbol: str | None = None) -> None:
    import torch

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if delete_symbol:
        for old_path in save_path.parent.glob(f"{delete_symbol}*.pt"):
            if old_path != save_path:
                old_path.unlink()
    torch.save(model.state_dict(), save_path)
    print(f"model is saved in {save_path}", flush=True)


def _dice_from_logits(logits: Any, target: Any, eps: float = 1e-8) -> float:
    import torch

    prediction = torch.sigmoid(logits) >= 0.5
    target = target >= 0.5
    dims = tuple(range(1, target.ndim))
    tp = torch.logical_and(prediction, target).float().sum(dim=dims)
    fp = torch.logical_and(prediction, ~target).float().sum(dim=dims)
    fn = torch.logical_and(~prediction, target).float().sum(dim=dims)
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    return float(dice.mean().detach().cpu())


def _format_fold_path(path: str | Path, fold: int) -> Path:
    return Path(str(path).format(fold=int(fold)))


def _sample_foreground_locations(
    seg: np.ndarray,
    positive_label_values: Sequence[int],
    *,
    seed: int = 1234,
) -> dict[int, np.ndarray]:
    rng = np.random.RandomState(int(seed))
    locations: dict[int, np.ndarray] = {}
    for label_value in positive_label_values:
        label = int(label_value)
        all_locations = np.argwhere(np.asarray(seg) == label)
        if len(all_locations) == 0:
            locations[label] = np.empty((0, 4), dtype=np.int64)
            continue
        target_count = min(10_000, len(all_locations))
        target_count = max(target_count, int(np.ceil(len(all_locations) * 0.01)))
        locations[label] = all_locations[rng.choice(len(all_locations), target_count, replace=False)]
    return locations


def _source_npz_path(record: Mapping[str, Any]) -> Path:
    npz_path = record.get("segmamba_npz") or record.get("image")
    if not npz_path:
        raise ValueError(f"Missing segmamba_npz/image for case_id={record.get('case_id', '<unknown>')}")
    return Path(str(npz_path))


def _original_case_filename(record: Mapping[str, Any]) -> str:
    return f"{_source_npz_path(record).stem}.npz"


def _link_or_copy_npz(source_path: Path, target_path: Path, *, link_mode: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() or target_path.is_symlink():
        target_path.unlink()
    if link_mode == "copy":
        shutil.copy2(source_path, target_path)
    elif link_mode == "hardlink":
        os.link(source_path, target_path)
    else:
        try:
            target_path.symlink_to(source_path.resolve())
        except OSError:
            shutil.copy2(source_path, target_path)


def _records_by_case(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(record["case_id"]): record for record in records}


def export_original_style_data(
    config_path: str | Path,
    *,
    output_dir: str | Path,
    folds: Sequence[int] | None = None,
    link_mode: str = "symlink",
    unpack: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    config = load_json(config_path)
    dataset_index = load_jsonl(config["dataset_index"])
    split_metadata = load_json(config["split_metadata"]) if config.get("split_metadata") else {}
    if folds is None:
        folds = [int(fold) for fold in sorted(split_metadata.get("folds", {}).keys(), key=int)]
    folds = list(folds or [0])

    output_dir = Path(output_dir)
    fullres_dir = output_dir / "fullres"
    splits_dir = output_dir / "splits"
    config_out = output_dir / "segmamba_original_config.json"
    positive_label_values = tuple(int(v) for v in config.get("positive_label_values", LAYER1_POSITIVE_LABEL_VALUES))

    summary: dict[str, Any] = {
        "mode": "export-original",
        "cases": len(dataset_index),
        "folds": [int(fold) for fold in folds],
        "fullres_dir": str(fullres_dir),
        "config": str(config_out),
        "link_mode": str(link_mode),
    }
    if dry_run:
        return summary

    from tqdm import tqdm

    fullres_dir.mkdir(parents=True, exist_ok=True)
    for record in tqdm(dataset_index, desc="Export original-style SegMamba data", mininterval=5):
        source_path = _source_npz_path(record)
        target_path = fullres_dir / _original_case_filename(record)
        _link_or_copy_npz(source_path, target_path, link_mode=link_mode)
        payload = np.load(str(source_path), allow_pickle=True)
        source = np.asarray(payload["seg_source"] if "seg_source" in payload else payload["seg"], dtype=np.uint8)
        properties = {
            "case_id": str(record["case_id"]),
            "class_locations": _sample_foreground_locations(source, positive_label_values),
            "bbox_zyx": np.asarray(payload["bbox_zyx"], dtype=np.int64) if "bbox_zyx" in payload else None,
            "native_shape_zyx": np.asarray(payload["native_shape_zyx"], dtype=np.int64)
            if "native_shape_zyx" in payload
            else None,
            "positive_label_values": positive_label_values,
        }
        save_pickle(properties, target_path.with_suffix(".pkl"))

    cases = _records_by_case(dataset_index)
    test_records = load_jsonl(config["test_list"]) if config.get("test_list") else []
    for fold in folds:
        train_records = _load_records(config, fold=int(fold), split="train")
        val_records = _load_records(config, fold=int(fold), split="val")
        split_payload = {
            "train": [_original_case_filename(cases[str(record["case_id"])]) for record in train_records],
            "validation": [_original_case_filename(cases[str(record["case_id"])]) for record in val_records],
            "test": [_original_case_filename(cases[str(record["case_id"])]) for record in test_records],
        }
        save_json(split_payload, splits_dir / f"fold_{int(fold)}_split.json")

    original_config = dict(config)
    original_config.update(
        {
            "data_format": "segmamba_original",
            "original_data_dir": str(fullres_dir),
            "original_split_pattern": str(splits_dir / "fold_{fold}_split.json"),
            "logdir": str(output_dir / "logs" / "fold_{fold}"),
            "checkpoint_dir": str(output_dir / "checkpoints"),
            "batch_size": int(config.get("batch_size", 2)),
            "train_process": int(config.get("train_process", 18)),
            "val_process": int(config.get("val_process", 6)),
            "steps_per_epoch": int(config.get("steps_per_epoch", 250)),
            "val_batches": int(config.get("val_batches", 100)),
            "optimizer": str(config.get("optimizer", "sgd")),
            "learning_rate": float(config.get("learning_rate", 1e-2)),
            "weight_decay": float(config.get("weight_decay", 3e-5)),
            "momentum": float(config.get("momentum", 0.99)),
            "nesterov": bool(config.get("nesterov", True)),
            "scheduler_type": str(config.get("scheduler_type", "poly")),
            "augmentation": config.get("augmentation", True),
            "pin_memory": bool(config.get("pin_memory", False)),
            "stop_on_nonfinite_loss": bool(config.get("stop_on_nonfinite_loss", True)),
        }
    )
    save_json(original_config, config_out)

    if unpack:
        repo_root = config.get("repo_root") or resolve_vendored_backend_root("segmamba")
        _insert_import_paths(repo_root)
        from light_training.dataloading.utils import unpack_dataset

        print(f"unpacking original-style data at {fullres_dir}", flush=True)
        unpack_dataset(str(fullres_dir), unpack_segmentation=True, overwrite_existing=False, num_processes=8)
    return summary


def _load_original_style_paths(config: Mapping[str, Any], *, fold: int) -> tuple[list[str], list[str]]:
    split_path = _format_fold_path(config["original_split_pattern"], int(fold))
    split_payload = load_json(split_path)
    data_dir = Path(str(config["original_data_dir"]))
    train_paths = [str(data_dir / name) for name in split_payload["train"]]
    val_paths = [str(data_dir / name) for name in split_payload["validation"]]
    return train_paths, val_paths


def _batch_source_to_target_and_weight(
    source: Any,
    positive_label_values: Sequence[int],
    source_weights: Mapping[str, float],
    background_weight: float,
) -> tuple[Any, Any]:
    import torch

    source = source[:, 0].long()
    target = torch.zeros_like(source, dtype=torch.float32)
    weight = torch.full_like(target, float(background_weight), dtype=torch.float32)
    for label_value in positive_label_values:
        label = int(label_value)
        mask = source == label
        target[mask] = 1.0
        weight[mask] = float(source_weights.get(str(label), source_weights.get(label, 1.0)))
    return target[:, None], weight[:, None]


def _build_original_optimizer(model: Any, config: Mapping[str, Any]) -> Any:
    import torch

    optimizer_name = str(config.get("optimizer", "sgd")).lower()
    lr = float(config.get("learning_rate", 1e-2 if optimizer_name == "sgd" else 1e-4))
    weight_decay = float(config.get("weight_decay", 3e-5 if optimizer_name == "sgd" else 0.0))
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return torch.optim.SGD(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        momentum=float(config.get("momentum", 0.99)),
        nesterov=bool(config.get("nesterov", True)),
    )


def _train_original_style(
    config: Mapping[str, Any],
    *,
    fold: int,
    max_epochs: int,
    summary: dict[str, Any],
) -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    repo_root = config.get("repo_root") or resolve_vendored_backend_root("segmamba")
    _insert_import_paths(repo_root)
    from light_training.dataloading.dataset import MedicalDataset
    from light_training.trainer import Trainer

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_type = "DDP" if world_size > 1 else "pytorch"
    logdir = _format_fold_path(config.get("logdir", Path("logs") / f"fold_{int(fold)}"), int(fold))
    checkpoint_dir = _format_fold_path(config.get("checkpoint_dir", Path("checkpoints")), int(fold))
    patch_size = list(config.get("patch_size", [128, 128, 128]))
    positive_label_values = tuple(int(v) for v in config.get("positive_label_values", LAYER1_POSITIVE_LABEL_VALUES))
    source_weights = {
        str(k): float(v) for k, v in dict(config.get("source_positive_weights", LAYER1_SOURCE_AWARE_WEIGHTS)).items()
    }
    background_weight = float(config.get("background_weight", LAYER1_BACKGROUND_WEIGHT))

    class Layer1OriginalStyleTrainer(Trainer):
        def get_dist_args(self) -> None:
            self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self.not_call_launch = True
            self.device = self.local_rank

        def __init__(self) -> None:
            super().__init__(
                env_type=env_type,
                max_epochs=int(max_epochs),
                batch_size=int(config.get("batch_size", 2)),
                device=str(config.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")),
                val_every=int(config.get("val_every", 2)),
                num_gpus=max(1, world_size),
                logdir=str(logdir),
                master_port=int(config.get("master_port", 17759)),
                training_script=__file__,
                train_process=int(config.get("train_process", 18)),
            )
            self.model = build_segmamba_model(
                repo_root=repo_root,
                in_channels=int(config.get("input_channels", len(LAYER1_INPUT_CHANNELS))),
                out_channels=1,
            )
            self.patch_size = patch_size
            self.best_mean_dice = 0.0
            self.augmentation = config.get("augmentation", True)
            self.pin_memory = bool(config.get("pin_memory", False))
            self.optimizer = _build_original_optimizer(self.model, config)
            self.scheduler_type = str(config.get("scheduler_type", "poly"))
            self.num_step_per_epoch = max(1, int(config.get("steps_per_epoch", 250)) // max(1, world_size))
            self.val_number = max(1, int(config.get("val_batches", 100)) // max(1, world_size))

        def training_step(self, batch: Mapping[str, Any]) -> Any:
            image = batch["data"]
            source = batch["seg"]
            target, voxel_weight = _batch_source_to_target_and_weight(
                source,
                positive_label_values,
                source_weights,
                background_weight,
            )
            logits = self.model(image)
            if logits.ndim == target.ndim - 1:
                logits = logits[:, None]
            loss = layer1_high_recall_loss(logits, target, voxel_weight)
            if bool(config.get("stop_on_nonfinite_loss", True)) and not torch.isfinite(loss).all():
                raise FloatingPointError(f"Non-finite SegMamba loss at epoch={self.epoch}, step={self.global_step}")
            self.log("training_loss", loss, step=self.global_step)
            return loss

        def validation_step(self, batch: Mapping[str, Any]) -> float:
            image = batch["data"]
            source = batch["seg"]
            target, _ = _batch_source_to_target_and_weight(
                source,
                positive_label_values,
                source_weights,
                background_weight,
            )
            logits = self.model(image)
            if logits.ndim == target.ndim - 1:
                logits = logits[:, None]
            return _dice_from_logits(logits, target)

        def validation_end(self, val_outputs: Any) -> None:
            values = torch.as_tensor(val_outputs, dtype=torch.float32)
            finite = values[torch.isfinite(values)]
            mean_dice = float(finite.mean().item()) if finite.numel() else float("nan")
            self.log("mean_dice", mean_dice, step=self.epoch)
            print(f"mean_dice is {mean_dice}", flush=True)
            model_to_save = self.model.module if hasattr(self.model, "module") else self.model
            if mean_dice > self.best_mean_dice:
                self.best_mean_dice = mean_dice
                _save_state_and_delete_last(
                    model_to_save,
                    checkpoint_dir / f"best_model_fold{int(fold)}_{mean_dice:.4f}.pt",
                    delete_symbol=f"best_model_fold{int(fold)}",
                )
            _save_state_and_delete_last(
                model_to_save,
                checkpoint_dir / f"final_model_fold{int(fold)}_{mean_dice:.4f}.pt",
                delete_symbol=f"final_model_fold{int(fold)}",
            )
            if (self.epoch + 1) % 100 == 0:
                torch.save(model_to_save.state_dict(), checkpoint_dir / f"tmp_model_fold{int(fold)}_ep{self.epoch}_{mean_dice:.4f}.pt")

    trainer = Layer1OriginalStyleTrainer()
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1 and rank == 0 and bool(config.get("preunpack_original_data", True)):
        from light_training.dataloading.utils import unpack_dataset

        unpack_dataset(str(config["original_data_dir"]), unpack_segmentation=True, overwrite_existing=False, num_processes=8)
    if world_size > 1 and dist.is_initialized():
        dist.barrier()

    train_paths, val_paths = _load_original_style_paths(config, fold=int(fold))
    train_ds = MedicalDataset(train_paths)
    val_ds = MedicalDataset(val_paths)
    trainer.train(train_dataset=train_ds, val_dataset=val_ds)

    is_main = int(os.environ.get("RANK", "0")) == 0
    final_checkpoint_path = checkpoint_dir / f"segmamba_layer1_fold{int(fold)}.pt"
    if is_main:
        model_to_save = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model_to_save.state_dict(), final_checkpoint_path)
    if getattr(trainer, "writer", None) is not None:
        trainer.writer.close()
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()
    summary["checkpoint_path"] = str(final_checkpoint_path)
    summary["best_mean_dice"] = float(trainer.best_mean_dice)
    return summary


def train(config_path: str | Path, *, fold: int, dry_run: bool = False, max_epochs: int = 1) -> dict[str, Any]:
    config = load_json(config_path)
    data_format = str(config.get("data_format", "segmamba_npz"))
    patch_size = list(config.get("patch_size", [128, 128, 128]))
    logdir = _format_fold_path(config.get("logdir", Path(config_path).parent / "logs" / f"fold_{int(fold)}"), int(fold))
    checkpoint_dir = _format_fold_path(config.get("checkpoint_dir", Path(config_path).parent / "checkpoints"), int(fold))
    batch_size = int(config.get("batch_size", 2))
    val_every = int(config.get("val_every", 2))
    steps_per_epoch = int(config.get("steps_per_epoch", 250))
    val_batches = int(config.get("val_batches", 100))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    per_rank_steps = max(1, steps_per_epoch // max(1, world_size))

    if data_format != "segmamba_original":
        summary = {
            "mode": "train",
            "ready": False,
            "fold": int(fold),
            "data_format": data_format,
            "required_data_format": "segmamba_original",
            "export_command": (
                "python -m segmoe_v2.segmamba_adapter export-original "
                f"--config {config_path} --output-dir data/exports/segmamba_original --fold {int(fold)} --link-mode symlink --unpack"
            ),
        }
        if dry_run:
            return summary
        raise ValueError(
            "SegMamba training now only uses the source-style light_training.Trainer path. "
            "Run export-original first and train with the generated segmamba_original_config.json."
        )

    train_paths, val_paths = _load_original_style_paths(config, fold=int(fold))
    summary = {
        "mode": "train-original",
        "fold": int(fold),
        "train_cases": len(train_paths),
        "val_cases": len(val_paths),
        "input_channels": int(config.get("input_channels", len(LAYER1_INPUT_CHANNELS))),
        "output_channels": 1,
        "positive_label_values": list(config.get("positive_label_values", LAYER1_POSITIVE_LABEL_VALUES)),
        "source_positive_weights": {
            str(k): float(v) for k, v in dict(config.get("source_positive_weights", LAYER1_SOURCE_AWARE_WEIGHTS)).items()
        },
        "background_weight": float(config.get("background_weight", LAYER1_BACKGROUND_WEIGHT)),
        "sampling_policy": dict(config.get("sampling_policy", {})),
        "data_format": data_format,
        "patch_size": patch_size,
        "batch_size": batch_size,
        "global_batch_size": batch_size * max(1, world_size),
        "max_epochs": int(max_epochs),
        "steps_per_epoch": steps_per_epoch,
        "per_rank_steps": per_rank_steps,
        "val_every": val_every,
        "val_batches": val_batches,
        "logdir": str(logdir),
        "checkpoint_dir": str(checkpoint_dir),
        "world_size": world_size,
        "trainer": "external/SegMamba/light_training/trainer.py",
        "optimizer": str(config.get("optimizer", "sgd")),
        "learning_rate": float(config.get("learning_rate", 1e-2)),
        "scheduler_type": str(config.get("scheduler_type", "poly")),
        "augmentation": config.get("augmentation", True),
        "pin_memory": bool(config.get("pin_memory", False)),
        "stop_on_nonfinite_loss": bool(config.get("stop_on_nonfinite_loss", True)),
    }
    if dry_run:
        return summary
    return _train_original_style(config, fold=int(fold), max_epochs=int(max_epochs), summary=summary)


def predict(
    config_path: str | Path,
    *,
    fold: int,
    split: str,
    checkpoint: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    config = load_json(config_path)
    records = _load_records(config, fold=int(fold), split=split)
    output_dir = Path(config.get("prediction_dir", Path(config_path).parent / "predictions" / f"fold_{fold}" / split))
    summary = {
        "mode": "predict",
        "fold": int(fold),
        "split": str(split),
        "cases": len(records),
        "output_dir": str(output_dir),
        "logit_field": "logits",
    }
    if dry_run:
        return summary

    import torch
    from torch.utils.data import DataLoader

    repo_root = config.get("repo_root") or resolve_vendored_backend_root("segmamba")
    model = build_segmamba_model(
        repo_root=repo_root,
        in_channels=int(config.get("input_channels", len(LAYER1_INPUT_CHANNELS))),
        out_channels=1,
    )
    checkpoint = checkpoint or config.get("checkpoint")
    if not checkpoint:
        raise ValueError("SegMamba prediction requires a checkpoint path.")
    state = torch.load(str(checkpoint), map_location="cpu")
    state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state_dict, strict=False)
    device = torch.device(str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    model.to(device)
    model.eval()
    dataset = SegMambaLayer1Dataset(
        records,
        positive_label_values=config.get("positive_label_values", LAYER1_POSITIVE_LABEL_VALUES),
        patch_size=None,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_collate)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_records: list[PredictionRecord] = []
    metric_per_case: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            data = batch["data"].to(device)
            logits = model(data)
            if logits.ndim == 4:
                logits = logits[:, None]
            logits_np = logits.detach().cpu().numpy()[0].astype(np.float32)
            target_np = batch["target"].numpy()[0]
            record = batch["record"][0]
            case_id = str(record["case_id"])
            target_positive = target_np[0] > 0.5 if target_np.ndim == 4 and target_np.shape[0] == 1 else target_np > 0.5
            prediction_positive = logits_np[0] >= 0.0 if logits_np.ndim == 4 and logits_np.shape[0] == 1 else logits_np >= 0.0
            tp = float(np.logical_and(prediction_positive, target_positive).sum())
            fp = float(np.logical_and(prediction_positive, ~target_positive).sum())
            fn = float(np.logical_and(~prediction_positive, target_positive).sum())
            tn = float(np.logical_and(~prediction_positive, ~target_positive).sum())
            denominator = 2.0 * tp + fp + fn
            dice = float(2.0 * tp / denominator) if denominator > 0 else float("nan")
            metric_per_case.append(
                {
                    "reference_file": case_id,
                    "prediction_file": str(output_dir / f"{case_id}.npz"),
                    "metrics": {
                        "candidate": {
                            "Dice": dice,
                            "FP": fp,
                            "TP": tp,
                            "FN": fn,
                            "TN": tn,
                        }
                    },
                }
            )
            logit_path = output_dir / f"{case_id}.npz"
            bbox = batch["bbox_zyx"][0]
            native_shape = batch["native_shape_zyx"][0]
            np.savez_compressed(
                logit_path,
                logits=logits_np,
                channel_names=np.asarray(["P_lesion_logit"]),
                bbox_zyx=bbox if bbox is not None else np.asarray(record.get("metadata", {}).get("bbox_zyx", ())),
                native_shape_zyx=native_shape
                if native_shape is not None
                else np.asarray(record.get("metadata", {}).get("native_shape_zyx", ())),
                positive_label_values=np.asarray(config.get("positive_label_values", LAYER1_POSITIVE_LABEL_VALUES)),
            )
            metadata = dict(record.get("metadata", {}))
            metadata["positive_label_values"] = list(config.get("positive_label_values", LAYER1_POSITIVE_LABEL_VALUES))
            prediction_records.append(
                PredictionRecord(
                    task="lesion",
                    stage="layer1",
                    model_name="SegMamba",
                    fold=int(fold),
                    split=str(split),
                    case_id=case_id,
                    predictor_fold=int(fold),
                    logit_path=logit_path,
                    channel_names=("P_lesion_logit",),
                    source_manifest_hash=str(record.get("source_manifest_hash", "")),
                    metadata=metadata,
                )
            )
    manifest_path = output_dir / "prediction_manifest.jsonl"
    save_jsonl((record.to_dict() for record in prediction_records), manifest_path)
    summary["prediction_manifest"] = str(manifest_path)
    if metric_per_case:
        min_dice = float(config.get("filtered_validation_min_dice", os.environ.get("SEGMOE_LAYER1_FILTER_MIN_DICE", 0.30)))
        mean_dice = float(np.nanmean([item["metrics"]["candidate"]["Dice"] for item in metric_per_case]))
        filtered_metric_per_case = [
            item
            for item in metric_per_case
            if np.isfinite(float(item["metrics"]["candidate"]["Dice"]))
            and float(item["metrics"]["candidate"]["Dice"]) >= min_dice
        ]
        filtered_mean_dice = (
            float(np.nanmean([item["metrics"]["candidate"]["Dice"] for item in filtered_metric_per_case]))
            if filtered_metric_per_case
            else float("nan")
        )
        filtered_excluded_case_ids = [
            str(item["reference_file"]) for item in metric_per_case if item not in filtered_metric_per_case
        ]
        summary.update(
            {
                "metric_per_case": metric_per_case,
                "mean": {"candidate": {"Dice": mean_dice}},
                "foreground_mean": {"Dice": mean_dice},
                "filtered": {
                    "min_dice": min_dice,
                    "case_count": len(filtered_metric_per_case),
                    "excluded_case_count": len(filtered_excluded_case_ids),
                    "excluded_case_ids": filtered_excluded_case_ids,
                    "mean": {"candidate": {"Dice": filtered_mean_dice}},
                },
            }
        )
        summary_path = output_dir / "summary.json"
        summary["summary_json"] = str(summary_path)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SegMoE SegMamba Layer1 adapter")
    sub = parser.add_subparsers(dest="command", required=True)
    export_parser = sub.add_parser("export-original")
    export_parser.add_argument("--config", required=True)
    export_parser.add_argument("--output-dir", required=True)
    export_parser.add_argument("--fold", type=int, action="append", dest="folds")
    export_parser.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    export_parser.add_argument("--unpack", action="store_true")
    export_parser.add_argument("--dry-run", action="store_true")
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--fold", type=int, required=True)
    train_parser.add_argument("--max-epochs", type=int, default=1)
    train_parser.add_argument("--dry-run", action="store_true")
    predict_parser = sub.add_parser("predict")
    predict_parser.add_argument("--config", required=True)
    predict_parser.add_argument("--fold", type=int, required=True)
    predict_parser.add_argument("--split", default="val")
    predict_parser.add_argument("--checkpoint", required=False)
    predict_parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "export-original":
        payload = export_original_style_data(
            args.config,
            output_dir=args.output_dir,
            folds=args.folds,
            link_mode=str(args.link_mode),
            unpack=bool(args.unpack),
            dry_run=bool(args.dry_run),
        )
    elif args.command == "train":
        payload = train(args.config, fold=int(args.fold), dry_run=bool(args.dry_run), max_epochs=int(args.max_epochs))
    else:
        payload = predict(
            args.config,
            fold=int(args.fold),
            split=str(args.split),
            checkpoint=args.checkpoint,
            dry_run=bool(args.dry_run),
        )
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
