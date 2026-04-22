from __future__ import annotations

import argparse
import json
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
from .io_utils import load_json, load_jsonl, save_jsonl
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


def train(config_path: str | Path, *, fold: int, dry_run: bool = False, max_epochs: int = 1) -> dict[str, Any]:
    config = load_json(config_path)
    train_records = _load_records(config, fold=int(fold), split="train")
    val_records = _load_records(config, fold=int(fold), split="val")
    summary = {
        "mode": "train",
        "fold": int(fold),
        "train_cases": len(train_records),
        "val_cases": len(val_records),
        "input_channels": int(config.get("input_channels", len(LAYER1_INPUT_CHANNELS))),
        "output_channels": 1,
        "positive_label_values": list(config.get("positive_label_values", LAYER1_POSITIVE_LABEL_VALUES)),
        "source_positive_weights": {
            str(k): float(v) for k, v in dict(config.get("source_positive_weights", LAYER1_SOURCE_AWARE_WEIGHTS)).items()
        },
        "background_weight": float(config.get("background_weight", LAYER1_BACKGROUND_WEIGHT)),
        "sampling_policy": dict(config.get("sampling_policy", {})),
        "patch_size": list(config.get("patch_size", [128, 128, 128])),
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
    device = torch.device(str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    model.to(device)
    dataset = SegMambaLayer1Dataset(
        train_records,
        positive_label_values=config.get("positive_label_values", LAYER1_POSITIVE_LABEL_VALUES),
        patch_size=config.get("patch_size", [128, 128, 128]),
        seed=int(config.get("seed", 42)),
    )
    loader = DataLoader(dataset, batch_size=int(config.get("batch_size", 1)), shuffle=True, collate_fn=_collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 1e-4)))
    model.train()
    for _epoch in range(int(max_epochs)):
        for batch in loader:
            data = batch["data"].to(device)
            target = batch["target"].to(device)
            voxel_weight = batch["voxel_weight"].to(device)
            logits = model(data)
            if logits.ndim == target.ndim - 1:
                logits = logits[:, None]
            loss = layer1_high_recall_loss(logits, target, voxel_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    checkpoint_dir = Path(config.get("checkpoint_dir", Path(config_path).parent / "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"segmamba_layer1_fold{int(fold)}.pt"
    torch.save({"state_dict": model.state_dict(), "config": dict(config), "fold": int(fold)}, checkpoint_path)
    summary["checkpoint_path"] = str(checkpoint_path)
    return summary


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
    with torch.no_grad():
        for batch in loader:
            data = batch["data"].to(device)
            logits = model(data)
            if logits.ndim == 4:
                logits = logits[:, None]
            logits_np = logits.detach().cpu().numpy()[0].astype(np.float32)
            record = batch["record"][0]
            case_id = str(record["case_id"])
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
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SegMoE SegMamba Layer1 adapter")
    sub = parser.add_subparsers(dest="command", required=True)
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
    if args.command == "train":
        payload = train(args.config, fold=int(args.fold), dry_run=bool(args.dry_run), max_epochs=int(args.max_epochs))
    else:
        payload = predict(
            args.config,
            fold=int(args.fold),
            split=str(args.split),
            checkpoint=args.checkpoint,
            dry_run=bool(args.dry_run),
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
