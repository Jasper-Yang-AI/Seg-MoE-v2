from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import PredictionRecord
from .io_utils import load_jsonl, save_jsonl, stable_hash


def merge_prediction_manifest_files(inputs: Sequence[str | Path], output: str | Path) -> Path:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for input_path in inputs:
        for record in load_jsonl(input_path):
            key = (
                str(record.get("case_id", "")),
                str(record.get("model_name", "")),
                str(record.get("split", "")),
                str(record.get("prob_path") or record.get("logit_path") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            records.append(dict(record))
    return save_jsonl(records, output)


def _matches_split(record: Mapping[str, Any], *, fold: int, split: str) -> bool:
    split = str(split)
    if split in {"val", "validation", f"val_{fold}"}:
        return record.get("fixed_split") == "trainval" and int(record.get("val_fold", -1)) == int(fold)
    if split == "train":
        return record.get("fixed_split") == "trainval" and int(record.get("val_fold", -1)) != int(fold)
    if split == "test":
        return record.get("fixed_split") == "test"
    return str(record.get("fixed_split")) == split


def _normalise_channel_names(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    arr = np.asarray(raw)
    return tuple(str(item.decode("utf-8") if isinstance(item, bytes) else item) for item in arr.tolist())


def _infer_prediction_payload(path: Path) -> tuple[str, tuple[str, ...]]:
    payload = np.load(path, allow_pickle=True)
    channel_names = _normalise_channel_names(payload["channel_names"] if "channel_names" in payload else None)
    for key in ("probabilities", "probs", "softmax", "logits"):
        if key not in payload:
            continue
        values = np.asarray(payload[key])
        if channel_names:
            return key, channel_names
        if key == "logits":
            return key, ("P_lesion_logit",) if values.shape[0] == 1 else tuple(f"logit_{idx}" for idx in range(values.shape[0]))
        if values.shape[0] == 1:
            return key, ("P_lesion",)
        if values.shape[0] == 2:
            return key, ("background", "P_lesion")
        return key, tuple(f"channel_{idx}" for idx in range(values.shape[0]))
    raise KeyError(f"{path} must contain one of probabilities, probs, softmax, or logits.")


def build_layer1_prediction_manifest(
    *,
    prediction_dir: str | Path,
    dataset_index: str | Path,
    output: str | Path,
    model_name: str,
    fold: int,
    split: str,
    allow_missing: bool = False,
) -> Path:
    prediction_dir = Path(prediction_dir)
    records = [record for record in load_jsonl(dataset_index) if _matches_split(record, fold=int(fold), split=split)]
    manifest_hash = stable_hash(records)
    prediction_records: list[dict[str, Any]] = []
    missing: list[str] = []
    for record in records:
        case_id = str(record["case_id"])
        prediction_path = prediction_dir / f"{case_id}.npz"
        if not prediction_path.exists():
            missing.append(case_id)
            continue
        key, channel_names = _infer_prediction_payload(prediction_path)
        payload = PredictionRecord(
            task="lesion",
            stage="layer1",
            model_name=str(model_name),
            fold=int(fold),
            split=f"val_{fold}" if split in {"val", "validation"} else str(split),
            case_id=case_id,
            predictor_fold=int(fold),
            prob_path=prediction_path if key in {"probabilities", "probs", "softmax"} else None,
            logit_path=prediction_path if key == "logits" else None,
            channel_names=channel_names,
            source_manifest_hash=str(record.get("source_manifest_hash", manifest_hash)),
            metadata={
                "prediction_key": key,
                "labels_available": bool(record.get("metadata", {}).get("labels_available", False)),
                "bbox_zyx": record.get("metadata", {}).get("bbox_zyx"),
                "native_shape_zyx": record.get("metadata", {}).get("native_shape_zyx"),
            },
        )
        prediction_records.append(payload.to_dict())
    if missing and not allow_missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing {len(missing)} Layer1 predictions in {prediction_dir}: {preview}")
    return save_jsonl(prediction_records, output)
