from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .contracts import PredictionRecord
from .io_utils import load_jsonl, save_jsonl


def save_prediction_manifest(records: Iterable[PredictionRecord], path: str | Path) -> Path:
    return save_jsonl((record.to_dict() for record in records), path)


def load_prediction_manifest(path: str | Path) -> list[PredictionRecord]:
    return [PredictionRecord(**row) for row in load_jsonl(path)]


def index_prediction_manifest(records: Iterable[PredictionRecord]) -> dict[str, list[PredictionRecord]]:
    grouped: dict[str, list[PredictionRecord]] = defaultdict(list)
    for record in records:
        grouped[record.case_id].append(record)
    return dict(grouped)
