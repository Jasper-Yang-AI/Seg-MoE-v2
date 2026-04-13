from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..contracts import CaseManifestRow, PredictionRecord, TaskSpec


class BaseRunner(abc.ABC):
    def __init__(self, *, name: str, workspace: str | Path) -> None:
        self.name = str(name)
        self.workspace = Path(workspace)

    @abc.abstractmethod
    def train_fold(
        self,
        fold: int,
        task_spec: TaskSpec,
        cases: Sequence[CaseManifestRow],
        experiment_cfg: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def predict_fold(
        self,
        fold: int,
        split_name: str,
        task_spec: TaskSpec,
        cases: Sequence[CaseManifestRow],
        checkpoint_ref: Mapping[str, Any],
        experiment_cfg: Mapping[str, Any],
    ) -> Sequence[Mapping[str, Any]]:
        raise NotImplementedError

    def export_probabilities(
        self,
        probability_map: np.ndarray,
        *,
        task_name: str,
        stage: str,
        model_name: str,
        fold: int,
        split: str,
        case_id: str,
        predictor_fold: int,
        channel_names: Sequence[str],
        out_dir: str | Path,
        source_manifest_hash: str = "",
        preprocess_fingerprint: str = "",
    ) -> PredictionRecord:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        prob_path = out_dir / f"{case_id}.npz"
        np.savez_compressed(prob_path, probs=np.asarray(probability_map, dtype=np.float32))
        return PredictionRecord(
            task=task_name,
            stage=stage,
            model_name=model_name,
            fold=int(fold),
            split=str(split),
            case_id=str(case_id),
            predictor_fold=int(predictor_fold),
            prob_path=prob_path,
            channel_names=tuple(channel_names),
            source_manifest_hash=str(source_manifest_hash),
            preprocess_fingerprint=str(preprocess_fingerprint),
        )
