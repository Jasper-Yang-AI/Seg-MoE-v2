from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..contracts import CaseManifestRow, TaskSpec
from .base import BaseRunner
from .utils import build_subprocess_env


class NnFormerRunner(BaseRunner):
    def __init__(self, *, workspace: str | Path, repo_root: str | Path | None = None) -> None:
        super().__init__(name="nnFormer_3d", workspace=workspace)
        from ..backend_data import resolve_vendored_backend_root

        self.repo_root = resolve_vendored_backend_root("nnformer", repo_root)

    def _env(self, experiment_cfg: Mapping[str, Any]) -> dict[str, str]:
        env = build_subprocess_env(
            backend="nnformer",
            repo_root=self.repo_root,
            extra_env=dict(experiment_cfg.get("env", {})),
        )
        env.setdefault("nnFormer_raw_data_base", str(self.workspace / "nnFormer_data"))
        env.setdefault("nnFormer_preprocessed", str(self.workspace / "nnFormer_preprocessed"))
        env.setdefault("RESULTS_FOLDER", str(self.workspace / "nnFormer_results"))
        Path(env["nnFormer_raw_data_base"]).mkdir(parents=True, exist_ok=True)
        Path(env["nnFormer_preprocessed"]).mkdir(parents=True, exist_ok=True)
        Path(env["RESULTS_FOLDER"]).mkdir(parents=True, exist_ok=True)
        return env

    def train_fold(
        self,
        fold: int,
        task_spec: TaskSpec,
        cases: Sequence[CaseManifestRow],
        experiment_cfg: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        task_id = experiment_cfg["task_id"]
        trainer = experiment_cfg.get("trainer", "nnFormerTrainerV2")
        command = [
            sys.executable,
            "-m",
            "nnformer.run.run_training",
            "3d_fullres",
            str(trainer),
            str(task_id),
            str(fold),
        ]
        if experiment_cfg.get("dry_run", False):
            return {"command": command, "cases": len(cases), "task": task_spec.name}
        subprocess.run(command, cwd=self.repo_root, env=self._env(experiment_cfg), check=True)
        return {"command": command, "task": task_spec.name, "fold": int(fold)}

    def predict_fold(
        self,
        fold: int,
        split_name: str,
        task_spec: TaskSpec,
        cases: Sequence[CaseManifestRow],
        checkpoint_ref: Mapping[str, Any],
        experiment_cfg: Mapping[str, Any],
    ) -> Sequence[Mapping[str, Any]]:
        input_dir = experiment_cfg["predict_input_dir"]
        output_dir = experiment_cfg["predict_output_dir"]
        task_id = experiment_cfg["task_id"]
        trainer = experiment_cfg.get("trainer", "nnFormerTrainerV2")
        plans = experiment_cfg.get("plans", "nnFormerPlans")
        command = [
            sys.executable,
            "-m",
            "nnformer.inference.predict_simple",
            "-i",
            str(input_dir),
            "-o",
            str(output_dir),
            "-m",
            "3d_fullres",
            "-t",
            str(task_id),
            "-f",
            str(fold),
            "-tr",
            str(trainer),
            "-p",
            str(plans),
            "-z",
        ]
        if experiment_cfg.get("dry_run", False):
            return [{"command": command, "cases": len(cases), "task": task_spec.name, "split": split_name}]
        subprocess.run(command, cwd=self.repo_root, env=self._env(experiment_cfg), check=True)
        return [{"command": command, "task": task_spec.name, "split": split_name}]
