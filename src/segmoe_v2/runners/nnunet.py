from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..contracts import CaseManifestRow, TaskSpec
from .base import BaseRunner
from .utils import build_subprocess_env, project_root


class NnUNetResEncRunner(BaseRunner):
    def __init__(self, *, workspace: str | Path, repo_root: str | Path | None = None) -> None:
        super().__init__(name="nnUNetv2_3d_fullres_resenc", workspace=workspace)
        from ..backend_data import resolve_vendored_backend_root

        self.repo_root = resolve_vendored_backend_root("nnunet", repo_root)

    def _env(self, experiment_cfg: Mapping[str, Any]) -> dict[str, str]:
        env = build_subprocess_env(
            backend="nnunet",
            repo_root=self.repo_root,
            extra_env=dict(experiment_cfg.get("env", {})),
        )
        root = Path(experiment_cfg.get("project_root", project_root()))
        env.setdefault("nnUNet_raw", str(root / "nnUNet_raw"))
        env.setdefault("nnUNet_preprocessed", str(root / "nnUNet_preprocessed"))
        env.setdefault("nnUNet_results", str(root / "nnUNet_results"))
        for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
            Path(env[key]).mkdir(parents=True, exist_ok=True)
        return env

    def train_fold(
        self,
        fold: int,
        task_spec: TaskSpec,
        cases: Sequence[CaseManifestRow],
        experiment_cfg: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        dataset_id = experiment_cfg["dataset_id"]
        trainer = experiment_cfg.get(
            "trainer",
            "nnUNetTrainerSegMoEAnatomy" if task_spec.name == "anatomy" else "nnUNetTrainer",
        )
        plans = experiment_cfg.get("plans", "nnUNetResEncUNetMPlans")
        export_validation_probabilities = bool(
            experiment_cfg.get("export_validation_probabilities", task_spec.name == "anatomy")
        )
        command = [
            sys.executable,
            "-m",
            "nnunetv2.run.run_training",
            str(dataset_id),
            "3d_fullres",
            str(fold),
            "-tr",
            str(trainer),
            "-p",
            str(plans),
        ]
        if export_validation_probabilities:
            command.append("--npz")
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
        dataset_id = experiment_cfg["dataset_id"]
        trainer = experiment_cfg.get(
            "trainer",
            "nnUNetTrainerSegMoEAnatomy" if task_spec.name == "anatomy" else "nnUNetTrainer",
        )
        plans = experiment_cfg.get("plans", "nnUNetResEncUNetMPlans")
        if task_spec.name == "anatomy":
            command = [
                sys.executable,
                "-m",
                "segmoe_v2.nnunet_anatomy_predict",
                "-d",
                str(dataset_id),
                "-i",
                str(input_dir),
                "-o",
                str(output_dir),
                "-f",
                str(fold),
                "-tr",
                str(trainer),
                "-p",
                str(plans),
                "--split-name",
                str(split_name),
            ]
        else:
            command = [
                sys.executable,
                "-c",
                "from nnunetv2.inference.predict_from_raw_data import predict_entry_point; predict_entry_point()",
                "-d",
                str(dataset_id),
                "-i",
                str(input_dir),
                "-o",
                str(output_dir),
                "-f",
                str(fold),
                "-tr",
                str(trainer),
                "-p",
                str(plans),
                "--save_probabilities",
            ]
        if experiment_cfg.get("dry_run", False):
            return [{"command": command, "cases": len(cases), "task": task_spec.name, "split": split_name}]
        subprocess.run(command, cwd=self.repo_root, env=self._env(experiment_cfg), check=True)
        if task_spec.name == "anatomy":
            return [
                {
                    "case_id": case.case_id,
                    "fold": int(fold),
                    "split": str(split_name),
                    "channel_names": ("P_WG", "P_PZ", "P_TZ"),
                    "prob_path": str(Path(output_dir) / f"{case.case_id}.npz"),
                    "source_manifest_hash": str(experiment_cfg.get("source_manifest_hash", "")),
                    "command": command,
                }
                for case in cases
            ]
        return [{"command": command, "task": task_spec.name, "split": split_name}]
