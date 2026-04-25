from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..contracts import CaseManifestRow, TaskSpec
from .base import BaseRunner
from .utils import build_subprocess_env, project_root


class MedNeXtRunner(BaseRunner):
    def __init__(self, *, workspace: str | Path, repo_root: str | Path | None = None) -> None:
        super().__init__(name="MedNeXt_3d", workspace=workspace)
        from ..backend_data import resolve_vendored_backend_root

        self.repo_root = resolve_vendored_backend_root("mednext", repo_root)

    def _env(self, experiment_cfg: Mapping[str, Any]) -> dict[str, str]:
        env = build_subprocess_env(
            backend="mednext",
            repo_root=self.repo_root,
            extra_env=dict(experiment_cfg.get("env", {})),
        )
        root = Path(experiment_cfg.get("project_root", project_root()))
        env.setdefault("nnUNet_raw_data_base", str(root / "MedNeXt_raw_data_base"))
        env.setdefault("nnUNet_preprocessed", str(root / "MedNeXt_preprocessed"))
        env.setdefault("RESULTS_FOLDER", str(root / "MedNeXt_results"))
        for key in ("nnUNet_raw_data_base", "nnUNet_preprocessed", "RESULTS_FOLDER"):
            Path(env[key]).mkdir(parents=True, exist_ok=True)
        return env

    @staticmethod
    def _task_identifier(experiment_cfg: Mapping[str, Any]) -> str:
        task = experiment_cfg.get("task", experiment_cfg.get("task_name", experiment_cfg.get("task_id")))
        if task is None:
            raise KeyError("MedNeXtRunner requires `task`, `task_name`, or `task_id` in experiment_cfg.")
        return str(task)

    @staticmethod
    def _entry_command(
        experiment_cfg: Mapping[str, Any],
        executable_key: str,
        module_key: str,
        default_module: str,
    ) -> list[str]:
        if executable_key in experiment_cfg:
            return [str(experiment_cfg[executable_key])]
        return [sys.executable, "-m", str(experiment_cfg.get(module_key, default_module))]

    def train_fold(
        self,
        fold: int,
        task_spec: TaskSpec,
        cases: Sequence[CaseManifestRow],
        experiment_cfg: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        trainer = experiment_cfg.get(
            "trainer",
            "nnUNetTrainerV2_MedNeXt_S_kernel3_SegMoELayer1"
            if task_spec.name == "lesion"
            else "nnUNetTrainerV2_MedNeXt_S_kernel3",
        )
        plans = experiment_cfg.get("plans", "nnUNetPlansv2.1_trgSp_1x1x1")
        command = self._entry_command(
            experiment_cfg,
            "train_executable",
            "train_module",
            "nnunet_mednext.run.run_training",
        ) + [
            str(experiment_cfg.get("network", "3d_fullres")),
            str(trainer),
            self._task_identifier(experiment_cfg),
            str(fold),
            "-p",
            str(plans),
        ]
        if experiment_cfg.get("pretrained_weights"):
            command.extend(["-pretrained_weights", str(experiment_cfg["pretrained_weights"])])
        if experiment_cfg.get("resample_weights"):
            command.append("-resample_weights")
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
        trainer = experiment_cfg.get(
            "trainer",
            "nnUNetTrainerV2_MedNeXt_S_kernel3_SegMoELayer1"
            if task_spec.name == "lesion"
            else "nnUNetTrainerV2_MedNeXt_S_kernel3",
        )
        plans = experiment_cfg.get("plans", "nnUNetPlansv2.1_trgSp_1x1x1")
        command = self._entry_command(
            experiment_cfg,
            "predict_executable",
            "predict_module",
            "nnunet_mednext.inference.predict_simple",
        ) + [
            "-i",
            str(input_dir),
            "-o",
            str(output_dir),
            "-t",
            self._task_identifier(experiment_cfg),
            "-m",
            str(experiment_cfg.get("network", "3d_fullres")),
            "-f",
            str(fold),
            "-tr",
            str(trainer),
            "-p",
            str(plans),
            "-z",
        ]
        checkpoint_name = checkpoint_ref.get("checkpoint_name") or experiment_cfg.get("checkpoint_name")
        if checkpoint_name:
            command.extend(["-chk", str(checkpoint_name)])
        if experiment_cfg.get("dry_run", False):
            return [{"command": command, "cases": len(cases), "task": task_spec.name, "split": split_name}]
        subprocess.run(command, cwd=self.repo_root, env=self._env(experiment_cfg), check=True)
        return [
            {
                "case_id": case.case_id,
                "fold": int(fold),
                "split": str(split_name),
                "channel_names": ("background", "P_lesion"),
                "prob_path": str(Path(output_dir) / f"{case.case_id}.npz"),
                "source_manifest_hash": str(experiment_cfg.get("source_manifest_hash", "")),
                "command": command,
            }
            for case in cases
        ]
