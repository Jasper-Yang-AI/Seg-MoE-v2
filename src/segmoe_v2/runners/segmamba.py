from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..contracts import CaseManifestRow, TaskSpec
from .base import BaseRunner
from .utils import append_pythonpath, build_subprocess_env


class SegMambaRunner(BaseRunner):
    def __init__(self, *, workspace: str | Path, repo_root: str | Path | None = None) -> None:
        super().__init__(name="SegMamba_3d", workspace=workspace)
        from ..backend_data import resolve_vendored_backend_root

        self.repo_root = resolve_vendored_backend_root("segmamba", repo_root)

    def _env(self, fold: int | None, split_name: str | None, experiment_cfg: Mapping[str, Any]) -> dict[str, str]:
        env = build_subprocess_env(
            backend="segmamba",
            repo_root=self.repo_root,
            extra_env=dict(experiment_cfg.get("env", {})),
        )
        env["PYTHONPATH"] = append_pythonpath(
            env.get("PYTHONPATH"),
            self.repo_root / "mamba",
            self.repo_root / "causal-conv1d",
        )
        env.setdefault("SEGMAMBA_DATA_DIR", str(experiment_cfg.get("data_dir", self.workspace / "segmamba_data")))
        env.setdefault("SEGMAMBA_LOGDIR", str(experiment_cfg.get("logdir", self.workspace / "segmamba_runs")))
        env.setdefault(
            "SEGMAMBA_PREDICTION_DIR",
            str(experiment_cfg.get("prediction_dir", self.workspace / "segmamba_predictions")),
        )
        if fold is not None:
            env["SEGMAMBA_FOLD"] = str(fold)
        if split_name is not None:
            env["SEGMAMBA_SPLIT"] = str(split_name)
        for key in ("SEGMAMBA_DATA_DIR", "SEGMAMBA_LOGDIR", "SEGMAMBA_PREDICTION_DIR"):
            Path(env[key]).mkdir(parents=True, exist_ok=True)
        return env

    def _ensure_import_paths(self) -> None:
        for path in (self.repo_root, self.repo_root / "mamba", self.repo_root / "causal-conv1d"):
            resolved = str(path)
            if resolved not in sys.path:
                sys.path.insert(0, resolved)

    def build_model(
        self,
        *,
        in_channels: int = 3,
        out_channels: int = 1,
        pretrained_path: str | Path | None = None,
        **kwargs: Any,
    ) -> Any:
        self._ensure_import_paths()
        try:
            import torch
            from model_segmamba.segmamba import SegMamba
        except ImportError as exc:
            raise ImportError(
                "SegMambaRunner.build_model requires the SegMamba repository on PYTHONPATH, PyTorch installed, "
                "and the mamba_ssm/causal_conv1d CUDA extensions available."
            ) from exc

        model = SegMamba(in_chans=int(in_channels), out_chans=int(out_channels), **kwargs)
        if pretrained_path:
            state = torch.load(str(pretrained_path), map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            missing, unexpected = model.load_state_dict(state, strict=False)
            return model, missing, unexpected
        return model, [], []

    def train_fold(
        self,
        fold: int,
        task_spec: TaskSpec,
        cases: Sequence[CaseManifestRow],
        experiment_cfg: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        script = experiment_cfg.get("train_script", "3_train.py")
        command = [sys.executable, str(script)]
        command.extend(str(arg) for arg in experiment_cfg.get("train_args", ()))
        if experiment_cfg.get("dry_run", False):
            return {
                "command": command,
                "runner": self.name,
                "fold": int(fold),
                "task": task_spec.name,
                "cases": len(cases),
                "status": "ready_for_segmamba_training_script",
            }
        subprocess.run(command, cwd=self.repo_root, env=self._env(fold, "train", experiment_cfg), check=True)
        return {"command": command, "runner": self.name, "task": task_spec.name, "fold": int(fold)}

    def predict_fold(
        self,
        fold: int,
        split_name: str,
        task_spec: TaskSpec,
        cases: Sequence[CaseManifestRow],
        checkpoint_ref: Mapping[str, Any],
        experiment_cfg: Mapping[str, Any],
    ) -> Sequence[Mapping[str, Any]]:
        script = experiment_cfg.get("predict_script", "4_predict.py")
        command = [sys.executable, str(script)]
        command.extend(str(arg) for arg in experiment_cfg.get("predict_args", ()))
        if experiment_cfg.get("dry_run", False):
            return [
                {
                    "command": command,
                    "runner": self.name,
                    "fold": int(fold),
                    "split": split_name,
                    "task": task_spec.name,
                    "cases": len(cases),
                    "status": "ready_for_segmamba_prediction_script",
                }
            ]
        subprocess.run(command, cwd=self.repo_root, env=self._env(fold, split_name, experiment_cfg), check=True)
        return [{"command": command, "runner": self.name, "task": task_spec.name, "split": split_name}]
