from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ..contracts import CaseManifestRow, TaskSpec
from .base import BaseRunner


class SwinUNETRRunner(BaseRunner):
    def __init__(self, *, workspace: str | Path, repo_root: str | Path | None = None) -> None:
        super().__init__(name="SwinUNETR_3d", workspace=workspace)
        from ..backend_data import resolve_vendored_backend_root

        self.repo_root = resolve_vendored_backend_root("swinunetr", repo_root)

    def build_model(
        self,
        *,
        in_channels: int,
        out_channels: int,
        feature_size: int = 48,
        pretrained_path: str | Path | None = None,
    ) -> Any:
        try:
            from monai.networks.nets import SwinUNETR
        except ImportError as exc:
            raise ImportError(
                "MONAI is required for SwinUNETRRunner. Install with `pip install monai SimpleITK`."
            ) from exc

        model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            feature_size=int(feature_size),
            use_checkpoint=True,
        )
        if pretrained_path:
            state = __import__("torch").load(str(pretrained_path), map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            missing, unexpected = model.load_state_dict(state, strict=False)
            return model, {"missing": missing, "unexpected": unexpected}
        return model, {"missing": [], "unexpected": []}

    def train_fold(
        self,
        fold: int,
        task_spec: TaskSpec,
        cases: Sequence[CaseManifestRow],
        experiment_cfg: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return {
            "runner": self.name,
            "fold": int(fold),
            "task": task_spec.name,
            "cases": len(cases),
            "status": "ready_for_custom_supervised_training_loop",
        }

    def predict_fold(
        self,
        fold: int,
        split_name: str,
        task_spec: TaskSpec,
        cases: Sequence[CaseManifestRow],
        checkpoint_ref: Mapping[str, Any],
        experiment_cfg: Mapping[str, Any],
    ) -> Sequence[Mapping[str, Any]]:
        return [
            {
                "runner": self.name,
                "fold": int(fold),
                "split": split_name,
                "task": task_spec.name,
                "cases": len(cases),
                "status": "ready_for_custom_supervised_inference_loop",
            }
        ]
