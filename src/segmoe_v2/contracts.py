from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


CohortType = Literal["pca", "nca"]
CalibrationStage = Literal["layer1_to_layer2", "layer2_to_gate"]
SourceLayer = Literal["layer1", "layer2"]


def _serialise(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_serialise(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialise(v) for k, v in value.items()}
    return value


@dataclass(frozen=True, slots=True)
class CaseManifestRow:
    case_id: str
    patient_id: str
    era_bin: str
    cohort_type: CohortType
    has_lesion_label3: bool = False
    label_unique_values: tuple[int, ...] = ()
    fixed_split: str = "unassigned"
    val_fold: int | None = None
    t2w_path: Path | str = Path()
    adc_path: Path | str = Path()
    dwi_path: Path | str = Path()
    label_path: Path | str = Path()
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    image_shape: tuple[int, int, int] = (0, 0, 0)
    affine_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: _serialise(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CaseManifestRow":
        return cls(
            case_id=str(payload["case_id"]),
            patient_id=str(payload.get("patient_id", payload["case_id"])),
            era_bin=str(payload.get("era_bin", "")),
            cohort_type=str(payload.get("cohort_type", "pca")).lower(),  # type: ignore[arg-type]
            has_lesion_label3=bool(payload.get("has_lesion_label3", False)),
            label_unique_values=tuple(int(v) for v in payload.get("label_unique_values", ())),
            fixed_split=str(payload.get("fixed_split", "unassigned")),
            val_fold=payload.get("val_fold"),
            t2w_path=Path(payload["t2w_path"]),
            adc_path=Path(payload["adc_path"]),
            dwi_path=Path(payload["dwi_path"]),
            label_path=Path(payload["label_path"]),
            spacing=tuple(float(v) for v in payload.get("spacing", (1.0, 1.0, 1.0))),
            image_shape=tuple(int(v) for v in payload.get("image_shape", (0, 0, 0))),
            affine_hash=str(payload.get("affine_hash", "")),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class TaskSpec:
    name: Literal["anatomy", "lesion"]
    input_modalities: tuple[str, ...]
    output_heads: tuple[str, ...]
    uses_priors: bool = False
    cohort_aware: bool = False
    notes: str = ""

    @classmethod
    def anatomy(cls) -> "TaskSpec":
        return cls(
            name="anatomy",
            input_modalities=("T2W", "ADC", "DWI"),
            output_heads=("WG", "PZ", "TZ"),
            uses_priors=False,
            cohort_aware=False,
            notes="Three independent sigmoid heads with lesion-aware ignore policy.",
        )

    @classmethod
    def lesion(cls, *, uses_priors: bool = True) -> "TaskSpec":
        channels = ("T2W", "ADC", "DWI")
        if uses_priors:
            channels = channels + ("P_WG", "P_PZ", "P_TZ")
        return cls(
            name="lesion",
            input_modalities=channels,
            output_heads=("lesion",),
            uses_priors=uses_priors,
            cohort_aware=True,
            notes="PCA label==3 is positive, NCA lesion target is forced empty.",
        )


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    task: str
    stage: str
    model_name: str
    fold: int
    split: str
    case_id: str
    predictor_fold: int
    prob_path: Path | str
    logit_path: Path | str | None = None
    channel_names: tuple[str, ...] = ()
    source_manifest_hash: str = ""
    preprocess_fingerprint: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: _serialise(v) for k, v in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class CalibrationRecord:
    stage: CalibrationStage
    fold: int
    expert: str
    temperature: float
    fit_case_count: int
    n_pos_voxels: int
    n_neg_voxels: int
    source_oof_manifest_hash: str
    fallback_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {k: _serialise(v) for k, v in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class FPComponentRecord:
    case_id: str
    source_layer: SourceLayer
    predictor_fold: int
    component_id: int
    bbox_zyx: tuple[int, int, int, int, int, int]
    centroid_zyx: tuple[float, float, float]
    volume_voxels: int
    dominant_zone: str
    signed_distance_to_wg_boundary_mm: float
    expert_prob_mean: tuple[float, ...]
    expert_prob_max: tuple[float, ...]
    entropy_mean: float
    entropy_max: float
    disagreement_mean: float
    disagreement_max: float
    intensity_mean: dict[str, float]
    intensity_std: dict[str, float]
    intensity_p10: dict[str, float]
    intensity_p90: dict[str, float]
    overlap_voxels: int
    max_iou: float
    fp_kind: str

    def to_dict(self) -> dict[str, Any]:
        return {k: _serialise(v) for k, v in asdict(self).items()}
