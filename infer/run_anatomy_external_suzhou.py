#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import nibabel as nib
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
NNUNET_ROOT = REPO_ROOT / "external" / "nnU-Net"
for path in (str(SRC_ROOT), str(NNUNET_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault("nnUNet_raw", str(REPO_ROOT / "nnUNet_raw"))
os.environ.setdefault("nnUNet_preprocessed", str(REPO_ROOT / "nnUNet_preprocessed"))
os.environ.setdefault("nnUNet_results", str(REPO_ROOT / "nnUNet_results"))

EXTERNAL_ROOT = Path("/mnt/z/multicenter_pca/PCa_SuZhou_Multicenter")
FOLDS = (0, 1, 2, 3, 4)
CHECKPOINT_NAME = "checkpoint_best.pth"
CASE_LIMIT: int | None = None

DATASET_ID = "501"
TRAINER = "nnUNetTrainerSegMoEAnatomy"
PLANS = "nnUNetResEncUNetMPlans"
CONFIGURATION = "3d_fullres"

OUTPUT_MASK_NAME = "segmoe_anatomy_best.nii.gz"
METRICS_CSV = "segmoe_anatomy_best_metrics.csv"
SUMMARY_JSON = "segmoe_anatomy_best_summary.json"
PREFLIGHT_ERRORS_CSV = "segmoe_anatomy_best_preflight_errors.csv"

TILE_STEP_SIZE = 0.5
USE_MIRRORING = True
PERFORM_EVERYTHING_ON_DEVICE = True
NUM_PREPROCESS_WORKERS = 2

MODALITY_PATTERNS = {
    "t2w": (r"(^|[/_\-.])t2w?([/_\-.]|$)", r"t2[_\- ]?tra", r"t2[_\- ]?ax"),
    "adc": (r"(^|[/_\-.])adc([/_\-.]|$)",),
    "dwi": (r"(^|[/_\-.])dwi([/_\-.]|$)", r"(^|[/_\-.])hbv([/_\-.]|$)", r"b[0-9]{3,4}"),
}
LABEL_PATTERNS = (
    r"(^|/)(label|labels|seg|segs|mask|masks|annotation|annotations)(/|$)",
    r"(^|[/_\-.])(label|seg|mask|annotation)([/_\-.]|$)",
)
LABEL_EXCLUDE_PATTERNS = (OUTPUT_MASK_NAME.lower(),)
MODALITY_EXCLUDE_PATTERNS = (
    r"(^|/)(label|labels|seg|segs|mask|masks|annotation|annotations)(/|$)",
    OUTPUT_MASK_NAME.lower(),
)

REGIONS = {
    "WG": (1, 2, 3),
    "PZ": (1,),
    "TZ": (2,),
}


@dataclass(frozen=True)
class ExternalCase:
    case_id: str
    case_dir: Path
    t2w_path: Path
    adc_path: Path
    dwi_path: Path
    label_path: Path
    output_mask_path: Path


def _nifti_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and (path.name.endswith(".nii.gz") or path.name.endswith(".nii")))


def _matches(path: Path, patterns: Sequence[str], *, base: Path) -> bool:
    rel = path.relative_to(base).as_posix().lower()
    return any(re.search(pattern, rel, flags=re.IGNORECASE) for pattern in patterns)


def _candidate_files(
    case_dir: Path,
    patterns: Sequence[str],
    *,
    exclude_patterns: Sequence[str] = (),
) -> list[Path]:
    candidates = []
    for path in _nifti_files(case_dir):
        rel = path.relative_to(case_dir).as_posix().lower()
        if any(re.search(pattern, rel, flags=re.IGNORECASE) for pattern in exclude_patterns):
            continue
        if _matches(path, patterns, base=case_dir):
            candidates.append(path)
    return candidates


def _find_unique(
    case_dir: Path,
    key: str,
    patterns: Sequence[str],
    errors: list[dict[str, str]],
    *,
    exclude_patterns: Sequence[str] = (),
) -> Path | None:
    candidates = _candidate_files(case_dir, patterns, exclude_patterns=exclude_patterns)
    if len(candidates) == 1:
        return candidates[0]
    errors.append(
        {
            "case_id": case_dir.name,
            "case_dir": str(case_dir),
            "field": key,
            "error": "missing" if len(candidates) == 0 else "ambiguous",
            "candidates": ";".join(str(path) for path in candidates[:20]),
        }
    )
    return None


def discover_cases(root: Path) -> tuple[list[ExternalCase], list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    if not root.exists():
        errors.append(
            {
                "case_id": "",
                "case_dir": str(root),
                "field": "EXTERNAL_ROOT",
                "error": "not_found",
                "candidates": "Mount Z: as /mnt/z or update EXTERNAL_ROOT at the top of this script.",
            }
        )
        return [], errors

    case_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    cases: list[ExternalCase] = []
    for case_dir in case_dirs:
        t2w = _find_unique(case_dir, "t2w", MODALITY_PATTERNS["t2w"], errors, exclude_patterns=MODALITY_EXCLUDE_PATTERNS)
        adc = _find_unique(case_dir, "adc", MODALITY_PATTERNS["adc"], errors, exclude_patterns=MODALITY_EXCLUDE_PATTERNS)
        dwi = _find_unique(case_dir, "dwi", MODALITY_PATTERNS["dwi"], errors, exclude_patterns=MODALITY_EXCLUDE_PATTERNS)
        label = _find_unique(case_dir, "label", LABEL_PATTERNS, errors, exclude_patterns=LABEL_EXCLUDE_PATTERNS)
        if t2w is None or adc is None or dwi is None or label is None:
            continue
        cases.append(
            ExternalCase(
                case_id=case_dir.name,
                case_dir=case_dir,
                t2w_path=t2w,
                adc_path=adc,
                dwi_path=dwi,
                label_path=label,
                output_mask_path=case_dir / OUTPUT_MASK_NAME,
            )
        )
    if CASE_LIMIT is not None:
        cases = cases[: int(CASE_LIMIT)]
    return cases, errors


def write_preflight_errors(root: Path, errors: Sequence[dict[str, str]]) -> Path:
    path = root / PREFLIGHT_ERRORS_CSV if root.exists() else REPO_ROOT / PREFLIGHT_ERRORS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "case_dir", "field", "error", "candidates"])
        writer.writeheader()
        writer.writerows(errors)
    return path


def check_nnunet_environment() -> Path:
    import nnunetv2
    from nnunetv2.utilities.file_path_utilities import get_output_folder

    nnunet_file = Path(nnunetv2.__file__).resolve()
    expected_root = NNUNET_ROOT.resolve()
    if expected_root not in nnunet_file.parents:
        raise RuntimeError(f"Wrong nnunetv2 import: {nnunet_file}. Expected it under {expected_root}.")

    model_folder = Path(get_output_folder(DATASET_ID, TRAINER, PLANS, CONFIGURATION))
    missing = [
        model_folder / f"fold_{fold}" / CHECKPOINT_NAME
        for fold in FOLDS
        if not (model_folder / f"fold_{fold}" / CHECKPOINT_NAME).exists()
    ]
    if missing:
        raise FileNotFoundError("Missing anatomy checkpoints:\n" + "\n".join(str(path) for path in missing))
    return model_folder


def orient_probabilities_xyz(probabilities: np.ndarray, reference_shape: tuple[int, int, int]) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.ndim != 4 or probabilities.shape[0] != 3:
        raise ValueError(f"Expected probabilities with shape (3, *, *, *), got {probabilities.shape}")
    spatial = tuple(int(v) for v in probabilities.shape[1:])
    if spatial == tuple(reference_shape):
        return probabilities
    for permutation in ((2, 1, 0), (0, 1, 2), (1, 0, 2), (0, 2, 1), (1, 2, 0), (2, 0, 1)):
        expected = tuple(reference_shape[index] for index in permutation)
        if spatial == expected:
            inverse = tuple(permutation.index(axis) + 1 for axis in range(3))
            return probabilities.transpose((0, *inverse))
    raise ValueError(f"Cannot orient probabilities {spatial} to reference shape {reference_shape}")


def probabilities_to_label_mask(probabilities: np.ndarray) -> np.ndarray:
    p_wg, p_pz, p_tz = probabilities
    pz = p_pz >= 0.5
    tz = p_tz >= 0.5
    pz_final = pz & ((~tz) | (p_pz >= p_tz))
    tz_final = tz & ((~pz) | (p_tz > p_pz))
    wg_residual = (p_wg >= 0.5) & ~(pz_final | tz_final)

    mask = np.zeros(p_wg.shape, dtype=np.uint8)
    mask[wg_residual] = 3
    mask[pz_final] = 1
    mask[tz_final] = 2
    return mask


def _region_metrics(prediction: np.ndarray, reference: np.ndarray, spacing: Sequence[float]) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=bool)
    ref = np.asarray(reference, dtype=bool)
    tp = float(np.logical_and(pred, ref).sum())
    fp = float(np.logical_and(pred, ~ref).sum())
    fn = float(np.logical_and(~pred, ref).sum())
    denom_dice = (2.0 * tp) + fp + fn
    denom_iou = tp + fp + fn
    voxel_ml = float(np.prod([float(v) for v in spacing[:3]]) / 1000.0)
    return {
        "dice": float((2.0 * tp / denom_dice) if denom_dice > 0 else np.nan),
        "iou": float((tp / denom_iou) if denom_iou > 0 else np.nan),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ref_volume_ml": float(ref.sum() * voxel_ml),
        "pred_volume_ml": float(pred.sum() * voxel_ml),
    }


def compute_case_metrics(case: ExternalCase, prediction_mask: np.ndarray) -> dict[str, float | str]:
    label_image = nib.load(str(case.label_path))
    reference = np.asanyarray(label_image.dataobj).astype(np.int16)
    if reference.shape != prediction_mask.shape:
        raise ValueError(
            f"{case.case_id}: label shape {reference.shape} does not match prediction shape {prediction_mask.shape}"
        )
    spacing = tuple(float(v) for v in label_image.header.get_zooms()[:3])
    row: dict[str, float | str] = {
        "case_id": case.case_id,
        "case_dir": str(case.case_dir),
        "prediction_mask": str(case.output_mask_path),
    }
    for region, values in REGIONS.items():
        pred_region = np.isin(prediction_mask, values)
        ref_region = np.isin(reference, values)
        metrics = _region_metrics(pred_region, ref_region, spacing)
        for key, value in metrics.items():
            row[f"{region}_{key}"] = value
    return row


def write_metrics_csv(path: Path, rows: Sequence[dict[str, float | str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["case_id", "case_dir", "prediction_mask"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _finite(values: Iterable[float | str]) -> np.ndarray:
    numeric = np.asarray([float(value) for value in values], dtype=np.float64)
    return numeric[np.isfinite(numeric)]


def summarize_metrics(rows: Sequence[dict[str, float | str]]) -> dict[str, object]:
    summary: dict[str, object] = {
        "external_root": str(EXTERNAL_ROOT),
        "checkpoint_name": CHECKPOINT_NAME,
        "folds": list(FOLDS),
        "case_count": len(rows),
        "regions": {},
    }
    region_summary: dict[str, dict[str, dict[str, float | int]]] = {}
    for region in REGIONS:
        region_summary[region] = {}
        for metric in ("dice", "iou", "ref_volume_ml", "pred_volume_ml"):
            values = _finite(row[f"{region}_{metric}"] for row in rows)
            region_summary[region][metric] = {
                "n": int(values.size),
                "mean": float(np.mean(values)) if values.size else float("nan"),
                "median": float(np.median(values)) if values.size else float("nan"),
                "std": float(np.std(values)) if values.size else float("nan"),
                "min": float(np.min(values)) if values.size else float("nan"),
                "max": float(np.max(values)) if values.size else float("nan"),
            }
    summary["regions"] = region_summary
    return summary


def save_prediction_mask(case: ExternalCase, mask: np.ndarray) -> None:
    t2_image = nib.load(str(case.t2w_path))
    if tuple(t2_image.shape) != tuple(mask.shape):
        raise ValueError(f"{case.case_id}: T2 shape {t2_image.shape} does not match prediction shape {mask.shape}")
    header = t2_image.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine=t2_image.affine, header=header), str(case.output_mask_path))


def build_predictor(model_folder: Path):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=TILE_STEP_SIZE,
        use_gaussian=True,
        use_mirroring=USE_MIRRORING,
        perform_everything_on_device=PERFORM_EVERYTHING_ON_DEVICE,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=tuple(int(fold) for fold in FOLDS),
        checkpoint_name=CHECKPOINT_NAME,
    )
    return predictor


def run_inference(cases: Sequence[ExternalCase], model_folder: Path) -> list[dict[str, float | str]]:
    from segmoe_v2.nnunet_anatomy import convert_anatomy_logits_to_probabilities_with_correct_shape

    predictor = build_predictor(model_folder)
    input_lists = [[str(case.t2w_path), str(case.adc_path), str(case.dwi_path)] for case in cases]
    output_truncated = [str(case.case_dir / case.case_id) for case in cases]
    case_by_ofile = {output: case for output, case in zip(output_truncated, cases, strict=True)}
    data_iterator = predictor._internal_get_data_iterator_from_lists_of_filenames(
        input_lists,
        [None] * len(input_lists),
        output_truncated,
        num_processes=NUM_PREPROCESS_WORKERS,
    )

    metric_rows: list[dict[str, float | str]] = []
    for preprocessed in data_iterator:
        ofile = str(preprocessed["ofile"])
        case = case_by_ofile[ofile]
        data = preprocessed["data"]
        if isinstance(data, str):
            npy_path = Path(data)
            data = torch.from_numpy(np.load(npy_path))
            npy_path.unlink(missing_ok=True)
        logits = predictor.predict_logits_from_preprocessed_data(data).cpu()
        probabilities = convert_anatomy_logits_to_probabilities_with_correct_shape(
            logits,
            plans_manager=predictor.plans_manager,
            configuration_manager=predictor.configuration_manager,
            properties_dict=preprocessed["data_properties"],
        )
        t2_shape = tuple(int(v) for v in nib.load(str(case.t2w_path)).shape)
        probabilities = orient_probabilities_xyz(probabilities, t2_shape)
        mask = probabilities_to_label_mask(probabilities)
        save_prediction_mask(case, mask)
        metric_rows.append(compute_case_metrics(case, mask))
        print(f"[done] {case.case_id} -> {case.output_mask_path}", flush=True)
    return sorted(metric_rows, key=lambda row: str(row["case_id"]))


def write_case_manifest(root: Path, cases: Sequence[ExternalCase]) -> Path:
    path = root / "segmoe_anatomy_best_cases.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps({key: str(value) for key, value in asdict(case).items()}, ensure_ascii=False) + "\n")
    return path


def main() -> None:
    cases, errors = discover_cases(EXTERNAL_ROOT)
    if errors:
        error_path = write_preflight_errors(EXTERNAL_ROOT, errors)
        raise SystemExit(f"Preflight failed for {len(errors)} fields. See {error_path}")
    if not cases:
        raise SystemExit(f"No valid case folders found under {EXTERNAL_ROOT}")

    model_folder = check_nnunet_environment()
    write_case_manifest(EXTERNAL_ROOT, cases)
    print(f"Found {len(cases)} cases under {EXTERNAL_ROOT}")
    print(f"Using model folder: {model_folder}")
    metric_rows = run_inference(cases, model_folder)

    metrics_path = write_metrics_csv(EXTERNAL_ROOT / METRICS_CSV, metric_rows)
    summary = summarize_metrics(metric_rows)
    summary_path = EXTERNAL_ROOT / SUMMARY_JSON
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Metrics written to {metrics_path}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
