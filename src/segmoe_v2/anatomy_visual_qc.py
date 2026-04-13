from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Mapping, Sequence

import nibabel as nib
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from .contracts import CaseManifestRow
from .io_utils import load_jsonl, save_json
from .manifest import load_case_manifest


def _load_prediction_manifest(path: str | Path) -> list[dict]:
    return [dict(row) for row in load_jsonl(path)]


def _normalize_grayscale(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(image, (1.0, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(image))
        hi = float(np.max(image))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _display_slice(volume: np.ndarray, slice_index: int) -> np.ndarray:
    return np.rot90(np.asarray(volume[:, :, slice_index], dtype=np.float32))


def _overlay_probability(
    base_gray: np.ndarray,
    probability: np.ndarray,
    *,
    color: tuple[float, float, float],
    alpha_scale: float = 0.6,
    lesion_outline: np.ndarray | None = None,
) -> np.ndarray:
    base_rgb = np.repeat(base_gray[..., None], 3, axis=-1).astype(np.float32)
    alpha = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)[..., None] * float(alpha_scale)
    color_rgb = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    blended = base_rgb * (1.0 - alpha) + color_rgb * alpha
    if lesion_outline is not None and lesion_outline.any():
        blended = blended.copy()
        blended[lesion_outline] = np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
    return np.clip(blended, 0.0, 1.0)


def _mask_outline(mask_2d: np.ndarray) -> np.ndarray:
    mask_2d = np.asarray(mask_2d, dtype=bool)
    return np.logical_xor(binary_dilation(mask_2d, iterations=1), mask_2d)


def _build_montage(panels: Sequence[np.ndarray], *, separator: int = 8) -> np.ndarray:
    if len(panels) != 4:
        raise ValueError(f"Expected 4 panels, got {len(panels)}")
    height, width, _ = panels[0].shape
    vertical_sep = np.ones((height, separator, 3), dtype=np.float32)
    horizontal_sep = np.ones((separator, width * 2 + separator, 3), dtype=np.float32)
    top = np.concatenate((panels[0], vertical_sep, panels[1]), axis=1)
    bottom = np.concatenate((panels[2], vertical_sep, panels[3]), axis=1)
    return np.concatenate((top, horizontal_sep, bottom), axis=0)


def _save_png(path: str | Path, image: np.ndarray) -> None:
    image_uint8 = np.clip(np.asarray(image, dtype=np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(image_uint8).save(str(path))


def _choose_slice_index(label_volume: np.ndarray, probabilities: np.ndarray) -> int:
    lesion_mask = np.asarray(label_volume == 3, dtype=np.uint8)
    if lesion_mask.any():
        lesion_scores = lesion_mask.sum(axis=(0, 1))
        return int(np.argmax(lesion_scores))
    wg_scores = np.asarray(probabilities[0], dtype=np.float32).sum(axis=(0, 1))
    return int(np.argmax(wg_scores))


def _select_cases(
    rows: Sequence[CaseManifestRow],
    *,
    available_case_ids: set[str],
    normal_count: int,
    lesion_count: int,
    geometry_fix_count: int,
    seed: int,
) -> list[tuple[str, CaseManifestRow]]:
    rng = random.Random(int(seed))
    trainval_rows = [row for row in rows if row.fixed_split == "trainval" and row.case_id in available_case_ids]

    def _sample(pool: Sequence[CaseManifestRow], count: int, used: set[str]) -> list[CaseManifestRow]:
        candidates = [row for row in pool if row.case_id not in used]
        if len(candidates) <= count:
            rng.shuffle(candidates)
            return candidates
        return rng.sample(candidates, count)

    used: set[str] = set()
    selected: list[tuple[str, CaseManifestRow]] = []

    geometry_rows = [row for row in trainval_rows if bool(row.metadata.get("geometry_fix_applied", False))]
    for row in _sample(geometry_rows, geometry_fix_count, used):
        selected.append(("geometry_fixed", row))
        used.add(row.case_id)

    lesion_rows = [row for row in trainval_rows if bool(row.has_lesion_label3)]
    for row in _sample(lesion_rows, lesion_count, used):
        selected.append(("lesion_case", row))
        used.add(row.case_id)

    normal_rows = [
        row
        for row in trainval_rows
        if (not bool(row.has_lesion_label3)) and (not bool(row.metadata.get("geometry_fix_applied", False)))
    ]
    for row in _sample(normal_rows, normal_count, used):
        selected.append(("normal_case", row))
        used.add(row.case_id)

    return selected


def _render_case_qc(
    *,
    row: CaseManifestRow,
    prediction_record: Mapping[str, object],
    category: str,
    output_dir: str | Path,
) -> dict[str, object]:
    t2 = np.asanyarray(nib.load(str(row.t2w_path)).dataobj).astype(np.float32)
    label = np.asanyarray(nib.load(str(row.label_path)).dataobj).astype(np.int16)
    payload = np.load(str(prediction_record["prob_path"]), allow_pickle=True)
    probabilities = np.asarray(payload["probabilities"], dtype=np.float32)

    slice_index = _choose_slice_index(label, probabilities)
    t2_slice = _normalize_grayscale(_display_slice(t2, slice_index))
    lesion_outline = _mask_outline(_display_slice((label == 3).astype(np.uint8), slice_index))
    wg_slice = _display_slice(probabilities[0], slice_index)
    pz_slice = _display_slice(probabilities[1], slice_index)
    tz_slice = _display_slice(probabilities[2], slice_index)

    panels = [
        np.repeat(t2_slice[..., None], 3, axis=-1),
        _overlay_probability(t2_slice, wg_slice, color=(0.1, 0.9, 0.2), lesion_outline=lesion_outline),
        _overlay_probability(t2_slice, pz_slice, color=(1.0, 0.75, 0.0), lesion_outline=lesion_outline),
        _overlay_probability(t2_slice, tz_slice, color=(0.1, 0.7, 1.0), lesion_outline=lesion_outline),
    ]
    montage = _build_montage(panels)
    case_output = Path(output_dir) / f"{category}__{row.case_id}.png"
    case_output.parent.mkdir(parents=True, exist_ok=True)
    _save_png(case_output, montage)
    return {
        "case_id": row.case_id,
        "category": category,
        "slice_index": int(slice_index),
        "output_png": str(case_output),
        "prob_path": str(prediction_record["prob_path"]),
        "has_lesion_label3": bool(row.has_lesion_label3),
        "geometry_fix_applied": bool(row.metadata.get("geometry_fix_applied", False)),
    }


def generate_anatomy_visual_qc(
    rows: Sequence[CaseManifestRow],
    *,
    prediction_manifest: Sequence[Mapping[str, object]],
    output_dir: str | Path,
    normal_count: int = 5,
    lesion_count: int = 3,
    geometry_fix_count: int = 2,
    seed: int = 42,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    row_by_case = {row.case_id: row for row in rows}
    prediction_by_case: dict[str, Mapping[str, object]] = {}
    for record in prediction_manifest:
        case_id = str(record["case_id"])
        prediction_by_case.setdefault(case_id, record)

    selected_cases = _select_cases(
        rows,
        available_case_ids=set(prediction_by_case.keys()),
        normal_count=int(normal_count),
        lesion_count=int(lesion_count),
        geometry_fix_count=int(geometry_fix_count),
        seed=int(seed),
    )

    rendered = [
        _render_case_qc(
            row=row_by_case[row.case_id],
            prediction_record=prediction_by_case[row.case_id],
            category=category,
            output_dir=output_dir,
        )
        for category, row in selected_cases
    ]

    summary = {
        "requested_counts": {
            "normal_case": int(normal_count),
            "lesion_case": int(lesion_count),
            "geometry_fixed": int(geometry_fix_count),
        },
        "actual_counts": {
            "normal_case": sum(1 for item in rendered if item["category"] == "normal_case"),
            "lesion_case": sum(1 for item in rendered if item["category"] == "lesion_case"),
            "geometry_fixed": sum(1 for item in rendered if item["category"] == "geometry_fixed"),
        },
        "cases": rendered,
    }
    save_json(summary, output_dir / "selection_summary.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate anatomy visual QC overlays from exported probability bundles")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prediction-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--normal-count", type=int, default=5)
    parser.add_argument("--lesion-count", type=int, default=3)
    parser.add_argument("--geometry-fix-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    rows = load_case_manifest(args.manifest)
    prediction_manifest = _load_prediction_manifest(args.prediction_manifest)
    summary = generate_anatomy_visual_qc(
        rows,
        prediction_manifest=prediction_manifest,
        output_dir=args.output_dir,
        normal_count=int(args.normal_count),
        lesion_count=int(args.lesion_count),
        geometry_fix_count=int(args.geometry_fix_count),
        seed=int(args.seed),
    )
    print(json.dumps(summary["actual_counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
