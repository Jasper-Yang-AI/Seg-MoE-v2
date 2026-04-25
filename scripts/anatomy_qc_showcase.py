#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nibabel as nib
import numpy as np


CHANNELS = ("WG", "PZ", "TZ")
CHANNEL_LABELS = ("P_WG", "P_PZ", "P_TZ")
COLORS = {
    "WG": "#4DFF73",
    "PZ": "#FFD45A",
    "TZ": "#4EC9FF",
    "lesion": "#FF4B5C",
    "fp": "#4EA5FF",
    "fn": "#FF4B5C",
    "tp": "#55FF8A",
}


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_.-]+", "_", value)
    return value.strip("_") or "case"


@dataclass(frozen=True)
class CaseQC:
    case_id: str
    category: str
    cohort_type: str
    era_bin: str
    geometry_fix_applied: bool
    prob_path: str
    output_png: str
    dice_wg: float
    dice_pz: float
    dice_tz: float
    mean_dice: float
    ref_wg_ml: float
    pred_wg_ml: float
    lesion_ml: float
    uncertain_wg_ratio: float
    hierarchy_violation_ratio: float
    slices: tuple[int, int, int]


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_prediction_path(raw_path: str | Path, prediction_manifest: str | Path) -> Path:
    path = Path(str(raw_path))
    if path.exists():
        return path
    fallback = Path(prediction_manifest).parent / PureWindowsPath(str(raw_path)).name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not resolve prediction path: {raw_path}")


def orient_probabilities_xyz(probabilities: np.ndarray, reference_shape: tuple[int, int, int]) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.ndim != 4 or probabilities.shape[0] != 3:
        raise ValueError(f"Expected probabilities with shape (3, *, *, *), got {probabilities.shape}")
    spatial = tuple(int(v) for v in probabilities.shape[1:])
    if spatial == tuple(reference_shape):
        return probabilities

    # nnU-Net exports here are commonly (C, Z, Y, X), while nibabel arrays are (X, Y, Z).
    for permutation in ((2, 1, 0), (0, 1, 2), (1, 0, 2), (0, 2, 1), (1, 2, 0), (2, 0, 1)):
        expected = tuple(reference_shape[index] for index in permutation)
        if spatial == expected:
            inverse = tuple(permutation.index(axis) + 1 for axis in range(3))
            return probabilities.transpose((0, *inverse))
    raise ValueError(f"Cannot orient probabilities {spatial} to reference shape {reference_shape}")


def reference_masks(label: np.ndarray) -> dict[str, np.ndarray]:
    label = np.asarray(label)
    return {
        "WG": np.isin(label, (1, 2, 3)),
        "PZ": label == 1,
        "TZ": label == 2,
        "lesion": label == 3,
    }


def dice_score(prediction: np.ndarray, reference: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    denominator = int(prediction.sum() + reference.sum())
    if denominator == 0:
        return float("nan")
    return float(2.0 * np.logical_and(prediction, reference).sum() / denominator)


def safe_nanmean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def normalize_display(image_2d: np.ndarray) -> np.ndarray:
    image_2d = np.asarray(image_2d, dtype=np.float32)
    finite = image_2d[np.isfinite(image_2d)]
    if finite.size == 0:
        return np.zeros_like(image_2d, dtype=np.float32)
    lo, hi = np.percentile(finite, (1.0, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros_like(image_2d, dtype=np.float32)
    return np.clip((image_2d - lo) / (hi - lo), 0.0, 1.0)


def display(image_2d: np.ndarray) -> np.ndarray:
    return np.rot90(np.asarray(image_2d))


def choose_slices(label: np.ndarray) -> tuple[int, int, int]:
    masks = reference_masks(label)
    wg_counts = masks["WG"].sum(axis=(0, 1))
    active = np.where(wg_counts > 0)[0]
    if active.size == 0:
        mid = int(label.shape[2] // 2)
        return (
            max(0, mid - 1),
            mid,
            min(label.shape[2] - 1, mid + 1),
        )

    lesion_counts = masks["lesion"].sum(axis=(0, 1))
    low = int(np.quantile(active, 0.18))
    high = int(np.quantile(active, 0.82))
    middle = int(np.argmax(lesion_counts)) if lesion_counts.max() > 0 else int(np.quantile(active, 0.50))
    slices = [low, middle, high]

    # Avoid duplicate columns in short glands by nudging repeated slice indices.
    used: set[int] = set()
    fixed: list[int] = []
    for value in slices:
        candidate = int(value)
        for delta in (0, -1, 1, -2, 2, -3, 3):
            shifted = min(max(candidate + delta, int(active[0])), int(active[-1]))
            if shifted not in used:
                candidate = shifted
                break
        used.add(candidate)
        fixed.append(candidate)
    return tuple(sorted(fixed))


def probability_overlay(base_gray: np.ndarray, probs: np.ndarray) -> np.ndarray:
    base = np.repeat(base_gray[..., None], 3, axis=-1) * 0.72
    color_arrays = np.asarray(
        [
            (0.20, 1.00, 0.35),
            (1.00, 0.78, 0.12),
            (0.12, 0.74, 1.00),
        ],
        dtype=np.float32,
    )
    probs = np.clip(np.asarray(probs, dtype=np.float32), 0.0, 1.0)
    color = np.tensordot(probs, color_arrays, axes=(0, 0))
    alpha = np.clip(np.max(probs, axis=0), 0.0, 1.0)[..., None] * 0.62
    return np.clip(base * (1.0 - alpha) + color * alpha, 0.0, 1.0)


def wg_error_overlay(base_gray: np.ndarray, pred_wg: np.ndarray, ref_wg: np.ndarray) -> np.ndarray:
    rgb = np.repeat(base_gray[..., None], 3, axis=-1) * 0.55
    pred_wg = np.asarray(pred_wg, dtype=bool)
    ref_wg = np.asarray(ref_wg, dtype=bool)
    tp = pred_wg & ref_wg
    fp = pred_wg & ~ref_wg
    fn = ~pred_wg & ref_wg
    rgb[tp] = np.asarray((0.20, 1.00, 0.45), dtype=np.float32)
    rgb[fp] = np.asarray((0.10, 0.52, 1.00), dtype=np.float32)
    rgb[fn] = np.asarray((1.00, 0.10, 0.18), dtype=np.float32)
    return np.clip(rgb, 0.0, 1.0)


def add_contour(ax: plt.Axes, mask: np.ndarray, *, color: str, linewidth: float = 1.25, linestyle: str = "-") -> None:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any() or mask.all():
        return
    ax.contour(display(mask.astype(np.float32)), levels=[0.5], colors=[color], linewidths=linewidth, linestyles=linestyle)


def load_case_arrays(row: dict[str, Any], record: dict[str, Any], prediction_manifest: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path, tuple[float, float, float]]:
    t2_image = nib.load(str(row["t2w_path"]))
    label_image = nib.load(str(row["label_path"]))
    t2 = np.asanyarray(t2_image.dataobj).astype(np.float32)
    label = np.asanyarray(label_image.dataobj).astype(np.int16)
    prob_path = resolve_prediction_path(record["prob_path"], prediction_manifest)
    payload = np.load(str(prob_path), allow_pickle=True)
    probabilities = orient_probabilities_xyz(np.asarray(payload["probabilities"], dtype=np.float32), tuple(label.shape))
    return t2, label, probabilities, prob_path, tuple(float(v) for v in label_image.header.get_zooms()[:3])


def compute_case_summary(
    row: dict[str, Any],
    record: dict[str, Any],
    prediction_manifest: str | Path,
    *,
    category: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, Path, tuple[float, float, float]]:
    t2, label, probabilities, prob_path, spacing = load_case_arrays(row, record, prediction_manifest)
    masks = reference_masks(label)
    hard = probabilities > 0.5
    dice = {
        "WG": dice_score(hard[0], masks["WG"]),
        "PZ": dice_score(hard[1], masks["PZ"]),
        "TZ": dice_score(hard[2], masks["TZ"]),
    }
    voxel_ml = float(np.prod(spacing) / 1000.0)
    wg_ref_voxels = int(masks["WG"].sum())
    uncertainty_mask = (probabilities[0] > 0.35) & (probabilities[0] < 0.65) & masks["WG"]
    hierarchy_violation = (
        (probabilities[1] > probabilities[0] + 1e-4)
        | (probabilities[2] > probabilities[0] + 1e-4)
        | ((probabilities[1] + probabilities[2]) > probabilities[0] + 1e-4)
    )
    summary = {
        "case_id": str(row["case_id"]),
        "category": category,
        "cohort_type": str(row.get("cohort_type", "")),
        "era_bin": str(row.get("era_bin", "")),
        "geometry_fix_applied": bool(row.get("metadata", {}).get("geometry_fix_applied", False)),
        "prob_path": str(prob_path),
        "dice": dice,
        "mean_dice": safe_nanmean(dice.values()),
        "ref_wg_ml": float(wg_ref_voxels * voxel_ml),
        "pred_wg_ml": float(hard[0].sum() * voxel_ml),
        "lesion_ml": float(masks["lesion"].sum() * voxel_ml),
        "uncertain_wg_ratio": float(uncertainty_mask.sum() / max(1, wg_ref_voxels)),
        "hierarchy_violation_ratio": float(hierarchy_violation.sum() / max(1, hierarchy_violation.size)),
    }
    return summary, t2, label, probabilities, prob_path, spacing


def render_case_board(
    *,
    row: dict[str, Any],
    record: dict[str, Any],
    prediction_manifest: str | Path,
    output_dir: Path,
    category: str,
) -> CaseQC:
    summary, t2, label, probabilities, prob_path, _spacing = compute_case_summary(
        row,
        record,
        prediction_manifest,
        category=category,
    )
    masks = reference_masks(label)
    hard = probabilities > 0.5
    slices = choose_slices(label)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{slugify(category)}__{slugify(str(row['case_id']))}.png"

    fig, axes = plt.subplots(4, 3, figsize=(13.6, 13.2), dpi=170)
    fig.patch.set_facecolor("#111418")
    row_titles = ("GT contours", "Prediction contours", "Probability overlay", "WG error map")
    col_titles = ("apex/low", "lesion or mid", "base/high")

    for col, z_index in enumerate(slices):
        base = normalize_display(t2[:, :, z_index])
        base_disp = display(base)
        prob_slice = np.stack([display(probabilities[index, :, :, z_index]) for index in range(3)], axis=0)
        hard_slice = {name: hard[index, :, :, z_index] for index, name in enumerate(CHANNELS)}
        ref_slice = {name: masks[name][:, :, z_index] for name in (*CHANNELS, "lesion")}

        for row_index in range(4):
            ax = axes[row_index, col]
            ax.set_facecolor("#05070A")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#2A3038")
                spine.set_linewidth(0.8)
            if row_index == 0:
                ax.set_title(f"{col_titles[col]}  z={z_index}", color="#E7EDF5", fontsize=10, pad=6)
            if col == 0:
                ax.set_ylabel(row_titles[row_index], color="#E7EDF5", fontsize=10, labelpad=8)

        axes[0, col].imshow(base_disp, cmap="gray", vmin=0, vmax=1)
        for name in CHANNELS:
            add_contour(axes[0, col], ref_slice[name], color=COLORS[name], linewidth=1.25)
        add_contour(axes[0, col], ref_slice["lesion"], color=COLORS["lesion"], linewidth=1.45)

        axes[1, col].imshow(base_disp, cmap="gray", vmin=0, vmax=1)
        for name in CHANNELS:
            add_contour(axes[1, col], hard_slice[name], color=COLORS[name], linewidth=1.35)
        add_contour(axes[1, col], ref_slice["lesion"], color=COLORS["lesion"], linewidth=1.25, linestyle="--")

        axes[2, col].imshow(probability_overlay(base_disp, prob_slice), vmin=0, vmax=1)
        add_contour(axes[2, col], ref_slice["lesion"], color=COLORS["lesion"], linewidth=1.25)

        axes[3, col].imshow(wg_error_overlay(base_disp, display(hard_slice["WG"]), display(ref_slice["WG"])), vmin=0, vmax=1)
        add_contour(axes[3, col], ref_slice["lesion"], color=COLORS["lesion"], linewidth=1.25)

    dice = summary["dice"]
    fig.suptitle(
        (
            f"{category}: {row['case_id']} | {row.get('cohort_type', '').upper()} {row.get('era_bin', '')} | "
            f"mean Dice {summary['mean_dice']:.3f}"
        ),
        color="#F5F7FA",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    subtitle = (
        f"Dice: WG {dice['WG']:.3f}  PZ {dice['PZ']:.3f}  TZ {dice['TZ']:.3f}  |  "
        f"WG volume ref/pred {summary['ref_wg_ml']:.1f}/{summary['pred_wg_ml']:.1f} ml  |  "
        f"lesion label-3 {summary['lesion_ml']:.2f} ml  |  "
        f"WG uncertain voxels {100.0 * summary['uncertain_wg_ratio']:.1f}%"
    )
    fig.text(0.5, 0.952, subtitle, ha="center", va="center", color="#B9C2CF", fontsize=9.2)
    fig.text(
        0.5,
        0.928,
        "Rows compare manual anatomy contours, predicted hard contours, calibrated probability fields, and WG TP/FP/FN error.",
        ha="center",
        va="center",
        color="#8D98A8",
        fontsize=8.6,
    )

    legend_handles = [
        Line2D([0], [0], color=COLORS["WG"], lw=2.0, label="WG"),
        Line2D([0], [0], color=COLORS["PZ"], lw=2.0, label="PZ"),
        Line2D([0], [0], color=COLORS["TZ"], lw=2.0, label="TZ"),
        Line2D([0], [0], color=COLORS["lesion"], lw=2.0, label="lesion label-3"),
        Line2D([0], [0], color=COLORS["tp"], lw=5.0, label="WG TP"),
        Line2D([0], [0], color=COLORS["fp"], lw=5.0, label="WG FP"),
        Line2D([0], [0], color=COLORS["fn"], lw=5.0, label="WG FN"),
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.5, 0.014),
        fontsize=8.5,
    )
    for text in legend.get_texts():
        text.set_color("#E7EDF5")

    plt.subplots_adjust(left=0.055, right=0.985, top=0.895, bottom=0.065, wspace=0.035, hspace=0.095)
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    return CaseQC(
        case_id=str(row["case_id"]),
        category=category,
        cohort_type=str(row.get("cohort_type", "")),
        era_bin=str(row.get("era_bin", "")),
        geometry_fix_applied=bool(row.get("metadata", {}).get("geometry_fix_applied", False)),
        prob_path=str(prob_path),
        output_png=str(out_path),
        dice_wg=float(dice["WG"]),
        dice_pz=float(dice["PZ"]),
        dice_tz=float(dice["TZ"]),
        mean_dice=float(summary["mean_dice"]),
        ref_wg_ml=float(summary["ref_wg_ml"]),
        pred_wg_ml=float(summary["pred_wg_ml"]),
        lesion_ml=float(summary["lesion_ml"]),
        uncertain_wg_ratio=float(summary["uncertain_wg_ratio"]),
        hierarchy_violation_ratio=float(summary["hierarchy_violation_ratio"]),
        slices=tuple(int(v) for v in slices),
    )


def evenly_spaced(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(items) <= count:
        return items
    indices = np.linspace(0, len(items) - 1, num=count, dtype=int)
    return [items[int(index)] for index in indices]


def select_cases(
    *,
    rows_by_case: dict[str, dict[str, Any]],
    records: list[dict[str, Any]],
    prediction_manifest: str | Path,
    candidate_per_group: int,
) -> list[tuple[str, dict[str, Any]]]:
    available = [record for record in records if record["case_id"] in rows_by_case]
    by_case = {str(record["case_id"]): record for record in available}

    pools = {
        "PCA representative": [by_case[case_id] for case_id in sorted(by_case) if rows_by_case[case_id].get("cohort_type") == "pca"],
        "NCA mimic representative": [by_case[case_id] for case_id in sorted(by_case) if rows_by_case[case_id].get("cohort_type") == "nca"],
        "Geometry-fixed check": [
            by_case[case_id]
            for case_id in sorted(by_case)
            if bool(rows_by_case[case_id].get("metadata", {}).get("geometry_fix_applied", False))
        ],
    }
    candidate_records: list[tuple[str, dict[str, Any]]] = []
    for category, pool in pools.items():
        candidate_records.extend((category, record) for record in evenly_spaced(pool, int(candidate_per_group)))

    scored: list[tuple[str, dict[str, Any], float]] = []
    for category, record in candidate_records:
        try:
            summary, *_ = compute_case_summary(
                rows_by_case[str(record["case_id"])],
                record,
                prediction_manifest,
                category=category,
            )
        except Exception as exc:
            print(f"Skipping {record.get('case_id')}: {exc}")
            continue
        scored.append((category, record, float(summary["mean_dice"])))

    selected: list[tuple[str, dict[str, Any]]] = []
    used: set[str] = set()
    for category in pools:
        group = sorted((item for item in scored if item[0] == category), key=lambda item: item[2])
        if not group:
            continue
        _, record, _score = group[len(group) // 2]
        selected.append((category, record))
        used.add(str(record["case_id"]))

    stress_pool = sorted((item for item in scored if str(item[1]["case_id"]) not in used), key=lambda item: item[2])
    if stress_pool:
        _category, record, _score = stress_pool[0]
        selected.append(("Stress test: low Dice in sample", record))

    return selected


def write_gallery(output_dir: Path, cases: list[CaseQC]) -> Path:
    cards = []
    for case in cases:
        rel = Path(case.output_png).name
        cards.append(
            f"""
            <article class="card">
              <img src="{html.escape(rel)}" alt="{html.escape(case.case_id)}">
              <h2>{html.escape(case.category)}</h2>
              <p><strong>{html.escape(case.case_id)}</strong> | {html.escape(case.cohort_type.upper())} {html.escape(case.era_bin)}</p>
              <p>Mean Dice {case.mean_dice:.3f} | WG {case.dice_wg:.3f} | PZ {case.dice_pz:.3f} | TZ {case.dice_tz:.3f}</p>
              <p>WG ref/pred {case.ref_wg_ml:.1f}/{case.pred_wg_ml:.1f} ml | lesion {case.lesion_ml:.2f} ml | slices {case.slices}</p>
            </article>
            """
        )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Anatomy QC Showcase</title>
  <style>
    body {{ margin: 0; background: #0f1217; color: #ecf2f8; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    header {{ padding: 28px 32px 8px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .sub {{ color: #96a3b4; margin: 0; max-width: 980px; line-height: 1.5; }}
    main {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 22px; padding: 24px 32px 40px; }}
    .card {{ background: #171c24; border: 1px solid #29313d; border-radius: 18px; padding: 14px; box-shadow: 0 18px 50px rgba(0,0,0,.35); }}
    .card img {{ width: 100%; display: block; border-radius: 12px; background: #05070a; }}
    .card h2 {{ margin: 14px 4px 8px; font-size: 18px; }}
    .card p {{ margin: 6px 4px; color: #b8c2d0; }}
  </style>
</head>
<body>
  <header>
    <h1>Anatomy QC Showcase</h1>
    <p class="sub">Each board uses three anatomy-relevant axial slices and four rows: manual contours, predicted contours, probability overlay, and WG TP/FP/FN error. Red contour marks lesion label 3.</p>
  </header>
  <main>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    out = output_dir / "index.html"
    out.write_text(html_text, encoding="utf-8")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create presentation-grade anatomy QC boards.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prediction-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-per-group", type=int, default=18)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    rows = load_jsonl(args.manifest)
    records = load_jsonl(args.prediction_manifest)
    rows_by_case = {str(row["case_id"]): row for row in rows}

    selected = select_cases(
        rows_by_case=rows_by_case,
        records=records,
        prediction_manifest=args.prediction_manifest,
        candidate_per_group=int(args.candidate_per_group),
    )
    rendered = [
        render_case_board(
            row=rows_by_case[str(record["case_id"])],
            record=record,
            prediction_manifest=args.prediction_manifest,
            output_dir=output_dir,
            category=category,
        )
        for category, record in selected
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "qc_selection.json"
    summary_path.write_text(json.dumps([asdict(item) for item in rendered], ensure_ascii=False, indent=2), encoding="utf-8")
    gallery_path = write_gallery(output_dir, rendered)
    print(json.dumps({"output_dir": str(output_dir), "cases": len(rendered), "summary": str(summary_path), "gallery": str(gallery_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
