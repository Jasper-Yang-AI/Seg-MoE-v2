from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ..anatomy_visual_qc import generate_anatomy_visual_qc
from ..backend_data import export_nnformer_task, export_nnunet_task, prepare_swinunetr_data
from ..geometry_audit import (
    GeometryAuditThresholds,
    audit_geometry,
    build_geometry_summary,
    default_geometry_csv_path,
    default_geometry_summary_path,
    format_geometry_summary,
    write_geometry_audit_artifacts,
)
from ..geometry_fix import (
    GEOMETRY_FIX_RECOMMENDATIONS,
    default_geometry_fix_report_csv,
    default_geometry_fix_report_json,
    default_geometry_fix_root,
    default_geometry_fixed_manifest_path,
    fix_geometry_to_t2,
    load_geometry_audit_csv,
    write_geometry_fix_artifacts,
)
from ..manifest import (
    audit_manifest_artifacts,
    build_case_manifest,
    default_summary_path,
    format_audit_report,
    load_case_manifest,
    scan_case_roots,
    write_manifest_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SegMoE v2 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    manifest_parser = sub.add_parser("build-manifest", help="Scan case roots, build manifest, summary, and splits")
    manifest_parser.add_argument("--roots", nargs="+", required=True)
    manifest_parser.add_argument("--manifest-out", required=True)
    manifest_parser.add_argument("--summary-out", required=False)
    manifest_parser.add_argument("--nnunet-splits-out", required=True)
    manifest_parser.add_argument("--nnformer-splits-out", required=True)
    manifest_parser.add_argument("--patient-map", required=False)
    manifest_parser.add_argument("--test-ratio", type=float, default=0.15)
    manifest_parser.add_argument("--folds", type=int, default=5)
    manifest_parser.add_argument("--seed", type=int, default=42)

    audit_parser = sub.add_parser("audit-manifest", help="Audit manifest and backend splits")
    audit_parser.add_argument("--manifest", required=True)
    audit_parser.add_argument("--nnunet-splits", required=True)
    audit_parser.add_argument("--nnformer-splits", required=True)

    nnunet_export = sub.add_parser("export-nnunet-task", help="Export canonical raw dataset layout for nnUNet")
    nnunet_export.add_argument("--manifest", required=True)
    nnunet_export.add_argument("--task-root", required=True)
    nnunet_export.add_argument("--dataset-id", type=int, required=True)
    nnunet_export.add_argument("--dataset-name", required=True)
    nnunet_export.add_argument("--task", choices=("anatomy", "lesion"), default="lesion")

    nnformer_export = sub.add_parser("export-nnformer-task", help="Export canonical raw dataset layout for nnFormer")
    nnformer_export.add_argument("--manifest", required=True)
    nnformer_export.add_argument("--task-root", required=True)
    nnformer_export.add_argument("--dataset-id", type=int, required=True)
    nnformer_export.add_argument("--dataset-name", required=True)
    nnformer_export.add_argument("--task", choices=("anatomy", "lesion"), default="lesion")

    swin_prep = sub.add_parser("prepare-swinunetr-data", help="Write dataset index and split lists for SwinUNETR")
    swin_prep.add_argument("--manifest", required=True)
    swin_prep.add_argument("--output-dir", required=True)
    swin_prep.add_argument("--task", choices=("anatomy", "lesion"), default="lesion")

    geometry_audit = sub.add_parser("audit-geometry", help="Audit multimodal geometry consistency for all manifest cases")
    geometry_audit.add_argument("--manifest", required=True)
    geometry_audit.add_argument("--csv-out", required=False)
    geometry_audit.add_argument("--summary-out", required=False)
    geometry_audit.add_argument("--soft-spacing-mm", type=float, default=1e-4)
    geometry_audit.add_argument("--hard-spacing-mm", type=float, default=1e-2)
    geometry_audit.add_argument("--soft-origin-mm", type=float, default=1e-2)
    geometry_audit.add_argument("--hard-origin-mm", type=float, default=1.0)
    geometry_audit.add_argument("--soft-direction", type=float, default=1e-4)
    geometry_audit.add_argument("--hard-direction", type=float, default=1e-3)

    geometry_fix = sub.add_parser("fix-geometry-to-t2", help="Repair flagged multimodal geometry issues using T2 as reference")
    geometry_fix.add_argument("--manifest", required=True)
    geometry_fix.add_argument("--audit-csv", required=True)
    geometry_fix.add_argument("--output-root", required=False)
    geometry_fix.add_argument("--manifest-out", required=False)
    geometry_fix.add_argument("--report-csv-out", required=False)
    geometry_fix.add_argument("--report-json-out", required=False)
    geometry_fix.add_argument(
        "--include-recommendations",
        nargs="+",
        choices=GEOMETRY_FIX_RECOMMENDATIONS,
        default=list(GEOMETRY_FIX_RECOMMENDATIONS),
    )
    geometry_fix.add_argument("--soft-spacing-mm", type=float, default=1e-4)
    geometry_fix.add_argument("--hard-spacing-mm", type=float, default=1e-2)
    geometry_fix.add_argument("--soft-origin-mm", type=float, default=1e-2)
    geometry_fix.add_argument("--hard-origin-mm", type=float, default=1.0)
    geometry_fix.add_argument("--soft-direction", type=float, default=1e-4)
    geometry_fix.add_argument("--hard-direction", type=float, default=1e-3)
    geometry_fix.add_argument("--overwrite", action="store_true")

    anatomy_qc = sub.add_parser("visualize-anatomy-qc", help="Generate anatomy visual QC overlays from exported probabilities")
    anatomy_qc.add_argument("--manifest", required=True)
    anatomy_qc.add_argument("--prediction-manifest", required=True)
    anatomy_qc.add_argument("--output-dir", required=True)
    anatomy_qc.add_argument("--normal-count", type=int, default=5)
    anatomy_qc.add_argument("--lesion-count", type=int, default=3)
    anatomy_qc.add_argument("--geometry-fix-count", type=int, default=2)
    anatomy_qc.add_argument("--seed", type=int, default=42)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "build-manifest":
        manifest_out = Path(args.manifest_out)
        summary_out = Path(args.summary_out) if args.summary_out else default_summary_path(manifest_out)
        discovered = scan_case_roots(args.roots, patient_map_path=args.patient_map)
        manifest = build_case_manifest(
            discovered,
            test_ratio=float(args.test_ratio),
            n_folds=int(args.folds),
            seed=int(args.seed),
        )
        write_manifest_artifacts(
            manifest,
            manifest_path=manifest_out,
            summary_path=summary_out,
            nnunet_splits_path=Path(args.nnunet_splits_out),
            nnformer_splits_path=Path(args.nnformer_splits_out),
        )
        report = audit_manifest_artifacts(
            manifest_path=manifest_out,
            nnunet_splits_path=args.nnunet_splits_out,
            nnformer_splits_path=args.nnformer_splits_out,
        )
        print(format_audit_report(report))
        if report.has_errors:
            raise SystemExit(1)
        return

    if args.command == "audit-manifest":
        report = audit_manifest_artifacts(
            manifest_path=args.manifest,
            nnunet_splits_path=args.nnunet_splits,
            nnformer_splits_path=args.nnformer_splits,
        )
        print(format_audit_report(report))
        if report.has_errors:
            raise SystemExit(1)
        return

    rows = load_case_manifest(args.manifest)

    if args.command == "export-nnunet-task":
        outputs = export_nnunet_task(
            rows,
            task_root=args.task_root,
            dataset_id=int(args.dataset_id),
            dataset_name=args.dataset_name,
            task=args.task,
        )
        print(f"nnUNet task exported to {outputs['dataset_dir']}")
        return

    if args.command == "export-nnformer-task":
        outputs = export_nnformer_task(
            rows,
            task_root=args.task_root,
            dataset_id=int(args.dataset_id),
            dataset_name=args.dataset_name,
            task=args.task,
        )
        print(f"nnFormer task exported to {outputs['dataset_dir']}")
        return

    if args.command == "prepare-swinunetr-data":
        outputs = prepare_swinunetr_data(rows, output_dir=args.output_dir, task=args.task)
        print(f"SwinUNETR data prepared at {outputs['split_metadata'].parent}")
        return

    if args.command == "audit-geometry":
        thresholds = GeometryAuditThresholds(
            soft_spacing_mm=float(args.soft_spacing_mm),
            hard_spacing_mm=float(args.hard_spacing_mm),
            soft_origin_mm=float(args.soft_origin_mm),
            hard_origin_mm=float(args.hard_origin_mm),
            soft_direction=float(args.soft_direction),
            hard_direction=float(args.hard_direction),
        )
        csv_out = Path(args.csv_out) if args.csv_out else default_geometry_csv_path(args.manifest)
        summary_out = Path(args.summary_out) if args.summary_out else default_geometry_summary_path(args.manifest)
        results = audit_geometry(rows, thresholds=thresholds)
        write_geometry_audit_artifacts(
            results,
            csv_path=csv_out,
            summary_path=summary_out,
            thresholds=thresholds,
        )
        print(format_geometry_summary(build_geometry_summary(results, thresholds=thresholds)))
        print(f"  csv_out: {csv_out}")
        print(f"  summary_out: {summary_out}")
        return

    if args.command == "fix-geometry-to-t2":
        thresholds = GeometryAuditThresholds(
            soft_spacing_mm=float(args.soft_spacing_mm),
            hard_spacing_mm=float(args.hard_spacing_mm),
            soft_origin_mm=float(args.soft_origin_mm),
            hard_origin_mm=float(args.hard_origin_mm),
            soft_direction=float(args.soft_direction),
            hard_direction=float(args.hard_direction),
        )
        output_root = Path(args.output_root) if args.output_root else default_geometry_fix_root(args.manifest)
        manifest_out = Path(args.manifest_out) if args.manifest_out else default_geometry_fixed_manifest_path(args.manifest)
        report_csv_out = Path(args.report_csv_out) if args.report_csv_out else default_geometry_fix_report_csv(output_root)
        report_json_out = Path(args.report_json_out) if args.report_json_out else default_geometry_fix_report_json(output_root)
        geometry_rows = load_geometry_audit_csv(args.audit_csv)
        patched_rows, reports = fix_geometry_to_t2(
            rows,
            geometry_audit_rows=geometry_rows,
            output_root=output_root,
            include_recommendations=args.include_recommendations,
            thresholds=thresholds,
            overwrite=bool(args.overwrite),
        )
        outputs = write_geometry_fix_artifacts(
            patched_rows,
            reports,
            manifest_out=manifest_out,
            report_csv_out=report_csv_out,
            report_json_out=report_json_out,
            include_recommendations=args.include_recommendations,
            output_root=output_root,
        )
        print(f"Geometry fix complete. Fixed cases: {len(reports)}")
        print(f"  manifest_out: {outputs['manifest']}")
        print(f"  report_csv_out: {outputs['report_csv']}")
        print(f"  report_json_out: {outputs['report_json']}")
        return

    if args.command == "visualize-anatomy-qc":
        from ..io_utils import load_jsonl

        summary = generate_anatomy_visual_qc(
            rows,
            prediction_manifest=load_jsonl(args.prediction_manifest),
            output_dir=args.output_dir,
            normal_count=int(args.normal_count),
            lesion_count=int(args.lesion_count),
            geometry_fix_count=int(args.geometry_fix_count),
            seed=int(args.seed),
        )
        print(f"Anatomy QC overlays written to {Path(args.output_dir)}")
        print(f"  actual_counts: {summary['actual_counts']}")
        return

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
