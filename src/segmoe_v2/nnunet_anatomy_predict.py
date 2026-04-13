from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .nnunet_anatomy import (
    ANATOMY_PROBABILITY_CHANNELS,
    convert_anatomy_logits_to_probabilities_with_correct_shape,
    write_anatomy_prediction_manifest,
    write_anatomy_probability_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SegMoE anatomy probability export for nnUNet models")
    parser.add_argument("-d", "--dataset-id", required=True)
    parser.add_argument("-i", "--input-dir", required=True)
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("-f", "--fold", nargs="+", required=True)
    parser.add_argument("-tr", "--trainer", default="nnUNetTrainerSegMoEAnatomy")
    parser.add_argument("-p", "--plans", default="nnUNetResEncUNetMPlans")
    parser.add_argument("-c", "--configuration", default="3d_fullres")
    parser.add_argument("-chk", "--checkpoint-name", default="checkpoint_final.pth")
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--save-quality-masks", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.file_path_utilities import get_output_folder

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    model_folder = get_output_folder(args.dataset_id, args.trainer, args.plans, args.configuration)
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=tuple(int(v) for v in args.fold),
        checkpoint_name=args.checkpoint_name,
    )

    input_lists, output_truncated, seg_prev = predictor._manage_input_and_output_lists(
        args.input_dir,
        str(output_dir),
        overwrite=True,
        save_probabilities=True,
    )
    data_iterator = predictor._internal_get_data_iterator_from_lists_of_filenames(
        input_lists,
        seg_prev,
        output_truncated,
        num_processes=2,
    )

    manifest_records = []
    for preprocessed in data_iterator:
        data = preprocessed["data"]
        if isinstance(data, str):
            npy_path = data
            data = torch.from_numpy(np.load(npy_path))
            Path(npy_path).unlink(missing_ok=True)
        ofile = Path(preprocessed["ofile"])
        probabilities = convert_anatomy_logits_to_probabilities_with_correct_shape(
            predictor.predict_logits_from_preprocessed_data(data).cpu(),
            plans_manager=predictor.plans_manager,
            configuration_manager=predictor.configuration_manager,
            properties_dict=preprocessed["data_properties"],
        )
        prob_path = write_anatomy_probability_bundle(
            probabilities=probabilities,
            output_file_truncated=ofile,
            channel_names=ANATOMY_PROBABILITY_CHANNELS,
            properties_dict=preprocessed["data_properties"],
            save_quality_masks=bool(args.save_quality_masks),
        )
        manifest_records.append(
            {
                "case_id": ofile.name,
                "fold": [int(v) for v in args.fold],
                "split": str(args.split_name),
                "channel_names": list(ANATOMY_PROBABILITY_CHANNELS),
                "prob_path": str(prob_path),
                "source_manifest_hash": str(predictor.dataset_json.get("segmoe_source_manifest_hash", "")),
                "hierarchy_consistency_applied": True,
            }
        )

    write_anatomy_prediction_manifest(
        manifest_records,
        output_path=output_dir / "prediction_manifest.jsonl",
    )


if __name__ == "__main__":
    main()
