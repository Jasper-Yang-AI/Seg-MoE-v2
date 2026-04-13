from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from torch import autocast

from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from segmoe_v2.nnunet_anatomy import (
    ANATOMY_PROBABILITY_CHANNELS,
    MaskedAnatomySegLoss,
    anatomy_tp_fp_fn,
    convert_anatomy_logits_to_probabilities_with_correct_shape,
    deep_supervision_weights,
    write_anatomy_prediction_manifest,
    write_anatomy_probability_bundle,
)


class nnUNetTrainerSegMoEAnatomy(nnUNetTrainer):
    head_channel_names = ANATOMY_PROBABILITY_CHANNELS
    head_loss_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> torch.nn.Module:
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            len(ANATOMY_PROBABILITY_CHANNELS),
            enable_deep_supervision,
        )

    def initialize(self):
        if not self.was_initialized:
            self._set_batch_size_and_oversample()
            self.num_input_channels = determine_num_input_channels(
                self.plans_manager,
                self.configuration_manager,
                self.dataset_json,
            )
            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                len(ANATOMY_PROBABILITY_CHANNELS),
                self.enable_deep_supervision,
            ).to(self.device)
            if self._do_i_compile():
                self.print_to_log_file("Using torch.compile...")
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = torch.nn.parallel.DistributedDataParallel(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized.")

    def _build_loss(self):
        return MaskedAnatomySegLoss(head_weights=self.head_loss_weights)

    def _compute_loss(self, output, target):
        if isinstance(output, (list, tuple)):
            assert isinstance(target, list), "Deep supervision expects the target to be a list of scales."
            weights = deep_supervision_weights(
                num_outputs=len(output),
                is_ddp=self.is_ddp,
                do_compile=self._do_i_compile(),
            )
            total = output[0].new_tensor(0.0)
            for weight, logits, raw_target in zip(weights, output, target):
                total = total + float(weight) * self.loss(logits, raw_target)
            return total
        return self.loss(output, target)

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [item.to(self.device, non_blocking=True) for item in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            loss = self._compute_loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [item.to(self.device, non_blocking=True) for item in target]
            target_for_metrics = target[0]
        else:
            target = target.to(self.device, non_blocking=True)
            target_for_metrics = target

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            loss = self._compute_loss(output, target)

        logits_for_metrics = output[0] if isinstance(output, (list, tuple)) else output
        tp_hard, fp_hard, fn_hard = anatomy_tp_fp_fn(logits_for_metrics, target_for_metrics)
        return {
            "loss": loss.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated["tp_hard"], 0)
        fp = np.sum(outputs_collated["fp_hard"], 0)
        fn = np.sum(outputs_collated["fn_hard"], 0)

        if self.is_ddp:
            losses_val = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(losses_val, outputs_collated["loss"])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated["loss"])

        dice_per_head = [2 * i / (2 * i + j + k) if (2 * i + j + k) > 0 else float("nan") for i, j, k in zip(tp, fp, fn)]
        mean_fg_dice = float(np.nanmean(dice_per_head))
        self.logger.log("mean_fg_dice", mean_fg_dice, self.current_epoch)
        self.logger.log("dice_per_class_or_region", dice_per_head, self.current_epoch)
        self.logger.log("val_losses", loss_here, self.current_epoch)

    def perform_actual_validation(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.manual_initialization(
            self.network,
            self.plans_manager,
            self.configuration_manager,
            None,
            self.dataset_json,
            self.__class__.__name__,
            self.inference_allowed_mirroring_axes,
        )

        validation_output_folder = join(self.output_folder, "validation")
        maybe_mkdir_p(validation_output_folder)
        _, val_keys = self.do_split()
        dataset_val = self.dataset_class(
            self.preprocessed_dataset_folder,
            val_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
        )

        manifest_records = []
        for case_id in dataset_val.identifiers:
            self.print_to_log_file(f"predicting anatomy probabilities for {case_id}")
            data, _, seg_prev, properties = dataset_val.load_case(case_id)
            data = data[:]

            if self.is_cascaded and seg_prev is not None:
                seg_prev = seg_prev[:]
                data = np.vstack(
                    (
                        data,
                        convert_labelmap_to_one_hot(seg_prev, self.label_manager.foreground_labels, output_dtype=data.dtype),
                    )
                )
            with torch.inference_mode():
                logits = predictor.predict_sliding_window_return_logits(torch.from_numpy(data)).cpu()
            if save_probabilities:
                probabilities = convert_anatomy_logits_to_probabilities_with_correct_shape(
                    logits,
                    plans_manager=self.plans_manager,
                    configuration_manager=self.configuration_manager,
                    properties_dict=properties,
                )
                output_truncated = Path(validation_output_folder) / case_id
                prob_path = write_anatomy_probability_bundle(
                    probabilities=probabilities,
                    output_file_truncated=output_truncated,
                    channel_names=self.head_channel_names,
                    properties_dict=properties,
                    save_quality_masks=False,
                )
                manifest_records.append(
                    {
                        "case_id": case_id,
                        "fold": int(self.fold),
                        "split": "val",
                        "channel_names": list(self.head_channel_names),
                        "prob_path": str(prob_path),
                        "source_manifest_hash": str(self.dataset_json.get("segmoe_source_manifest_hash", "")),
                    }
                )
        if save_probabilities and manifest_records:
            write_anatomy_prediction_manifest(
                manifest_records,
                output_path=Path(validation_output_folder) / "prediction_manifest.jsonl",
            )
            with (Path(validation_output_folder) / "prediction_summary.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "task": "anatomy",
                        "fold": int(self.fold),
                        "split": "val",
                        "channel_names": list(self.head_channel_names),
                        "case_count": len(manifest_records),
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
        self.set_deep_supervision_enabled(True)
