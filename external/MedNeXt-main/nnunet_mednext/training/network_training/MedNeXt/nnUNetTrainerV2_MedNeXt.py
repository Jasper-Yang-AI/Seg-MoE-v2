import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt as MedNeXt_Orig
from nnunet_mednext.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet_mednext.training.network_training.nnUNetTrainerV2_DDP import nnUNetTrainerV2_DDP
from nnunet_mednext.network_architecture.neural_network import SegmentationNetwork
from nnunet_mednext.utilities.nd_softmax import softmax_helper


class MedNeXt(MedNeXt_Orig, SegmentationNetwork):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Segmentation Network Params. Needed for the nnUNet evaluation pipeline
        self.conv_op = nn.Conv3d
        self.inference_apply_nonlin = softmax_helper
        self.input_shape_must_be_divisible_by = 2**5
        self.num_classes = kwargs['n_classes']
        # self.do_ds = False        Already added this in the main class


class nnUNetTrainerV2_Optim_and_LR(nnUNetTrainerV2):

    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.initial_lr = 1e-3

    def process_plans(self, plans):
        super().process_plans(plans)
        # Please don't do this for nnunet. This is only for MedNeXt for all the DS to be used
        num_of_outputs_in_mednext = 5
        self.net_num_pool_op_kernel_sizes = [[2,2,2] for i in range(num_of_outputs_in_mednext+1)]    
    
    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(self.network.parameters(), 
                                            self.initial_lr, 
                                            weight_decay=self.weight_decay,
                                            eps=1e-4        # 1e-8 might cause nans in fp16
                                        )
        self.lr_scheduler = None


class nnUNetTrainerV2_DDP_Optim_and_LR(nnUNetTrainerV2_DDP):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 1e-3

    def process_plans(self, plans):
        super().process_plans(plans)
        num_of_outputs_in_mednext = 5
        self.net_num_pool_op_kernel_sizes = [[2,2,2] for i in range(num_of_outputs_in_mednext+1)]

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(self.network.parameters(),
                                            self.initial_lr,
                                            weight_decay=self.weight_decay,
                                            eps=1e-4
                                        )
        self.lr_scheduler = None


class nnUNetTrainerV2_MedNeXt_S_kernel3(nnUNetTrainerV2_Optim_and_LR):   
    
    def initialize_network(self):
        self.network = MedNeXt(
            in_channels = self.num_input_channels, 
            n_channels = 32,
            n_classes = self.num_classes, 
            exp_r=2                 ,         # Expansion ratio as in Swin Transformers
            kernel_size=3,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            block_counts = [2,2,2,2,2,2,2,2,2]
        )

        if torch.cuda.is_available():
            self.network.cuda()


class nnUNetTrainerV2_DDP_MedNeXt_S_kernel3(nnUNetTrainerV2_DDP_Optim_and_LR):

    def initialize_network(self):
        self.network = MedNeXt(
            in_channels = self.num_input_channels,
            n_channels = 32,
            n_classes = self.num_classes,
            exp_r=2,
            kernel_size=3,
            deep_supervision=True,
            do_res=True,
            do_res_up_down = True,
            block_counts = [2,2,2,2,2,2,2,2,2]
        )

        if torch.cuda.is_available():
            self.network.cuda()


class Layer1SourceAwareCELoss(nn.Module):
    """Candidate-first Layer1 loss for source labels 0/1/2.

    MedNeXt keeps a 2-channel softmax head, but the exported labels preserve
    source semantics: 1=PCA lesion and 2=NCA mimic are both candidate-positive.
    """

    def __init__(self, *, background_weight=1.0, source_positive_weights=None, smooth=1e-5):
        super().__init__()
        self.background_weight = float(background_weight)
        self.source_positive_weights = dict(source_positive_weights or {1: 1.0, 2: 1.5})
        self.smooth = float(smooth)

    @staticmethod
    def _squeeze_target(target):
        if target.ndim >= 4 and target.shape[1] == 1:
            return target[:, 0]
        return target

    def _binary_target_and_weights(self, raw_target, device):
        target = self._squeeze_target(raw_target).long().to(device)
        binary_target = ((target == 1) | (target == 2)).long()
        weights = torch.full(binary_target.shape, self.background_weight, dtype=torch.float32, device=device)
        for label_value, weight in self.source_positive_weights.items():
            weights = torch.where(target == int(label_value), torch.as_tensor(float(weight), device=device), weights)
        return binary_target, weights

    def forward(self, net_output, target):
        if net_output.shape[1] != 2:
            raise ValueError(f"Layer1 MedNeXt expects a 2-channel softmax head, got {tuple(net_output.shape)}")
        binary_target, weights = self._binary_target_and_weights(target, net_output.device)
        ce = F.cross_entropy(net_output, binary_target, reduction="none")
        ce = (ce * weights).sum() / weights.sum().clamp_min(1.0)

        probabilities = softmax_helper(net_output)[:, 1:2]
        target_float = binary_target[:, None].float()
        weights = weights[:, None]
        spatial_axes = tuple(range(2, probabilities.ndim))
        intersection = (probabilities * target_float * weights).sum(dim=spatial_axes)
        denominator = ((probabilities + target_float) * weights).sum(dim=spatial_axes)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return ce + (1.0 - dice).mean()


class _Layer1SourceAwareMedNeXtMixin:
    source_positive_weights = {1: 1.0, 2: 1.5}
    filtered_validation_min_dice = 0.30

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = Layer1SourceAwareCELoss(
            background_weight=1.0,
            source_positive_weights=self.source_positive_weights,
        )
        self.pin_memory = False

    def process_plans(self, plans):
        super().process_plans(plans)
        self.num_classes = 2
        self.classes = [1]

    @staticmethod
    def _target_positive(target, device):
        if target.ndim >= 4 and target.shape[1] == 1:
            target = target[:, 0]
        target = target.long().to(device)
        return (target == 1) | (target == 2)

    def run_online_evaluation(self, output, target):
        with torch.no_grad():
            target = target[0] if isinstance(target, (tuple, list)) else target
            output = output[0] if isinstance(output, (tuple, list)) else output
            output_seg = softmax_helper(output).argmax(1)
            target_positive = self._target_positive(target, output_seg.device)
            output_positive = output_seg == 1
            axes = tuple(range(1, target_positive.ndim))
            tp_hard = (output_positive & target_positive).float().sum(dim=axes).sum()
            fp_hard = (output_positive & (~target_positive)).float().sum(dim=axes).sum()
            fn_hard = ((~output_positive) & target_positive).float().sum(dim=axes).sum()

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(tp_hard, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(fp_hard, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(fn_hard, op=torch.distributed.ReduceOp.SUM)

            tp_hard = float(tp_hard.detach().cpu().item())
            fp_hard = float(fp_hard.detach().cpu().item())
            fn_hard = float(fn_hard.detach().cpu().item())
            self.online_eval_foreground_dc.append([float((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8))])
            self.online_eval_tp.append([tp_hard])
            self.online_eval_fp.append([fp_hard])
            self.online_eval_fn.append([fn_hard])

    def compute_loss(self, output, target):
        outputs = output if isinstance(output, (tuple, list)) else [output]
        targets = target if isinstance(target, (tuple, list)) else [target] * len(outputs)
        weights = self.ds_loss_weights if self.ds_loss_weights is not None else np.ones(len(outputs), dtype=np.float32)
        total_loss = None
        for i, net_output in enumerate(outputs):
            weight = float(weights[i]) if i < len(weights) else 0.0
            if weight == 0.0:
                zero_loss = net_output.sum() * 0.0
                total_loss = zero_loss if total_loss is None else total_loss + zero_loss
                continue
            loss = weight * self.loss(net_output, targets[i])
            total_loss = loss if total_loss is None else total_loss + loss
        return total_loss if total_loss is not None else outputs[0].sum() * 0.0

    @staticmethod
    def _write_binary_reference(source_path, destination_path):
        image = nib.load(str(source_path))
        data = (np.asanyarray(image.dataobj) > 0).astype(np.uint8)
        destination_path = Path(destination_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(data, affine=image.affine, header=image.header.copy()), str(destination_path))

    def _binary_gt_niftis_folder(self):
        source_folder = Path(self.gt_niftis_folder)
        destination_folder = Path(self.output_folder_base) / "gt_segmentations_layer1_binary"
        destination_folder.mkdir(parents=True, exist_ok=True)
        for source_path in source_folder.glob("*.nii.gz"):
            destination_path = destination_folder / source_path.name
            if not destination_path.exists() or source_path.stat().st_mtime > destination_path.stat().st_mtime:
                self._write_binary_reference(source_path, destination_path)
        return str(destination_folder)

    @staticmethod
    def _case_id_from_summary_path(value):
        name = Path(str(value)).name
        return name[:-7] if name.endswith(".nii.gz") else Path(name).stem

    def _add_filtered_validation_summary(self, summary_path):
        summary_path = Path(summary_path)
        if not summary_path.exists():
            return
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = list(summary.get("results", {}).get("all", []))
        min_dice = float(os.environ.get("SEGMOE_LAYER1_FILTER_MIN_DICE", self.filtered_validation_min_dice))
        filtered_rows = [
            row
            for row in rows
            if np.isfinite(float(row.get("1", {}).get("Dice", float("nan"))))
            and float(row.get("1", {}).get("Dice", float("nan"))) >= min_dice
        ]
        filtered_mean_dice = (
            float(np.nanmean([row["1"]["Dice"] for row in filtered_rows])) if filtered_rows else float("nan")
        )
        filtered_excluded_case_ids = [
            self._case_id_from_summary_path(row.get("reference") or row.get("test") or "")
            for row in rows
            if row not in filtered_rows
        ]
        summary["filtered"] = {
            "min_dice": min_dice,
            "case_count": len(filtered_rows),
            "excluded_case_count": len(filtered_excluded_case_ids),
            "excluded_case_ids": filtered_excluded_case_ids,
            "mean": {"candidate": {"Dice": filtered_mean_dice}},
        }
        summary_path.write_text(json.dumps(summary, indent=4, sort_keys=True), encoding="utf-8")
        self.print_to_log_file(
            "Filtered Layer1 Candidate Dice:",
            np.round(filtered_mean_dice, decimals=4),
            f"kept {len(filtered_rows)}/{len(rows)} cases",
        )

    @staticmethod
    def _distributed_rank():
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0

    def validate(self, *args, **kwargs):
        original_gt_folder = self.gt_niftis_folder
        binary_gt_folder = str(Path(self.output_folder_base) / "gt_segmentations_layer1_binary")
        if self._distributed_rank() == 0:
            self.gt_niftis_folder = self._binary_gt_niftis_folder()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
            self.gt_niftis_folder = binary_gt_folder
        try:
            result = super().validate(*args, **kwargs)
            validation_folder_name = args[6] if len(args) > 6 else kwargs.get("validation_folder_name", "validation_raw")
            if self._distributed_rank() == 0:
                self._add_filtered_validation_summary(Path(self.output_folder) / validation_folder_name / "summary.json")
            return result
        finally:
            self.gt_niftis_folder = original_gt_folder


class nnUNetTrainerV2_MedNeXt_S_kernel3_SegMoELayer1(
    _Layer1SourceAwareMedNeXtMixin,
    nnUNetTrainerV2_MedNeXt_S_kernel3,
):
    pass


class nnUNetTrainerV2_DDP_MedNeXt_S_kernel3_SegMoELayer1(
    _Layer1SourceAwareMedNeXtMixin,
    nnUNetTrainerV2_DDP_MedNeXt_S_kernel3,
):
    pass


class nnUNetTrainerV2_MedNeXt_B_kernel3(nnUNetTrainerV2_Optim_and_LR):   
        
    def initialize_network(self):
        self.network = MedNeXt(
            in_channels = self.num_input_channels, 
            n_channels = 32,
            n_classes = self.num_classes, 
            exp_r=[2,3,4,4,4,4,4,3,2],         # Expansion ratio as in Swin Transformers
            kernel_size=3,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            block_counts = [2,2,2,2,2,2,2,2,2]
        )

        if torch.cuda.is_available():
            self.network.cuda()


class nnUNetTrainerV2_MedNeXt_M_kernel3(nnUNetTrainerV2_Optim_and_LR):   
        
    def initialize_network(self):
        self.network = MedNeXt(
            in_channels = self.num_input_channels, 
            n_channels = 32,
            n_classes = self.num_classes, 
            exp_r=[2,3,4,4,4,4,4,3,2],         # Expansion ratio as in Swin Transformers
            kernel_size=3,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            block_counts = [3,4,4,4,4,4,4,4,3],
            checkpoint_style = 'outside_block'
        )

        if torch.cuda.is_available():
            self.network.cuda()


class nnUNetTrainerV2_MedNeXt_L_kernel3(nnUNetTrainerV2_Optim_and_LR):   
        
    def initialize_network(self):
        self.network = MedNeXt(
            in_channels = self.num_input_channels, 
            n_channels = 32,
            n_classes = self.num_classes, 
            exp_r=[3,4,8,8,8,8,8,4,3],         # Expansion ratio as in Swin Transformers
            # exp_r=[3,4,8,8,8,8,8,4,3],         # Expansion ratio as in Swin Transformers
            kernel_size=3,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            # block_counts = [6,6,6,6,4,2,2,2,2],
            block_counts = [3,4,8,8,8,8,8,4,3],
            checkpoint_style = 'outside_block'
        )

        if torch.cuda.is_available():
            self.network.cuda()


# Kernels of size 5
class nnUNetTrainerV2_MedNeXt_S_kernel5(nnUNetTrainerV2_Optim_and_LR):   

    def initialize_network(self):
        self.network = MedNeXt(
            in_channels = self.num_input_channels, 
            n_channels = 32,
            n_classes = self.num_classes, 
            exp_r=2,                           # Expansion ratio as in Swin Transformers
            kernel_size=5,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                       # Can be used to individually test residual connection
            do_res_up_down = True,
            block_counts = [2,2,2,2,2,2,2,2,2]
        )

        if torch.cuda.is_available():
            self.network.cuda()


class nnUNetTrainerV2_MedNeXt_S_kernel5_lr_1e_4(nnUNetTrainerV2_MedNeXt_S_kernel5):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 1e-4


class nnUNetTrainerV2_MedNeXt_S_kernel5_lr_25e_5(nnUNetTrainerV2_MedNeXt_S_kernel5):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 25e-5


class nnUNetTrainerV2_MedNeXt_B_kernel5(nnUNetTrainerV2_Optim_and_LR):   

    def initialize_network(self):
        self.network = MedNeXt(
            in_channels = self.num_input_channels, 
            n_channels = 32,
            n_classes = self.num_classes, 
            exp_r=[2,3,4,4,4,4,4,3,2],         # Expansion ratio as in Swin Transformers
            kernel_size=5,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            block_counts = [2,2,2,2,2,2,2,2,2]
        )

        if torch.cuda.is_available():
            self.network.cuda()


class nnUNetTrainerV2_MedNeXt_B_kernel5_lr_5e_4(nnUNetTrainerV2_MedNeXt_B_kernel5):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 5e-4


class nnUNetTrainerV2_MedNeXt_B_kernel5_lr_25e_5(nnUNetTrainerV2_MedNeXt_B_kernel5):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 25e-5


class nnUNetTrainerV2_MedNeXt_B_kernel5_lr_1e_4(nnUNetTrainerV2_MedNeXt_B_kernel5):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 1e-4


class nnUNetTrainerV2_MedNeXt_M_kernel5(nnUNetTrainerV2_Optim_and_LR):   

    def initialize_network(self):
        self.network = MedNeXt(
            in_channels = self.num_input_channels, 
            n_channels = 32,
            n_classes = self.num_classes, 
            exp_r=[2,3,4,4,4,4,4,3,2],         # Expansion ratio as in Swin Transformers
            kernel_size=5,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            block_counts = [3,4,4,4,4,4,4,4,3],
            checkpoint_style = 'outside_block'
        )

        if torch.cuda.is_available():
            self.network.cuda()


class nnUNetTrainerV2_MedNeXt_M_kernel5_lr_5e_4(nnUNetTrainerV2_MedNeXt_M_kernel5):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 5e-4


class nnUNetTrainerV2_MedNeXt_M_kernel5_lr_25e_5(nnUNetTrainerV2_MedNeXt_M_kernel5):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 25e-5


class nnUNetTrainerV2_MedNeXt_M_kernel5_lr_1e_4(nnUNetTrainerV2_MedNeXt_M_kernel5):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 1e-4


class nnUNetTrainerV2_MedNeXt_L_kernel5(nnUNetTrainerV2_Optim_and_LR):   
    
    def initialize_network(self):
        self.network = MedNeXt(
            in_channels = self.num_input_channels, 
            n_channels = 32,
            n_classes = self.num_classes, 
            exp_r=[3,4,8,8,8,8,8,4,3],         # Expansion ratio as in Swin Transformers
            kernel_size=5,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            # block_counts = [6,6,6,6,4,2,2,2,2],
            block_counts = [3,4,8,8,8,8,8,4,3],
            checkpoint_style = 'outside_block'
        )

        if torch.cuda.is_available():
            self.network.cuda()


class nnUNetTrainerV2_MedNeXt_L_kernel5_lr_5e_4(nnUNetTrainerV2_MedNeXt_L_kernel5):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 5e-4


class nnUNetTrainerV2_MedNeXt_L_kernel5_lr_25e_5(nnUNetTrainerV2_MedNeXt_L_kernel5):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 25e-5


class nnUNetTrainerV2_MedNeXt_L_kernel5_lr_1e_4(nnUNetTrainerV2_MedNeXt_L_kernel5):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 1e-4
