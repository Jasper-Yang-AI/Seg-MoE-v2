import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt as MedNeXt_Orig
from nnunet_mednext.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
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


class Layer1SourceAwareCELoss(nn.Module):
    """Candidate-first Layer1 loss for source labels 0/1/2.

    MedNeXt keeps a 2-channel softmax head, but the exported labels preserve
    source semantics: 1=PCA lesion and 2=NCA mimic are both candidate-positive.
    """

    def __init__(self, *, background_weight=1.0, source_positive_weights=None, smooth=1e-5):
        super().__init__()
        self.background_weight = float(background_weight)
        self.source_positive_weights = dict(source_positive_weights or {1: 1.25, 2: 0.75})
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
    source_positive_weights = {1: 1.25, 2: 0.75}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = Layer1SourceAwareCELoss(
            background_weight=1.0,
            source_positive_weights=self.source_positive_weights,
        )

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
            tp_hard = (output_positive & target_positive).float().sum(dim=axes).sum().detach().cpu().numpy()
            fp_hard = (output_positive & (~target_positive)).float().sum(dim=axes).sum().detach().cpu().numpy()
            fn_hard = ((~output_positive) & target_positive).float().sum(dim=axes).sum().detach().cpu().numpy()
            self.online_eval_foreground_dc.append([float((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8))])
            self.online_eval_tp.append([float(tp_hard)])
            self.online_eval_fp.append([float(fp_hard)])
            self.online_eval_fn.append([float(fn_hard)])

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

    def validate(self, *args, **kwargs):
        original_gt_folder = self.gt_niftis_folder
        self.gt_niftis_folder = self._binary_gt_niftis_folder()
        try:
            return super().validate(*args, **kwargs)
        finally:
            self.gt_niftis_folder = original_gt_folder


class nnUNetTrainerV2_MedNeXt_S_kernel3_SegMoELayer1(
    _Layer1SourceAwareMedNeXtMixin,
    nnUNetTrainerV2_MedNeXt_S_kernel3,
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
