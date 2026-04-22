from __future__ import annotations

import sys

import torch
from segmoe_v2.backend_data import resolve_vendored_backend_root

_NNUNET_SRC = resolve_vendored_backend_root("nnunet")
if str(_NNUNET_SRC) not in sys.path:
    sys.path.insert(0, str(_NNUNET_SRC))

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


class _FakeLabelManager:
    num_segmentation_heads = 4


class _FakeConfigurationManager:
    patch_size = (2, 2, 2)


class _ThreeHeadNetwork(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat(1, 3, 1, 1, 1)


def test_predictor_uses_network_output_channels_for_sliding_window_accumulator() -> None:
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=False,
        use_mirroring=False,
        perform_everything_on_device=False,
        device=torch.device("cpu"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.network = _ThreeHeadNetwork()
    predictor.configuration_manager = _FakeConfigurationManager()
    predictor.label_manager = _FakeLabelManager()
    predictor.allowed_mirroring_axes = None

    input_image = torch.ones((1, 2, 2, 2), dtype=torch.float32)
    predicted_logits = predictor.predict_sliding_window_return_logits(input_image)

    assert predicted_logits.shape == (3, 2, 2, 2)
    assert torch.allclose(predicted_logits, torch.ones_like(predicted_logits))
