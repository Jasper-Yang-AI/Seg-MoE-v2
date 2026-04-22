from __future__ import annotations

import numpy as np
import pytest

from segmoe_v2.roi import expand_bbox_to_min_size, reinflate_crop


def test_expand_bbox_to_min_size_clips_at_native_boundaries() -> None:
    bbox = expand_bbox_to_min_size(
        (2, 4, 10, 20, 10, 20),
        min_size_zyx=(8, 32, 32),
        shape_zyx=(6, 40, 50),
    )

    assert bbox == (0, 6, 0, 32, 0, 32)


def test_reinflate_crop_supports_spatial_and_channel_shapes() -> None:
    crop = np.ones((2, 3, 4), dtype=np.float32)
    full = reinflate_crop(crop, (1, 3, 2, 5, 3, 7), (5, 8, 9), fill_value=-1.0)
    assert full.shape == (5, 8, 9)
    assert np.all(full[1:3, 2:5, 3:7] == 1.0)
    assert full[0, 0, 0] == -1.0

    channel_crop = np.ones((3, 2, 3, 4), dtype=np.float32)
    channel_full = reinflate_crop(channel_crop, (1, 3, 2, 5, 3, 7), (5, 8, 9))
    assert channel_full.shape == (3, 5, 8, 9)


def test_reinflate_crop_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match bbox"):
        reinflate_crop(np.zeros((2, 2, 2)), (0, 3, 0, 2, 0, 2), (4, 4, 4))
