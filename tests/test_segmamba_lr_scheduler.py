from __future__ import annotations

import sys
from pathlib import Path

import pytest


def test_segmamba_poly_lr_scheduler_matches_current_torch_signature() -> None:
    torch = pytest.importorskip("torch")
    repo_root = Path(__file__).resolve().parents[1]
    segmamba_root = repo_root / "external" / "SegMamba"
    sys.path.insert(0, str(segmamba_root))
    try:
        from light_training.utils.lr_scheduler import PolyLRScheduler
    finally:
        sys.path.pop(0)

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = PolyLRScheduler(optimizer, initial_lr=0.01, max_steps=10)

    scheduler.step()

    assert scheduler is not None
    assert 0.0 < optimizer.param_groups[0]["lr"] <= 0.01
