from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Layer1CropChoice:
    cohort_type: str
    mode: str


def choose_layer1_crop_mode(cohort_type: str, rng: random.Random | None = None) -> Layer1CropChoice:
    rng = rng or random.Random()
    cohort_type = str(cohort_type).lower()
    if cohort_type == "pca":
        mode = "lesion_positive" if rng.random() < (2.0 / 3.0) else "regular_background"
    else:
        mode = "wg_or_boundary_background" if rng.random() < 0.5 else "regular_background"
    return Layer1CropChoice(cohort_type=cohort_type, mode=mode)


class Layer1BalancedBatchSampler:
    def __init__(
        self,
        cohort_types: Sequence[str],
        *,
        batch_size: int,
        pca_ratio: float = 2.0 / 3.0,
        steps_per_epoch: int | None = None,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        self.pca_ratio = float(pca_ratio)
        self.seed = int(seed)
        self.epoch = 0
        self.pca_indices = [idx for idx, cohort in enumerate(cohort_types) if str(cohort).lower() == "pca"]
        self.nca_indices = [idx for idx, cohort in enumerate(cohort_types) if str(cohort).lower() == "nca"]
        if not self.pca_indices or not self.nca_indices:
            raise ValueError("Layer1BalancedBatchSampler requires both PCA and NCA samples")
        total = len(cohort_types)
        self.steps_per_epoch = int(steps_per_epoch or max(1, math.ceil(total / self.batch_size)))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.steps_per_epoch):
            if self.batch_size == 1:
                pool = self.pca_indices if rng.random() < self.pca_ratio else self.nca_indices
                yield [rng.choice(pool)]
                continue

            n_pca = int(round(self.batch_size * self.pca_ratio))
            n_pca = min(max(n_pca, 1), self.batch_size - 1)
            n_nca = self.batch_size - n_pca
            batch = [rng.choice(self.pca_indices) for _ in range(n_pca)]
            batch.extend(rng.choice(self.nca_indices) for _ in range(n_nca))
            rng.shuffle(batch)
            yield batch
