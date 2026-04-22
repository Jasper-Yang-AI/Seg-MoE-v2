from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence

LAYER1_HIGH_RECALL_POLICY: dict[str, float] = {
    "pca_lesion": 0.50,
    "nca_mimic": 0.25,
    "random_gland": 0.25,
}


class Layer1HighRecallBatchSampler:
    def __init__(
        self,
        cohort_types: Sequence[str],
        *,
        batch_size: int,
        steps_per_epoch: int | None = None,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.pca_indices = [idx for idx, cohort in enumerate(cohort_types) if str(cohort).lower() == "pca"]
        self.nca_indices = [idx for idx, cohort in enumerate(cohort_types) if str(cohort).lower() == "nca"]
        self.all_indices = list(range(len(cohort_types)))
        if not self.pca_indices or not self.nca_indices:
            raise ValueError("Layer1HighRecallBatchSampler requires both PCA and NCA samples")
        total = len(cohort_types)
        self.steps_per_epoch = int(steps_per_epoch or max(1, math.ceil(total / self.batch_size)))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def _mode_counts(self) -> dict[str, int]:
        if self.batch_size == 1:
            return {"pca_lesion": 1, "nca_mimic": 0, "random_gland": 0}
        n_pca = max(1, int(round(self.batch_size * LAYER1_HIGH_RECALL_POLICY["pca_lesion"])))
        n_nca = max(1, int(round(self.batch_size * LAYER1_HIGH_RECALL_POLICY["nca_mimic"])))
        if n_pca + n_nca >= self.batch_size:
            n_nca = max(1, self.batch_size - n_pca)
        n_random = max(0, self.batch_size - n_pca - n_nca)
        return {"pca_lesion": n_pca, "nca_mimic": n_nca, "random_gland": n_random}

    def __iter__(self) -> Iterator[list[tuple[int, str]]]:
        rng = random.Random(self.seed + self.epoch)
        counts = self._mode_counts()
        for _ in range(self.steps_per_epoch):
            batch: list[tuple[int, str]] = []
            batch.extend((rng.choice(self.pca_indices), "pca_lesion") for _ in range(counts["pca_lesion"]))
            batch.extend((rng.choice(self.nca_indices), "nca_mimic") for _ in range(counts["nca_mimic"]))
            batch.extend((rng.choice(self.all_indices), "random_gland") for _ in range(counts["random_gland"]))
            rng.shuffle(batch)
            yield batch
