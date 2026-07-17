# Copyright (c) 2026 SaferAI. GEPA forecaster-optimisation experiment.
"""Bin-stratified gate minibatch sampler.

Each GEPA iteration resamples `n_tasks_per_bin` target tasks per FST bin from
the train pool and expands them across the search-model panel, yielding the
gate/reflection cells (default 5 bins x 1 task x 4 models = 20 cells).
Reshuffled every iteration; deterministic given the seed, with RNG state
persisted through the adapter for exact resume.
"""

from __future__ import annotations

import random
from typing import Sequence

from forecaster_gepa.data import cell_key


class BinStratifiedBatchSampler:
    """Implements the GEPA `BatchSampler` protocol over cell-key ids."""

    def __init__(
        self,
        tasks_by_bin: dict[int, list[str]],
        models: Sequence[str],
        n_tasks_per_bin: int = 1,
        rng: random.Random | None = None,
    ):
        for b, tasks in tasks_by_bin.items():
            if len(tasks) < n_tasks_per_bin:
                raise ValueError(f"Bin {b}: {len(tasks)} tasks < n_tasks_per_bin={n_tasks_per_bin}")
        self.tasks_by_bin = {b: list(tasks) for b, tasks in sorted(tasks_by_bin.items())}
        self.models = list(models)
        self.n_tasks_per_bin = n_tasks_per_bin
        self.rng = rng if rng is not None else random.Random(0)

    def next_minibatch_ids(self, loader, state) -> list[str]:
        ids: list[str] = []
        for _b, tasks in self.tasks_by_bin.items():
            for task_id in self.rng.sample(tasks, self.n_tasks_per_bin):
                ids.extend(cell_key(task_id, m) for m in self.models)
        known = set(loader.all_ids())
        missing = [i for i in ids if i not in known]
        assert not missing, f"Sampled cell ids missing from the train loader: {missing}"
        return ids
