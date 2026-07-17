# Copyright (c) 2026 SaferAI. GEPA forecaster-optimisation experiment.
"""Parent selection and stop-condition helpers.

Native per-instance Pareto selection is the default
(`n_cells_won_needed_for_pareto_frontier = 1`). Raising the threshold keeps
the native machinery but restricts the sampleable frontier to candidates
winning at least k val cells (Jeff's k-of-n stability suggestion) — a config
knob for the follow-up run, not used by the diagnostic run.
"""

from __future__ import annotations

import random

from gepa.core.state import GEPAState
from gepa.gepa_utils import idxmax, select_program_candidate_from_pareto_front
from gepa.strategies.candidate_selector import ParetoCandidateSelector


def make_candidate_selector(n_cells_won_needed: int, rng: random.Random):
    if n_cells_won_needed <= 1:
        return ParetoCandidateSelector(rng=rng)
    return MinCellsWonParetoSelector(k=n_cells_won_needed, rng=rng)


class MinCellsWonParetoSelector:
    """Pareto selection restricted to candidates winning >= k val cells."""

    def __init__(self, k: int, rng: random.Random):
        assert k >= 1
        self.k = k
        self.rng = rng

    def select_candidate_idx(self, state: GEPAState) -> int:
        mapping = state.get_pareto_front_mapping()
        freq: dict[int, int] = {}
        for front in mapping.values():
            for prog in front:
                freq[prog] = freq.get(prog, 0) + 1
        eligible = {prog for prog, cells in freq.items() if cells >= self.k}
        if not eligible:
            return idxmax(state.per_program_tracked_scores)
        filtered = {key: front & eligible for key, front in mapping.items() if front & eligible}
        return select_program_candidate_from_pareto_front(filtered, state.per_program_tracked_scores, self.rng)


class MaxIterationsStopper:
    """Stop after exactly `max_iterations` GEPA iterations (candidates proposed)."""

    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations

    def __call__(self, gepa_state: GEPAState) -> bool:
        return gepa_state.i + 1 >= self.max_iterations
