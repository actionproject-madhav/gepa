# Copyright (c) 2026 SaferAI. GEPA forecaster-optimisation experiment.
"""Acceptance-gate criteria for the child-vs-parent minibatch comparison.

Native GEPA's acceptance gate (`gepa.strategies.acceptance.StrictImprovementAcceptance`)
accepts a child iff `sum(child_scores) > sum(parent_scores)` over the gate
minibatch — a single **aggregate** comparison. As our pilot run showed, this
lets one lucky/unlucky cell (a confidently-wrong-but-outcome-favoured
forecast can swing a Brier sum a lot) dominate the accept/reject decision:
only ~40-50% of accepted children also improved on the held-out 84-cell val
set, i.e. the 20-cell gate was close to a coin flip.

This module adds a **task-win** criterion (Jeff's suggestion from the design
discussion): require the child to beat the parent on at least
`min_task_wins` of the individual gate cells (a "broad, distributed
improvement" requirement, more robust to one cell dominating the aggregate
sum), optionally ANDed with the aggregate-sum condition (with an adjustable
margin `tau`, so "improvement" can require more than a razor-thin sum
advantage).

Per-cell alignment: `subsample_scores_before[i]` and `subsample_scores_after[i]`
are both evaluations of the SAME minibatch cell id at index `i` (GEPA
evaluates parent and child on identical `task.minibatch_ids`), so pairing by
index is valid — see `ReflectiveMutationProposer.propose` in
`gepa/proposer/reflective_mutation/reflective_mutation.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from gepa.core.state import GEPAState
    from gepa.proposer.base import CandidateProposal

# Valid values for `ForecasterGEPAConfig.acceptance_criterion`.
ACCEPTANCE_CRITERIA = ("aggregate_sum", "min_task_wins", "aggregate_sum_and_min_task_wins")


@dataclass
class TaskWinAcceptance:
    """Accept a child based on per-cell win count, optionally ANDed with the
    aggregate-sum criterion.

    Args:
        min_task_wins: child must strictly beat the parent's score on at
            least this many of the gate minibatch cells.
        require_aggregate_improvement: if True, ALSO require
            `sum(after) - sum(before) > margin` (native GEPA's criterion,
            with an adjustable margin). If False, the task-win count is the
            only criterion (Jeff's suggestion in isolation).
        margin: tau in "sum(after) - sum(before) > tau"; 0.0 reproduces
            native `StrictImprovementAcceptance`'s strict-inequality
            behaviour exactly. Only used when
            `require_aggregate_improvement=True`.
    """

    min_task_wins: int
    require_aggregate_improvement: bool = True
    margin: float = 0.0

    def __post_init__(self) -> None:
        if self.min_task_wins < 1:
            raise ValueError(f"min_task_wins must be >= 1, got {self.min_task_wins}")

    def _stats(self, proposal: CandidateProposal) -> tuple[int, int, float, float]:
        before = proposal.subsample_scores_before or []
        after = proposal.subsample_scores_after or []
        assert len(before) == len(after), (
            f"subsample_scores_before ({len(before)}) and _after ({len(after)}) length mismatch "
            "-- expected parent and child to be evaluated on the same minibatch cells"
        )
        wins = sum(1 for b, a in zip(before, after, strict=True) if a > b)
        return wins, len(before), sum(before), sum(after)

    def should_accept(self, proposal: CandidateProposal, state: GEPAState) -> bool:
        wins, n, old_sum, new_sum = self._stats(proposal)
        task_win_ok = wins >= self.min_task_wins
        if not self.require_aggregate_improvement:
            return task_win_ok
        aggregate_ok = (new_sum - old_sum) > self.margin
        return task_win_ok and aggregate_ok

    def reject_reason(self, proposal: CandidateProposal, state: GEPAState) -> str:
        wins, n, old_sum, new_sum = self._stats(proposal)
        reasons = []
        if wins < self.min_task_wins:
            reasons.append(f"won only {wins}/{n} gate cells (need >= {self.min_task_wins})")
        if self.require_aggregate_improvement and not (new_sum - old_sum > self.margin):
            reasons.append(f"aggregate improvement {new_sum - old_sum:.4f} did not exceed margin {self.margin}")
        return "; ".join(reasons) if reasons else f"won {wins}/{n} cells, aggregate {old_sum:.4f} -> {new_sum:.4f}"


def make_acceptance_criterion(
    criterion: str,
    *,
    min_task_wins: int,
    margin: float,
) -> Literal["strict_improvement"] | TaskWinAcceptance:
    """Build the acceptance-gate object from the config's string knob.

    - "aggregate_sum": native GEPA behaviour (returns the string
      "strict_improvement" for `gepa.optimize` to resolve internally;
      identical to `TaskWinAcceptance(min_task_wins=1, margin=0.0)` since
      winning >=1 cell is not actually required by the aggregate-only path,
      but we keep the native object rather than reimplementing it).
    - "min_task_wins": task-win count only, ignoring the aggregate sum.
    - "aggregate_sum_and_min_task_wins": Jeff's suggestion ANDed with the
      native aggregate criterion (with margin `margin`) -- the child must
      BOTH beat the parent's aggregate Brier AND beat it on >= k individual
      cells.
    """
    if criterion == "aggregate_sum":
        return "strict_improvement"
    if criterion == "min_task_wins":
        return TaskWinAcceptance(min_task_wins=min_task_wins, require_aggregate_improvement=False)
    if criterion == "aggregate_sum_and_min_task_wins":
        return TaskWinAcceptance(min_task_wins=min_task_wins, require_aggregate_improvement=True, margin=margin)
    raise ValueError(f"Unknown acceptance_criterion {criterion!r}; choose one of {ACCEPTANCE_CRITERIA}")
