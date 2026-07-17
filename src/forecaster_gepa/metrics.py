# Copyright (c) 2026 SaferAI. GEPA forecaster-optimisation experiment.
"""Diagnostic metrics (spec section 6): Brier decompositions, frontier stats
and confidence-vs-wins correlations. Diagnostics only — never optimised on.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

from gepa.gepa_utils import remove_dominated_programs

# ---------------------------------------------------------------------------
# Brier decompositions (p50 vs binary outcome)
# ---------------------------------------------------------------------------


def yates_decomposition(p50s: Sequence[float], outcomes: Sequence[float]) -> dict[str, float]:
    """Yates (1982) covariance partition of the Brier score.

    Brier = Var(o) + Var(f) + bias^2 - 2*cov(f,o), with Var(f) split into
    slope-explained variance (Var_min = slope^2 * Var(o)) and excess scatter
    (delta_var). Population (biased) moments throughout, so the identity is
    exact; `check_residual` should be ~0 up to float error.
    """
    f = np.asarray(p50s, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    n = len(f)
    assert n == len(o) and n > 0
    f_bar, o_bar = float(f.mean()), float(o.mean())
    var_o = o_bar * (1.0 - o_bar)
    var_f = float(f.var())
    bias_sq = (f_bar - o_bar) ** 2
    if 0.0 < o_bar < 1.0:
        slope = float(f[o == 1].mean() - f[o == 0].mean())
    else:
        slope = float("nan")
    cov_fo = float((f * o).mean() - f_bar * o_bar)
    var_min_f = slope**2 * var_o if not math.isnan(slope) else float("nan")
    delta_var_f = var_f - var_min_f if not math.isnan(slope) else float("nan")
    brier = float(((f - o) ** 2).mean())
    check_residual = brier - (var_o + var_f + bias_sq - 2.0 * cov_fo)
    return {
        "brier": brier,
        "uncertainty_var_o": var_o,
        "var_f": var_f,
        "bias_sq": bias_sq,
        "slope": slope,
        "var_min_f": var_min_f,
        "scatter_delta_var_f": delta_var_f,
        "cov_fo": cov_fo,
        "check_residual": check_residual,
        "n": float(n),
    }


def murphy_decomposition(p50s: Sequence[float], outcomes: Sequence[float], n_prob_bins: int = 10) -> dict[str, float]:
    """Murphy (1973) partition: Brier ~= reliability - resolution + uncertainty.

    Forecasts are grouped into `n_prob_bins` equal-width probability bins;
    the identity is exact only for within-bin-constant forecasts, so the
    within-bin variance is reported as `residual`.
    """
    f = np.asarray(p50s, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    n = len(f)
    assert n == len(o) and n > 0
    o_bar = float(o.mean())
    edges = np.linspace(0.0, 1.0, n_prob_bins + 1)
    idx = np.clip(np.digitize(f, edges[1:-1]), 0, n_prob_bins - 1)
    reliability = 0.0
    resolution = 0.0
    for k in range(n_prob_bins):
        mask = idx == k
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        f_k = float(f[mask].mean())
        o_k = float(o[mask].mean())
        reliability += n_k * (f_k - o_k) ** 2
        resolution += n_k * (o_k - o_bar) ** 2
    reliability /= n
    resolution /= n
    uncertainty = o_bar * (1.0 - o_bar)
    brier = float(((f - o) ** 2).mean())
    return {
        "brier": brier,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "residual": brier - (reliability - resolution + uncertainty),
        "n": float(n),
    }


# ---------------------------------------------------------------------------
# Frontier statistics (spec section 6.2)
# ---------------------------------------------------------------------------


def frontier_stats(
    program_at_pareto_front_valset: Mapping[Any, set[int]],
    per_program_scores: Sequence[float],
) -> dict[str, Any]:
    """Frontier size, cells-won distribution and parent-selection entropy.

    Uses the same dominance filtering + frequency weighting as GEPA's native
    Pareto parent sampler, so `effective_parents` reflects the actual
    sampling distribution.
    """
    filtered = remove_dominated_programs(
        {k: set(v) for k, v in program_at_pareto_front_valset.items()},
        scores=dict(enumerate(per_program_scores)),
    )
    freq: dict[int, int] = {}
    for front in filtered.values():
        for prog in front:
            freq[prog] = freq.get(prog, 0) + 1
    total = sum(freq.values())
    if total == 0:
        return {"frontier_size": 0}
    probs = [c / total for c in freq.values()]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    cells_won = dict(sorted(freq.items()))
    return {
        "frontier_size": len(freq),
        "cells_won_by_program": cells_won,
        "share_winning_exactly_one_cell": sum(1 for c in freq.values() if c == 1) / len(freq),
        "parent_selection_entropy_nats": entropy,
        "effective_parents": math.exp(entropy),
        "max_cells_won": max(freq.values()),
    }


# ---------------------------------------------------------------------------
# Confidence-vs-wins correlations (spec section 6.4)
# ---------------------------------------------------------------------------


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 3:
        return float("nan")
    rho: Any = stats.spearmanr(x, y)[0]
    return float(rho) if rho is not None else float("nan")


def partial_spearman(x: Sequence[float], y: Sequence[float], z: Sequence[float]) -> float:
    """Spearman partial correlation of x and y controlling for z
    (Pearson partial correlation on the rank-transformed variables)."""
    if len(x) < 4:
        return float("nan")
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)
    r_xy = np.corrcoef(rx, ry)[0, 1]
    r_xz = np.corrcoef(rx, rz)[0, 1]
    r_yz = np.corrcoef(ry, rz)[0, 1]
    denom = math.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if denom == 0 or any(math.isnan(v) for v in (r_xy, r_xz, r_yz)):
        return float("nan")
    return float((r_xy - r_xz * r_yz) / denom)


def mean_extremity(p50s: Sequence[float | None]) -> float:
    vals = [abs(p - 0.5) for p in p50s if p is not None]
    return float(np.mean(vals)) if vals else float("nan")
