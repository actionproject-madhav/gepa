#!/usr/bin/env python3
"""Offline pre-screen of two never-run search-setting ideas, from logged data.

SCOPE (Jakub's caveat, stated up front): changing an acceptance rule or the
Pareto instance changes which prompts enter the pool and everything
downstream, so nothing here SIMULATES an alternative run. What a replay can
do is grade a candidate setting against the decisions we already know the
answers to. A setting that misgrades those is dead before we pay for a live
run; a setting that grades them well is worth buying one.

(4) Acceptance-criterion variants (Sec. 4 item 4)
    For every logged proposal we have the 20 (or 100) gate cell scores of
    parent and child. For ACCEPTED proposals we also know the truth: the
    child got a full val pass, so "good accept" = child's parsed-only val
    Brier beat its parent's. Replay each rule on the accepted set: how many
    good/bad accepts would it keep? Also count how many of the (truth-
    unknown) rejects it would newly admit — a rule admitting most rejects is
    near-vacuous.

(5) (model, bin) aggregated Pareto instances (Sec. 4 item 5, Jeff's idea)
    Rebuild the val scoreboard at both granularities from the logged
    per-cell scores (parsed-only) and compare: frontier concentration, how
    well instance-wins track overall quality, and whether the mild
    "confidently-wrong wins the cell" tilt (extremity correlation) shrinks.

Usage:
    uv run python scripts/offline_prescreen.py
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from gepa.core.state import GEPAState  # noqa: E402
from forecaster_gepa.metrics import spearman, partial_spearman  # noqa: E402

RUNS = ["pilot_baseline", "pilot_gate100", "pilot_reflection_v2"]
FAIL = -0.9999  # scores are -Brier; parse failures are exactly -1.0


def parsed_mean_brier(sub) -> float:
    briers = [-v for v in (sub.values() if isinstance(sub, dict) else sub)]
    parsed = [b for b in briers if b <= 0.9999]
    return statistics.mean(parsed)


# ---------------------------------------------------------------- (4)
def replay_acceptance() -> None:
    print("=" * 78)
    print("(4) ACCEPTANCE-RULE REPLAY on logged proposals (parsed-only scoring)")
    print("=" * 78)
    for run in RUNS:
        run_dir = REPO / "runs" / run
        props = [json.loads(l) for l in (run_dir / "candidates.jsonl").open()
                 if '"proposal"' in l]
        st = GEPAState.load(str(run_dir))
        vals = [parsed_mean_brier(s) for s in st.prog_candidate_val_subscores]

        accepted, rejected = [], []
        for p in props:
            ps = p.get("gate_scores_parent") or []
            cs = p.get("gate_scores_child") or []
            if len(ps) != len(cs) or not ps:
                continue
            # parsed-only replay: drop cells where EITHER side is a failure
            pairs = [(a, b) for a, b in zip(ps, cs) if a > FAIL and b > FAIL]
            if p["status"] == "accepted" and p.get("candidate_idx") is not None:
                parent_idx = p.get("parent_idx")
                if parent_idx is None:
                    continue
                good = vals[p["candidate_idx"]] < vals[parent_idx]
                accepted.append((pairs, good))
            elif p["status"] == "rejected":
                rejected.append(pairs)

        gate_n = len(accepted[0][0]) if accepted else 0
        n_good = sum(1 for _p, g in accepted if g)
        print(f"\n{run}: {len(accepted)} accepts with truth "
              f"({n_good} good / {len(accepted) - n_good} bad by parsed val), "
              f"{len(rejected)} rejects (truth unknown), ~{gate_n}-cell gate")

        def rule_sum(pairs, tau=0.0):
            return sum(b for _a, b in pairs) - sum(a for a, _b in pairs) > tau

        def rule_wins(pairs, k):
            return sum(1 for a, b in pairs if b > a) >= k

        ks = range(2, 11) if gate_n <= 30 else range(10, 55, 5)
        rows = [("native sum>0 (clean)", lambda p: rule_sum(p))]
        rows += [(f"min_task_wins k={k}", (lambda k: lambda p: rule_wins(p, k))(k))
                 for k in ks]
        rows += [(f"sum>0 AND wins>={k}",
                  (lambda k: lambda p: rule_sum(p) and rule_wins(p, k))(k))
                 for k in ks]

        print(f"  {'rule':<24} {'good kept':>9} {'bad kept':>9} {'rejects admitted':>17}")
        for name, fn in rows:
            gk = sum(1 for pr, g in accepted if g and fn(pr))
            bk = sum(1 for pr, g in accepted if not g and fn(pr))
            ra = sum(1 for pr in rejected if fn(pr))
            print(f"  {name:<24} {gk:>4}/{n_good:<4} {bk:>4}/{len(accepted) - n_good:<4} "
                  f"{ra:>7}/{len(rejected):<7}")


# ---------------------------------------------------------------- (5)
def replay_pareto_aggregation() -> None:
    print()
    print("=" * 78)
    print("(5) PARETO INSTANCES: per-cell vs (model, bin) aggregate  [July val data]")
    print("=" * 78)
    man = json.load(open(REPO / "runs/task_manifest_seed42.json"))
    task_bin = {t: int(b) for b, e in man["bins"].items() for t in e.get("val", [])}

    st = GEPAState.load(str(REPO / "runs/pilot_baseline"))
    subs = st.prog_candidate_val_subscores
    n_cand = len(subs)
    vals = [parsed_mean_brier(s) for s in subs]

    # parsed-only per-cell score tables
    cell_scores: dict[str, dict[int, float]] = defaultdict(dict)
    agg_scores: dict[tuple, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for idx, sub in enumerate(subs):
        for key, score in sub.items():
            if score <= FAIL:
                continue
            cell_scores[key][idx] = score
            task, model = key.split("::")
            agg_scores[(model, task_bin[task])][idx].append(score)

    def wins_from(table) -> dict[int, int]:
        wins: dict[int, int] = defaultdict(int)
        for _inst, per_cand in table.items():
            if len(per_cand) < n_cand:  # only instances every candidate has
                continue
            best = max(per_cand.values())
            for idx, sc in per_cand.items():
                if sc == best:
                    wins[idx] += 1
        return wins

    cell_wins = wins_from(cell_scores)
    agg_mean = {inst: {i: statistics.mean(v) for i, v in per.items()}
                for inst, per in agg_scores.items()}
    agg_wins = wins_from(agg_mean)

    def describe(name, wins, n_inst):
        holders = {i for i, w in wins.items() if w > 0}
        top3 = sum(sorted(wins.values(), reverse=True)[:3])
        total = sum(wins.values())
        rho = spearman([wins.get(i, 0) for i in range(n_cand)],
                       [-v for v in vals])  # + = wins track LOWER Brier
        print(f"  {name:<22} instances={n_inst:<4} frontier holders={len(holders):<3} "
              f"top-3 share={top3 / total:.2f}  spearman(wins, quality)={rho:+.3f}")
        return wins

    print(f"\n  {n_cand} candidates; quality = parsed-only val Brier (single-pass, directional)")
    describe("per-cell (native)", cell_wins, len(cell_scores))
    describe("(model, bin) agg", agg_wins, len(agg_mean))

    # extremity tilt (metric-4): does aggregation shrink it?
    ext: dict[int, list[float]] = defaultdict(list)
    for line in (REPO / "runs/pilot_baseline/gate_traces.jsonl").open():
        r = json.loads(line)
        idx = r.get("candidate_idx")
        if idx is None:
            continue
        for t in r.get("trajectories") or []:
            if t.get("p50") is not None:
                ext[idx].append(abs(t["p50"] - 0.5))
    common = [i for i in range(n_cand) if len(ext.get(i, [])) >= 10]
    e = [statistics.mean(ext[i]) for i in common]
    b = [vals[i] for i in common]
    for name, wins in (("per-cell", cell_wins), ("(model,bin)", agg_wins)):
        w = [wins.get(i, 0) for i in common]
        print(f"  extremity tilt {name:<12} partial spearman(wins, extremity | Brier) = "
              f"{partial_spearman(w, e, b):+.3f}   (n={len(common)}; + = confident prompts over-win)")

    print("\n  sealed-validated candidates' standing (parsed-only sealed edge vs seed):")
    sealed = {9: +0.0046, 12: +0.0265, 14: -0.0071, 15: +0.0140, 20: +0.0173}
    for idx, edge in sealed.items():
        print(f"    cand {idx:>2} (sealed {edge:+.4f}): cell-wins {cell_wins.get(idx, 0):>3}, "
              f"(model,bin)-wins {agg_wins.get(idx, 0):>2}")


if __name__ == "__main__":
    replay_acceptance()
    replay_pareto_aggregation()
    print("\nREMINDER: screening only — a setting that passes still needs a live run;")
    print("a changed rule changes the pool and every later iteration (not simulable).")
