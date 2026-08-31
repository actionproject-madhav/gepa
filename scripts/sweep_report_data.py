#!/usr/bin/env python3
"""Data behind the five parameter-sweep figures (points 4/5 pre-run screen).

One number source for summary/make_param_sweep_figs.py (LLM_elicitation).
Everything is recomputed from the logged runs — no simulation of
counterfactual runs, only replay of decisions whose inputs were recorded:

  1. gate size            n_train_tasks_per_iter        ONLINE (runs done)
  2. wins-only acceptance acceptance_criterion           OFFLINE replay
  3. joint acceptance     aggregate_sum_and_min_task_wins OFFLINE -> arm A
  4. frontier threshold   n_cells_won_needed_for_pareto_frontier OFFLINE
  5. pareto instance      pareto_instance: model_bin     OFFLINE -> arm B

Conventions (also stored under meta in the JSON):
  - Acceptance replay pools the two 20-cell-gate runs (pilot_baseline,
    pilot_baseline_clean): 80 proposals. Scores are parsed-only (cells where
    either side is a parse failure, score -1.0, are dropped from the pair
    list — July only; the clean run has zero failures). The 3 July
    rejections that the parse-failure audit showed were themselves flipped
    by the bug (parsed-only sum would accept) are EXCLUDED from the reject
    pool — their recorded verdicts are measurement errors, not decisions.
  - Truth labels exist only for recorded accepts (they received a full val
    pass): good accept = child parsed-only val Brier < parent's.
  - Frontier replay uses raw scores (what the engine actually saw), ties
    shared, over the candidates that existed at each iteration.
  - (model, bin) replay uses parsed-only scores on instances where every
    candidate has data (prescreen convention).

Output: runs/sweep_report_data.json + printed tables.
Usage:  uv run python scripts/sweep_report_data.py
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
from forecaster_gepa.metrics import spearman  # noqa: E402

RUNS20 = ["pilot_baseline", "pilot_baseline_clean"]
FAIL = -0.9999  # scores are -Brier; parse failures are exactly -1.0

# Multi-pass sealed verdicts (runs index FORECASTER_GEPA_RUNS.md, all >=3
# paired held-out passes; + = better than seed).
SEALED = {
    "pilot_baseline": {12: +0.0266, 20: +0.0173, 15: +0.0143, 9: +0.0046, 14: -0.0071},
    "pilot_baseline_clean": {7: +0.0161, 3: +0.0027},
}


def parsed_mean_brier(sub) -> float:
    briers = [-v for v in (sub.values() if isinstance(sub, dict) else sub)]
    return statistics.mean(b for b in briers if b <= 0.9999)


def load_proposals(run: str) -> list[dict]:
    return [json.loads(l) for l in (REPO / "runs" / run / "candidates.jsonl").open()
            if '"proposal"' in l]


# ---------------------------------------------------------------- acceptance
def acceptance_sweep() -> dict:
    accepted: list[tuple[list, bool, str]] = []   # (pairs, good, run)
    rejected: list[tuple[list, str]] = []
    per_run = {}
    for run in RUNS20:
        st = GEPAState.load(str(REPO / "runs" / run))
        vals = [parsed_mean_brier(s) for s in st.prog_candidate_val_subscores]
        n_acc = n_good = n_rej = 0
        for p in load_proposals(run):
            ps, cs = p.get("gate_scores_parent") or [], p.get("gate_scores_child") or []
            if len(ps) != len(cs) or not ps:
                continue
            pairs = [(a, b) for a, b in zip(ps, cs) if a > FAIL and b > FAIL]
            if p["status"] == "accepted" and p.get("candidate_idx") is not None:
                good = vals[p["candidate_idx"]] < vals[p["parent_idx"]]
                accepted.append((pairs, good, run)); n_acc += 1; n_good += good
            elif p["status"] == "rejected":
                rejected.append((pairs, run)); n_rej += 1
        per_run[run] = {"n_accept": n_acc, "n_good": n_good,
                        "n_bad": n_acc - n_good, "n_reject": n_rej}

    n_good = sum(1 for _p, g, _r in accepted if g)
    n_bad = len(accepted) - n_good

    def rule_sum(pairs):
        return sum(b for _a, b in pairs) > sum(a for a, _b in pairs)

    def rule_wins(pairs, k):
        return sum(1 for a, b in pairs if b > a) >= k

    # drop parse-flipped rejections (recorded reject, parsed-only sum accepts)
    n_bug_rejects = sum(1 for pr, _r in rejected if rule_sum(pr))
    rejected = [(pr, r) for pr, r in rejected if not rule_sum(pr)]

    def sweep(fn) -> dict:
        out = {}
        for k in range(0, 21):
            gk = sum(1 for pr, g, _r in accepted if g and fn(pr, k))
            bk = sum(1 for pr, g, _r in accepted if not g and fn(pr, k))
            ra = sum(1 for pr, _r in rejected if fn(pr, k))
            flips = (len(accepted) - gk - bk) + ra  # accepts lost + rejects gained
            out[k] = {"good_kept": gk, "bad_kept": bk, "rejects_admitted": ra,
                      "decisions_changed": flips}
        return out

    wins_only = sweep(lambda pr, k: rule_wins(pr, k))
    joint = sweep(lambda pr, k: rule_sum(pr) and rule_wins(pr, k))
    return {"per_run": per_run,
            "pooled": {"n_good": n_good, "n_bad": n_bad, "n_reject": len(rejected)},
            "n_bug_rejects_excluded": n_bug_rejects,
            "wins_only": wins_only, "joint": joint}


# ---------------------------------------------------------------- frontier
def _wins(per_cell_best_tables) -> dict[int, int]:
    wins: dict[int, int] = defaultdict(int)
    for per_cand in per_cell_best_tables.values():
        best = max(per_cand.values())
        for idx, sc in per_cand.items():
            if sc == best:
                wins[idx] += 1
    return dict(wins)


def frontier_sweep() -> dict:
    out = {}
    for run in RUNS20:
        st = GEPAState.load(str(REPO / "runs" / run))
        subs = st.prog_candidate_val_subscores
        added = {0: 0}
        for p in load_proposals(run):
            if p["status"] == "accepted" and p.get("candidate_idx") is not None:
                added[p["candidate_idx"]] = p["iteration"]

        def wins_at(existing: list[int]) -> dict[int, int]:
            table = {key: {i: subs[i][key] for i in existing if key in subs[i]}
                     for key in subs[0]}
            return _wins({k: v for k, v in table.items() if v})

        end_wins = wins_at(list(range(len(subs))))
        eligible_vs_k = {k: sum(1 for w in end_wins.values() if w >= k)
                         for k in range(1, 9)}
        drops = {str(idx): {"cells_won": end_wins.get(idx, 0),
                            "sealed": SEALED[run].get(idx),
                            "drops_at_k": end_wins.get(idx, 0) + 1}
                 for idx in SEALED[run]}
        # first iteration where raising k would have changed the parent lottery
        first_div = {}
        iters = sorted({t for t in added.values() if t > 0})
        for k in (2, 3, 4):
            div = None
            for t in iters:
                existing = [i for i, a in added.items() if a < t]
                if len(existing) < 2:
                    continue
                w = wins_at(existing)
                if any(0 < ww < k for ww in w.values()):
                    div = t
                    break
            first_div[k] = div
        out[run] = {"cells_won": {str(i): w for i, w in sorted(end_wins.items())},
                    "n_candidates": len(subs), "eligible_vs_k": eligible_vs_k,
                    "sealed_candidates": drops, "first_divergence_iter": first_div}
    return out


# ---------------------------------------------------------------- pareto inst
def pareto_sweep() -> dict:
    man = json.load(open(REPO / "runs/task_manifest_seed42.json"))
    task_bin = {t: int(b) for b, e in man["bins"].items() for t in e.get("val", [])}
    out = {}
    for run in RUNS20:
        st = GEPAState.load(str(REPO / "runs" / run))
        subs = st.prog_candidate_val_subscores
        n_cand = len(subs)
        vals = [parsed_mean_brier(s) for s in subs]
        cell: dict[str, dict[int, float]] = defaultdict(dict)
        grp: dict[tuple, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        for idx, sub in enumerate(subs):
            for key, sc in sub.items():
                if sc <= FAIL:
                    continue
                cell[key][idx] = sc
                task, model = key.split("::")
                grp[(model, task_bin[task])][idx].append(sc)
        full_cell = {k: v for k, v in cell.items() if len(v) == n_cand}
        full_grp = {k: {i: statistics.mean(v) for i, v in per.items()}
                    for k, per in grp.items() if len(per) == n_cand}
        cw, gw = _wins(full_cell), _wins(full_grp)
        rho = lambda wins: spearman([wins.get(i, 0) for i in range(n_cand)],
                                    [-v for v in vals])
        out[run] = {
            "n_cell_instances": len(full_cell), "n_group_instances": len(full_grp),
            "cell_wins": {str(i): w for i, w in sorted(cw.items())},
            "group_wins": {str(i): w for i, w in sorted(gw.items())},
            "spearman_cell": rho(cw), "spearman_group": rho(gw),
            "sealed": {str(i): v for i, v in SEALED[run].items()},
        }
    return out


# ---------------------------------------------------------------- gate size
GATE_SIZE = {
    # verified multi-pass paired sealed gains vs seed (runs index):
    "runs": [
        {"label": "search run 1", "gate_cells": 20, "best_gain": 0.0266,
         "passes": "8 paired passes", "run_dir": "pilot_baseline"},
        {"label": "search run 2", "gate_cells": 20, "best_gain": 0.0161,
         "passes": "3 paired passes", "run_dir": "pilot_baseline_clean"},
        {"label": "100-cell run", "gate_cells": 100, "best_gain": None,
         "passes": "no prompt beat the seed on the search set",
         "run_dir": "pilot_gate100"},
    ],
    # runs/pilot_gate100/retro_gate_analysis.txt: subsampled-gate disagreement
    # with the 100-cell verdict (P(accept) resampling, 200 draws/proposal)
    "subsample_flip_rate": {"20": 0.30, "40": 0.22, "60": 0.15},
}


def main() -> int:
    payload = {
        "meta": {"date": "2026-08-31", "runs_pooled": RUNS20,
                 "scoring": "parsed-only pairs; truth = child parsed val < parent",
                 "frontier_scoring": "raw engine scores, ties shared"},
        "acceptance": acceptance_sweep(),
        "frontier": frontier_sweep(),
        "pareto_instance": pareto_sweep(),
        "gate_size": GATE_SIZE,
    }
    (REPO / "runs/sweep_report_data.json").write_text(json.dumps(payload, indent=1))

    a = payload["acceptance"]
    print(f"acceptance pooled: {a['pooled']} ("
          f"{a['n_bug_rejects_excluded']} parse-flipped July rejections excluded)")
    print(f"{'k':>3} {'wins-only g/b/adm/flip':>24} {'joint g/b/adm/flip':>22}")
    for k in range(0, 21):
        w, j = a["wins_only"][k], a["joint"][k]
        print(f"{k:>3} {w['good_kept']:>6}/{w['bad_kept']}/{w['rejects_admitted']}/"
              f"{w['decisions_changed']:<6} {j['good_kept']:>10}/{j['bad_kept']}/"
              f"{j['rejects_admitted']}/{j['decisions_changed']}")
    for run, f in payload["frontier"].items():
        print(f"\n{run}: eligible vs k {f['eligible_vs_k']}  "
              f"first divergence iter {f['first_divergence_iter']}")
        for idx, d in f["sealed_candidates"].items():
            print(f"  cand {idx}: {d['cells_won']} cells won, sealed {d['sealed']:+.4f}, "
                  f"ineligible at k>={d['drops_at_k']}")
    for run, p in payload["pareto_instance"].items():
        print(f"\n{run}: {p['n_cell_instances']} cells / {p['n_group_instances']} groups; "
              f"spearman(wins, quality) {p['spearman_cell']:+.3f} -> {p['spearman_group']:+.3f}")
        for idx in p["sealed"]:
            print(f"  cand {idx}: cell-wins {p['cell_wins'].get(idx, 0)}, "
                  f"group-wins {p['group_wins'].get(idx, 0)}, sealed {p['sealed'][idx]:+.4f}")
    print("\nwritten: runs/sweep_report_data.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
