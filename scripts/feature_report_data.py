#!/usr/bin/env python3
"""Data pipeline for the one-page feature-synthesis report.

Produces every number the report needs, from logged data only:

  A. Feature tags for every evolved prompt across all four GEPA runs
     (deterministic regex rubric; the tag table is saved so anyone can audit
     or re-tag by hand).
  B. Per-feature accuracy screen: within-run parsed-only val Brier delta
     (with-feature minus without), pooled across runs weighted by n.
     Within-run deltas remove the run-level measurement-condition confound
     (July ran the forecaster at T=1, the clean rerun at T=0).
  C. Per-feature output-shift screen: Wasserstein-1 distance between the
     pooled gate-trace p50 distribution of feature-carrying prompts and the
     same run's seed p50 distribution (July + clean runs, which log full
     gate traces), pooled across runs weighted by n.
  D. Matched-cell shift + accuracy + CRPS for the sealed-measured prompts
     and the causal ablation arms (same 230 cells, multi-pass): mean Brier,
     mean beta-fit CRPS (repo-standard, report_analyses.calculate_baselines
     machinery; pinball fallback), W1 of p50 distributions vs seed.

Output: runs/feature_report_data.json + printed tables.
Usage:  uv run python scripts/feature_report_data.py
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from gepa.core.state import GEPAState  # noqa: E402
from forecaster_gepa.config import COMPONENT_NAME  # noqa: E402

RUNS = ["pilot_baseline", "pilot_gate100", "pilot_reflection_v2", "pilot_baseline_clean"]
TRACE_RUNS = ["pilot_baseline", "pilot_baseline_clean"]  # full gate traces

FEATURES = [
    ("empirical_anchoring", re.compile(r"\banchor"), 1),
    ("nearest_example", re.compile(
        r"near[- ]twin|individual example|closest analog|most similar|near[- ]est (?:individual|example)|closely (?:mirrors|matched)"), 1),
    ("numeric_bands", re.compile(r"0\.\d+\s*(?:-|–|—|\bto\b)\s*0\.\d+"), 2),
    ("anti_hedging", re.compile(
        r"avoid.{0,30}0\.5|0\.5.{0,30}trap|decisiv|do not hedge|don't hedge|commit to|hedging|park.{0,20}(?:midpoint|fifty)"), 1),
    ("task_type_rules", re.compile(
        r"interactive|multi-?step|multi-?stage|single[- ](?:command|step|shot)|one-?liner|static"), 1),
]


def w1(a, b, grid=101) -> float:
    qs = np.linspace(0, 1, grid)
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def parsed_val(sub) -> float:
    b = [-v for v in (sub.values() if isinstance(sub, dict) else sub)]
    return statistics.mean(x for x in b if x <= 0.9999)


# ---------------------------------------------------------------- A. tags
def build_tags():
    rows = []
    for run in RUNS:
        st = GEPAState.load(str(REPO / "runs" / run))
        for idx, cand in enumerate(st.program_candidates):
            text = cand[COMPONENT_NAME]
            low = text.lower()
            feats = {n: (len(rx.findall(low)) >= k) for n, rx, k in FEATURES}
            rows.append({"run": run, "idx": idx, "is_seed": idx == 0,
                         "chars": len(text), "val": parsed_val(st.prog_candidate_val_subscores[idx]),
                         **feats})
    return rows


# ---------------------------------------------------------------- B. accuracy screen
def accuracy_screen(rows):
    out = {}
    for name, _rx, _k in FEATURES:
        deltas, weights = [], []
        for run in RUNS:
            w = [r["val"] for r in rows if r["run"] == run and not r["is_seed"] and r[name]]
            wo = [r["val"] for r in rows if r["run"] == run and not r["is_seed"] and not r[name]]
            if len(w) >= 2 and len(wo) >= 2:
                deltas.append(statistics.mean(w) - statistics.mean(wo))
                weights.append(min(len(w), len(wo)))
        if deltas:
            pooled = sum(d * n for d, n in zip(deltas, weights)) / sum(weights)
            out[name] = {"delta_val_brier": pooled, "runs_used": len(deltas)}
        else:
            out[name] = {"delta_val_brier": None, "runs_used": 0}
        nw = sum(1 for r in rows if not r["is_seed"] and r[name])
        out[name]["n_with"] = nw
        out[name]["n_without"] = sum(1 for r in rows if not r["is_seed"]) - nw
    return out


# ---------------------------------------------------------------- C. shift screen
def gate_p50_pools(run):
    """p50 pools per candidate from gate traces (parents + accepted children)."""
    props_path = REPO / "runs" / run / "candidates.jsonl"
    iter_to_new = {}
    for line in props_path.open():
        if '"proposal"' not in line:
            continue
        p = json.loads(line)
        if p["status"] == "accepted" and p.get("candidate_idx") is not None:
            iter_to_new[p["iteration"]] = p["candidate_idx"]
    pools = defaultdict(list)
    for line in (REPO / "runs" / run / "gate_traces.jsonl").open():
        r = json.loads(line)
        idx = r.get("candidate_idx")
        if idx is None:
            idx = iter_to_new.get(r.get("iteration"))
        if idx is None:
            continue
        for t in r.get("trajectories") or []:
            if t.get("p50") is not None:
                pools[idx].append(t["p50"])
    return pools


def shift_screen(rows):
    out = {name: {"w1": [], "n": []} for name, _rx, _k in FEATURES}
    for run in TRACE_RUNS:
        pools = gate_p50_pools(run)
        seed_pool = pools.get(0, [])
        if len(seed_pool) < 30:
            continue
        tag = {r["idx"]: r for r in rows if r["run"] == run}
        for name, _rx, _k in FEATURES:
            feat_pool = [p for idx, ps in pools.items()
                         if idx != 0 and idx in tag and tag[idx][name] for p in ps]
            if len(feat_pool) >= 50:
                out[name]["w1"].append(w1(feat_pool, seed_pool))
                out[name]["n"].append(len(feat_pool))
    return {name: {
        "w1_vs_seed": (sum(a * b for a, b in zip(d["w1"], d["n"])) / sum(d["n"])
                       if d["n"] else None)}
        for name, d in out.items()}


# ---------------------------------------------------------------- D. matched sealed
def crps_fns():
    import os
    # Sibling checkout, same convention as the editable install in
    # pyproject.toml; override with LLM_ELICITATION_DIR if yours differs.
    elicit = Path(os.environ.get("LLM_ELICITATION_DIR", REPO.parent / "LLM_elicitation"))
    sys.path.insert(0, str(elicit))
    cwd = os.getcwd()
    try:
        # calculate_baselines locates its repo root from CWD
        os.chdir(elicit)
        from report_analyses.calculate_baselines import fit_beta_to_percentiles, crps_beta
        os.chdir(cwd)
        def crps(p25, p50, p75, outcome):
            a, b = fit_beta_to_percentiles(p25, p50, p75)
            return crps_beta(a, b, int(outcome >= 0.5))
        return crps, "beta-fit CRPS (report_analyses.calculate_baselines)"
    except Exception:
        os.chdir(cwd)
        def crps(p25, p50, p75, outcome):  # pinball fallback
            o = float(outcome >= 0.5)
            loss = 0.0
            for q, x in ((0.25, p25), (0.5, p50), (0.75, p75)):
                loss += (q - (o < x)) * (o - x)
            return loss / 3
        return crps, "mean pinball loss at q=0.25/0.5/0.75 (beta-fit unavailable)"


def load_sealed_cells():
    """All temp-0 sealed measurements: {prompt: {repeat_key: {cell: rec}}}."""
    files = ["noise_cells_temp0.jsonl", "noise_cells_temp0_confirm.jsonl",
             "noise_cells_clean_rerun.jsonl", "noise_cells_feature_ablation.jsonl"]
    data = defaultdict(lambda: defaultdict(dict))
    for i, f in enumerate(files):
        p = REPO / "runs/noise_study" / f
        if not p.exists():
            continue
        for line in p.open():
            r = json.loads(line)
            if r["set"] != "sealed":
                continue
            data[r["prompt"]][(i, r["repeat"])][(r["task_id"], r["model"])] = r
    return data


def matched_sealed():
    crps, crps_name = crps_fns()
    data = load_sealed_cells()
    out = {"crps_method": crps_name, "prompts": {}}
    seed_p50 = defaultdict(list)
    for rep, cells in data.get("seed", {}).items():
        for cell, r in cells.items():
            if r["p50"] is not None:
                seed_p50[cell].append(r["p50"])
    seed_mean_p50 = {c: statistics.mean(v) for c, v in seed_p50.items()}
    for prompt, reps in data.items():
        briers, crpss = [], []
        per_cell = defaultdict(list)  # cell -> [(p25,p50,p75,outcome)]
        for rep, cells in reps.items():
            for cell, r in cells.items():
                briers.append(r["brier"] if r["brier"] is not None else 1.0)
                if r["p50"] is not None:
                    per_cell[cell].append((r["p25"], r["p50"], r["p75"], r["outcome"]))
        # CRPS on the per-cell MEAN percentile triple (one Beta fit per cell,
        # not per pass-cell) — same estimand, ~8x fewer scipy fits.
        for cell, triples in per_cell.items():
            m25 = statistics.mean(t[0] for t in triples if t[0] is not None)
            m50 = statistics.mean(t[1] for t in triples)
            m75 = statistics.mean(t[2] for t in triples if t[2] is not None)
            try:
                crpss.append(crps(m25, m50, m75, triples[0][3]))
            except Exception:
                pass
        mean_p50 = {c: statistics.mean(t[1] for t in v) for c, v in per_cell.items()}
        common = [c for c in mean_p50 if c in seed_mean_p50]
        out["prompts"][prompt] = {
            "n_passes": len(reps),
            "brier": statistics.mean(briers),
            "crps": statistics.mean(crpss) if crpss else None,
            "w1_p50_vs_seed": w1([mean_p50[c] for c in common],
                                 [seed_mean_p50[c] for c in common]) if prompt != "seed" and common else 0.0,
        }
    return out


def main() -> int:
    rows = build_tags()
    acc = accuracy_screen(rows)
    shf = shift_screen(rows)
    sealed = matched_sealed()

    n_evolved = sum(1 for r in rows if not r["is_seed"])
    print(f"{n_evolved} evolved prompts tagged across {len(RUNS)} runs\n")
    print(f"{'feature':<22} {'with':>5} {'w/o':>5} {'dBrier(val)':>12} {'W1 shift':>9}")
    for name, _rx, _k in FEATURES:
        a, s = acc[name], shf[name]
        d = f"{a['delta_val_brier']:+.4f}" if a["delta_val_brier"] is not None else "--"
        wv = f"{s['w1_vs_seed']:.3f}" if s["w1_vs_seed"] is not None else "--"
        print(f"{name:<22} {a['n_with']:>5} {a['n_without']:>5} {d:>12} {wv:>9}")

    print(f"\nmatched sealed (temp 0, multi-pass; CRPS = {sealed['crps_method']}):")
    print(f"{'prompt':<24} {'passes':>6} {'Brier':>8} {'CRPS':>8} {'W1 vs seed':>11}")
    for prompt, d in sorted(sealed["prompts"].items(), key=lambda kv: kv[1]["brier"]):
        c = f"{d['crps']:.4f}" if d["crps"] is not None else "--"
        print(f"{prompt:<24} {d['n_passes']:>6} {d['brier']:>8.4f} {c:>8} {d['w1_p50_vs_seed']:>11.4f}")

    payload = {"tags": rows, "accuracy": acc, "shift": shf, "sealed": sealed}
    (REPO / "runs/feature_report_data.json").write_text(json.dumps(payload, indent=1))
    print("\nwritten: runs/feature_report_data.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
