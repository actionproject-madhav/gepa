#!/usr/bin/env python3
"""Prompt-feature scan across every candidate from every run (Matt's idea).

Treat all candidate prompts the three GEPA runs produced — winners AND
losers — as a dataset. Tag each prompt with a deterministic regex rubric of
features, then compare scores across feature groups: which features
correlate with better forecasts, which merely change the text.

Scores used:
  * parsed-only val Brier (failures dropped): every candidate has one, but a
    single val pass has rerun sd ~0.0065, so treat val deltas as DIRECTIONAL
    screening only;
  * sealed parsed-only edge vs the run's own seed, for the candidates that
    got a finalist evaluation (July top-5, v2 top-5) — the reliable subset.

Honest caveats (also printed in the output):
  * candidates within a run share lineage (children inherit parent text), so
    prompts are NOT independent samples — this is feature screening, not
    hypothesis testing;
  * features co-occur (e.g. numeric bands and anti-hedging arrive together
    in July's lineage), so single-feature deltas are confounded with their
    co-travellers.

Usage:
    uv run python scripts/prompt_feature_scan.py
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from gepa.core.state import GEPAState  # noqa: E402
from forecaster_gepa.config import COMPONENT_NAME  # noqa: E402

RUNS = ["pilot_baseline", "pilot_gate100", "pilot_reflection_v2"]

# Deterministic rubric. Each feature: (name, compiled regex, min count to
# count as present). Patterns operate on the lower-cased prompt text.
FEATURES = [
    ("numeric_bands", re.compile(r"0\.\d+\s*(?:-|–|—|\bto\b)\s*0\.\d+"), 2),
    ("anti_hedging", re.compile(
        r"avoid.{0,30}0\.5|0\.5.{0,30}trap|decisiv|do not hedge|don't hedge|commit to|hedging"), 1),
    ("task_type_rules", re.compile(
        r"interactive|multi-?step|single[- ](?:command|step|shot)|one-?liner|static"), 1),
    ("platform_terms", re.compile(
        r"cybashbench|nl2bash|nyuctf|intercode|cybench|\bctf\b"), 1),
    ("evidence_anchoring", re.compile(
        r"pass[- ]rate|base[- ]rate|cluster|reference task|similar task"), 1),
    ("calibration_language", re.compile(
        r"calibrat|overconfiden|underconfiden|brier"), 1),
    ("extreme_prescriptions", re.compile(r"0\.9\d|0\.0\d"), 2),
]


def parsed_only_val(state: GEPAState) -> list[float]:
    out = []
    for sub in state.prog_candidate_val_subscores:
        briers = [-v for v in (sub.values() if isinstance(sub, dict) else sub)]
        parsed = [b for b in briers if b <= 0.9999]
        out.append(statistics.mean(parsed))
    return out


def sealed_edges(run_dir: Path) -> dict[int, float]:
    """Parsed-only sealed edge vs the run's own seed, per finalist candidate."""
    path = run_dir / "finalist_cells.jsonl"
    if not path.exists():
        return {}
    per: dict[int, dict[tuple, float]] = {}
    for line in path.open():
        r = json.loads(line)
        if r.get("brier") is None:
            continue  # parsed-only
        per.setdefault(r["candidate_idx"], {})[(r["task_id"], r["model"])] = r["brier"]
    if 0 not in per:
        return {}
    seed = per[0]
    edges = {}
    for idx, cells in per.items():
        if idx == 0:
            continue
        common = [c for c in cells if c in seed]
        edges[idx] = statistics.mean(seed[c] - cells[c] for c in common)
    return edges


def main() -> int:
    rows = []  # (run, idx, text, feats: dict, val_brier, sealed_edge|None)
    for run in RUNS:
        run_dir = REPO / "runs" / run
        st = GEPAState.load(str(run_dir))
        vals = parsed_only_val(st)
        sealed = sealed_edges(run_dir)
        for idx, cand in enumerate(st.program_candidates):
            text = cand[COMPONENT_NAME].lower()
            feats = {name: (len(rx.findall(text)) >= k) for name, rx, k in FEATURES}
            rows.append({
                "run": run, "idx": idx, "chars": len(text), "feats": feats,
                "val": vals[idx], "sealed": sealed.get(idx),
                "is_seed": idx == 0,
            })

    n = len(rows)
    print(f"{n} candidate prompts scanned across {len(RUNS)} runs "
          f"(seeds included; {sum(1 for r in rows if r['sealed'] is not None)} have sealed edges)\n")

    print(f"{'feature':<22} {'n_with':>6} {'n_without':>9} "
          f"{'val Brier with':>14} {'without':>9} {'delta':>8}   sealed-known with-feature")
    print("-" * 100)
    evolved = [r for r in rows if not r["is_seed"]]
    for name, _rx, _k in FEATURES:
        w = [r for r in evolved if r["feats"][name]]
        wo = [r for r in evolved if not r["feats"][name]]
        if not w or not wo:
            print(f"{name:<22} {'--- all prompts on one side ---':>40}")
            continue
        vw = statistics.mean(r["val"] for r in w)
        vwo = statistics.mean(r["val"] for r in wo)
        sealed_w = [(r["run"][6:], r["idx"], round(r["sealed"], 4))
                    for r in w if r["sealed"] is not None]
        print(f"{name:<22} {len(w):>6} {len(wo):>9} {vw:>14.4f} {vwo:>9.4f} "
              f"{vw - vwo:>+8.4f}   {sealed_w if sealed_w else ''}")

    # length
    lens = [(r["chars"], r["val"]) for r in evolved]
    lens.sort()
    half = len(lens) // 2
    short_v = statistics.mean(v for _c, v in lens[:half])
    long_v = statistics.mean(v for _c, v in lens[half:])
    print(f"\nlength: shorter half mean val {short_v:.4f} vs longer half {long_v:.4f} "
          f"(median split at {lens[half][0]} chars)")

    # the validated ground truth
    print("\nsealed-validated candidates (parsed-only edge vs own seed, + = better):")
    for r in rows:
        if r["sealed"] is not None:
            feats_on = [k for k, v in r["feats"].items() if v]
            print(f"  {r['run']:<22} cand {r['idx']:>2}  edge {r['sealed']:+.4f}  features: {feats_on}")

    print("\nCAVEATS: prompts within a run share lineage (not independent); features")
    print("co-occur; val numbers are single-pass (rerun sd ~0.0065) — directional only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
