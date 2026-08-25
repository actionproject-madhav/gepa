#!/usr/bin/env python3
"""Audit the impact of parse failures on every scored decision in a GEPA run.

Background: when the forecaster's response does not contain a parseable
<p50>, the pipeline scores that cell `failure_score` (-1.0, i.e. Brier 1.0 —
the worst possible value). A normal cell scores ~0.10, so ONE parse failure
costs about as much as nine ordinary cells. On a 20-cell gate batch a single
failure shifts the batch mean by ~0.045; on an 84-cell val pass by ~0.011;
on the 230-cell finalist set by ~0.004. Our real prompt effects are ~0.02,
so a handful of failures can erase or invent an entire result.

Failures are not hidden: they are recorded as the exact sentinel score
-1.0 in the per-cell logs, so their impact on any SCORE can be recomputed
exactly, with no new API calls:

  * gate cells      -> <run_dir>/candidates.jsonl (gate_scores_parent/child)
  * val cells       -> <run_dir>/gepa_state.bin (prog_candidate_val_subscores)
  * finalist cells  -> <run_dir>/finalist_cells.jsonl

What this script reports per run:
  1. failure counts and rates in each scoring path;
  2. which accept/reject decisions had a failure in their gate batch, and
     which of those WOULD FLIP if the failure were removed (a rejected child
     whose sum-margin is smaller than the ~0.9 penalty, or an accepted child
     that only cleared the bar because its PARENT was penalised);
  3. operational vs parsed-only val Brier per affected candidate.

IMPORTANT — what cannot be recomputed: the optimisation trajectory. GEPA is
sequential; a flipped accept/reject changes the candidate pool, hence parent
selection, hence every later iteration. Removing a failure from the logs
re-scores a decision but cannot reconstruct the run that would have
followed. That counterfactual is only obtainable by re-running (and, at
temperature 1.0, a re-run differs from the original anyway).

Usage:
    uv run python scripts/parse_failure_audit.py                 # all known runs
    uv run python scripts/parse_failure_audit.py --run-dir runs/pilot_baseline
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

FAIL_SCORE = -0.9999  # scores are -Brier; a failure is exactly -1.0
FAIL_PENALTY = 0.9  # a failure (1.0) vs a typical cell (~0.1)


def audit_gate(run_dir: Path) -> None:
    path = run_dir / "candidates.jsonl"
    if not path.exists():
        print("  gate cells: candidates.jsonl not found")
        return
    props = [json.loads(l) for l in path.open() if '"proposal"' in l]
    tot = fails = 0
    touched: list[tuple] = []
    for p in props:
        ps = p.get("gate_scores_parent") or []
        cs = p.get("gate_scores_child") or []
        if len(ps) != len(cs) or not ps:
            continue
        pf = sum(1 for s in ps if s <= FAIL_SCORE)
        cf = sum(1 for s in cs if s <= FAIL_SCORE)
        tot += len(ps) + len(cs)
        fails += pf + cf
        if pf or cf:
            margin = sum(cs) - sum(ps)
            # child penalised -> would the rejection reverse without it?
            # parent penalised -> did the acceptance depend on that penalty?
            flip = (p["status"] == "rejected" and margin > -FAIL_PENALTY * cf) or (
                p["status"] == "accepted" and margin < FAIL_PENALTY * pf
            )
            touched.append((p["iteration"], p["status"], pf, cf, margin, flip))
    rate = 100 * fails / tot if tot else 0.0
    print(f"  gate cells:      {fails}/{tot} failed ({rate:.2f}%) | decisions touched {len(touched)}/{len(props)}")
    for it, st, pf, cf, m, flip in touched:
        mark = "  <== WOULD FLIP without the failure" if flip else ""
        print(f"      iter {it:>3} {st:<9} parent_fails={pf} child_fails={cf} sum-margin={m:+.3f}{mark}")


def audit_val(run_dir: Path) -> None:
    try:
        from gepa.core.state import GEPAState
    except ImportError:
        print("  val cells: gepa not importable (run via `uv run`)")
        return
    if not (run_dir / "gepa_state.bin").exists():
        print("  val cells: gepa_state.bin not found")
        return
    st = GEPAState.load(str(run_dir))
    tot = fails = 0
    rows = []
    for idx, sub in enumerate(st.prog_candidate_val_subscores):
        vals = list(sub.values()) if isinstance(sub, dict) else list(sub)
        briers = [-v for v in vals]
        f = sum(1 for b in briers if b > 0.9999)
        tot += len(briers)
        fails += f
        if f:
            parsed = [b for b in briers if b <= 0.9999]
            rows.append((idx, f, statistics.mean(briers), statistics.mean(parsed)))
    rate = 100 * fails / tot if tot else 0.0
    print(f"  val cells:       {fails}/{tot} failed ({rate:.2f}%)")
    for idx, f, op, pa in rows:
        print(f"      cand {idx:>3}: {f} failure(s) | val Brier operational {op:.4f} -> parsed-only {pa:.4f}")


def audit_finalist(run_dir: Path) -> None:
    path = run_dir / "finalist_cells.jsonl"
    if not path.exists():
        return
    per: dict[int, list[float]] = {}
    fails: dict[int, int] = {}
    for line in path.open():
        r = json.loads(line)
        idx = r["candidate_idx"]
        b = r.get("brier")
        if b is None:
            fails[idx] = fails.get(idx, 0) + 1
            b = 1.0
        per.setdefault(idx, []).append(b)
    print("  finalist cells:")
    for idx in sorted(per):
        vals = per[idx]
        f = fails.get(idx, 0)
        parsed = [b for b in vals if b < 0.9999]
        print(f"      cand {idx:>3}: {f}/{len(vals)} failed | Brier operational "
              f"{statistics.mean(vals):.4f} -> parsed-only {statistics.mean(parsed):.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=None, help="single run dir; default audits all known runs")
    args = ap.parse_args()

    runs = [Path(args.run_dir)] if args.run_dir else [
        REPO / "runs/pilot_baseline",
        REPO / "runs/pilot_gate100",
        REPO / "runs/pilot_reflection_v2",
    ]
    for run_dir in runs:
        if not run_dir.exists():
            print(f"\n=== {run_dir.name}: NOT PRESENT locally "
                  f"(raw runs live on the results/* branches of the fork) ===")
            continue
        print(f"\n=== {run_dir.name} ===")
        audit_gate(run_dir)
        audit_val(run_dir)
        audit_finalist(run_dir)

    print("\nNOTE: scores can be recomputed exactly from the logs; the optimisation")
    print("TRAJECTORY cannot — a flipped accept/reject changes every later iteration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
