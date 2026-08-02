#!/usr/bin/env python3
"""Retrospective acceptance-gate analysis for any forecaster-GEPA run.

Every proposal record in <run_dir>/candidates.jsonl logs the parent's and the
child's per-cell gate scores (-Brier, higher better) on the SAME minibatch,
plus (for accepted children) the child's full-valset mean score and the
parent's. That is enough to simulate, offline, what ANY acceptance criterion
(task-win count k, aggregate margin tau, or both) would have decided for each
proposal, and — for accepted proposals — whether the decision agreed with the
val outcome ("val improved" = child val mean > parent val mean).

With --subsample-tasks-per-bin N (needs --manifest for task->bin lookup), it
additionally simulates SMALLER gates by drawing task-stratified subsets of
each proposal's gate cells many times, giving the distribution of decisions a
smaller gate would have made. Subsampling can only go DOWN from the run's
actual gate size — run wide gates if you want the size-response curve.

Caveats printed with the results:
  - "val improved" is itself a noisy 84-cell estimate; treat per-criterion
    precision as directional evidence, not a definitive ranking.
  - Rejected proposals never got a val eval, so criteria that would ACCEPT a
    natively-rejected child (e.g. min_task_wins alone) cannot be scored
    against val truth here.
  - Trajectory counterfactuals are NOT recoverable: flipping one decision
    changes every downstream iteration. This measures per-decision quality
    only.

Usage:
    python scripts/retrospective_gate_analysis.py --run-dir runs/pilot_baseline
    python scripts/retrospective_gate_analysis.py --run-dir runs/pilot_gate100 \
        --manifest runs/task_manifest_seed42.json --subsample-tasks-per-bin 1 2 3
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

CELL_KEY_SEPARATOR = "::"


def load_proposals(run_dir: Path) -> list[dict]:
    props = []
    with (run_dir / "candidates.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("kind") != "proposal":
                continue
            ps, cs = r.get("gate_scores_parent") or [], r.get("gate_scores_child") or []
            if not ps or len(ps) != len(cs):
                continue
            rec = {
                "iter": r.get("iteration"),
                "status": r["status"],
                "cell_ids": r.get("gate_cell_ids") or [],
                "parent_scores": ps,
                "child_scores": cs,
                "wins": sum(1 for p, c in zip(ps, cs) if c > p),
                "n": len(ps),
                "margin": sum(cs) - sum(ps),
                "candidate_idx": r.get("candidate_idx"),
                "val_child": r.get("val_mean_score"),
                "val_parent": r.get("parent_val_score"),
            }
            if rec["status"] == "accepted" and rec["val_child"] is not None and rec["val_parent"] is not None:
                rec["val_improved"] = rec["val_child"] > rec["val_parent"]
                rec["val_delta"] = rec["val_child"] - rec["val_parent"]
            props.append(rec)
    return props


def criterion_table(acc: list[dict]) -> None:
    print("\n=== Retrospective stricter gates (accepted set only) ===")
    print("criterion                          kept_good blocked_bad blocked_good kept_bad")

    def evaluate(kfn, label):
        kept = [p for p in acc if kfn(p)]
        blocked = [p for p in acc if not kfn(p)]
        kg = sum(1 for p in kept if p.get("val_improved"))
        kb = sum(1 for p in kept if p.get("val_improved") is False)
        bg = sum(1 for p in blocked if p.get("val_improved"))
        bb = sum(1 for p in blocked if p.get("val_improved") is False)
        prec = kg / max(1, kg + kb)
        print(f"{label:<34} {kg:>9} {bb:>11} {bg:>12} {kb:>8}   precision_of_kept={prec:.2f}")

    n = acc[0]["n"] if acc else 20
    for k in sorted({round(f * n) for f in (0.2, 0.3, 0.4, 0.5, 0.6)}):
        evaluate(lambda p, k=k: p["wins"] >= k, f"wins>={k}/{n} (+native sum)")
    for tau in (0.1, 0.25, 0.5, 1.0):
        evaluate(lambda p, tau=tau: p["margin"] > tau, f"margin>tau={tau}")


def load_task_bins(manifest_path: Path) -> dict[str, int]:
    manifest = json.loads(manifest_path.read_text())
    task_bin: dict[str, int] = {}
    for b, entry in manifest["bins"].items():
        for key in ("train_pool", "val", "finalist"):
            for tid in entry.get(key, []):
                task_bin[tid] = int(b)
    return task_bin


def subsample_decisions(props: list[dict], task_bin: dict[str, int], tasks_per_bin: int, draws: int, rng: random.Random) -> None:
    """P(accept | smaller gate) per proposal via task-stratified subsampling."""
    print(f"\n=== Simulated smaller gate: {tasks_per_bin} task(s)/bin, {draws} draws/proposal ===")
    print(f"{'iter':>4} {'status':>9} {'full_margin':>11} {'P(accept)':>10}  (native sum rule on subsampled cells)")
    for p in props:
        by_task: dict[str, list[int]] = {}
        for i, cid in enumerate(p["cell_ids"]):
            by_task.setdefault(cid.split(CELL_KEY_SEPARATOR)[0], []).append(i)
        bins: dict[int, list[str]] = {}
        for t in by_task:
            b = task_bin.get(t)
            if b is None:
                continue
            bins.setdefault(b, []).append(t)
        if any(len(ts) < tasks_per_bin for ts in bins.values()) or not bins:
            continue
        accepts = 0
        for _ in range(draws):
            idxs: list[int] = []
            for ts in bins.values():
                for t in rng.sample(ts, tasks_per_bin):
                    idxs.extend(by_task[t])
            m = sum(p["child_scores"][i] for i in idxs) - sum(p["parent_scores"][i] for i in idxs)
            accepts += m > 0
        print(f"{p['iter']:>4} {p['status']:>9} {p['margin']:>11.4f} {accepts / draws:>10.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--manifest", default=None, help="task manifest for bin-stratified subsampling")
    ap.add_argument("--subsample-tasks-per-bin", nargs="*", type=int, default=[], help="simulate smaller gates")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    props = load_proposals(run_dir)
    acc = [p for p in props if p["status"] == "accepted"]
    rej = [p for p in props if p["status"] == "rejected"]
    print(f"{run_dir}: {len(props)} proposals with usable gate data (accepted {len(acc)}, rejected {len(rej)})")

    good = [p for p in acc if p.get("val_improved") is True]
    bad = [p for p in acc if p.get("val_improved") is False]
    if good or bad:
        print(f"native gate quality: {len(good)}/{len(good) + len(bad)} accepts genuinely improved val "
              f"(agreement {len(good) / max(1, len(good) + len(bad)):.2f})")
        print(f"good accepts: wins {sorted(p['wins'] for p in good)} | margins "
              f"{sorted(round(p['margin'], 3) for p in good)}")
        print(f"bad  accepts: wins {sorted(p['wins'] for p in bad)} | margins "
              f"{sorted(round(p['margin'], 3) for p in bad)}")
        if good and bad:
            print(f"margin medians: good {statistics.median(p['margin'] for p in good):.3f} "
                  f"vs bad {statistics.median(p['margin'] for p in bad):.3f}")
        criterion_table(acc)

    if args.subsample_tasks_per_bin:
        if not args.manifest:
            ap.error("--subsample-tasks-per-bin requires --manifest")
        task_bin = load_task_bins(Path(args.manifest))
        rng = random.Random(args.seed)
        for tpb in args.subsample_tasks_per_bin:
            subsample_decisions(props, task_bin, tpb, args.draws, rng)

    print("\nCAVEATS: 'val improved' is a noisy 84-cell estimate; rejected proposals have no val truth;")
    print("trajectory counterfactuals are not recoverable — this measures per-decision quality only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
