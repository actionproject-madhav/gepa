#!/usr/bin/env python3
"""One-shot evaluation on the 27-task confirmatory holdout (pre-registered).

The 27 tasks are every CVEBench/CyberGym task that (a) carries full 11-panel
outcomes in the Lyptus corpus, (b) has non-empty estimation_instructions, and
(c) was NOT part of the frozen 94-task reserved test set spent on 2026-09-01
(they were excluded from the headline set only for lacking a curated
best_available_minutes difficulty label). No LLM has ever been prompted on
them. Resolution rule asserted at runtime: 7 cvebench + 20 cybergym.

Prompt construction: the evidence side of every reserved-test prompt depends
only on (forecasted model, target bin) — verified byte-identical across
targets — so each holdout cell reuses the frozen test-phase plan for its
(model, proxy bin) and swaps in the holdout target task. Difficulty ruler
(pre-registered): model_estimate_minutes mapped through the standard bin
edges (0.46, 2.81, 12.82, 60, 180, 2160], clipped to the test-template range
[1, 4]; the target's time/bin is never shown to the forecaster (unchanged).

Measurement conditions match the spent test exactly: config
pilot_baseline_clean.yaml (temperature 0.0), 2 repeats, failure cells
recorded not halted. Output: runs/noise_study/noise_cells_holdout27.jsonl
(set="holdout27", same row schema as the test logs).

Usage (from the gepa repo root):
    uv run python scripts/holdout27_run.py --limit-cells 2 --repeats 1  # smoke
    uv run python scripts/holdout27_run.py                              # full
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from forecaster_gepa.adapter import ForecasterAdapter  # noqa: E402
from forecaster_gepa.config import COMPONENT_NAME, load_config  # noqa: E402
from forecaster_gepa.data import CellSpec, ExperimentData, cell_key  # noqa: E402
from intra_benchmark_calibration.lyptus_data import LyptusTask  # noqa: E402

EDGES = [0.46, 2.81, 12.82, 60.0, 180.0, 2160.0]
FAM_DIRS = {"cvebench": "cvebench", "cybergym": "cybergym"}


def proxy_bin(minutes: float) -> int:
    b = int(np.searchsorted(np.array(EDGES[1:]), minutes, side="left"))
    return min(max(b, 1), 4)  # clip to the bins test templates exist for


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO / "configs/pilot_baseline_clean.yaml"))
    ap.add_argument("--prompts", default="seed", help="comma-separated stems from configs/noise_study_prompts")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--limit-cells", type=int, default=None, help="smoke-testing only")
    ap.add_argument("--tag", default="holdout27")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.halt_on_cell_failure = False
    data = ExperimentData(cfg)
    raw = Path(cfg.resolved("lyptus_repo_dir"))

    # --- resolve the 27 holdout tasks by rule -----------------------------
    d = raw / "analysis/figures/data"
    td = pd.read_parquet(d / "task_difficulties.parquet").set_index("task_id")
    mr = pd.read_parquet(d / "model_runs.parquet")
    headline = set(td.dropna(subset=["best_available_minutes"]).index) & set(mr["task_id"])
    spent_test = set(data.manifest["test"]["task_ids"])

    meta = {}
    for fam, dname in FAM_DIRS.items():
        for fp in (raw / "data/tasks" / dname).glob("*_tasks.jsonl"):
            for line in open(fp):
                if line.strip():
                    r = json.loads(line)
                    m = r.get("dataset_task_metadata") or {}
                    ei = str(m.get("estimation_instructions") or "").strip()
                    if ei:
                        meta[r["task_id"]] = {"family": fam, "ei": ei}

    holdout_ids = sorted(
        t for t in meta
        if t not in headline and t not in spent_test and t in set(mr["task_id"])
    )
    fams = pd.Series([meta[t]["family"] for t in holdout_ids]).value_counts().to_dict()
    print(f"holdout tasks: {len(holdout_ids)} {fams}")
    assert len(holdout_ids) == 27 and fams == {"cybergym": 20, "cvebench": 7}, "resolution rule drifted"
    assert not (set(holdout_ids) & spent_test)

    # --- outcomes + proxy bins -------------------------------------------
    panel = list(cfg.forecasted_models_full)
    runs = mr[mr["task_id"].isin(holdout_ids) & mr["alias"].isin(panel)]
    y = {(r.task_id, r.alias): float(r.score_binarized) for r in runs.itertuples()}
    print(f"outcome cells available: {len(y)} of {27 * len(panel)} possible")

    # --- clone test-phase plans with swapped targets ----------------------
    tplans, tspecs = data.test_phase()
    template_by = {}
    for plan in tplans.values():
        template_by.setdefault((plan.forecasted_model, plan.target_bin_j), plan)
    print(f"templates: {len(template_by)} (model x bin) combos from the frozen test phase")

    plans, specs, clipped = {}, [], []
    for tid in holdout_ids:
        minutes = float(td.at[tid, "model_estimate_minutes"])
        rawbin = int(np.searchsorted(np.array(EDGES[1:]), minutes, side="left"))
        binj = proxy_bin(minutes)
        if rawbin != binj:
            clipped.append((tid, rawbin, binj))
        task = LyptusTask(
            task_id=tid, task_family=meta[tid]["family"], fst_minutes=minutes,
            fst_source="model_estimate_proxy", estimation_instructions=meta[tid]["ei"],
            solution_walkthrough=None,
        )
        for m in panel:
            if (tid, m) not in y:
                continue
            tpl = template_by.get((m, binj))
            assert tpl is not None, (m, binj)
            plan = dataclasses.replace(
                tpl, target_task=task, target_outcome=y[(tid, m)], target_bin_j=binj
            )
            key = cell_key(tid, m)
            plans[key] = plan
            specs.append(CellSpec(cell_key=key, task_id=tid, model=m, bin=binj,
                                  outcome=y[(tid, m)]))
    print(f"cells: {len(specs)}; proxy-bin clips: {clipped or 'none'}")
    if args.limit_cells:
        specs = specs[: args.limit_cells]

    adapter = ForecasterAdapter(plans, cfg)
    out = REPO / "runs/noise_study"
    out.mkdir(parents=True, exist_ok=True)
    cells_path = out / f"noise_cells_{args.tag}.jsonl"

    prompt_dir = REPO / "configs/noise_study_prompts"
    for stem in [s.strip() for s in args.prompts.split(",")]:
        template = (prompt_dir / f"{stem}.txt").read_text(encoding="utf-8")
        for rep in range(args.repeats):
            eb = adapter.evaluate(specs, {COMPONENT_NAME: template}, capture_traces=False)
            vals = [o["brier"] if o["brier"] is not None else 1.0 for o in eb.outputs]
            n_fail = sum(1 for o in eb.outputs if o["error"])
            with cells_path.open("a", encoding="utf-8") as fh:
                for o in eb.outputs:
                    fh.write(json.dumps({"prompt": stem, "set": "holdout27", "repeat": rep, **o}) + "\n")
            print(f"{stem} repeat {rep + 1}/{args.repeats}: grand Brier "
                  f"{sum(vals) / len(vals):.4f} ({len(vals)} cells, {n_fail} failures)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
