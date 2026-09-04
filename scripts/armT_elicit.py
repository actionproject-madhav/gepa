#!/usr/bin/env python3
"""ARM T: evidence-matched time elicitation (pre-registered head-to-head).

The strict minimal pair against the seed prompt: identical skeleton, identical
example tasks per difficulty bin (the modal anchor + 2 easier roster from the
frozen test-phase plans, byte-identical across targets sharing a bin), but the
examples are labeled with their estimated human solve times (minutes, printed
verbatim from fst_minutes) instead of SOLVED/FAILED tags and pass rates, the
model name is absent (time is model-independent), and the ask is the target's
human solve time as three percentiles in the seed's XML shape. One call per
TASK, 3 repeats, temperature 0, claude-sonnet-4-6 — matching the D-arm's
estimator so the T-vs-D contrast isolates the evidence pack.

Shown bins = all except the target's own bin (the seed's all_except_target
structure). Target bin: curated-FST bin for train/sealed/test tasks; the
holdout27 tasks use holdout27_run.py's ME-proxy rule verbatim.

Leakage guards asserted at runtime: the target never appears among the
example tasks; the target's own time never appears in its prompt; all
example tasks are train-family.

Usage (gepa repo root):
    uv run python scripts/armT_elicit.py --limit-tasks 2 --repeats 1  # smoke
    uv run python scripts/armT_elicit.py                              # full
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from asyncio import Semaphore
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from forecaster_gepa.config import load_config  # noqa: E402
from forecaster_gepa.data import ExperimentData  # noqa: E402
from intra_benchmark_calibration.estimation_api import resolve_anthropic_api_key  # noqa: E402
from shared.llm_client import LLMSettings, initialize_client, make_api_call  # noqa: E402

MODEL = "claude-sonnet-4-6"
EDGES = [0.46, 2.81, 12.82, 60.0, 180.0, 2160.0]
FAM_DIRS = {"cvebench": "cvebench", "cybergym": "cybergym"}

# FROZEN template (committed in runs/headtohead_prereg.md before launch).
# Line 1 folds the D-arm persona into the user turn; the system prompt is
# empty, mirroring the seed harness (config.py system_prompt="").
USER_TEMPLATE = """You are an experienced cybersecurity professional estimating how long \
security tasks take skilled humans to complete.

Below are tasks from the same benchmark suite, grouped by difficulty level, \
each tagged with its estimated human solve time:

{evidence_block}

Target task:
{target_task_text}

Estimate how long a skilled human professional (an expert in the relevant \
specialty, working alone, with standard professional tooling and internet \
access) would need to complete the target task, from first reading the \
specification to delivering the required artifact. Estimate the human effort \
the task itself demands. Do not consider whether an AI system could do it, \
and do not give any probability. Answer in minutes; you should be roughly \
90% confident the true value lies between your p25 and p75... report three \
percentiles of your subjective distribution over the required time using \
EXACTLY this format:

<rationale>
Your rationale here.
</rationale>

<percentile_estimates>
<p25_minutes>number</p25_minutes>
<p50_minutes>number</p50_minutes>
<p75_minutes>number</p75_minutes>
</percentile_estimates>"""


def parse_reply(text: str):
    def grab(tag):
        m = re.findall(rf"<{tag}>\s*([0-9][0-9,.]*)\s*</{tag}>", text or "")
        return float(m[-1].replace(",", "")) if m else None

    lo, mid, hi = grab("p25_minutes"), grab("p50_minutes"), grab("p75_minutes")
    if lo and mid and hi and 0 < lo <= mid <= hi:
        return lo, mid, hi
    return None


def proxy_bin(minutes: float) -> int:
    b = int(np.searchsorted(np.array(EDGES[1:]), minutes, side="left"))
    return min(max(b, 1), 4)


def build_roster(data) -> dict[int, list]:
    """Modal (anchor, easier1, easier2) per source bin over the frozen
    test-phase plans (identical across models in bins 0-3; bin 4 has one
    GLM-5 variant — the mode wins, footnoted in the prereg)."""
    tplans, _ = data.test_phase()
    per_bin: dict[int, Counter] = {}
    for plan in tplans.values():
        for p in plan.profiles:
            ids = (p.anchor.task_id, tuple(t.task_id for t in p.easier_tasks))
            per_bin.setdefault(p.bin_index, Counter())[ids] += 1
    task_by_id = {t.task_id: t for t in data.dataset.tasks}
    roster = {}
    for b, ctr in per_bin.items():
        anchor_id, easier_ids = ctr.most_common(1)[0][0]
        roster[b] = [task_by_id[anchor_id]] + [task_by_id[i] for i in easier_ids]
    assert set(roster) == {0, 1, 2, 3, 4}
    return roster


def render_block(task, minutes: float) -> str:
    body = task.estimation_instructions.strip().replace("\n", "\n    ")
    return (f"    --- (task_id={task.task_id}) ---\n"
            f"    [estimated human solve time: {minutes:.1f} minutes]\n"
            f"    {body}")


def render_evidence(roster, target_bin: int, fst_of) -> str:
    chunks = []
    for b in sorted(roster):
        if b == target_bin:
            continue
        chunks.append(f"=== Source bin {b} ===\n")
        labels = ["ANCHOR (representative of this bin)", "Easier example #1",
                  "Easier example #2"]
        for lab, t in zip(labels, roster[b]):
            chunks.append(render_block(t, fst_of[t.task_id]).replace(
                "--- (", f"--- {lab} (", 1))
            chunks.append("")
    return "\n".join(chunks).rstrip()


def collect_targets(cfg):
    data = ExperimentData(cfg)
    raw = Path(cfg.resolved("lyptus_repo_dir"))
    test_ids = set(data.manifest["test"]["task_ids"])
    sealed_ids = {tid for entry in data.manifest["bins"].values()
                  for tid in entry["finalist"]]
    binned = np.searchsorted(np.array(EDGES[1:]),
                             [t.fst_minutes for t in data.dataset.tasks], side="left")
    bin_of = {t.task_id: int(b) for t, b in zip(data.dataset.tasks, binned)}
    rows = []
    for t in data.dataset.tasks:
        which = ("test" if t.task_id in test_ids
                 else "sealed" if t.task_id in sealed_ids else "train_fit")
        rows.append({"task_id": t.task_id, "set": which,
                     "text": t.estimation_instructions, "bin": bin_of[t.task_id]})
    d = raw / "analysis/figures/data"
    td = pd.read_parquet(d / "task_difficulties.parquet").set_index("task_id")
    mr = pd.read_parquet(d / "model_runs.parquet")
    headline = set(td.dropna(subset=["best_available_minutes"]).index) & set(mr["task_id"])
    have = {r["task_id"] for r in rows}
    n_hold = 0
    for fam, dname in FAM_DIRS.items():
        for fp in (raw / "data/tasks" / dname).glob("*_tasks.jsonl"):
            for line in open(fp):
                if not line.strip():
                    continue
                r = json.loads(line)
                m = r.get("dataset_task_metadata") or {}
                ei = str(m.get("estimation_instructions") or "").strip()
                tid = r["task_id"]
                if (ei and tid not in headline and tid not in have
                        and tid in set(mr["task_id"])):
                    rows.append({"task_id": tid, "set": "holdout27", "text": ei,
                                 "bin": proxy_bin(float(td.at[tid, "model_estimate_minutes"]))})
                    n_hold += 1
    assert n_hold == 27, n_hold
    counts = Counter(r["set"] for r in rows)
    assert counts["test"] == 94 and counts["sealed"] == 21, counts
    assert len(rows) == len({r["task_id"] for r in rows}), "duplicate task ids"
    return data, rows


async def run(prompts, repeats, out_path):
    client = initialize_client(resolve_anthropic_api_key(), None, MODEL)
    settings = LLMSettings(model=MODEL, temperature=0.0,
                           max_concurrent_calls=8, rate_limit_calls=45)
    sem = Semaphore(8)
    done = 0

    async def one(item, rep):
        nonlocal done
        row = {"task_id": item["task_id"], "set": item["set"], "repeat": rep,
               "low_minutes": None, "mid_minutes": None, "high_minutes": None,
               "error": None}
        for _attempt in range(2):
            try:
                reply = await make_api_call(client, sem, settings, "",
                                            item["prompt"], max_tokens=2000)
            except Exception as e:
                row["error"] = f"api:{type(e).__name__}"
                break
            parsed = parse_reply(reply or "")
            if parsed:
                row["low_minutes"], row["mid_minutes"], row["high_minutes"] = parsed
                row["error"] = None
                break
            row["error"] = "parse"
        done += 1
        if done % 50 == 0:
            print(f"  {done} calls done", flush=True)
        return row

    rows = await asyncio.gather(*[one(t, r) for r in range(repeats) for t in prompts])
    with open(out_path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    n_err = sum(1 for r in rows if r["error"])
    print(f"wrote {len(rows)} rows ({n_err} errors) -> {out_path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO / "configs/pilot_baseline_clean.yaml"))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--limit-tasks", type=int, default=None)
    ap.add_argument("--tag", default="armT_times")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data, targets = collect_targets(cfg)
    roster = build_roster(data)
    roster_ids = {t.task_id for tasks in roster.values() for t in tasks}
    fst_of = {t.task_id: t.fst_minutes for t in data.dataset.tasks}
    fam_of = {t.task_id: t.task_family for t in data.dataset.tasks}
    assert all(fam_of[i] in {"cybashbench", "nl2bash", "intercode_ctf", "nyuctf",
                             "cybench"} for i in roster_ids), "non-train evidence task"

    rendered_by_bin = {}
    prompts = []
    for r in targets:
        assert r["task_id"] not in roster_ids, f"target in evidence: {r['task_id']}"
        ev = rendered_by_bin.setdefault(r["bin"],
                                        render_evidence(roster, r["bin"], fst_of))
        prompt = USER_TEMPLATE.format(evidence_block=ev, target_task_text=r["text"].strip())
        if r["task_id"] in fst_of:  # holdout tasks have no curated fst
            assert f"{fst_of[r['task_id']]:.1f} minutes" not in prompt.split("Target task:")[1], \
                f"target time leaked: {r['task_id']}"
        prompts.append({**r, "prompt": prompt})
    # byte-identity guarantee: one rendered block per bin, reused for all
    # targets in that bin (rendered_by_bin construction enforces it).
    print(f"targets: {len(prompts)} {Counter(p['set'] for p in prompts)}; "
          f"evidence blocks: {len(rendered_by_bin)} bins; roster tasks: {sorted(roster_ids)}")
    if args.limit_tasks:
        prompts = prompts[: args.limit_tasks]

    out = REPO / "runs/noise_study"
    out.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(prompts, args.repeats, out / f"noise_cells_{args.tag}.jsonl"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
