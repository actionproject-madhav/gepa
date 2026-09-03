#!/usr/bin/env python3
"""D-arm step 1: elicit human solve-time estimates per task (pre-registered).

One call per TASK (not per model-task cell). Input to the LLM: the task's
estimation_instructions text and nothing else — no FST, no bin, no family,
no pass rates, no example tasks, no outcomes (leakage statement in
runs/darm_prereg.md). Estimator: claude-sonnet-4-6, temperature 0.0,
reasoning off — the identical model/settings every probability arm used, so
the head-to-head isolates the question asked.

Tasks: all usable train-family tasks + the frozen 94-task reserved test +
the 27-task holdout (resolution rules identical to prior runs). 3 repeats.
Output: runs/noise_study/darm_times.jsonl rows
  {task_id, set, repeat, low_minutes, mid_minutes, high_minutes, error}

Usage (gepa repo root):
    uv run python scripts/darm_elicit.py --limit-tasks 2 --repeats 1  # smoke
    uv run python scripts/darm_elicit.py                              # full
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from asyncio import Semaphore
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from forecaster_gepa.config import load_config  # noqa: E402
from forecaster_gepa.data import ExperimentData  # noqa: E402
from intra_benchmark_calibration.estimation_api import resolve_anthropic_api_key  # noqa: E402
from shared.llm_client import LLMSettings, initialize_client, make_api_call  # noqa: E402

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = (
    "You are an experienced cybersecurity professional estimating how long "
    "security tasks take skilled humans to complete."
)

# FROZEN elicitation template (committed in runs/darm_prereg.md before launch).
USER_TEMPLATE = """Below is the specification of one cybersecurity task.

Estimate how long a skilled human professional (an expert in the relevant \
specialty, working alone, with standard professional tooling and internet \
access) would need to complete this task, from first reading the \
specification to delivering the required artifact.

Rules:
- Estimate the human effort the task itself demands. Do not consider whether \
an AI system could do it, and do not give any probability.
- Answer in minutes. Give a plausible low value, your best guess, and a \
plausible high value (you should be roughly 90% confident the true value \
lies between low and high).
- Reply with ONLY a JSON object on one line, no other text:
{{"low_minutes": <number>, "mid_minutes": <number>, "high_minutes": <number>}}

TASK SPECIFICATION:
{task_text}"""

FAM_DIRS = {"cvebench": "cvebench", "cybergym": "cybergym"}


def parse_reply(text: str):
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        lo, mid, hi = (float(d["low_minutes"]), float(d["mid_minutes"]),
                       float(d["high_minutes"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not (0 < lo <= mid <= hi):
        return None
    return lo, mid, hi


def collect_tasks(cfg) -> list[dict]:
    data = ExperimentData(cfg)
    raw = Path(cfg.resolved("lyptus_repo_dir"))
    rows = []
    test_ids = set(data.manifest["test"]["task_ids"])
    for t in data.dataset.tasks:
        which = "test" if t.task_id in test_ids else "train"
        rows.append({"task_id": t.task_id, "set": which,
                     "text": t.estimation_instructions})
    # holdout-27: same resolution rule as holdout27_run.py
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
                    rows.append({"task_id": tid, "set": "holdout27", "text": ei})
                    n_hold += 1
    assert n_hold == 27, f"holdout rule drifted: {n_hold}"
    # leakage guard: the prompt is built from `text` only.
    for r in rows:
        assert r["text"].strip(), r["task_id"]
    return rows


async def run(tasks, repeats, out_path):
    client = initialize_client(resolve_anthropic_api_key(), None, MODEL)
    settings = LLMSettings(model=MODEL, temperature=0.0,
                           max_concurrent_calls=8, rate_limit_calls=45)
    sem = Semaphore(8)
    done = 0

    async def one(task, rep):
        nonlocal done
        prompt = USER_TEMPLATE.format(task_text=task["text"])
        row = {"task_id": task["task_id"], "set": task["set"], "repeat": rep,
               "low_minutes": None, "mid_minutes": None, "high_minutes": None,
               "error": None}
        for attempt in range(2):  # one parse retry
            try:
                reply = await make_api_call(client, sem, settings,
                                            SYSTEM_PROMPT, prompt, max_tokens=300)
            except Exception as e:  # client retries exhausted
                row["error"] = f"api:{type(e).__name__}"
                break
            parsed = parse_reply(reply or "")
            if parsed:
                row["low_minutes"], row["mid_minutes"], row["high_minutes"] = parsed
                break
            row["error"] = "parse"
        done += 1
        if done % 50 == 0:
            print(f"  {done} calls done", flush=True)
        return row

    jobs = [one(t, r) for r in range(repeats) for t in tasks]
    rows = await asyncio.gather(*jobs)
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
    ap.add_argument("--tag", default="darm_times")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tasks = collect_tasks(cfg)
    counts = pd.Series([t["set"] for t in tasks]).value_counts().to_dict()
    print(f"tasks: {len(tasks)} {counts}")
    if args.limit_tasks:
        tasks = tasks[: args.limit_tasks]
    out = REPO / "runs/noise_study" / f"noise_cells_{args.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(tasks, args.repeats, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
