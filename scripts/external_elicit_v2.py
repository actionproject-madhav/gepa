#!/usr/bin/env python3
"""External elicitation v2: standardized-quantile time template
(pre-registered: runs/external_v2_prereg.md).

The template mirrors the seed probability prompt structure EXACTLY — same
opening-role sentence pattern, same "Target task:" block, same ask-sentence
shape, same <rationale> + <percentile_estimates><p25><p50><p75> scaffold,
empty system prompt — with only the subject swapped from probability to
human solve time in minutes, and explicit 25th/50th/75th percentiles
(replacing v1's "~90% confident low/high" interval). No anchors. Same 270
frozen stripped inputs as v1 (runs/external_corpora/*_tasks.jsonl), so
v2-vs-v1 is a paired wording-robustness replication.

Usage:
    uv run python scripts/external_elicit_v2.py --limit-tasks 2 --repeats 1  # smoke
    uv run python scripts/external_elicit_v2.py                              # full
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

from intra_benchmark_calibration.estimation_api import resolve_anthropic_api_key  # noqa: E402
from shared.llm_client import LLMSettings, initialize_client, make_api_call  # noqa: E402
from external_elicit import fetch_secbench, fetch_airtbench  # noqa: E402

# FROZEN v2 template — seed-prompt structure, subject swapped to time.
USER_TEMPLATE_V2 = """You are estimating how long a skilled human professional would need to solve a target task under this benchmark's standard evaluation.

Target task:
{task_text}

Estimate the time in minutes that a skilled human professional (an expert in the relevant specialty, working alone, with standard professional tooling and internet access) would need to solve the target task. Report three percentiles of your subjective distribution over the required time, in minutes, using EXACTLY this format:

<rationale>
Your rationale here.
</rationale>

<percentile_estimates>
<p25>x</p25>
<p50>x</p50>
<p75>x</p75>
</percentile_estimates>"""


def parse_reply_v2(text: str):
    def grab(tag):
        m = re.findall(rf"<{tag}>\s*([0-9][0-9,.]*)\s*</{tag}>", text or "")
        return float(m[-1].replace(",", "")) if m else None

    lo, mid, hi = grab("p25"), grab("p50"), grab("p75")
    if lo and mid and hi and 0 < lo <= mid <= hi:
        return lo, mid, hi
    return None


async def run(tasks: pd.DataFrame, repeats: int, out_path: Path):
    key = resolve_anthropic_api_key()
    sem = Semaphore(8)
    done = 0

    async def one(client, settings, item, elicitor, rep):
        nonlocal done
        prompt = USER_TEMPLATE_V2.format(task_text=item["text"])
        row = {"corpus": item["corpus"], "task_id": item["task_id"],
               "elicitor": elicitor, "repeat": rep, "low_minutes": None,
               "mid_minutes": None, "high_minutes": None, "error": None}
        for _ in range(2):
            try:
                reply = await make_api_call(client, sem, settings, "",
                                            prompt, max_tokens=2000)
            except Exception as e:
                row["error"] = f"api:{type(e).__name__}"
                break
            parsed = parse_reply_v2(reply or "")
            if parsed:
                row["low_minutes"], row["mid_minutes"], row["high_minutes"] = parsed
                row["error"] = None
                break
            row["error"] = "parse"
        done += 1
        if done % 100 == 0:
            print(f"  {done} calls", flush=True)
        return row

    items = tasks.to_dict("records")
    for model in ["claude-sonnet-4-6", "claude-haiku-4-5"]:
        client = initialize_client(key, None, model)
        settings = LLMSettings(model=model, temperature=0.0,
                               max_concurrent_calls=8, rate_limit_calls=45)
        rows = await asyncio.gather(*[one(client, settings, t, model, r)
                                      for r in range(repeats) for t in items])
        with open(out_path, "a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        n_err = sum(1 for r in rows if r["error"])
        print(f"{model}: wrote {len(rows)} rows ({n_err} errors)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--limit-tasks", type=int, default=None)
    args = ap.parse_args()
    tasks = pd.concat([fetch_secbench(), fetch_airtbench()], ignore_index=True)
    assert len(tasks) == 270
    print(f"inputs: {len(tasks)} (frozen v1 files reused byte-identically)")
    if args.limit_tasks:
        tasks = tasks.groupby("corpus").head(args.limit_tasks).reset_index(drop=True)
    out = REPO / "runs/noise_study/noise_cells_external_times_v2.jsonl"
    asyncio.run(run(tasks, args.repeats, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
