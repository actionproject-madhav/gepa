#!/usr/bin/env python3
"""LLM re-tagging of the five prompt features, as a robustness check on the
deterministic keyword rubric in feature_report_data.py (Jakub's request).

Claude Sonnet 4.6 at temperature 0 reads each evolved prompt and answers
yes/no for each feature definition (definitions phrased functionally, not as
keywords). Output: per-feature agreement with the rubric + every
disagreement listed, and the accuracy screen recomputed under LLM tags.

Usage:  uv run python scripts/feature_tags_llm_check.py
Cost:   82 calls, ~$2.
"""
from __future__ import annotations

import asyncio
import json
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from shared.llm_client import LLMSettings, initialize_client, make_api_call  # noqa: E402
from intra_benchmark_calibration.estimation_api import resolve_anthropic_api_key  # noqa: E402

FEATURES = {
    "empirical_anchoring": "instructs the forecaster to base/anchor its probability on the empirical pass rates or observed outcomes shown in the evidence",
    "nearest_example": "instructs that a single closely-matching example task (a 'near twin') should be weighted above, or used to correct, the group/cluster average",
    "numeric_bands": "prescribes explicit numeric probability ranges or values for kinds of tasks (e.g. 'trivial tasks: 0.92-0.99')",
    "anti_hedging": "instructs the forecaster not to default to 0.5 (or the midpoint) when unsure, or demands decisive estimates",
    "task_type_rules": "instructs different treatment by task type, e.g. interactive/live-service vs static file-analysis tasks",
}

PROMPT = """You will read one forecasting prompt template. For each of five features, answer whether the TEMPLATE TEXT ITSELF contains an instruction with that meaning (not merely a placeholder that might inject such content).

Features:
{feats}

Template:
<template>
{template}
</template>

Answer with EXACTLY five lines, one per feature, format `feature_name: yes` or `feature_name: no`. No other text."""


async def main() -> int:
    data = json.load(open(REPO / "runs/feature_report_data.json"))
    rows = [r for r in data["tags"] if not r["is_seed"]]
    from gepa.core.state import GEPAState
    from forecaster_gepa.config import COMPONENT_NAME
    texts = {}
    for run in {r["run"] for r in rows}:
        st = GEPAState.load(str(REPO / "runs" / run))
        for r in rows:
            if r["run"] == run:
                texts[(run, r["idx"])] = st.program_candidates[r["idx"]][COMPONENT_NAME]

    settings = LLMSettings(model="claude-sonnet-4-6", temperature=0.0,
                           max_concurrent_calls=8, rate_limit_calls=200,
                           rate_limit_period=60, reasoning_effort="off")
    client = initialize_client(api_key_anthropic=resolve_anthropic_api_key(),
                               api_key_openai=None, model=settings.model)
    sem = asyncio.Semaphore(settings.max_concurrent_calls)
    feats_txt = "\n".join(f"- {k}: {v}" for k, v in FEATURES.items())

    async def tag(row):
        text = texts[(row["run"], row["idx"])]
        resp = await make_api_call(client, sem, settings, "",
                                   PROMPT.format(feats=feats_txt, template=text), 2000)
        out = {}
        for name in FEATURES:
            m = re.search(rf"{name}\s*:\s*(yes|no)", resp, re.I)
            out[name] = (m.group(1).lower() == "yes") if m else None
        return row, out

    results = await asyncio.gather(*[tag(r) for r in rows])
    close = getattr(client, "close", None)
    if close:
        await close()

    print(f"{len(results)} prompts re-tagged by LLM\n")
    print(f"{'feature':<22} {'agree':>7} {'rubric_only':>11} {'llm_only':>9}")
    llm_tags = {}
    for name in FEATURES:
        a = ro = lo = 0
        for row, out in results:
            r_tag, l_tag = row[name], out[name]
            llm_tags[(row["run"], row["idx"], name)] = l_tag
            if l_tag is None:
                continue
            if r_tag == l_tag:
                a += 1
            elif r_tag:
                ro += 1
            else:
                lo += 1
        print(f"{name:<22} {a:>4}/82 {ro:>11} {lo:>9}")

    print("\naccuracy screen under LLM tags (within-run pooled delta, with - without):")
    runs = sorted({r['run'] for r, _ in results})
    for name in FEATURES:
        deltas, weights = [], []
        for run in runs:
            w = [r["val"] for r, o in results if r["run"] == run and o[name]]
            wo = [r["val"] for r, o in results if r["run"] == run and o[name] is False]
            if len(w) >= 2 and len(wo) >= 2:
                deltas.append(statistics.mean(w) - statistics.mean(wo))
                weights.append(min(len(w), len(wo)))
        d = (sum(d * n for d, n in zip(deltas, weights)) / sum(weights)) if deltas else None
        print(f"  {name:<22} {'%+.4f' % d if d is not None else '--- (one-sided)'}")

    out_path = REPO / "runs/feature_tags_llm.json"
    out_path.write_text(json.dumps(
        [{"run": r["run"], "idx": r["idx"], **{n: o[n] for n in FEATURES}}
         for r, o in results], indent=1))
    print(f"\nwritten: {out_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
