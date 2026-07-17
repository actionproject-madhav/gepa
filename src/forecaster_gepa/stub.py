# Copyright (c) 2026 SaferAI. GEPA forecaster-optimisation experiment.
"""Deterministic stubs for dry runs: no API calls, fully reproducible.

The stub forecaster produces a pseudo-random but deterministic p50 per
(template, cell) pair, centred near the panel base rate so the seed template
scores reasonably. A template containing `BAD_PROMPT_MARKER` forecasts 0.99
everywhere, which is strictly worse in expected Brier — used to verify the
metric sign end-to-end. The stub reflection LM applies a deterministic
mutation to the current template.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from intra_benchmark_calibration.estimation_api import CellResult
from intra_benchmark_calibration.task_selector import CellPlan

BAD_PROMPT_MARKER = "always report p50 = 0.99"


def _unit_hash(*parts: str) -> float:
    digest = hashlib.sha256("||".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def stub_estimate_cells(plans: list[CellPlan], template: str) -> list[CellResult]:
    """Deterministic fake forecasts, one per plan (mirrors estimate_cells)."""
    results: list[CellResult] = []
    bad = BAD_PROMPT_MARKER in template.lower()
    for plan in plans:
        if bad:
            p50 = 0.99
        else:
            jitter = _unit_hash(template, plan.cell_id) - 0.5
            p50 = min(0.95, max(0.05, 0.7 + 0.3 * jitter))
        p25 = max(0.01, p50 - 0.1)
        p75 = min(0.99, p50 + 0.1)
        brier = (p50 - plan.target_outcome) ** 2
        results.append(
            CellResult(
                cell_id=plan.cell_id,
                target_task_id=plan.target_task.task_id,
                target_bin=plan.target_bin_j,
                forecasted_model=plan.forecasted_model,
                outcome=plan.target_outcome,
                p25=p25,
                p50=p50,
                p75=p75,
                brier=brier,
                rationale="(stub rationale)",
                system_prompt="",
                user_prompt=f"(stub prompt for {plan.cell_id})",
                raw_response=(
                    f"(stub rollout)\n<rationale>\nStub.\n</rationale>\n\n"
                    f"<percentile_estimates>\n<p25>{p25:.2f}</p25>\n<p50>{p50:.2f}</p50>\n"
                    f"<p75>{p75:.2f}</p75>\n</percentile_estimates>"
                ),
                error=None,
                duration_seconds=0.0,
            )
        )
    return results


class StubReflectionLM:
    """LanguageModel-protocol callable: deterministic template mutation.

    Extracts the current template from the reflection prompt (the first
    ``` block, where the default proposal template places <curr_param>) and
    appends a deterministic marker line, returning it inside ``` blocks as
    the instruction-proposal extractor expects.
    """

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        text = prompt if isinstance(prompt, str) else str(prompt)
        match = re.search(r"```\n(.*?)\n```", text, re.DOTALL)
        current = match.group(1) if match else "(no template found)"
        tag = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        mutated = (
            current.rstrip() + f"\n\nCalibration note (stub mutation {tag}): weigh the pass-rate evidence carefully."
        )
        return f"```\n{mutated}\n```"
