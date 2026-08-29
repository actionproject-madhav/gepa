# Copyright (c) 2026 SaferAI. GEPA forecaster-optimisation experiment.
"""GEPAAdapter for the LLM forecaster (spec section 3).

- `evaluate` runs the forecaster (via the llm-estimator per-cell callable) on
  a batch of cells and returns per-cell **-Brier** (GEPA maximises; Brier is
  lower-is-better). Cells that fail to format/parse score `failure_score`
  (-1.0, the worst possible -Brier) instead of raising.
- `make_reflective_dataset` packages gate traces + per-cell Brier outcomes
  for the reflection LLM, optionally subsetting via worst-k-stochastic
  selection (traces are 10-20k tokens each).
- Adapter state (all experiment RNGs) is persisted through GEPA's
  get/set_adapter_state hooks so a resumed run continues exactly.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any, Mapping, Sequence

from intra_benchmark_calibration.estimation_api import (
    CellResult,
    estimate_cells,
    make_llm_settings,
    resolve_anthropic_api_key,
)
from intra_benchmark_calibration.task_selector import CellPlan

from forecaster_gepa.config import COMPONENT_NAME, ForecasterGEPAConfig
from forecaster_gepa.data import CellSpec
from forecaster_gepa.stub import stub_estimate_cells
from gepa.core.adapter import EvaluationBatch, ProposalFn

logger = logging.getLogger(__name__)


class CellFailureHalt(BaseException):
    """A cell still scored failure_score after the parse retry and
    cfg.halt_on_cell_failure is set.

    Derives from BaseException ON PURPOSE: the optimize phase runs the GEPA
    engine with raise_on_exception=False, which catches Exception and keeps
    iterating — this tripwire must penetrate that and stop the process so a
    failure is never silently averaged into the metric.
    """


class ForecasterAdapter:
    """GEPAAdapter[CellSpec, dict, dict] over the intra-benchmark forecaster."""

    def __init__(
        self,
        plans: dict[str, CellPlan],
        cfg: ForecasterGEPAConfig,
        *,
        stub: bool = False,
    ):
        self.plans = plans
        self.cfg = cfg
        self.stub = stub
        self.llm_settings = make_llm_settings(
            model=cfg.forecaster_model,
            temperature=cfg.temperature,
            reasoning_effort=cfg.reasoning_effort,
            max_concurrent_calls=cfg.max_concurrent_calls,
            rate_limit_calls=cfg.rate_limit_calls,
            rate_limit_period=cfg.rate_limit_period,
        )
        self._api_key = None if stub else resolve_anthropic_api_key()
        if not stub and not self._api_key:
            raise ValueError("No Anthropic API key found (checked LLM_elicitation .env chain and process env).")

        # All experiment RNGs are registered here and snapshotted into the
        # GEPA checkpoint via get/set_adapter_state, so resume is exact.
        self.reflection_rng = random.Random(cfg.seed + 101)
        self._rngs: dict[str, random.Random] = {"reflection": self.reflection_rng}

    def register_rng(self, name: str, rng: random.Random) -> None:
        assert name not in self._rngs, f"RNG {name!r} already registered"
        self._rngs[name] = rng

    # ------------------------------------------------------------------
    # GEPAAdapter protocol
    # ------------------------------------------------------------------

    propose_new_texts: ProposalFn | None = None

    def evaluate(
        self,
        batch: list[CellSpec],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[dict, dict]:
        template = candidate[COMPONENT_NAME]
        plans = [self.plans[spec.cell_key] for spec in batch]

        if self.stub:
            results = stub_estimate_cells(plans, template)
        else:
            results = estimate_cells(
                plans,
                template,
                llm_settings=self.llm_settings,
                system_prompt=self.cfg.system_prompt,
                api_key_anthropic=self._api_key,
                max_tokens=self.cfg.max_tokens,
                parse_retries=self.cfg.parse_retries,
            )

        outputs: list[dict] = []
        scores: list[float] = []
        trajectories: list[dict] | None = [] if capture_traces else None
        for spec, res in zip(batch, results, strict=True):
            score = -res.brier if res.brier is not None else self.cfg.failure_score
            scores.append(score)
            outputs.append(
                {
                    "cell_key": spec.cell_key,
                    "cell_id": res.cell_id,
                    "task_id": res.target_task_id,
                    "model": res.forecasted_model,
                    "bin": res.target_bin,
                    "outcome": res.outcome,
                    "p25": res.p25,
                    "p50": res.p50,
                    "p75": res.p75,
                    "brier": res.brier,
                    "error": res.error,
                }
            )
            if trajectories is not None:
                trajectories.append(self._trajectory(spec, res, score))

        # Loud per-batch failure summary: failed cells score failure_score
        # (worst possible), which contaminates the metric — persistent
        # rate-limit failures mean the throughput knobs are set too high.
        errors = [(o["cell_key"], o["error"]) for o in outputs if o["error"]]
        if errors:
            n_rate_limited = sum(1 for _, e in errors if "Rate Limit" in e)
            detail = "; ".join(f"{k}: {e}" for k, e in errors[:3])
            logger.warning(
                f"{len(errors)}/{len(outputs)} cells FAILED in this batch and scored "
                f"{self.cfg.failure_score} ({n_rate_limited} rate-limited). First few: {detail}. "
                + (
                    "Rate-limit exhaustion after SDK retries — lower max_concurrent_calls / "
                    "rate_limit_calls in the config."
                    if n_rate_limited
                    else ""
                )
            )
            if self.cfg.halt_on_cell_failure:
                detail_all = "; ".join(f"{k}: {e}" for k, e in errors[:5])
                raise CellFailureHalt(
                    f"{len(errors)}/{len(outputs)} cell(s) failed even after "
                    f"{self.cfg.parse_retries} parse retr{'y' if self.cfg.parse_retries == 1 else 'ies'} "
                    f"and would have scored {self.cfg.failure_score}: {detail_all}. "
                    "Run halted (halt_on_cell_failure=true). The checkpoint from the last "
                    "completed iteration is intact — inspect these cells, then re-run the "
                    "same command to resume."
                )
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    @staticmethod
    def _trajectory(spec: CellSpec, res: CellResult, score: float) -> dict:
        return {
            "cell_key": spec.cell_key,
            "cell_id": res.cell_id,
            "task_id": res.target_task_id,
            "model": res.forecasted_model,
            "bin": res.target_bin,
            "outcome": res.outcome,
            "p25": res.p25,
            "p50": res.p50,
            "p75": res.p75,
            "brier": res.brier,
            "score": score,
            "error": res.error,
            "system_prompt": res.system_prompt,
            "user_prompt": res.user_prompt,
            "raw_response": res.raw_response,
            "duration_seconds": res.duration_seconds,
        }

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[dict, dict],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        assert components_to_update == [COMPONENT_NAME], f"Single-component system; got {components_to_update}"
        trajectories = eval_batch.trajectories or []
        chosen = self._select_traces(trajectories, eval_batch.scores)

        records = []
        for traj in chosen:
            records.append(
                {
                    "Inputs": {"forecaster_prompt": traj["user_prompt"]},
                    "Generated Outputs": traj["raw_response"],
                    "Feedback": self._feedback(traj),
                    "score": traj["score"],
                }
            )
        return {COMPONENT_NAME: records}

    def _select_traces(self, trajectories: list[dict], scores: list[float]) -> list[dict]:
        """Worst-k-stochastic subset (spec section 4). k >= n returns all,
        ordered worst-first so the reflection LM sees failures first."""
        k = self.cfg.n_reflection_traces
        order = sorted(range(len(trajectories)), key=lambda i: scores[i])
        if k >= len(trajectories):
            return [trajectories[i] for i in order]
        if self.cfg.reflection_trace_selection != "worst_k_stochastic":
            raise ValueError(f"Unknown reflection_trace_selection: {self.cfg.reflection_trace_selection}")
        # Exponentially decaying weight in the worst-first rank, sampled
        # without replacement for variety across iterations.
        temp = self.cfg.reflection_selection_temperature
        remaining = list(order)
        chosen_idx: list[int] = []
        for _ in range(k):
            weights = [math.exp(-temp * rank) for rank in range(len(remaining))]
            total = sum(weights)
            r = self.reflection_rng.random() * total
            acc = 0.0
            pick_pos = len(remaining) - 1
            for pos, w in enumerate(weights):
                acc += w
                if r <= acc:
                    pick_pos = pos
                    break
            chosen_idx.append(remaining.pop(pick_pos))
        chosen_idx.sort(key=lambda i: scores[i])
        return [trajectories[i] for i in chosen_idx]

    @staticmethod
    def _feedback(traj: dict) -> str:
        model = traj["model"]
        solved = traj["outcome"] >= 0.5
        if traj["error"] is not None:
            return (
                f"FAILURE: {traj['error']}. This cell scored {traj['score']:.2f} (the worst possible). "
                "The response must end with the exact <percentile_estimates> XML block "
                "containing <p25>, <p50> and <p75> probabilities in [0, 1]."
            )
        return (
            f"Ground truth: {model} {'SOLVED' if solved else 'FAILED'} the target task "
            f"(binary outcome {int(traj['outcome'])}). "
            f"Forecast: p25={traj['p25']}, p50={traj['p50']}, p75={traj['p75']}. "
            f"Brier score on p50: {traj['brier']:.4f} (0 is best, 1 is worst; "
            "the optimisation target is the mean Brier over many such forecasts, so "
            "well-calibrated probabilities beat confident guessing on average)."
        )

    # ------------------------------------------------------------------
    # Checkpoint persistence (exact resume)
    # ------------------------------------------------------------------

    def get_adapter_state(self) -> dict[str, Any]:
        return {f"rng:{name}": rng.getstate() for name, rng in self._rngs.items()}

    def set_adapter_state(self, state: dict[str, Any]) -> None:
        for name, rng in self._rngs.items():
            key = f"rng:{name}"
            if key in state:
                rng.setstate(state[key])
