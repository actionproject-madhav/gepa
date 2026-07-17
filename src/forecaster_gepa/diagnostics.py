# Copyright (c) 2026 SaferAI. GEPA forecaster-optimisation experiment.
"""Diagnostics callback (spec sections 6-7).

Logs, per candidate (accepted AND rejected) and per iteration:
  1. gate->val agreement for accepted children,
  2. frontier stats (size, cells-won distribution, parent-selection entropy),
  3. best-val-Brier-so-far trajectory,
  4. confidence-vs-wins correlations across frontier members,
  6. full per-candidate records,
plus Yates (primary) and Murphy (secondary) Brier decompositions on the val
set of every accepted candidate.

Scalar metrics go to W&B (when active); detail goes to local JSONL files in
the run directory:
  - candidates.jsonl   one record per proposal (accepted + rejected) + seed
  - diagnostics.jsonl  one record per iteration
  - gate_traces.jsonl  full rollout traces for gate/reflection cells only
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, TextIO

from forecaster_gepa.metrics import (
    frontier_stats,
    mean_extremity,
    murphy_decomposition,
    partial_spearman,
    spearman,
    yates_decomposition,
)

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4  # crude token estimate for cost instrumentation


def _jsonl_append(handle: TextIO, record: dict) -> None:
    handle.write(json.dumps(record, default=repr) + "\n")
    handle.flush()


class DiagnosticsCallback:
    """GEPACallback implementation for the forecaster experiment."""

    def __init__(self, run_dir: str | Path, use_wandb: bool = False):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        self._candidates_fh = (self.run_dir / "candidates.jsonl").open("a", encoding="utf-8")
        self._diag_fh = (self.run_dir / "diagnostics.jsonl").open("a", encoding="utf-8")
        self._traces_fh = (self.run_dir / "gate_traces.jsonl").open("a", encoding="utf-8")

        # candidate_idx -> {avg, scores_by_val_id, p50/outcome pairs, parents}
        self._cand: dict[int, dict[str, Any]] = {}
        self._agreements: list[int] = []  # 1 if val confirms gate direction
        self._best_val_score = float("-inf")
        self._ctx: dict[str, Any] = {}

    # ------------------------------------------------------------------
    def _wandb_log(self, metrics: dict[str, Any], step: int) -> None:
        if not self.use_wandb:
            return
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(metrics, step=step)
        except Exception as e:  # never break the run over logging
            logger.warning(f"wandb logging failed: {e}")

    @staticmethod
    def _val_forecasts(outputs_by_val_id: dict | None) -> tuple[list[float], list[float]]:
        p50s, outcomes = [], []
        for out in (outputs_by_val_id or {}).values():
            if isinstance(out, dict) and out.get("p50") is not None:
                p50s.append(float(out["p50"]))
                outcomes.append(float(out["outcome"]))
        return p50s, outcomes

    # ------------------------------------------------------------------
    # Iteration lifecycle
    # ------------------------------------------------------------------

    def on_iteration_start(self, event) -> None:
        self._ctx = {"iteration": event["iteration"], "t_start": time.time()}

    def on_candidate_selected(self, event) -> None:
        self._ctx["parent_idx"] = event["candidate_idx"]
        self._ctx["parent_val_score"] = event["score"]

    def on_minibatch_sampled(self, event) -> None:
        self._ctx["gate_cell_ids"] = list(event["minibatch_ids"])

    def on_evaluation_end(self, event) -> None:
        role = "child" if event["candidate_idx"] is None else "parent"
        self._ctx[f"{role}_gate_scores"] = list(event["scores"])
        n_errors = sum(1 for o in event["outputs"] if isinstance(o, dict) and o.get("error"))
        self._ctx[f"{role}_gate_errors"] = n_errors
        self._wandb_log({f"diagnostics/gate_cell_errors_{role}": n_errors}, step=event["iteration"])
        trajectories = event["trajectories"] or []
        approx_tokens = sum(
            (len(t.get("user_prompt", "")) + len(t.get("raw_response", ""))) // _CHARS_PER_TOKEN
            for t in trajectories
            if isinstance(t, dict)
        )
        self._ctx[f"{role}_gate_approx_tokens"] = approx_tokens
        _jsonl_append(
            self._traces_fh,
            {
                "iteration": event["iteration"],
                "role": role,
                "candidate_idx": event["candidate_idx"],
                "parent_ids": list(event["parent_ids"]),
                "scores": list(event["scores"]),
                "trajectories": trajectories,
                "timestamp": time.time(),
            },
        )

    def on_proposal_end(self, event) -> None:
        self._ctx["proposal_new_instructions"] = dict(event["new_instructions"])
        self._ctx["reflection_raw_output_chars"] = sum(
            len(v) for v in event["raw_lm_outputs"].values() if isinstance(v, str)
        )

    # ------------------------------------------------------------------
    # Valset evaluations (seed + accepted children)
    # ------------------------------------------------------------------

    def on_valset_evaluated(self, event) -> None:
        idx = event["candidate_idx"]
        p50s, outcomes = self._val_forecasts(event["outputs_by_val_id"])
        self._cand[idx] = {
            "avg_score": event["average_score"],
            "scores_by_val_id": dict(event["scores_by_val_id"]),
            "extremity": mean_extremity(p50s),
            "parents": list(event["parent_ids"]),
        }
        val_p50s = {
            key: out.get("p50") for key, out in (event["outputs_by_val_id"] or {}).items() if isinstance(out, dict)
        }
        n_val_errors = sum(
            1 for out in (event["outputs_by_val_id"] or {}).values() if isinstance(out, dict) and out.get("error")
        )
        self._wandb_log({"diagnostics/val_cell_errors": n_val_errors}, step=event["iteration"])
        record: dict[str, Any] = {
            "kind": "valset_evaluation",
            "iteration": event["iteration"],
            "candidate_idx": idx,
            "val_mean_score": event["average_score"],
            "val_mean_brier": -event["average_score"],
            "n_val_cells": event["num_examples_evaluated"],
            "n_val_errors": n_val_errors,
            "is_best": event["is_best_program"],
            "scores_by_val_id": dict(event["scores_by_val_id"]),
            "p50_by_val_id": val_p50s,
            "timestamp": time.time(),
        }
        if len(p50s) >= 2 and 0.0 < (sum(outcomes) / len(outcomes)) < 1.0:
            record["yates"] = yates_decomposition(p50s, outcomes)
            record["murphy"] = murphy_decomposition(p50s, outcomes)
            self._wandb_log(
                {
                    "val/yates_bias_sq": record["yates"]["bias_sq"],
                    "val/yates_slope": record["yates"]["slope"],
                    "val/yates_scatter": record["yates"]["scatter_delta_var_f"],
                    "val/murphy_reliability": record["murphy"]["reliability"],
                    "val/murphy_resolution": record["murphy"]["resolution"],
                },
                step=event["iteration"],
            )

        # Gate->val agreement (accepted children only; gate positive by construction).
        if idx != 0 and event["parent_ids"]:
            parent = event["parent_ids"][0]
            parent_avg = self._cand.get(parent, {}).get("avg_score")
            if parent_avg is not None:
                confirmed = int(event["average_score"] > parent_avg)
                self._agreements.append(confirmed)
                record["val_delta_vs_parent"] = event["average_score"] - parent_avg
                record["gate_val_direction_confirmed"] = confirmed
                self._wandb_log(
                    {
                        "diagnostics/gate_val_agreement_running": sum(self._agreements) / len(self._agreements),
                        "diagnostics/val_delta_vs_parent": record["val_delta_vs_parent"],
                    },
                    step=event["iteration"],
                )
        _jsonl_append(self._candidates_fh, record)

    # ------------------------------------------------------------------
    # Accept / reject records (spec section 6.6)
    # ------------------------------------------------------------------

    def _proposal_record(self, status: str, extra: dict[str, Any]) -> dict[str, Any]:
        ctx = self._ctx
        record = {
            "kind": "proposal",
            "status": status,
            "iteration": ctx.get("iteration"),
            "parent_idx": ctx.get("parent_idx"),
            "parent_val_score": ctx.get("parent_val_score"),
            "gate_cell_ids": ctx.get("gate_cell_ids"),
            "gate_scores_parent": ctx.get("parent_gate_scores"),
            "gate_scores_child": ctx.get("child_gate_scores"),
            "gate_sum_parent": sum(ctx.get("parent_gate_scores") or []),
            "gate_sum_child": sum(ctx.get("child_gate_scores") or []),
            "candidate_text": ctx.get("proposal_new_instructions"),
            "approx_gate_tokens": (
                (ctx.get("parent_gate_approx_tokens") or 0) + (ctx.get("child_gate_approx_tokens") or 0)
            ),
            "reflection_raw_output_chars": ctx.get("reflection_raw_output_chars"),
            "duration_seconds": time.time() - ctx.get("t_start", time.time()),
            "timestamp": time.time(),
        }
        record.update(extra)
        return record

    def on_candidate_accepted(self, event) -> None:
        idx = event["new_candidate_idx"]
        extra = {
            "candidate_idx": idx,
            "val_mean_score": self._cand.get(idx, {}).get("avg_score"),
        }
        _jsonl_append(self._candidates_fh, self._proposal_record("accepted", extra))
        self._wandb_log({"diagnostics/accepted": 1}, step=event["iteration"])

    def on_candidate_rejected(self, event) -> None:
        extra = {"reject_reason": event["reason"]}
        _jsonl_append(self._candidates_fh, self._proposal_record("rejected", extra))
        self._wandb_log({"diagnostics/accepted": 0}, step=event["iteration"])

    # ------------------------------------------------------------------
    # Per-iteration frontier diagnostics
    # ------------------------------------------------------------------

    def on_iteration_end(self, event) -> None:
        state = event["state"]
        iteration = event["iteration"]
        stats = frontier_stats(state.program_at_pareto_front_valset, state.per_program_tracked_scores)

        best_avg = max(state.program_full_scores_val_set)
        self._best_val_score = max(self._best_val_score, best_avg)

        # Confidence-vs-wins across frontier members (spec section 6.4).
        conf: dict[str, float] = {}
        cells_won = stats.get("cells_won_by_program") or {}
        members = sorted(cells_won)
        if len(members) >= 3:
            wins = [cells_won[m] for m in members]
            extremity = [self._cand.get(m, {}).get("extremity", float("nan")) for m in members]
            val_brier = [-self._cand.get(m, {}).get("avg_score", float("nan")) for m in members]
            conf = {
                "spearman_wins_vs_extremity": spearman(wins, extremity),
                "partial_spearman_wins_extremity_given_brier": partial_spearman(wins, extremity, val_brier),
            }

        record = {
            "kind": "iteration",
            "iteration": iteration,
            "proposal_accepted": event["proposal_accepted"],
            "n_candidates": len(state.program_candidates),
            "total_metric_calls": state.total_num_evals,
            "best_val_score_so_far": self._best_val_score,
            "best_val_brier_so_far": -self._best_val_score,
            "gate_val_agreement_running": (sum(self._agreements) / len(self._agreements) if self._agreements else None),
            "frontier": stats,
            "confidence_vs_wins": conf,
            "timestamp": time.time(),
        }
        _jsonl_append(self._diag_fh, record)

        wandb_metrics = {
            "diagnostics/best_val_brier_so_far": -self._best_val_score,
            "diagnostics/n_candidates": len(state.program_candidates),
            "diagnostics/frontier_size": stats.get("frontier_size", 0),
            "diagnostics/share_one_cell_winners": stats.get("share_winning_exactly_one_cell", 0.0),
            "diagnostics/effective_parents": stats.get("effective_parents", 1.0),
        }
        for key, value in conf.items():
            wandb_metrics[f"diagnostics/{key}"] = value
        self._wandb_log(wandb_metrics, step=iteration)

    # ------------------------------------------------------------------
    def on_optimization_end(self, event) -> None:
        _jsonl_append(
            self._diag_fh,
            {
                "kind": "end",
                "best_candidate_idx": event["best_candidate_idx"],
                "total_iterations": event["total_iterations"],
                "total_metric_calls": event["total_metric_calls"],
                "gate_val_agreement_final": (
                    sum(self._agreements) / len(self._agreements) if self._agreements else None
                ),
                "timestamp": time.time(),
            },
        )
        for fh in (self._candidates_fh, self._diag_fh, self._traces_fh):
            fh.flush()


# Files sufficient to resume a run on another machine (small, a few MB).
_CHECKPOINT_FILES = ("gepa_state.bin", "candidates.json", "run_config.json", "wandb_run_id.txt", "run_log.json")
# Bulky diagnostics/traces, uploaded once at the end of the run.
_LOG_FILES = ("candidates.jsonl", "diagnostics.jsonl", "gate_traces.jsonl", "run_log.txt")


class WandbArtifactCallback:
    """Uploads run_dir contents as W&B artifacts for cross-machine hand-off.

    Two artifacts per run (versioned; the name embeds the W&B run id so
    resumed runs extend the same lineage):
      - ``checkpoint-<run_id>`` (type ``gepa-checkpoint``): everything needed
        to resume — GEPA state pickle, task manifest, run config, candidate
        texts, the W&B run id. Uploaded every ``every`` iterations and at the
        end. Small, so frequent versions are cheap.
      - ``logs-<run_id>`` (type ``gepa-logs``): the append-only JSONL
        diagnostics and gate traces. These grow to hundreds of MB and W&B
        re-uploads a changed file in full per version, so they are uploaded
        ONCE, at the end of the run (each resume adds one more version).
    """

    def __init__(self, run_dir: str | Path, manifest_path: str | Path, every: int = 10, log_files_at_end: bool = True):
        self.run_dir = Path(run_dir)
        self.manifest_path = Path(manifest_path)
        self.every = every
        self.log_files_at_end = log_files_at_end

    @staticmethod
    def _active_run():
        try:
            import wandb

            return wandb.run
        except Exception:
            return None

    def _upload(self, kind: str, files: list[Path]) -> None:
        run = self._active_run()
        if run is None:
            return
        try:
            import wandb

            artifact = wandb.Artifact(f"{kind}-{run.id}", type=f"gepa-{kind}")
            n_added = 0
            for path in files:
                if path.exists():
                    artifact.add_file(str(path))
                    n_added += 1
            if n_added:
                run.log_artifact(artifact)
        except Exception as e:  # never break the run over logging
            logger.warning(f"W&B artifact upload ({kind}) failed: {e}")

    def _checkpoint_files(self) -> list[Path]:
        files = [self.run_dir / name for name in _CHECKPOINT_FILES]
        files.append(self.manifest_path)
        return files

    def on_state_saved(self, event) -> None:
        if self.every > 0 and event["iteration"] % self.every == 0:
            self._upload("checkpoint", self._checkpoint_files())

    def on_optimization_end(self, event) -> None:
        self._upload("checkpoint", self._checkpoint_files())
        if self.log_files_at_end:
            self._upload("logs", [self.run_dir / name for name in _LOG_FILES])
