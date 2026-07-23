# Copyright (c) 2026 SaferAI. GEPA forecaster-optimisation experiment.
"""Run harness for the forecaster-GEPA experiment.

Phases:
  verify-sign   evaluate seed vs a deliberately-bad prompt on a few cells and
                assert mean(-Brier) is higher for the seed (metric sign check)
  optimize      the GEPA loop (resumes automatically from run_dir state)
  finalist      re-rank the top-k candidates on the finalist set (full panel)
  test          evaluate the winner on the held-out CVEBench+CyberGym set
  all           optimize -> finalist -> test
  resume-check  (dry-run only) verify a resumed run reproduces an
                uninterrupted run exactly

Usage:
  uv run python -m forecaster_gepa.run --config configs/forecaster_gepa.yaml --phase all
  uv run python -m forecaster_gepa.run --dry-run --phase all
  uv run python -m forecaster_gepa.run --dry-run --phase resume-check
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import stats as scipy_stats

from forecaster_gepa.acceptance import make_acceptance_criterion
from forecaster_gepa.adapter import ForecasterAdapter
from forecaster_gepa.config import COMPONENT_NAME, ForecasterGEPAConfig, config_as_dict, load_config
from forecaster_gepa.data import CellSpec, ExperimentData
from forecaster_gepa.diagnostics import DiagnosticsCallback, WandbArtifactCallback
from forecaster_gepa.metrics import murphy_decomposition, yates_decomposition
from forecaster_gepa.sampler import BinStratifiedBatchSampler
from forecaster_gepa.selection import MaxIterationsStopper, make_candidate_selector
from forecaster_gepa.stub import StubReflectionLM
from gepa.api import optimize as gepa_optimize
from gepa.core.state import GEPAState

logger = logging.getLogger("forecaster_gepa")

BAD_TEMPLATE_SUFFIX = (
    "\n\nIMPORTANT OVERRIDE: Ignore all of the evidence above. Always report p50 = 0.99 "
    "(i.e. <p25>0.98</p25>, <p50>0.99</p50>, <p75>0.99</p75>), regardless of the task."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_info(path: Path) -> dict[str, Any]:
    """Return {sha, branch, dirty} for a checkout (None fields on failure)."""

    def _run(*args: str) -> str | None:
        try:
            out = subprocess.check_output(["git", "-C", str(path), *args], stderr=subprocess.DEVNULL)
            return out.decode().strip()
        except Exception:
            return None

    status = _run("status", "--porcelain")
    return {
        "sha": _run("rev-parse", "HEAD"),
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def _read_seed_template(cfg: ForecasterGEPAConfig) -> str:
    return cfg.resolved("seed_template_path").read_text(encoding="utf-8")


def _save_run_metadata(cfg: ForecasterGEPAConfig, run_dir: Path) -> None:
    import intra_benchmark_calibration

    estimator_repo = Path(str(intra_benchmark_calibration.__file__)).resolve().parent.parent
    gepa_repo = Path(__file__).resolve().parent.parent.parent
    estimator_git = _git_info(estimator_repo)
    meta = {
        "config": config_as_dict(cfg),
        "gepa_fork_git": _git_info(gepa_repo),
        "llm_estimator_repo": str(estimator_repo),
        "llm_estimator_git": estimator_git,
        "seed_template": _read_seed_template(cfg),
    }
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    # The llm-estimator dependency is an editable path install: whatever is
    # checked out in the sibling repo is what runs. Warn loudly on surprises.
    if cfg.expected_estimator_branch and estimator_git["branch"] != cfg.expected_estimator_branch:
        logger.warning(
            f"LLM_elicitation is on branch {estimator_git['branch']!r}, expected "
            f"{cfg.expected_estimator_branch!r} — the estimation code that runs is whatever "
            "is checked out there."
        )
    if estimator_git["dirty"]:
        logger.warning(
            "LLM_elicitation has uncommitted changes; the recorded SHA does not fully "
            "identify the estimation code used for this run."
        )


def _wandb_init_kwargs(cfg: ForecasterGEPAConfig, run_dir: Path) -> dict[str, Any]:
    """Persist the W&B run id in the run dir so a resumed run reattaches."""
    id_file = run_dir / "wandb_run_id.txt"
    if id_file.exists():
        run_id = id_file.read_text().strip()
    else:
        run_id = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8))
        id_file.write_text(run_id)
    kwargs: dict[str, Any] = {
        "project": cfg.wandb_project,
        "id": run_id,
        "resume": "allow",
        "name": run_dir.name,
        "config": config_as_dict(cfg),
    }
    if cfg.wandb_entity:
        kwargs["entity"] = cfg.wandb_entity
    return kwargs


def _grand_brier(outputs: Sequence[dict], failure_brier: float = 1.0) -> float:
    vals = [o["brier"] if o["brier"] is not None else failure_brier for o in outputs]
    return float(np.mean(vals))


def _forecast_arrays(outputs: Sequence[dict]) -> tuple[list[float], list[float]]:
    p50s = [o["p50"] for o in outputs if o["p50"] is not None]
    outcomes = [o["outcome"] for o in outputs if o["p50"] is not None]
    return p50s, outcomes


def _decompositions(outputs: Sequence[dict]) -> dict[str, Any]:
    p50s, outcomes = _forecast_arrays(outputs)
    if len(p50s) < 2 or not (0.0 < np.mean(outcomes) < 1.0):
        return {}
    return {
        "yates": yates_decomposition(p50s, outcomes),
        "murphy": murphy_decomposition(p50s, outcomes),
    }


def _write_cells_jsonl(path: Path, candidate_idx: int, outputs: Sequence[dict]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        for out in outputs:
            fh.write(json.dumps({"candidate_idx": candidate_idx, **out}) + "\n")


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


def phase_optimize(cfg: ForecasterGEPAConfig, data: ExperimentData, run_dir: Path):
    plans, train_loader, val_loader, tasks_by_bin = data.search_phase()
    logger.info(
        f"Search phase: {len(train_loader)} train-pool cells, {len(val_loader)} val cells, "
        f"gate = {cfg.n_bins} bins x {cfg.n_train_tasks_per_iter} task(s) x "
        f"{len(cfg.forecasted_models_search)} models"
    )

    adapter = ForecasterAdapter(plans, cfg, stub=cfg.dry_run)
    sampler_rng = random.Random(cfg.seed + 202)
    adapter.register_rng("sampler", sampler_rng)
    sampler = BinStratifiedBatchSampler(
        tasks_by_bin,
        cfg.forecasted_models_search,
        n_tasks_per_bin=cfg.n_train_tasks_per_iter,
        rng=sampler_rng,
    )
    pareto_rng = random.Random(cfg.seed + 303)
    adapter.register_rng("pareto", pareto_rng)
    selector = make_candidate_selector(cfg.n_cells_won_needed_for_pareto_frontier, pareto_rng)

    gate_size = cfg.n_bins * cfg.n_train_tasks_per_iter * len(cfg.forecasted_models_search)
    if cfg.acceptance_criterion != "aggregate_sum" and cfg.acceptance_min_task_wins > gate_size:
        raise ValueError(
            f"acceptance_min_task_wins={cfg.acceptance_min_task_wins} exceeds the gate size "
            f"{gate_size} ({cfg.n_bins} bins x {cfg.n_train_tasks_per_iter} task(s) x "
            f"{len(cfg.forecasted_models_search)} models) -- no child could ever be accepted."
        )
    acceptance_criterion = make_acceptance_criterion(
        cfg.acceptance_criterion,
        min_task_wins=cfg.acceptance_min_task_wins,
        margin=cfg.acceptance_margin_tau,
    )
    logger.info(f"Acceptance gate: {cfg.acceptance_criterion!r} (gate size {gate_size} cells)")

    if cfg.dry_run:
        reflection_lm: Any = StubReflectionLM()
        reflection_lm_kwargs = None
        max_iterations = cfg.dry_run_max_iterations
        use_wandb = False
    else:
        # litellm (the reflection LM) reads the key from the environment;
        # make the estimator's .env-resolved key visible to it.
        import os

        from intra_benchmark_calibration.estimation_api import resolve_anthropic_api_key

        if not os.environ.get("ANTHROPIC_API_KEY"):
            key = resolve_anthropic_api_key()
            if key:
                os.environ["ANTHROPIC_API_KEY"] = key
        reflection_lm = cfg.reflection_model
        reflection_lm_kwargs = {
            "max_tokens": cfg.reflection_max_tokens,
            "temperature": cfg.reflection_temperature,
        }
        max_iterations = cfg.max_iterations
        use_wandb = cfg.use_wandb

    seed_candidate = {COMPONENT_NAME: _read_seed_template(cfg)}
    callbacks: list[Any] = [DiagnosticsCallback(run_dir, use_wandb=use_wandb)]
    if use_wandb:
        callbacks.append(
            WandbArtifactCallback(
                run_dir,
                cfg.resolved("manifest_path"),
                every=cfg.wandb_checkpoint_every,
                log_files_at_end=cfg.wandb_log_files_at_end,
            )
        )

    result = gepa_optimize(
        seed_candidate=seed_candidate,
        trainset=train_loader,
        valset=val_loader,
        adapter=adapter,
        reflection_lm=reflection_lm,
        reflection_lm_kwargs=reflection_lm_kwargs,
        reflection_prompt_template=cfg.reflection_prompt_template,
        candidate_selection_strategy=selector,
        batch_sampler=sampler,
        reflection_minibatch_size=None,
        perfect_score=0.0,
        skip_perfect_score=True,
        use_merge=cfg.use_merge,
        acceptance_criterion=acceptance_criterion,
        max_metric_calls=cfg.max_metric_calls,
        stop_callbacks=[MaxIterationsStopper(max_iterations)],
        run_dir=str(run_dir),
        callbacks=callbacks,
        use_wandb=use_wandb,
        wandb_init_kwargs=_wandb_init_kwargs(cfg, run_dir) if use_wandb else None,
        track_best_outputs=cfg.track_best_outputs,
        seed=cfg.seed,
        raise_on_exception=False,
        display_progress_bar=False,
    )
    logger.info(
        f"Optimisation finished: {result.num_candidates} candidates, "
        f"best val Brier {-max(result.val_aggregate_scores):.4f}, "
        f"{result.total_metric_calls} metric calls."
    )
    return result


def _load_candidates_from_state(run_dir: Path) -> tuple[list[dict[str, str]], list[float]]:
    state = GEPAState.load(str(run_dir))
    return list(state.program_candidates), list(state.program_full_scores_val_set)


def phase_finalist(cfg: ForecasterGEPAConfig, data: ExperimentData, run_dir: Path) -> dict:
    candidates, val_scores = _load_candidates_from_state(run_dir)
    order = sorted(range(len(candidates)), key=lambda i: val_scores[i], reverse=True)
    top = order[: cfg.finalist_top_k]
    logger.info(f"Finalist re-ranking of candidates {top} on the finalist set (full panel).")

    plans, specs = data.finalist_phase()
    adapter = ForecasterAdapter(plans, cfg, stub=cfg.dry_run)
    cells_path = run_dir / "finalist_cells.jsonl"

    entries = []
    for idx in top:
        eb = adapter.evaluate(specs, candidates[idx], capture_traces=False)
        _write_cells_jsonl(cells_path, idx, eb.outputs)
        entries.append(
            {
                "candidate_idx": idx,
                "val_brier": -val_scores[idx],
                "finalist_brier": _grand_brier(eb.outputs),
                "n_cells": len(eb.outputs),
                **_decompositions(eb.outputs),
            }
        )

    val_rank = [-e["val_brier"] for e in entries]
    fin_rank = [-e["finalist_brier"] for e in entries]
    rho_raw: Any = scipy_stats.spearmanr(val_rank, fin_rank)[0] if len(entries) >= 3 else float("nan")
    rho = float(rho_raw)
    winner = min(entries, key=lambda e: e["finalist_brier"])
    results = {
        "top_k_by_val": top,
        "entries": entries,
        "val_vs_finalist_spearman": rho,
        "ranking_preserved": [e["candidate_idx"] for e in sorted(entries, key=lambda e: e["finalist_brier"])] == top,
        "winner_candidate_idx": winner["candidate_idx"],
    }
    with (run_dir / "finalist_results.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info(
        f"Finalist winner: candidate {winner['candidate_idx']} "
        f"(finalist Brier {winner['finalist_brier']:.4f}); val-vs-finalist Spearman rho={rho:.3f}"
    )
    return results


def phase_test(cfg: ForecasterGEPAConfig, data: ExperimentData, run_dir: Path) -> dict:
    finalist_path = run_dir / "finalist_results.json"
    if not finalist_path.exists():
        raise FileNotFoundError("finalist_results.json not found — run the finalist phase first.")
    with finalist_path.open("r", encoding="utf-8") as fh:
        winner_idx = json.load(fh)["winner_candidate_idx"]
    candidates, val_scores = _load_candidates_from_state(run_dir)
    winner = candidates[winner_idx]
    seed_candidate = candidates[0]

    plans, specs = data.test_phase()
    adapter = ForecasterAdapter(plans, cfg, stub=cfg.dry_run)
    cells_path = run_dir / "test_cells.jsonl"

    def eval_on_test(idx: int, candidate: dict[str, str]) -> dict:
        eb = adapter.evaluate(specs, candidate, capture_traces=False)
        _write_cells_jsonl(cells_path, idx, eb.outputs)
        per_bin: dict[int, list[dict]] = {}
        for out in eb.outputs:
            per_bin.setdefault(out["bin"], []).append(out)
        return {
            "candidate_idx": idx,
            "test_brier": _grand_brier(eb.outputs),
            "n_cells": len(eb.outputs),
            "per_bin_brier": {str(b): _grand_brier(outs) for b, outs in sorted(per_bin.items())},
            **_decompositions(eb.outputs),
        }

    # Evaluate the winner AND the seed, so the headline transfer claim has a baseline.
    results = {
        "winner": eval_on_test(winner_idx, winner),
        "seed_baseline": eval_on_test(0, seed_candidate),
        "winner_val_brier": -val_scores[winner_idx],
    }
    with (run_dir / "test_results.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info(
        f"Test (held-out benchmarks): winner Brier {results['winner']['test_brier']:.4f} "
        f"vs seed {results['seed_baseline']['test_brier']:.4f}"
    )
    return results


def phase_verify_sign(cfg: ForecasterGEPAConfig, data: ExperimentData, run_dir: Path) -> dict:
    """Metric-sign check: the seed must outscore a deliberately-bad prompt."""
    plans, _train_loader, val_loader, _tasks = data.search_phase()
    adapter = ForecasterAdapter(plans, cfg, stub=cfg.dry_run)
    all_ids = val_loader.all_ids()
    n = min(cfg.verify_sign_n_cells, len(all_ids))
    picked = [all_ids[(i * len(all_ids)) // n] for i in range(n)]  # spread across bins
    batch: list[CellSpec] = val_loader.fetch(picked)

    seed_template = _read_seed_template(cfg)
    bad_template = seed_template + BAD_TEMPLATE_SUFFIX
    eb_seed = adapter.evaluate(batch, {COMPONENT_NAME: seed_template})
    eb_bad = adapter.evaluate(batch, {COMPONENT_NAME: bad_template})
    mean_seed = float(np.mean(eb_seed.scores))
    mean_bad = float(np.mean(eb_bad.scores))
    passed = mean_seed > mean_bad
    results = {
        "n_cells": n,
        "cells": picked,
        "seed_mean_score": mean_seed,
        "bad_mean_score": mean_bad,
        "seed_mean_brier": -mean_seed,
        "bad_mean_brier": -mean_bad,
        "sign_check_passed": passed,
    }
    with (run_dir / "verify_sign.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    status = "PASSED" if passed else "FAILED"
    logger.info(
        f"Sign check {status}: seed -Brier {mean_seed:.4f} vs bad-prompt -Brier {mean_bad:.4f} (higher must be better)"
    )
    if not passed:
        raise AssertionError("Metric sign check failed: bad prompt scored higher than the seed.")
    return results


def pull_checkpoint(artifact_ref: str, run_dir: Path) -> None:
    """Download a `gepa-checkpoint` artifact into run_dir (fresh dirs only).

    `artifact_ref` is e.g. `<entity>/<project>/checkpoint-<wandb_run_id>:latest`.
    Afterwards the usual command resumes the run — including logging into the
    same W&B run, whose id ships inside the checkpoint.
    """
    if (run_dir / "gepa_state.bin").exists():
        raise SystemExit(
            f"{run_dir} already contains gepa_state.bin — refusing to overwrite an existing "
            "checkpoint. Use a fresh --run-dir or remove the state file first."
        )
    import wandb

    artifact = wandb.Api().artifact(artifact_ref, type="gepa-checkpoint")
    artifact.download(root=str(run_dir))
    logger.info(f"Pulled checkpoint {artifact_ref} into {run_dir}; re-run to resume.")


def phase_resume_check(cfg: ForecasterGEPAConfig, run_dir: Path) -> None:
    """Dry-run only: a killed+resumed run must match an uninterrupted one."""
    assert cfg.dry_run, "resume-check runs with the stub forecaster only"
    n_full, n_partial = 8, 5
    dir_a = run_dir / "resume_check_A"
    dir_b = run_dir / "resume_check_B"
    for d in (dir_a, dir_b):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    def run_to(n_iters: int, target: Path):
        cfg_i = dataclasses.replace(cfg, dry_run_max_iterations=n_iters)
        data_i = ExperimentData(cfg_i)
        phase_optimize(cfg_i, data_i, target)

    run_to(n_full, dir_a)  # uninterrupted
    run_to(n_partial, dir_b)  # "killed" after n_partial iterations
    run_to(n_full, dir_b)  # resumed to the same total

    with (dir_a / "candidates.json").open() as fh:
        cands_a = json.load(fh)
    with (dir_b / "candidates.json").open() as fh:
        cands_b = json.load(fh)
    state_a = GEPAState.load(str(dir_a))
    state_b = GEPAState.load(str(dir_b))
    assert cands_a == cands_b, "Resume check FAILED: candidate pools differ"
    assert state_a.prog_candidate_val_subscores == state_b.prog_candidate_val_subscores, (
        "Resume check FAILED: per-cell val scores differ"
    )
    assert state_a.i == state_b.i, "Resume check FAILED: iteration counters differ"
    logger.info(
        f"Resume check PASSED: {len(cands_a)} identical candidates after "
        f"{state_a.i + 1} iterations (interrupted at {n_partial})."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="YAML config overriding the defaults")
    parser.add_argument("--run-dir", default=None, help="Override run_dir from the config")
    parser.add_argument(
        "--phase",
        default="all",
        choices=["all", "verify-sign", "optimize", "finalist", "test", "resume-check"],
    )
    parser.add_argument("--dry-run", action="store_true", help="Stub forecaster + stub reflection")
    parser.add_argument(
        "--pull-checkpoint",
        default=None,
        metavar="ARTIFACT_REF",
        help="Download a gepa-checkpoint W&B artifact (e.g. entity/project/checkpoint-<id>:latest) "
        "into --run-dir before running, to resume someone else's run on this machine",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    overrides: dict[str, Any] = {}
    if args.run_dir:
        overrides["run_dir"] = args.run_dir
    if args.dry_run:
        overrides["dry_run"] = True
    cfg = load_config(args.config, overrides)
    if cfg.dry_run and args.run_dir is None and args.config is None:
        cfg.run_dir = "runs/forecaster_gepa_dry"
        cfg.manifest_path = "runs/forecaster_gepa_dry/task_manifest.json"

    run_dir = cfg.resolved("run_dir")
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.pull_checkpoint:
        pull_checkpoint(args.pull_checkpoint, run_dir)
    _save_run_metadata(cfg, run_dir)

    if args.phase == "resume-check":
        phase_resume_check(cfg, run_dir)
        return 0

    data = ExperimentData(cfg)
    if args.phase == "verify-sign":
        phase_verify_sign(cfg, data, run_dir)
    elif args.phase == "optimize":
        phase_optimize(cfg, data, run_dir)
    elif args.phase == "finalist":
        phase_finalist(cfg, data, run_dir)
    elif args.phase == "test":
        phase_test(cfg, data, run_dir)
    elif args.phase == "all":
        phase_optimize(cfg, data, run_dir)
        phase_finalist(cfg, data, run_dir)
        phase_test(cfg, data, run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
