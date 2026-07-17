# Copyright (c) 2026 SaferAI. GEPA forecaster-optimisation experiment.
"""Data layer: task manifest, Lyptus dataset, cell plans and GEPA data loaders.

A GEPA data instance ("cell") is a (forecasted model, target task) pair with
`all_except_target` evidence; the corresponding `CellPlan` (prompt evidence)
is prebuilt deterministically by the llm-estimator package. Bin labels are a
design tool only and never appear in any prompt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from intra_benchmark_calibration.binning import BinAssignment
from intra_benchmark_calibration.config import DEFAULT_DROP_MODELS
from intra_benchmark_calibration.estimation_api import build_plans_for_targets, canonical_family
from intra_benchmark_calibration.gepa_task_sets import (
    TRAIN_FAMILIES,
    build_task_manifest,
    load_manifest,
    write_manifest,
)
from intra_benchmark_calibration.lyptus_data import LyptusDataset, load_lyptus_dataset
from intra_benchmark_calibration.task_selector import CellPlan

from forecaster_gepa.config import ForecasterGEPAConfig

logger = logging.getLogger(__name__)

CELL_KEY_SEPARATOR = "::"


def cell_key(task_id: str, model: str) -> str:
    return f"{task_id}{CELL_KEY_SEPARATOR}{model}"


@dataclass(frozen=True)
class CellSpec:
    """One GEPA data instance: forecast (model, target task)."""

    cell_key: str
    task_id: str
    model: str
    bin: int
    outcome: float  # ground-truth binary outcome


class CellLoader:
    """Dict-backed GEPA DataLoader keyed by cell_key strings."""

    def __init__(self, specs: Sequence[CellSpec]):
        self._by_key = {s.cell_key: s for s in specs}
        self._order = [s.cell_key for s in specs]

    def all_ids(self) -> list[str]:
        return list(self._order)

    def fetch(self, ids: Sequence[str]) -> list[CellSpec]:
        return [self._by_key[i] for i in ids]

    def __len__(self) -> int:
        return len(self._order)


class ExperimentData:
    """Loads the manifest + Lyptus data and builds plans/loaders per phase."""

    def __init__(self, cfg: ForecasterGEPAConfig):
        self.cfg = cfg
        self.dataset: LyptusDataset = load_lyptus_dataset(
            cfg.resolved("lyptus_repo_dir"), drop_models=list(DEFAULT_DROP_MODELS)
        )
        self.manifest = self._load_or_build_manifest()
        # Bins over the FULL usable pool with the manifest's fixed explicit
        # edges, so train/val/finalist/test bin membership is consistent.
        from intra_benchmark_calibration.estimation_api import compute_explicit_bins

        self.bins: BinAssignment = compute_explicit_bins(self.dataset, self.manifest["bin_edges_minutes"])
        # Evidence shown in prompts is restricted to the 5 training benchmarks
        # in every phase (search, finalist AND test) so held-out benchmark
        # content never appears in any prompt.
        train_fams = set(self.manifest["train_families"])
        self.evidence_task_ids = frozenset(
            t.task_id for t in self.dataset.tasks if canonical_family(t.task_family) in train_fams
        )
        self._bin_of = dict(zip(self.dataset.task_ids, self.bins.bin_index_per_task, strict=False))

        self._validate_models()

    # ------------------------------------------------------------------
    def _load_or_build_manifest(self) -> dict:
        path = self.cfg.resolved("manifest_path")
        if path.exists():
            manifest = load_manifest(path)
            logger.info(f"Loaded existing task manifest from {path} (seed={manifest['seed']})")
            if manifest["seed"] != self.cfg.seed:
                raise ValueError(
                    f"Manifest seed {manifest['seed']} != config seed {self.cfg.seed}. "
                    "Use a fresh manifest_path or match the seeds."
                )
            return manifest
        manifest = build_task_manifest(
            self.dataset,
            seed=self.cfg.seed,
            edges_minutes=self.cfg.bin_edges_minutes,
            val_tasks_per_bin=self.cfg.val_tasks_per_bin,
            finalist_tasks_per_bin=self.cfg.finalist_tasks_per_bin,
        )
        write_manifest(manifest, path)
        return manifest

    def _validate_models(self) -> None:
        available = set(self.dataset.outcomes.models)
        for name, models in (
            ("forecasted_models_search", self.cfg.forecasted_models_search),
            ("forecasted_models_full", self.cfg.forecasted_models_full),
        ):
            missing = [m for m in models if m not in available]
            if missing:
                raise ValueError(f"{name} not in the outcomes matrix: {missing} (have {sorted(available)})")
        forecaster_aliases = {"sonnet 4.6"}
        clash = [m for m in self.cfg.forecasted_models_full if m.strip().lower() in forecaster_aliases]
        if clash:
            raise ValueError(f"Forecaster model must not appear in the forecasted panel: {clash}")

    # ------------------------------------------------------------------
    def _targets_by_bin(self, which: str) -> dict[int, list[str]]:
        """`which` in {'train_pool', 'val', 'finalist'} from the manifest bins."""
        return {int(b): list(entry[which]) for b, entry in self.manifest["bins"].items()}

    def _test_targets_by_bin(self) -> dict[int, list[str]]:
        out: dict[int, list[str]] = {}
        for tid in self.manifest["test"]["task_ids"]:
            out.setdefault(self._bin_of[tid], []).append(tid)
        return out

    def build_cells(
        self, targets_by_bin: dict[int, list[str]], models: Sequence[str]
    ) -> tuple[dict[str, CellPlan], list[CellSpec], list[tuple[str, str]]]:
        """Build plans + specs for (targets x models). Returns (plans, specs, dropped)."""
        plans_by_pair, dropped = build_plans_for_targets(
            dataset=self.dataset,
            bins=self.bins,
            forecasted_models=list(models),
            targets_by_bin=targets_by_bin,
            n_examples_per_source_bin=self.cfg.n_examples_per_source_bin,
            evidence_task_ids=self.evidence_task_ids,
        )
        plans: dict[str, CellPlan] = {}
        specs: list[CellSpec] = []
        for (task_id, model), plan in sorted(plans_by_pair.items()):
            key = cell_key(task_id, model)
            plans[key] = plan
            specs.append(
                CellSpec(
                    cell_key=key,
                    task_id=task_id,
                    model=model,
                    bin=plan.target_bin_j,
                    outcome=plan.target_outcome,
                )
            )
        # Stable order: (bin, task, model)
        specs.sort(key=lambda s: (s.bin, s.task_id, s.model))
        if dropped:
            logger.warning(
                f"{len(dropped)} inadmissible cells dropped: {dropped[:10]}{'...' if len(dropped) > 10 else ''}"
            )
        return plans, specs, dropped

    # ------------------------------------------------------------------
    # Phase-specific bundles
    # ------------------------------------------------------------------

    def search_phase(self):
        """Plans + loaders for the GEPA loop (train pool + val, search models).

        Returns (plans, train_loader, val_loader, sampler_tasks_by_bin) where
        `sampler_tasks_by_bin` contains only train-pool tasks whose FULL
        search-model cell set is admissible (so every gate minibatch has
        exactly n_bins x n_models cells).
        """
        models = self.cfg.forecasted_models_search
        pool_by_bin = self._targets_by_bin("train_pool")
        val_by_bin = self._targets_by_bin("val")
        merged = {b: pool_by_bin.get(b, []) + val_by_bin.get(b, []) for b in range(self.cfg.n_bins)}
        plans, specs, dropped = self.build_cells(merged, models)

        dropped_tasks = {task_id for task_id, _model in dropped}
        val_ids = {tid for ids in val_by_bin.values() for tid in ids}
        train_specs = [s for s in specs if s.task_id not in val_ids]
        val_specs = [s for s in specs if s.task_id in val_ids]

        sampler_tasks_by_bin: dict[int, list[str]] = {}
        for b in range(self.cfg.n_bins):
            complete = [t for t in pool_by_bin.get(b, []) if t not in dropped_tasks]
            incomplete = [t for t in pool_by_bin.get(b, []) if t in dropped_tasks]
            if incomplete:
                logger.warning(
                    f"Bin {b}: excluding {len(incomplete)} train-pool tasks with incomplete "
                    f"model coverage from gate sampling: {incomplete}"
                )
            assert complete, f"Bin {b}: train pool empty after admissibility filtering"
            sampler_tasks_by_bin[b] = sorted(complete)

        dropped_val = sorted({t for t, _m in dropped if t in val_ids})
        if dropped_val:
            logger.warning(f"Val tasks with missing cells (kept, with fewer cells): {dropped_val}")

        return plans, CellLoader(train_specs), CellLoader(val_specs), sampler_tasks_by_bin

    def finalist_phase(self):
        """Plans + specs for finalist re-ranking (finalist tasks x full panel)."""
        plans, specs, _dropped = self.build_cells(self._targets_by_bin("finalist"), self.cfg.forecasted_models_full)
        return plans, specs

    def test_phase(self):
        """Plans + specs for the held-out test evaluation (test tasks x full panel)."""
        targets = self._test_targets_by_bin()
        dist = {b: len(v) for b, v in sorted(targets.items())}
        logger.warning(
            f"Test set bin distribution (targets): {dist} — bins 0-1 are (near-)empty by "
            "construction; joint domain + hard-difficulty transfer."
        )
        plans, specs, _dropped = self.build_cells(targets, self.cfg.forecasted_models_full)
        return plans, specs


__all__ = [
    "CELL_KEY_SEPARATOR",
    "CellLoader",
    "CellSpec",
    "ExperimentData",
    "TRAIN_FAMILIES",
    "cell_key",
]
