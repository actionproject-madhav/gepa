# Copyright (c) 2026 SaferAI. GEPA forecaster-optimisation experiment.
"""Single configuration module for the forecaster-GEPA run (spec section 4).

Every adjustable parameter lives here with its default; a YAML file can
override any field. We deliberately stay near-native GEPA: the widened gate,
trainset=valset caching and stratified parent selection are future work.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# The single text component GEPA evolves: the round-1 estimation user-prompt
# template (single API call per cell; capability analysis stage is never run).
COMPONENT_NAME = "estimation_template"

# The full 12-model modern panel minus Sonnet 4.6 (the forecaster itself;
# self-forecasting confound). Used for finalist re-ranking and the test set.
FULL_FORECASTED_PANEL = [
    "GPT-4o",
    "Claude 3 Opus",
    "Gemini 2.5 Pro",
    "o3",
    "Opus 4",
    "DeepSeek V3.1",
    "GLM-5",
    "GPT-5.1 Codex Max",
    "GPT-5.2 Codex",
    "Opus 4.6",
    "GPT-5.3 Codex",
]

# Default custom reflection prompt: the generic GEPA template plus hard
# constraints the evolved template must keep to remain executable (str.format
# placeholders, XML output block). Must contain <curr_param> and <side_info>.
DEFAULT_REFLECTION_PROMPT = """I provided an assistant with the following prompt template to perform a probabilistic forecasting task for me:
```
<curr_param>
```

The template is instantiated per task via Python str.format. The following are examples of instantiated prompts given to the assistant, the assistant's full response for each, and feedback comparing the forecast against the ground-truth outcome (Brier score on p50; lower is better, 0 is best, 1 is worst):
```
<side_info>
```

Your task is to write an improved prompt template for the assistant, so that its p50 forecasts achieve a lower Brier score against the binary ground truth.

Read the examples and feedback carefully. Identify systematic failure modes (e.g. over/under-confidence, ignoring the pass-rate evidence, miscalibration on hard or easy tasks) and revise the template to address them. You may restructure the template freely, subject to these HARD CONSTRAINTS:

1. The template must contain the literal placeholders {forecasted_model}, {capability_profile} and {target_task_text} exactly as written (single braces). Do not add any other placeholder in single braces; any other literal brace character must be doubled ({{ or }}).
2. The template must instruct the assistant to end its response with EXACTLY this XML structure (percentiles as probabilities in [0, 1]):

<rationale>
...
</rationale>

<percentile_estimates>
<p25>0.xx</p25>
<p50>0.xx</p50>
<p75>0.xx</p75>
</percentile_estimates>

3. Do not mention difficulty bins, bin indices or held-out task sets.

Provide the new template within ``` blocks."""


@dataclass
class ForecasterGEPAConfig:
    # ---- paths ----
    # Relative paths are resolved against this repo's root (not the cwd).
    lyptus_repo_dir: str = "~/gitrepos/cyber-task-horizons-data"
    # Seed prompt: the minimal single-call template (rationale first, then the
    # XML percentiles). Derived from the prompt-ablation experiment's
    # `minimal` condition in the LLM_elicitation repo.
    seed_template_path: str = "configs/seed_prompt_minimal.txt"
    run_dir: str = "runs/forecaster_gepa"
    # Manifest is created (seeded) on first run and reused thereafter.
    manifest_path: str = "runs/forecaster_gepa/task_manifest.json"

    # ---- models ----
    forecaster_model: str = "claude-sonnet-4-6"  # reasoning_effort=off
    reflection_model: str = "anthropic/claude-opus-4-8"  # litellm slug
    reflection_max_tokens: int = 32000
    reflection_temperature: float = 1.0
    forecasted_models_search: list[str] = field(
        default_factory=lambda: ["GPT-4o", "Gemini 2.5 Pro", "Opus 4", "GPT-5.3 Codex"]
    )
    forecasted_models_full: list[str] = field(default_factory=lambda: list(FULL_FORECASTED_PANEL))

    # ---- forecaster call settings ----
    temperature: float = 1.0
    reasoning_effort: str = "off"
    max_concurrent_calls: int = 10
    rate_limit_calls: int = 200
    rate_limit_period: int = 60
    max_tokens: int = 20000
    # The forecaster receives ONLY the evolved template: empty system prompt,
    # no expert persona, no benchmark description, no ground-truth summary.
    system_prompt: str = ""

    # ---- task counts (task units; cells = tasks x models) ----
    n_bins: int = 5
    bin_edges_minutes: list[float] = field(default_factory=lambda: [0.46, 2.81, 12.82, 60.0, 180.0, 2160.0])
    n_train_tasks_per_iter: int = 1  # per bin -> 5 tasks -> 20 cells with 4 models
    val_tasks_per_bin: dict[int, int] = field(default_factory=lambda: {0: 5, 1: 5, 2: 5, 3: 3, 4: 3})
    finalist_tasks_per_bin: dict[int, int] = field(default_factory=lambda: {0: 5, 1: 5, 2: 5, 3: 3, 4: 3})
    n_examples_per_source_bin: int = 2

    # ---- GEPA loop ----
    # Native per-instance Pareto frontier; >1 filters the frontier to
    # candidates winning at least this many val cells (config knob only;
    # default 1 = exactly native behaviour).
    n_cells_won_needed_for_pareto_frontier: int = 1
    # Budget: one metric call = one evaluated cell = one Sonnet generation.
    # Cost model: seed valset (84) + 40 gate cells/iteration + 84 val cells
    # per accepted child. ~18k calls ~= 250-300 candidates at ~30% acceptance.
    max_metric_calls: int = 18000
    max_iterations: int = 300  # hard cap on candidates proposed
    use_merge: bool = False
    selection_strategy: str = "pareto"  # native per-instance Pareto parent selection

    # ---- reflection dataset ----
    n_reflection_traces: int = 20  # subset of the 20 gate cells fed to reflection
    reflection_trace_selection: str = "worst_k_stochastic"
    # Exponential bias towards worst-Brier traces when subsetting (only
    # relevant when n_reflection_traces < gate size).
    reflection_selection_temperature: float = 0.5
    reflection_prompt_template: str = DEFAULT_REFLECTION_PROMPT

    # ---- scoring ----
    # Score returned to GEPA is -Brier (GEPA maximises). Cells whose response
    # cannot be parsed (or whose template fails to format) score -1.0, the
    # worst possible -Brier.
    failure_score: float = -1.0

    # ---- finalist / test ----
    finalist_top_k: int = 5  # candidates re-ranked on the finalist set

    # ---- reproducibility / logging ----
    seed: int = 42
    wandb_project: str = "gepa-forecaster"
    wandb_entity: str | None = None
    use_wandb: bool = True
    track_best_outputs: bool = True
    # The estimator is an editable path dependency; warn if the sibling
    # LLM_elicitation checkout is not on this branch (None disables the check).
    expected_estimator_branch: str | None = "feat/ss2026_intrabenchmark_package"
    # Checkpoint hand-off via W&B artifacts: upload the (small) resume
    # checkpoint every N iterations (0 disables), and the bulky JSONL logs
    # once at the end of the run. Collaborators restart with
    # `--pull-checkpoint <entity>/<project>/checkpoint-<run_id>:latest`.
    wandb_checkpoint_every: int = 10
    wandb_log_files_at_end: bool = True

    # ---- dry-run / verification ----
    dry_run: bool = False  # stub forecaster + stub reflection, tiny budget
    dry_run_max_iterations: int = 5
    verify_sign_n_cells: int = 8  # cells used by the --verify-sign check

    def resolved(self, name: str) -> Path:
        """Expand `~` and anchor relative paths at the gepa repo root."""
        path = Path(getattr(self, name)).expanduser()
        if not path.is_absolute():
            repo_root = Path(__file__).resolve().parent.parent.parent
            path = repo_root / path
        return path


def _coerce_int_keys(d: dict) -> dict[int, int]:
    return {int(k): int(v) for k, v in d.items()}


def load_config(yaml_path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> ForecasterGEPAConfig:
    """Defaults, overridden by YAML file fields, overridden by `overrides`."""
    cfg = ForecasterGEPAConfig()
    data: dict[str, Any] = {}
    if yaml_path is not None:
        with Path(yaml_path).expanduser().open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    data.update(overrides or {})

    valid = {f.name for f in dataclasses.fields(ForecasterGEPAConfig)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    for key, value in data.items():
        if key in ("val_tasks_per_bin", "finalist_tasks_per_bin") and isinstance(value, dict):
            value = _coerce_int_keys(value)
        setattr(cfg, key, value)
    return cfg


def config_as_dict(cfg: ForecasterGEPAConfig) -> dict[str, Any]:
    return dataclasses.asdict(cfg)
