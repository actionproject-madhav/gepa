# GEPA forecaster-optimisation experiment (`forecaster_gepa`)

This fork of [gepa-ai/gepa](https://github.com/gepa-ai/gepa) runs GEPA over the
SaferAI intra-benchmark LLM-forecasting pipeline: it evolves the forecaster's
prompt template to minimise the grand Brier score of its `p50` forecasts
against binary (model, task) outcomes from the Lyptus data. Design background:
`GEPA_experiment_design_summary.md` (task allocation, gate/val/finalist/test
sets, diagnostic metrics). Every run kicked off against this repo (config,
run_dir, question, status, result) is tracked in `FORECASTER_GEPA_RUNS.md` —
check it before starting a new run to avoid duplicating one that already
exists.

The **GEPA core (`src/gepa/`) is unchanged upstream code** apart from one
resume optimisation (see "Changes to this fork" below). Everything
experiment-specific lives in `**src/forecaster_gepa/`**; the estimation
pipeline itself lives in the sibling `**LLM_elicitation`** repo, installed as
the editable package `llm-estimator` (see its `README_GEPA.md`).

## Repo layout of the experiment code


| Path                                 | What it is                                                            |
| ------------------------------------ | --------------------------------------------------------------------- |
| `src/forecaster_gepa/config.py`      | All experiment knobs (spec §4) in one dataclass; YAML overrides       |
| `src/forecaster_gepa/data.py`        | Manifest + Lyptus loading, cell plans, GEPA data loaders              |
| `src/forecaster_gepa/adapter.py`     | `GEPAAdapter`: per-cell **−Brier** scores, traces, reflective dataset |
| `src/forecaster_gepa/sampler.py`     | Gate sampler: `n_train_tasks_per_iter` tasks per FST bin × search models |
| `src/forecaster_gepa/selection.py`   | Native Pareto **parent-selection** (frontier-inclusion) helper — see below |
| `src/forecaster_gepa/acceptance.py`  | **Acceptance-gate** criteria (child vs parent on the gate minibatch) — see below |
| `src/forecaster_gepa/diagnostics.py` | W&B/JSONL instrumentation (spec §6)                                   |
| `src/forecaster_gepa/metrics.py`     | Yates/Murphy Brier decompositions, frontier stats, correlations       |
| `src/forecaster_gepa/run.py`         | CLI harness: all phases, checkpoint/resume, W&B wiring                |
| `src/forecaster_gepa/stub.py`        | Deterministic stub forecaster/reflection for dry runs                 |
| `configs/forecaster_gepa.yaml`       | Full-run config                                                       |
| `configs/forecaster_gepa_pilot.yaml` | Small real-API pilot, native acceptance gate (go/no-go diagnostics)   |
| `configs/forecaster_gepa_pilot_taskwin.yaml` | Same pilot, joint acceptance gate (aggregate AND ≥k task wins) |
| `configs/seed_prompt_minimal.txt`    | The GEPA seed prompt (minimal single-call template, rationale first)  |
| `configs/pilot_*.yaml`, `configs/verify_sign_n24.yaml` | One committed, commented config per run — the run's question and result live in `FORECASTER_GEPA_RUNS.md` |
| `configs/noise_study_prompts*/`      | Frozen prompt texts the measurement scripts evaluate (seed, measured winners, sabotage control, single-feature ablation arms) |
| `runs/task_manifest_seed42.json`     | The frozen seed-42 task split every config points at (identical gate/val/finalist/test cells across all runs) |
| `scripts/`                           | Measurement + offline-analysis tools (section below)                  |


## Changes to this fork vs upstream

- `pyproject.toml`: new `forecaster` extra; `[tool.uv.sources]` declares
`llm-estimator` as an **editable path dependency on `../LLM_elicitation`**.
- `src/gepa/core/engine.py`: on resume, the seed candidate's valset
re-evaluation is skipped (its scores are already in the checkpoint). No
algorithmic change; saves 84 API calls per restart.
- `pyrightconfig.json`: `extraPaths` so pyright resolves the editable install.
- Everything else new is under `src/forecaster_gepa/` + `configs/`.

Three forecaster knobs added after the 2026-08 parse-failure audit (full
rationale in `config.py` comments):

- `parse_retries` (default 1): an unparsable response is re-sampled once
  before scoring, instead of scoring Brier 1.0 (~9 ordinary cells of
  penalty, which landed almost entirely on evolved candidates).
- `halt_on_cell_failure` (default true): a cell that STILL fails after the
  retry halts the run loudly; checkpoint intact, re-running the same
  command resumes. Measurement scripts opt out (they study failures).
- `finalist_include_seed` (default false): `--phase finalist` also scores
  the seed on the finalist cells as a baseline row, excluded from the
  ranking stats.

## One-time setup

1. **Clone the two repos as siblings, with these names** (the editable path
  dependency is literally `../LLM_elicitation`).  
  Also clone the Lyptus data repo (parquets are committed, no LFS pull needed for our purposes):
2. **Create the environment** (from the `gepa` repo root):
  ```bash
   uv sync --extra forecaster --extra dev
  ```
   This installs the GEPA core, `llm-estimator` (editable), litellm, wandb,
   scipy etc. into `.venv`. Re-run it after pulling either repo.
3. **API key**: put `ANTHROPIC_API_KEY=...` in `LLM_elicitation/.env` (or the
  `intra_benchmark_calibration/.env`), or export it in your shell. The
   forecaster (Sonnet 4.6, native SDK) and the reflection LLM (Opus 4.8 via
   litellm) both use it; the harness propagates the `.env`-resolved key to
   litellm automatically.
4. **W&B**: `uv run wandb login` once (or set `WANDB_API_KEY`). Set
  `wandb_project` / `wandb_entity` in the config YAML. Dry runs never touch
   W&B; real runs log automatically. With `wandb_entity: null` runs go to
   the launcher's personal entity; collaborators view them via a shared
   project link (view-only). Team entities with write access for everyone
   are a paid W&B feature (free research licences exist for non-profits) —
   not required for this experiment.
5. **Check paths in the config YAML** (`configs/forecaster_gepa.yaml`):
  `lyptus_repo_dir` defaults to `~/gitrepos/cyber-task-horizons-data` —
   adjust if your checkout lives elsewhere. Relative paths (seed template,
   run dirs) are resolved against the gepa repo root, so commands work from
   any cwd.

## Running an experiment, in order

All commands from the `gepa` repo root.

```bash
# 0. Wiring check, no API calls, no W&B (stub forecaster + stub reflection):
#    runs 5 iterations end-to-end incl. finalist + test, then verifies that a
#    killed+resumed run reproduces an uninterrupted one exactly.
uv run python -m forecaster_gepa.run --dry-run --phase all
uv run python -m forecaster_gepa.run --dry-run --phase resume-check

# 1. Metric-sign check with the REAL forecaster (~16 Sonnet calls):
#    the seed prompt must outscore a deliberately-overconfident prompt.
uv run python -m forecaster_gepa.run --phase verify-sign --run-dir runs/verify_sign

# 2. PILOT (go/no-go diagnostics, ~$40, ~40 candidates — see below):
uv run python -m forecaster_gepa.run --config configs/forecaster_gepa_pilot.yaml --phase optimize

# 2b. Optional: same pilot with the joint acceptance gate, for comparison
#     (see "Two distinct noise-mitigation knobs" below):
uv run python -m forecaster_gepa.run --config configs/forecaster_gepa_pilot_taskwin.yaml --phase optimize

# 3. FULL run (~18k Sonnet cells + ~250 Opus reflections, roughly $300):
uv run python -m forecaster_gepa.run --config configs/forecaster_gepa.yaml --phase all
```

Notes:

- **Task split (§5)**: you do *not* need to run a separate split script — on
first use the harness builds the seeded manifest
(`<run_dir>/task_manifest.json`: train pool / val / finalist / test ids per
bin) via `intra_benchmark_calibration/gepa_task_sets.py` and reuses it on
every subsequent phase/resume. To inspect or pre-build it standalone:
`python intra_benchmark_calibration/gepa_task_sets.py --lyptus-repo <path> --seed 42 --output manifest.json`
(from the `LLM_elicitation` repo).
- **Single API call per cell**: the GEPA path never runs the stage-1
capability-analysis call by construction — the estimation API only issues
the estimation call (equivalent to `workflow_settings.skip_analysis: true`
in the batch pipeline). The forecaster receives the evolved template only:
empty system prompt, no expert persona, no benchmark description, no
ground-truth summary.
- **Phases** (`--phase ...`; each can also be run on its own — `finalist` and
`test` read the checkpoint left in `run_dir` by `optimize`):
  - `verify-sign` — score the seed template and a deliberately overconfident
    variant on a few val cells; assert the seed wins (−Brier sign check,
    ~16 forecaster calls).
  - `optimize` — the GEPA loop (gate eval → reflection → acceptance → full
    valset eval); resumes automatically if `run_dir` already has a checkpoint.
  - `finalist` — re-score the top-`finalist_top_k` candidates (by val Brier)
    on the finalist set × the full 11-model panel; log ranking stability
    (val-vs-finalist Spearman) and pick the winner (~1,150 calls).
  - `test` — evaluate the finalist winner AND the seed baseline on the
    held-out CVEBench+CyberGym tasks × 11 models, with per-bin Brier
    breakdown (~2,050 calls).
  - `all` — `optimize` → `finalist` → `test`.
  - `resume-check` — dry-run only: assert a killed+resumed run reproduces an
    uninterrupted one candidate-for-candidate.
- **Parallelism**: the engine hands the adapter whole batches (20 gate cells,
84 val cells, ~1,000 test cells); within a batch all cells run concurrently
via asyncio, bounded by `max_concurrent_calls` (default 10 in-flight
requests) and a client-side rate limiter (`rate_limit_calls`/
`rate_limit_period`, default 200/min). Raise these in the YAML if your
Anthropic tier allows.
- **Stop / resume**: create `<run_dir>/gepa.stop` to stop gracefully, or just
kill the process. Re-running the same command resumes exactly (candidate
pool, per-cell scores, all RNGs, W&B run id). Budget = `max_metric_calls`
(1 call = 1 cell = 1 Sonnet generation) plus a `max_iterations` cap on
candidates; both in the YAML.

## The reflection prompt

GEPA's stock reflection prompt lives in
`src/gepa/strategies/instruction_proposal.py`
(`InstructionProposalSignature.default_prompt_template`) — a generic "here is
the instruction, here are examples with feedback, write a better instruction"
template. We do **not** use it as-is: `DEFAULT_REFLECTION_PROMPT` in
`src/forecaster_gepa/config.py` replaces it via the `reflection_prompt_template`
argument to `gepa.optimize` (an official override hook — no core code was
modified). It keeps the same mechanics (the `<curr_param>` / `<side_info>`
placeholders and the answer-in-```-blocks convention that GEPA's extractor
parses) but reframes the task as probabilistic-forecast improvement (lower
Brier on p50) and adds three HARD CONSTRAINTS the evolved template must obey
to remain executable: (1) keep the literal `{forecasted_model}`,
`{capability_profile}`, `{target_task_text}` placeholders and double any other
brace, (2) end with the exact `<rationale>` + `<percentile_estimates>` XML
structure the parser expects, (3) never mention difficulty bins or held-out
sets. It is a config field, so it can be overridden per run from the YAML
(`reflection_prompt_template: ...`).

## Cross-machine resume via W&B artifacts

When W&B is active, the run uploads two artifacts:

- `checkpoint-<wandb_run_id>` (type `gepa-checkpoint`) — everything needed to
resume (`gepa_state.bin`, task manifest, run config, candidate texts, the
W&B run id). Small (a few MB), uploaded every `wandb_checkpoint_every`
iterations (default 10) and at the end.
- `logs-<wandb_run_id>` (type `gepa-logs`) — the JSONL diagnostics and gate
traces. These grow to hundreds of MB and W&B re-uploads changed files in
full per artifact version, so they are uploaded **once, at the end of the
run** (each resume adds a version). Set `wandb_log_files_at_end: false` to
skip.

Resuming a run into the *same W&B run* requires write access to the project,
so on personal entities only the person who launched it can do this (fine in
practice — it still covers the "continue my own run on another machine"
case). Anyone else can pull the checkpoint files manually and continue the
*computation* under their own W&B entity as a new run:

```bash
uv run python -m forecaster_gepa.run --config configs/forecaster_gepa.yaml \
    --run-dir runs/continued \
    --pull-checkpoint <entity>/<project>/checkpoint-<wandb_run_id>:latest \
    --phase optimize
```

This downloads the checkpoint into the (fresh) run dir and continues both the
optimisation state and the same W&B run seamlessly. Note the checkpoint does
NOT include the JSONL diagnostics accumulated so far — those keep growing in
separate files per machine (merge on iteration number post-hoc if needed) —
and a killed run's last few un-uploaded iterations are lost to W&B, so for a
same-machine restart always prefer the local `run_dir` (which is exact) over
pulling the artifact.

## Outputs (in `run_dir`)


| File                                         | Contents                                                                                  |
| -------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `gepa_state.bin`, `candidates.json`          | GEPA checkpoint + all candidate texts                                                     |
| `task_manifest.json`                         | The seeded task split (reused on resume)                                                  |
| `run_config.json`                            | Resolved config + git SHAs of both repos + seed template                                  |
| `candidates.jsonl`                           | Per-proposal records, accepted AND rejected (gate scores, task-win count, text, val scores, Yates/Murphy) |
| `diagnostics.jsonl`                          | Per-iteration frontier stats, agreement, correlations                                     |
| `gate_traces.jsonl`                          | Full rollout traces for the gate/reflection cells only                                    |
| `finalist_results.json`, `test_results.json` | End-of-run evaluations                                                                    |
| `verify_sign.json`                           | Metric-sign check result                                                                  |


## Two distinct noise-mitigation knobs — don't conflate them

Native GEPA makes two separate decisions from noisy per-cell scores, and
this fork adds an independent knob for each (see the design summary §4/§7):

1. **Parent selection / frontier inclusion** (`selection.py`,
   `n_cells_won_needed_for_pareto_frontier` in the config): which of the
   *already-accepted* candidates in the pool are eligible to be sampled as a
   parent next iteration. Native GEPA's frontier is "every candidate that is
   the best scorer on ≥1 val cell" (default `= 1`); raising it to e.g. 3
   requires ≥3 val cells won before a candidate joins the sampleable
   frontier. This does not affect whether a child gets into the pool in the
   first place — only who gets to be a parent afterwards.
2. **Acceptance gate** (`acceptance.py`, `acceptance_criterion` /
   `acceptance_min_task_wins` / `acceptance_margin_tau`): whether a proposed
   child even *enters* the pool. Native GEPA's gate is
   `sum(child gate scores) > sum(parent gate scores)` over the 20-cell
   minibatch — a single aggregate comparison. This is what a pilot run
   showed to be unreliable: only ~40–50% of accepted children also improved
   on the 84-cell val set (`diagnostics/gate_val_agreement_running`), i.e.
   close to a coin flip. The failure mode is concrete: a child that wins big
   on one gate cell and loses slightly on the other 19 can still win the
   *aggregate* Brier sum, even though it is only better on 1/20 individual
   cells — the aggregate criterion cannot tell "broad, consistent
   improvement" apart from "one lucky/unlucky cell dominating the sum".
   `TaskWinAcceptance` in `acceptance.py` adds a **task-win count**
   criterion as an alternative or an AND-joined addition:

   | `acceptance_criterion` | Accepts iff |
   |---|---|
   | `aggregate_sum` (default; native) | `sum(child) - sum(parent) > 0` |
   | `min_task_wins` | child beats parent on `>= acceptance_min_task_wins` of the individual gate cells (ignores the sum) |
   | `aggregate_sum_and_min_task_wins` | **both**: `sum(child) - sum(parent) > acceptance_margin_tau` **AND** `>= acceptance_min_task_wins` individual cells won |

   `configs/forecaster_gepa_pilot_taskwin.yaml` runs the pilot with the joint
   criterion (`min_task_wins=6` of the 20-cell gate, i.e. Jeff's "3 of 10"
   scaled up); every accept/reject decision also logs `gate_task_wins` to
   `candidates.jsonl` and W&B (`diagnostics/gate_task_wins`) **regardless of
   which criterion is active**, so the two gates can be compared side by
   side from the same run if you want. `acceptance_min_task_wins` must not
   exceed the actual gate size (`n_bins × n_train_tasks_per_iter × len(forecasted_models_search)`,
   20 by default) — the harness raises at startup if it does.

   Gate size itself is also adjustable: `n_train_tasks_per_iter` (default 1)
   sets how many tasks are sampled per FST bin per iteration; raising it
   widens the gate (e.g. `3` → 5 bins × 3 tasks × 4 models = 60 cells) at
   proportional extra cost per iteration. If you use the joint acceptance
   criterion with a wider gate, scale `acceptance_min_task_wins` accordingly.

## The pilot: what to look at for go/no-go

The pilot (`configs/forecaster_gepa_pilot.yaml`, ~40 candidates) exists to
answer the design summary's §4/§7/§8 question: is the native per-instance
Pareto frontier usable under our noisy binary per-cell signal, or do we need
the k-cells-won threshold / stratified selection? Watch these W&B panels
(all also in `diagnostics.jsonl`):


| Panel                                                                      | Healthy                                    | Pathological                                                                                                   |
| -------------------------------------------------------------------------- | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| `diagnostics/gate_val_agreement_running`                                   | ≳ 0.8 (val confirms gate direction)        | ~0.5 — gate accepts coin-flips                                                                                 |
| `diagnostics/share_one_cell_winners`                                       | low / falling                              | ~1.0 — frontier of lucky one-cell winners                                                                      |
| `diagnostics/effective_parents`                                            | small vs frontier size (mass concentrates) | ≈ frontier size (uniform noise)                                                                                |
| `diagnostics/spearman_wins_vs_extremity` and `..._partial_..._given_brier` | ≈ 0                                        | strongly positive — confident-wrong prompts win cells (improper-selection pathology, §4 of the design summary) |
| `diagnostics/best_val_brier_so_far`                                        | decreasing                                 | flat — optimiser not learning                                                                                  |
| `subsample/before` vs `subsample/after`, `val_program_average`             | native GEPA progress views                 | —                                                                                                              |


Caveat: with ~40 candidates and a realistic acceptance rate you get perhaps
10–15 accepted children, so the agreement fraction and the frontier
correlations are indicative, not conclusive — extend `max_iterations` if the
signals are borderline. If the pathologies bite, there are now two
independent remedies (see "Two distinct noise-mitigation knobs" above — pick
the one matching which mechanism looks noisy):

- Frontier looks noise-dominated (many one-cell winners, high effective
  parents) → raise `n_cells_won_needed_for_pareto_frontier` (e.g. to 3).
- Gate looks noise-dominated (`gate_val_agreement_running` near 0.5, as our
  first pilot showed) → switch `acceptance_criterion` to
  `aggregate_sum_and_min_task_wins` (see `forecaster_gepa_pilot_taskwin.yaml`).

**Stratified parent selection is deliberately NOT implemented** — per the
design summary it is future work, justified only if these cheaper remedies
prove insufficient.

## Future option (not implemented): optional data placeholders

The seed is deliberately barer than even the prompt-ablation experiment's
`minimal` condition (which kept an expert-persona system prompt; here the
system prompt is empty): the forecaster sees only `{forecasted_model}`,
`{capability_profile}` and `{target_task_text}`. GEPA's reflection can add
*text* to the template — including base-rate heuristics it distils from the
gate feedback — but it cannot add *data channels* it was never given: as
wired, `{ground_truth_summary}` (panel base-rate statistics) is filled with
an empty string and `{benchmark_description}` is not advertised, so the
computed values are unreachable.

A GEPA-native extension, should reflection visibly struggle to calibrate
base rates: supply those two values in the template data and amend the
reflection-prompt constraints to say the placeholders
`{benchmark_description}` and `{ground_truth_summary}` are *optionally
available* — then whether to include them becomes a decision the reflection
LLM makes on its own, not one we impose. Two caveats when implementing:
compute the ground-truth summary over **training-benchmark tasks only** (the
existing `LyptusOutcomes.ground_truth_summary()` aggregates the full 269-task
matrix, which leaks aggregate outcome statistics about val/finalist/test
tasks), and remember that base-rate constants distilled against the 4-model
search panel may transfer imperfectly to the 11-model finalist/test panel.

## Pitfalls

- **The estimator dependency is not pinned**: `llm-estimator` is an editable
path install, so whatever is checked out in `../LLM_elicitation` is what
runs. Keep it on `feat/ss2026_intrabenchmark_package`. The harness warns at
startup if the branch differs or the checkout is dirty, and records the
exact SHA in `run_config.json`.
- **Binning is right-closed**: the fixed FST edges (10^[−0.34…3.33] minutes)
use (a, b] intervals — this is what reproduces the 54/52/40/18/11
allocation table. Don't swap in `binning.compute_bins(strategy= 'explicit_edges')`, which is right-open and shifts ~15 boundary tasks.
- **Evidence is train-benchmarks-only in every phase, including test**: the
evidence pool (anchor/easier example tasks and pass-rate denominators in the
capability profile) is restricted to the 5 training benchmarks. During the
optimisation and finalist phases no CVEBench/CyberGym content appears in any
prompt at all; in the test phase the *target task* is of course a
CVEBench/CyberGym task (that's what is being forecast), but the evidence
around it is still exclusively from the training benchmarks — so the test
measures generalisation to fully unseen contexts (deliberate default; change
`evidence_task_ids` in `data.py::ExperimentData` if you ever want in-domain
test evidence).
- **Evolved templates must keep the `{...}` placeholders** and the XML output
block; the reflection prompt hard-constrains this, and a template that
breaks `str.format` or the parser scores −1.0 per cell (worst possible), so
the gate rejects it — but expect the occasional wasted iteration.
- **Bin labels never appear in prompts** — bins only choose cells and
evidence. Don't add them when editing prompt construction.
- **Rate limits (429s) never crash the run** — the defence is layered:
  (1) the client-side limiter (`max_concurrent_calls`, `rate_limit_calls`/
  `rate_limit_period`, all YAML-settable) throttles before requests go out;
  (2) the Anthropic SDK auto-retries 429s up to 8 times with backoff,
  honouring `Retry-After` (silent); (3) only if that exhausts does the cell
  fail — it scores −1.0 and the run continues. Failures are flagged loudly:
  a per-batch WARNING in the console/`run_log_stderr.txt` naming the failed
  cells (with an explicit "lower the throughput knobs" hint when they are
  rate-limit failures), and W&B panels `diagnostics/gate_cell_errors_*` /
  `diagnostics/val_cell_errors` (should be flat zero — any bump means failed
  cells are contaminating scores; lower `max_concurrent_calls` first).
- **Scores are −Brier** (GEPA maximises): 0 is perfect, −1 worst. Run
`--phase verify-sign` after any change to the scoring path.
- Changing `seed` invalidates an existing manifest (the harness refuses to
mix a manifest built under a different seed — use a fresh `run_dir`).



## Measurement & analysis scripts

The optimisation phases above produce runs; these tools measure and dissect
them. All read committed logs or call the API directly; none mutate runs.

| Script | What it does | Reproduce |
| --- | --- | --- |
| `scripts/val_noise_study.py` | Repeated evaluation of fixed prompts on the val/finalist cells — the reliable instrument (single val passes have rerun sd ≈ 0.0065) | `uv run python scripts/val_noise_study.py --config configs/pilot_baseline_clean.yaml --prompts seed,july_cand12 --repeats-sealed 3 --tag mytag` |
| `scripts/parse_failure_audit.py` | Exact per-run impact of Brier-1.0 parse failures (gate/val/finalist; which accept/reject decisions would flip) | `uv run python scripts/parse_failure_audit.py` |
| `scripts/retrospective_gate_analysis.py` | Replays gate decisions from a run's logs (gate↔val agreement, alternative thresholds) | `uv run python scripts/retrospective_gate_analysis.py --run-dir runs/pilot_baseline` |
| `scripts/offline_prescreen.py` | Screens acceptance-rule variants and (model, bin) Pareto aggregation against logged proposals — screening only, not a simulation of an alternative run | `uv run python scripts/offline_prescreen.py` |
| `scripts/feature_report_data.py` | Tags every evolved prompt for recurring features; accuracy + output-shift screens; matched-cell Brier/CRPS for all sealed-measured prompts | `uv run python scripts/feature_report_data.py` |

Curated results, figures and the write-ups live in the sibling repo:
`LLM_elicitation/intra_benchmark_calibration/experiments/III_gepa_optimization/`.
Raw run outputs are archived on this fork's `results/*` branches.
