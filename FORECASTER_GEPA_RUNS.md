# Forecaster-GEPA runs index

Tracks every run kicked off against this repo: which config, which run_dir,
what question it answers, and where the result lives. `runs/` itself is
gitignored (regenerable scratch); this file is the durable pointer into it.
Same pattern as `LLM_elicitation/intra_benchmark_calibration/experiments/README.md`.

All runs share one task split — `runs/task_manifest_seed42.json` (seed 42) —
so results across rows are directly comparable unless noted otherwise.

## Verification / wiring checks (free or near-free)

| Config | run_dir | What it checked | Result |
|---|---|---|---|
| `--dry-run --phase all` | `runs/forecaster_gepa_dry/` | Stub forecaster+reflection, full phase pipeline wiring | PASSED |
| `--dry-run --phase resume-check` | `runs/forecaster_gepa_dry/resume_check_{A,B}/` | Kill+resume reproduces an uninterrupted run exactly | PASSED (6 identical candidates after 8 iters, interrupted at 5) |
| default (`verify-sign`, no config) | `runs/verify_sign_n8/` | Seed must outscore a deliberately-bad "always p=0.99" prompt, N=8 cells | **FAILED** — near-tie (seed Brier 0.0967 vs bad 0.0960); N=8 too noisy, not a sign bug (see below) |
| `configs/verify_sign_n24.yaml` | `runs/verify_sign_n24/` | Same check, N=24 (3x, superset of the N=8 cells) | PASSED clearly (seed Brier 0.1193 vs bad 0.1448) |

## Real optimization runs (cost real $)

| Config | run_dir | Question | Status | Headline |
|---|---|---|---|---|
| `configs/forecaster_gepa_pilot.yaml` (Jakub's) | not shared (Jakub's local machine / personal W&B) | Is the native per-instance Pareto frontier usable under noisy binary Brier, or do we need the k-cells-won threshold? | Smoke-tested by Jakub, ~40 candidates | not shared/available yet |
| `configs/pilot_baseline.yaml` (Madhav's) | `runs/pilot_baseline/` | Same question — independent run, same config/seed, own W&B account | **COMPLETE** (2026-07-19 13:44–14:39) | 26 candidates, best val Brier 0.0937 (from seed ~0.11+), 3784/4000 metric calls. Signals mixed — see below |
| `configs/pilot_baseline.yaml` `--phase finalist` (E1) | `runs/pilot_baseline/finalist_*.{json,jsonl}` | Spec §8 metric 5, never run in July: does the seed→winner val gap (0.112→0.094) survive on the 21 sealed finalist tasks × full 11-model panel? (top-5 + seed baseline via `finalist_include_seed`) | **planned** (~1,380 cells, ~$5–12) | pre-declared readout: seed-vs-winner finalist gap + top-5 ranking stability (Spearman). A vanishing gap = the July gain was val-selection luck |
| `configs/pilot_gate100.yaml` (E3) | `runs/pilot_gate100/` | Is the coin-flip acceptance gate (0.60 gate↔val agreement) fixed by evidence, i.e. a 100-cell gate? One change vs baseline: `n_train_tasks_per_iter: 5` | **planned** (~9,400–10,200 cells, ~$60–95, ~2–2.5 h) | pre-declared readout: `gate_val_agreement_running` (healthy ≥~0.8) + offline task-stratified subsample curve at 20/40/60 via `scripts/retrospective_gate_analysis.py`. Best-val Brier NOT compared across runs (single stochastic draws) |
| `configs/pilot_reflection_v2.yaml` (E4) | `runs/pilot_reflection_v2/` | Is the winning prompts' overfit texture (numeric bands, platform rules, forced anti-hedging — see rubric scan) caused by our own reflection instruction? One change vs baseline: `reflection_prompt_template` (v2 text approved 2026-08-02; only the two middle paragraphs differ from `DEFAULT_REFLECTION_PROMPT` — verified by diff) | **ready to run** (~3,800–4,000 cells, ~$25–40) | pre-declared readout: rubric features (bands/platform-rules/anti-hedging ↓?) with val Brier not collapsing to seed level |
| `configs/forecaster_gepa_pilot_taskwin.yaml` (Jakub's) | — | Joint acceptance gate (sum AND ≥6/20 task-wins) | **deferred with evidence** | Retro-analysis of baseline logs: every bad accept won 7–11/20 cells, so k=6 blocks only good accepts (precision 0.60→0.57). Gate is information-starved, not lucky-cell-dominated; thresholds to be re-picked from baseline+gate100 subsample curves. Audit: config safe to run, but (a) resume must re-pass the same YAML or the gate silently reverts to native, (b) parent-side rate-limit failures (−1.0) count as free child wins |

### `pilot_baseline` go/no-go readout (against README's healthy/pathological table)

From `runs/pilot_baseline/diagnostics.jsonl`, final iteration (40):

| Panel | Value | Read |
|---|---|---|
| `best_val_brier_so_far` | 0.1118 → 0.0937, monotonically decreasing | **Healthy** — optimizer is finding real improvements, not flat |
| `share_winning_exactly_one_cell` | 0.095 | **Healthy** — most frontier members win multiple cells, not lucky one-offs |
| `gate_val_agreement_running` | dipped to 0.375–0.45 mid-run, recovered to 0.6 by the end | **Borderline** — never reached the "healthy" ≳0.8 bar; spent a third of the run in pathological (~0.4–0.5, "gate accepts coin-flips") territory |
| `effective_parents` vs `frontier_size` | 16.46 / 21 | **Borderline-to-concerning** — close to frontier size, i.e. selection is closer to spread-uniformly-over-noise than concentrated-on-a-few-strong-candidates |
| `parent_selection_entropy_nats` | 2.80 (max possible ln(21)=3.05) | Consistent with the above — near-maximum entropy, not concentrated |
| `spearman_wins_vs_extremity` (+ partial) | NaN | Undefined this run — can't read at all |

**Net read:** mixed, exactly the "indicative, not conclusive" outcome the README predicted for a 40-candidate pilot. The clearest positive signal (best-val-Brier decreasing) says GEPA is doing *something* real. But both frontier-concentration diagnostics lean toward the noise-dominated end, and gate/val agreement sits below the healthy threshold — a real empirical case for trying `n_cells_won_needed_for_pareto_frontier: 3` next (`pilot_kwon3.yaml`), rather than treating the native frontier as validated.

### Known instrumentation quirk (spec §8 metric 4)

`diagnostics/spearman_wins_vs_extremity` was NaN throughout the baseline run:
per-candidate `extremity` needs val-eval outputs (`outputs_by_val_id`), which
are not populated on the accepted-child valset path. $0 remedy used instead
of patching the harness: compute each candidate's mean |p50−0.5| from its
own gate traces (`gate_traces.jsonl`) post-hoc. Flagged to Jakub.

## Conventions

- One committed config per row in `configs/`, filename == `run_dir` basename.
- Every new config explicitly sets `manifest_path: runs/task_manifest_seed42.json`
  (don't rely on the per-config default — that's what fragmented the two
  verify-sign runs above before this file existed).
- `run_config.json` (auto-written into every run_dir) is the actual
  provenance record — full resolved config + git SHA of both repos + seed
  template. This file is just the human index into that.
- W&B: `use_wandb: true` with no `wandb_entity` set logs under whichever
  personal account is currently `wandb login`-ed — don't hardcode a username,
  so this file works unmodified for any collaborator with their own free tier
  account.
