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
| `configs/pilot_baseline.yaml` `--phase finalist` (E1) | `runs/pilot_baseline/finalist_*.{json,jsonl}` | Spec §8 metric 5, never run in July: does the seed→winner val gap (0.112→0.094) survive on the 21 sealed finalist tasks × full 11-model panel? (top-5 + seed baseline via `finalist_include_seed`) | **COMPLETE** (2026-08-02, ran in cloud) | ranking scrambled (ρ=0.10); paired bootstrap: only cand 20 CI-solid (+0.0173 [+0.0075,+0.0269]); on val itself no candidate distinguishable from seed |
| `configs/pilot_gate100.yaml` (E3) | `runs/pilot_gate100/` | Is the coin-flip acceptance gate (0.60 gate↔val agreement) fixed by evidence, i.e. a 100-cell gate? One change vs baseline: `n_train_tasks_per_iter: 5` | **COMPLETE** (2026-08-02, ran in cloud) | FALSIFIED: agreement fell to 0.10, seed never beaten (its val draw was 0.0995; July's 0.1144 later found to include a parse-failure cell — see measurement rows below; definitive same-day noise: sd 0.0065). Flip ladder from subsampling: 20-cell gate flips 30% of decisions vs 100-cell verdict (40→22%, 60→15%) |
| `configs/pilot_reflection_v2.yaml` (E4) | `runs/pilot_reflection_v2/` | Is the winning prompts' overfit texture (numeric bands, platform rules, forced anti-hedging — see rubric scan) caused by our own reflection instruction? One change vs baseline: `reflection_prompt_template` (v2 text approved 2026-08-02; only the two middle paragraphs differ from `DEFAULT_REFLECTION_PROMPT` — verified by diff) | **COMPLETE** (2026-08-02, cloud + local finalist check) | texture: fixed (0 bands/platform-rules/anti-hedging vs July's 5/4/2). Gain: NONE — winner's sealed edge −0.0013 [−0.0083,+0.0053] vs seed; run's sealed winner = the seed. (Earlier '0.1029 vs 0.1144' claim was an invalid cross-run comparison; v2's own seed drew 0.1034) |
| `configs/pilot_baseline_clean.yaml` | `runs/pilot_baseline_clean/` | What does the SAME search find on the FIXED measurement pipeline (temp 0, parse retry, halt-on-failure tripwire)? July's run was handicapped: 12 failed cells scored Brier 1.0, 3 rejections flipped by format typos (`scripts/parse_failure_audit.py`). Re-run is a fresh sample, judged on its own sealed passes | **COMPLETE** (2026-08-29 local, 11:41–12:43 optimize + finalist + 3-pass sealed) | **Zero parse failures in ~8,300 calls** (retry fired rarely, tripwire never). 26 candidates, 25/40 accepted. Val: only 2 candidates beat the seed (cand 3: 0.0972 vs seed 0.1031) — the de-noised search looks sober, July's '24 of 26 better' was the seed's inflated draw. Sealed single pass re-ranks again (val winner cand 3 → +0.004; val-rank-4 cand 7 → +0.016). **3-pass paired sealed verdict (tag clean_rerun): cand 7 +0.0161 (3/3) — a fresh bug-free run again produces a real winner; July cand 12 re-confirmed champion +0.0266 (now 8/8 lifetime paired passes); July cand 15 confirmed third winner +0.0143 (matches parsed-only prediction +0.0140 exactly); clean val-winner cand 3 ≈ null (+0.0027).** Lesson: GEPA reliably finds real sealed winners, but val-based final selection stays the weak link — select on multi-pass sealed, not val |
| `configs/pilot_accept_joint.yaml` | `runs/pilot_accept_joint/` | Sweep arm A (points 4/5): does the joint acceptance gate (sum>0 AND wins>=8/20 — the only variant that passed offline screening) select better children? | **PLANNED** | — |
| `configs/pilot_pareto_modelbin.yaml` | `runs/pilot_pareto_modelbin/` | Sweep arm B (points 4/5): do 20 aggregated (model, bin) Pareto instances (Jeff's fix for confidently-wrong-wins-the-cell; newly implemented, dry-run + resume-check PASSED) select better children? | **PLANNED** | — |
| `scripts/val_noise_study.py` feature-ablation arm (`--prompts-dir configs/noise_study_prompts_ablation --tag feature_ablation`) | `runs/noise_study/` + branch `results/clean-rerun-2026-08-29` | Causal decomposition of the winning recipe: seed + NUMERIC BANDS only vs seed + ANCHORING PROCEDURE only, 3 paired sealed passes each at temp 0 | **COMPLETE** (2026-08-30, local, ~$30) | **One-sided: bands alone +0.0201 (3/3), procedure alone +0.0062 (3/3)** — explicit numeric anchors carry ~74% of cand 12's edge; the reference-class procedure alone is real but 3x smaller. Retro-explains the v2 null (its instruction banned numbers). Beta-CRPS ordering identical (seed 0.216 / bands 0.189 / cand12 0.178). Feeds `summary/feature_synthesis_2026-08.pdf` |
| `scripts/val_noise_study.py` PRIORITY arms | `runs/noise_study/` + branch `results/priority-2026-08-10` | (1) temp-0: noise killable? (2) sealed re-check of cand 20; (3) always-99 control | **COMPLETE** (cloud 2026-08-10; container restarts on arm 2, stitched from complete blocks per `PROVENANCE_sealed_check.md`; independently re-verified 2026-08-15) | **cand 20 replicated**: better than seed in all 3 paired sealed repeats (+0.0153/+0.0201/+0.0162; 8 total cand-20 sealed passes mean 0.1153 sd 0.0018 vs seed 0.1306). **Temp-0 is NOT deterministic** (seed val sd 0.0048; serving nondeterminism) but nearly eliminates parse failures and slightly improves scores; **cand12@temp0 CONFIRMED (2026-08-15, 3 more local paired passes): sealed edge +0.0258/+0.0289/+0.0256; pooled over 5 passes: cand12 0.1045±0.0031 vs seed 0.1307±0.0010, edge +0.026, distributions non-overlapping — the project's best validated result**. Control: always-99 val passes 0.1334/0.1116/0.1561 — one pass overlaps the seed's own range, i.e. a single val pass cannot reliably detect deliberate sabotage. Bootstrap-from-one-pass overestimates rerun sd ~2.5× (val: 0.016 vs 0.0065) to ~7× (sealed: 0.012 vs 0.002) |
| local seed re-measurement (2026-08-03) | `runs/noise_study/noise_*_seed_local_5x.*` | Definitive same-day self-consistency of one val pass: seed × 5 fresh passes, same 84 cells, same hour | **COMPLETE** (~$3) | 0.1027/0.1021/0.0956/0.1041/0.1137 — **sd 0.0065, range 0.018, zero parse failures**. A single val pass cannot resolve the ~0.01–0.02 effects being optimized |
| local temp0 confirmation (2026-08-15) | `runs/noise_study/noise_*_temp0_confirm.*` | Confirm cand12@temp0's sealed edge with 3 more paired passes | **COMPLETE** (~$10) | edges +0.0258/+0.0289/+0.0256 — confirmed; pooled 5 passes +0.026 |
| `scripts/sweep_report_data.py` (pre-run screen for the two PLANNED arms above) | `runs/sweep_report_data.json` + figs 12–16 in LLM_elicitation `experiments/III_gepa_optimization/summary/` | Points-4/5 sweep collapsed offline: replay of all 77 usable logged accept/reject decisions (both 20-cell runs, parsed-only, 3 parse-flipped July rejections excluded) + engine-faithful frontier and (model, bin) scoreboard replays | **COMPLETE** (2026-08-31, $0) | `min_task_wins` alone DEAD (at every k that keeps the 18 known-good accepts it re-admits ~half of the 27 rejects; ≥23/77 decisions change at every k). Frontier threshold DEAD (cells-won misranks verified quality: winners at 3 and 6 cells vs a −0.007 prompt at 13; k=2 identical until iter 34/28 of 40). Joint k=8 keeps 15/18 good, cuts 10/32 bad, admits 0 → arm A. (model, bin): spearman(wins, quality) +0.16→+0.32 in run 1 but +0.16→−0.04 in run 2, and verified winners hold 0 groups in both runs → genuinely undecidable → arm B, flagged mixed |
| `configs/forecaster_gepa_pilot_taskwin.yaml` (Jakub's) | — | Joint acceptance gate (sum AND ≥6/20 task-wins) | **deferred with evidence** | Retro-analysis of baseline logs: every bad accept won 7–11/20 cells, so k=6 blocks only good accepts (precision 0.60→0.57). Gate is information-starved, not lucky-cell-dominated; thresholds to be re-picked from baseline+gate100 subsample curves. Audit: config safe to run, but (a) resume must re-pass the same YAML or the gate silently reverts to native, (b) parent-side rate-limit failures (−1.0) count as free child wins |

### `pilot_baseline` go/no-go readout (against README's healthy/pathological table)

> **Superseded 2026-08:** the "Healthy — optimizer is finding real improvements" reading of the descending best-val curve was wrong — best-so-far is a running minimum over single passes with sd 0.0065, so it descends under zero real improvement. Kept as the dated record; see the measurement rows above.

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
