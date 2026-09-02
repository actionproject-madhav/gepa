# Pre-registration: 27-task confirmatory holdout run (directional replication)

Committed BEFORE any API call for this run. Purpose: a fresh, never-prompted
directional check of the audit's central architecture result — "a train-family
difficulty curve sets the forecast level, the LLM contributes within-level
ranking" — after an audit day (2026-09-02) in which ~40 zero-label
constructions were scored offline against the already-spent 1,033-cell
reserved test set. Those numbers are honest (no test label entered any fit)
but not confirmatory; this run buys the only fresh cells available.

## Holdout definition (resolution rule, asserted in scripts/holdout27_run.py)

Every CVEBench/CyberGym task with (a) full-panel outcomes in the Lyptus
corpus, (b) non-empty estimation_instructions, (c) NOT in the frozen 94-task
reserved test set. Resolves to 27 tasks (7 cvebench + 20 cybergym), ~297
cells on the 11-model panel. No LLM has ever been prompted on these tasks.
They were excluded from the headline set only for lacking a curated
best_available_minutes label; they are known to skew easier than the spent
test (reservoir solve rates ~0.31-0.36 vs 0.240) and are not a random sample
— hence DIRECTIONAL replication, not confirmation.

## Frozen artifacts (recomputed 2026-09-02, three independent derivations
## agree to <=5e-4; audit-workspace analysis/freeze/frozen_artifacts.json)

- PRIMARY curve A4 ("decontaminated"): logistic P(solve) = sigmoid(a_m + b*ln
  t_minutes), fit on the 4 train families excluding intercode_ctf (the one
  family whose leave-one-family-out offset, +1.74 logits, and flat 0.88-1.00
  solve profile identify it as contamination-inflated; external evidence:
  Palisade 95% InterCode-CTF with memorization signs). b = -1.09499; a_m =
  {Claude 3 Opus 2.07601, DeepSeek V3.1 2.50880, GLM-5 4.11105, GPT-4o
  1.71192, GPT-5.1 Codex Max 4.22421, GPT-5.2 Codex 4.66212, GPT-5.3 Codex
  5.58558, Gemini 2.5 Pro 2.95258, Opus 4 3.60717, Opus 4.6 5.68292, o3
  3.47414}. Known spent-test performance: Brier 0.1466, mean prediction
  0.2409 vs realized 0.2401.
- SENSITIVITY curve A5 (all 5 train families): b = -1.02103; a_m = {Claude 3
  Opus 2.11439, DeepSeek V3.1 2.94331, GLM-5 4.16003, GPT-4o 2.07620,
  GPT-5.1 Codex Max 4.34788, GPT-5.2 Codex 4.57713, GPT-5.3 Codex 5.48137,
  Gemini 2.5 Pro 3.28201, Opus 4 3.83139, Opus 4.6 5.56683, o3 3.73544}.
  Spent-test 0.1522.
- BLEND: p = sigmoid(0.64*logit(curve) + 0.36*logit(seed p50 clipped to
  [0.01, 0.99])); w = 0.36 was chosen by leave-one-family-out on train-family
  cells only (spent-test values: A4 blend 0.1398, A5 blend 0.1448).
- CAP rule: p50 <- min(p50, printed pass rate of the model's nearest easier
  shown bin) (spent-test 0.1633 -> 0.1556; sealed 0.1305 -> 0.1111).
- Difficulty ruler for holdout tasks (they have no curated FST):
  model_estimate_minutes -> ln minutes into the curve; proxy bin for
  evidence-template selection via edges (0.46, 2.81, 12.82, 60, 180, 2160],
  clipped to [1, 4]. The target's time/bin is never shown to the forecaster.

## Run protocol

1. Smoke: `uv run python scripts/holdout27_run.py --limit-cells 2 --repeats 1`
   (~$0.05) — verifies plumbing + cost model.
2. Main: `uv run python scripts/holdout27_run.py` — seed prompt only, 2
   repeats x ~297 cells ~= 594 calls ~= $8 at the verified ~$0.012/call.
   Config pilot_baseline_clean.yaml (temperature 0.0), identical to the
   spent-test measurement.
3. Offline ($0): score seed, cap-clamped seed, curves A4/A5, blends A4/A5 on
   the same cells, task-clustered CIs.
4. OPTIONAL second arm (~$8), only if step 3 is directionally positive:
   seed + one-sentence cap instruction, scored for instruction compliance vs
   the mechanical clamp.
- NOT spent: the 1,033-cell reserved test set (no 9th arm); sealed set (no
  new passes).

## Pre-registered readouts and wording

- POWERED check (level transfer): |mean prediction - realized solve rate| for
  curve A4 vs for the seed; and whether A4's realized holdout family offsets
  fall inside the clean-corpus LOFO band (mean -0.06, sd 0.40 logits).
- DIRECTIONAL checks (MDE ~0.022-0.027 at 27 task clusters vs expected edges
  0.007-0.019 — power ~50-68% vs seed, ~11-15% vs curve; stated in advance):
  blend A4 <= seed; capped seed <= seed; blend not worse than curve by >MDE.
- Outcome sentences (committed now): WIN -> "the architecture directionally
  replicated on 27 never-prompted tasks (point estimates, CIs; below-MDE
  margins stay directional)". NULL -> "direction failed to replicate; the
  architecture is unconfirmed on fresh data; spent-set results stand as
  descriptive with an explicit no-fresh-confirmation flag". INVERSION ->
  same as NULL plus the inversion reported prominently.
- Known-in-advance disclosures: curve-side offsets on these 27 tasks were
  already computed during the audit (reservoir deltas cybergym -0.49,
  cvebench -0.36/-0.37 under two time proxies); the seed has never produced
  p50s on them; the 7 cvebench tasks' human_minutes is a constant 60.0
  placeholder (uninformative); model_estimate_minutes is model-generated and
  rank-correlates only ~0.33 with curated FST on the spent test.

2026-09-02. Runner: scripts/holdout27_run.py (committed with this file).
