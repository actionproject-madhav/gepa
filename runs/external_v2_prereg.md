# Pre-registration: external replication v2 — standardized-quantile template

Committed BEFORE any API call. Owner directive 2026-09-06: the difficulty
elicitation must mirror the probability (seed) prompt EXACTLY — same
structure, same output scaffold — with only the subject swapped, explicit
25th/50th/75th percentiles, and no anchors, so comparisons are apples to
apples.

## What changes vs the completed v1 external run (2ef9e04/6a263ea)

ONLY the elicitation template (frozen in scripts/external_elicit_v2.py):
- Structure mirrors configs/seed_prompt_minimal.txt line-for-line where the
  subject allows: opening role sentence, "Target task:" block, ask-sentence
  shape, <rationale> + <percentile_estimates><p25><p50><p75> XML scaffold,
  EMPTY system prompt (the seed harness convention; v1 used a persona
  system prompt + JSON — both disclosed v1 nuisances now removed).
- Explicit named percentiles of TIME in minutes (v1 asked "low / best
  guess / high, ~90% confident between low and high" — the interval-level
  inconsistency this version fixes).
- No anchors (bare), unchanged from v1.
Everything else identical and reused byte-for-byte: the 270 frozen stripped
inputs (runs/external_corpora/{secbench,airtbench}_tasks.jsonl), elicitors
(sonnet-4-6 primary, haiku-4-5 secondary), temp 0, 3 repeats, median
mid_minutes, one parse retry then drop-and-count. Recognition probe NOT
repeated (same inputs, already measured: 14% / 2.9%).

## Endpoints (identical to v1's prereg, plus the pairing)

- Same two primary gates: SEC-bench mean within-scaffold AUC over
  {OpenHands, SWE-agent}, instance-clustered 10k bootstrap (seed 0), CI
  must exclude 0.5; AIRTBench mean within-model AUC over the same 9
  pre-named models, per-run binary, challenge-clustered bootstrap, CI
  excluding 0.5. Aider and the rest descriptive, as before.
- NEW paired readouts (the replication's purpose): per-corpus Spearman
  between v1 and v2 elicited mids (same tasks), and the v2−v1 AUC deltas
  with cluster-bootstrap CIs — descriptive, no gate.
- Outcome sentences: PASS-BOTH -> "the external result is robust to the
  elicitation wording; the standardized template becomes the frozen
  instrument going forward." FAIL-EITHER where v1 passed -> "the external
  signal is wording-sensitive; both templates' results reported side by
  side, claims scoped accordingly." Haiku secondary: same asymmetric rule
  as v1.

## Ledger

~1,620 calls ~= $14-16. Zero Lyptus contact. v1 raw untouched; v2 writes
noise_cells_external_times_v2.jsonl. 2026-09-06.
