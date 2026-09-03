# Pre-registration: D-arm — difficulty elicitation instead of probability elicitation

Committed before any API call. 2026-09-03.

## Question

Does asking the SAME LLM (claude-sonnet-4-6, temperature 0.0, same harness as
every probability arm) for the task's human solve time — and converting to a
probability through a 12-parameter curve fit on audited training benchmarks —
forecast better than asking it for the probability directly (the seed and
GEPA-optimized prompts)?

## Design

- Step 1 (`scripts/darm_elicit.py`): one call per TASK, 3 repeats. Prompt =
  the task's estimation_instructions text plus the frozen instructions below;
  nothing else. Output: low/mid/high minutes (90%-style bounds + best guess).
  Tasks: all usable train-family tasks + the frozen 94-task reserved test +
  the 27-task holdout (~296 tasks, ~890 calls, ~$10-15).
- Step 2: per task, median across repeats of low/mid/high.
- Step 3 (offline, zero test/holdout outcomes — asserted in code): logistic
  fit on train families EXCLUDING intercode_ctf (the audited corpus): one
  intercept per panel model + one shared slope on ln(mid_minutes).
- Step 4: per cell, p50 = sigmoid(a_m + b*ln mid); p25/p75 = same curve at
  high/low minutes (wider time -> lower probability, so p25 uses high_minutes).
- Step 5: paired per-cell Brier on identical cells vs seed and GEPA arms:
  reserved test (1,033), sealed (230), holdout27 (297); cell-level and
  task-clustered t. Reference lines reported, clearly labeled: GT LOO table
  0.1488 (oracle: target label + test outcomes), curve-on-curated-FST 0.1466
  (target label, no outcomes), curve-on-dataset-ME (prior $0 preview).

## Frozen elicitation prompt

System: "You are an experienced cybersecurity professional estimating how
long security tasks take skilled humans to complete."

User template (verbatim in scripts/darm_elicit.py USER_TEMPLATE): estimate
minutes for a skilled human professional working alone with standard tooling,
from reading the spec to delivering the artifact; do NOT consider whether an
AI could do it and give no probability; reply with one JSON object
{"low_minutes", "mid_minutes", "high_minutes"}.

## Pre-registered readouts and interpretation rules

- PRIMARY: paired delta vs the seed prompt on (a) the reserved test and
  (b) the holdout27 cells, task-clustered t reported. The reserved test is
  comparability-grade (its labels have been studied since 2026-09-01); the
  holdout is the cleaner readout.
- SECONDARY: paired deltas vs july_cand12 / clean_cand7 / joint_cand5 /
  modelbin_cand18 on the reserved test; sealed non-regression vs seed
  (D-arm sealed Brier not worse than seed's 0.1305 by >0.005); level check
  (mean p50 vs realized rate per set); monotonicity by construction (noted,
  not tested).
- Elicitation-quality diagnostics (reported regardless): parse-failure rate,
  repeat agreement of mid_minutes, rank correlation of elicited mids with
  curated FST on train tasks (descriptive only).
- Outcome sentences, committed now. WIN (D-arm beats seed on both primary
  sets, task-clustered): "directly supported; still one dataset, one
  estimator model, and the reserved test is comparability-grade." MIXED
  (wins one set): "suggestive; report both, no strong claim." LOSS: "the
  elicited-time route underperforms its dataset-ME preview; the idea's
  support reverts to the pre-existing ME evidence only, and the write-up
  must say the fresh elicitation failed."
- No success criterion may be revised after data arrives. Nothing further is
  spent on the reserved test beyond this scoring (no new prompt arms).

## Leakage statement

The elicitation prompt contains only the target task's specification text.
No FST, no difficulty bin, no benchmark family label, no pass rates, no
example tasks, no solution walkthrough, no outcome reaches the estimator.
The curve fit uses train-family outcomes only. The target's curated FST is
used nowhere in the D-arm pipeline. Residual, disclosed, not fixable: the
estimator may recognize public CVEs from pretraining — legitimate world
knowledge available to any deployed forecaster; per-cell model outcomes are
not public. Design-level disclosure: this arm was designed after the
reserved-test labels had been studied (hence the holdout readout and the
frozen outcome sentences above).

## Information-fairness ordering (who sees what about the target)

D-arm: task text only < seed/GEPA prompts: task text + other-bin pass-rate
tables (target bin partially inferable) < curve-on-FST: target's curated
difficulty label < GT LOO table: target label + test outcomes.
