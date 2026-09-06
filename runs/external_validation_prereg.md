# Pre-registration: external-corpus validation of elicited-time ranking

Committed BEFORE any API call. Owner approved 2026-09-06 ("u can run it if u
are 100% sure it's sanity checked"). Design reviewed by two adversarial
reviewers (both RUN_WITH_CHANGES); every required change incorporated.
Purpose: the time-elicitation signal has only ever been measured on Lyptus
(2 held-out families, one harness). This tests it on two EXTERNAL corpora
with published per-task ground truth we never generated. Zero contact with
any Lyptus set.

## Corpora and inputs (frozen)

- SEC-bench (arXiv 2506.11791): 200 real-CVE PoC-generation instances.
  Outcomes: per-instance success for OpenHands (36/200), SWE-agent (25/200),
  Aider (6/200), all Claude-3.7-Sonnet, from github.com/SEC-bench/experiments
  report.jsonl (git-LFS: use the fetched real files, not pointer stubs;
  verified success==sanitizer_triggered 600/600; single $1.5/75-iteration
  budget per run — outcomes are budget-conditional single Bernoulli draws,
  disclosed). Elicitation input: fixed objective line ("Task: given the bug
  report below, craft a proof-of-concept input that triggers the described
  sanitizer crash in the affected project.") + the instance's bug_report
  text (never patch, never standalone sanitizer_report), truncated at 8,000
  chars, stripped by frozen regexes: CVE-\d{4}-\d+; 4-digit years; all URLs
  (the Source/GitHub/Huntr links identify vulns more precisely than CVE
  ids); date lines. Residual identifiability (product/version/function
  names) cannot be removed and is treated as world knowledge, bounded by
  the recognition probe below.
- AIRTBench (arXiv 2506.14682; HF dreadnode/AIRTBench, Apache-2.0): 70
  AI-red-teaming challenges x 12 models, per-run binary flag_found (~10
  runs/cell; gpt-4.5-preview 366 rows only, disclosed). Elicitation input:
  the challenge-info block of the first user turn, deduplicated to one text
  per challenge (70 elicitations), stripped of challenge slug, all
  dreadnode/crucible URLs, and API-key placeholders.

## Instruments (frozen)

- Elicitor 1 (primary): claude-sonnet-4-6, temp 0, the D-arm USER_TEMPLATE
  verbatim (scripts/darm_elicit.py), 3 repeats, median mid_minutes, one
  parse retry then drop-and-count (D-arm precedent: 77/888 first-pass
  errors).
- Elicitor 2 (secondary, +cheap): claude-haiku-4-5, identical protocol.
  Asymmetric interpretation committed now: if it also clears the gate where
  sonnet passes -> "elicited-human-time signal, not elicitor-specific"; if
  it fails while sonnet passes -> every claim scoped to "the frozen
  sonnet-4-6 instrument". Caveat disclosed: same provider (no OpenAI
  plumbing in this harness); cross-provider replication stays open.
- Recognition probe (all 270 stripped texts, 1 call each, sonnet): "name
  the specific CVE / challenge if you recognize it, else UNKNOWN." Report
  recognition rate per corpus and, as a robustness row, the primary AUCs on
  the unrecognized subset. Both corpora predate the elicitor cutoff, so all
  claims are "pre-cutoff external corpora, recognition-probe-bounded".

## Endpoints (frozen; baselines computed ONLY after elicited times are frozen)

PRIMARY GATE (both must hold; nothing else gates):
- A. SEC-bench: mean within-scaffold AUC over {OpenHands, SWE-agent} of
  elicited time ranking solved vs unsolved (orientation: solved = shorter
  time; ties 0.5); 10k bootstrap resamples clustered on instance, fixed
  seed 0, percentile 95% CI must exclude 0.5. (Null SD 0.047; MDE ~0.63;
  power ~0.99 if the Lyptus-range effect 0.70-0.73 transfers.)
- B. AIRTBench: mean within-model AUC over the 9 pre-named models with
  >=10 solved and >=10 unsolved challenge outcomes (claude-3-7-sonnet,
  gemini-2.5-pro-preview, gpt-4.5-preview, o3-mini,
  gemini-2.5-flash-preview, DeepSeek-R1, gemini-2.0-flash, gpt-4o,
  gemini-1.5-pro), per-run binary flag_found, predictor constant within
  challenge; 10k bootstrap clustered on the 70 challenges, seed 0, 95% CI
  excluding 0.5. (Null SD 0.060; MDE ~0.67; power 0.92 at true 0.70 but
  0.71 at 0.65 — a near-miss reads as underpower, not refutation.)
SECONDARY (directional; "beats" claimed only if a clustered CI on the
paired delta excludes 0; otherwise "comparable"):
- vs CVSS severity AUC (SEC-bench; construct mismatch severity-not-
  difficulty and range restriction 7.0-10.0 disclosed — context, not gate);
  vs patch-length (SEC-bench, context); vs the 3-level challenge_difficulty
  labels (AIRTBench; coarseness handicap disclosed).
- Aider (6/200) descriptive only. Per-scaffold and per-model tables
  descriptive.
- Level DIAGNOSTICS, non-gating: Gemini-2.5-Pro (preview-vs-GA mismatch,
  harness/budget differences) and GPT-4o family offsets on AIRTBench via
  the frozen D-arm elicited-time curve; reference band = the D-curve's own
  train-side LOFO band, frozen pre-launch at mean +0.112, sd 0.353 (clean 4
  families; computed 2026-09-06 before any external call).

## Outcome sentences (committed)

- PASS (both primary CIs exclude 0.5): "The elicited-time ranking signal
  replicates on two external corpora, three external harnesses, and two
  other labs' ground truth (pre-registered; recognition-probe-bounded;
  pre-cutoff corpora)."
- FAIL (either CI includes 0.5): "The elicited-time ranking signal did not
  replicate externally; every time-route claim in the paper is scoped to
  the Lyptus corpus."
- MIXED: report per-corpus, no aggregation, no spin; the paper carries the
  passing corpus as supporting evidence and the failing one as a stated
  limit.

## Scope disclosures

SEC-bench draws from OSS-Fuzz projects — the same source population as
CyberGym (zero CVE-id overlap with Lyptus cvebench verified) — so claim
"external corpus/harness/ground-truth provider", not "unrelated task
distribution". SEC-bench licenses a scaffold-conditional ranking claim only
(no level, no cross-model there; Claude-3.7-Sonnet is not in the Lyptus
panel); cross-model generality rests on AIRTBench's 9 models. Published
outcome distributions were inspected to set power/inclusion rules before
elicitation (public data; the predictor did not exist yet); no baseline AUC
is computed until the elicited times are frozen.

## Budget and ledger

~1,890 calls (810 sonnet + 810 haiku + 270 probe) ~= $15-20. No Lyptus-set
contact of any kind. Runner: scripts/external_elicit.py, committed with
this file; scorer runs offline at $0.

2026-09-06.
