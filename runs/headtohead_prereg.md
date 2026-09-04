# Pre-registration: head-to-head elicitation experiment (ARM T + ARM P0)

Committed BEFORE any API call. Owner-approved 2026-09-03 ("if u think that
this experiment is principled then u can run this... make sure this is head
to head and no bs and sanity check from different angles"). Design reviewed
by three independent adversarial reviewers before this file was written; all
verdicts RUN_WITH_CHANGES; every required change is incorporated below.

## Supersession (explicit)

`runs/darm_prereg.md` (commit cc8f678) froze: "Nothing further is spent on
the reserved test beyond this scoring (no new prompt arms)." The owner has
explicitly approved this one further experiment. That commitment is hereby
SUPERSEDED with a terminal replacement: ARM T and ARM P0 are the FINAL
reserved-test arms of this program. After them: reserved-test lifetime
prompt-arms = 11, all reserved-test numbers comparability-grade forever,
holdout27 lifetime touches = 3 (ARM T only; ARM P0 never touches it),
sealed LLM passes unchanged (both new arms score sealed offline or not at
all). No further arms on any of these sets regardless of outcome.

## The question

The D-arm changed two things at once (probability->time ask AND rich
evidence->bare prompt). These two arms complete the 2x2 so the changes
separate. Factor impurity disclosed: the "ask" factor is a bundle
{probability<->time, model name present<->absent, per-cell<->per-task call,
direct p50<->curve mapping}; the "evidence" factor is ask-appropriate
anchoring content (outcome labels for probability, time labels for time),
not identical text.

## Arms (frozen)

- ARM T (time ask + evidence): `scripts/armT_elicit.py`, committed with this
  file; template frozen in the script. Seed skeleton; evidence block = the
  modal (anchor + 2 easier) roster per source bin from the frozen test-phase
  plans (identical across the 11 models in bins 0-3; bin 4 has one GLM-5
  variant, mode wins), shown all-except-target-bin, each example labeled
  "[estimated human solve time: X minutes]" with fst_minutes printed
  verbatim; NO model name; NO pass rates; NO outcome tags; ask = target's
  time as p25/p50/p75 minutes in the seed's XML shape. One call per task;
  296 unique tasks (154 train_fit + 21 sealed + 94 test + 27 holdout27; the
  21 sealed tasks are INSIDE the train families - counted once); 3 repeats;
  claude-sonnet-4-6, temperature 0, empty system prompt (persona folded into
  the user turn); parse failure -> one retry -> task dropped from pairing
  and counted (never scored 1.0). ~888 calls, ~$11.
  Disclosures: the modal roster resolves to 15 unique tasks (the audit's
  count of 16 included a GLM-5 bin-4 variant that the mode collapses); their
  time-label sources are mixed (completion-timed, expert-estimated, and
  first-blood), printed verbatim - named, part of the treatment; the all-except-target structure leaves the target's bin
  inferable from the missing header (same as seed; MORE target information
  than the bare D-arm: information ordering D < T ~= seed), so the T-vs-D
  contrast is "does the anchored evidence pack, including the inferable
  target bin, help time estimation?", not pure anchoring. Output shape XML
  vs D's JSON: disclosed nuisance. Holdout27 target bins use
  holdout27_run.py's ME-proxy rule verbatim.
- ARM P0 (probability ask, bare): `configs/noise_study_prompts/p0_bare.txt`
  = seed.txt with ONLY the evidence intro line and {capability_profile}
  block deleted; model name kept; run via the standard
  `val_noise_study.py --prompts p0_bare --repeats-test 2 --tag p0_bare`
  (reserved test only; NOT sealed - the in-distribution answer is
  predetermined by the verified sufficiency result; NOT holdout27 - no
  spend of the semi-fresh set on a mechanism control). ~2,066 calls, ~$25.

## Conversion and scoring (frozen)

Curve: per-model intercepts + one shared slope on ln(median-over-repeats
elicited mid minutes), logistic MLE, fit on clean-train-minus-sealed cells
ONLY (families cybashbench, nl2bash, nyuctf, cybench; intercode_ctf
excluded per the audit; the 21 sealed tasks excluded so sealed scoring is
out-of-sample). p50 = curve(mid); p25/p75 = curve at high/low minutes (the
darm_prereg step-4 inversion). $0 correction adopted from review: the
D-arm is RE-SCORED with this identical sealed-excluded refit for every
T-vs-D comparison (its original numbers reported alongside, labeled; its
published sealed 0.1420 was partly in-sample - 13.8% of fit cells).

## Readouts (all pre-committed; no other slicing will be reported)

1. THE ONE CONFIRMATORY TEST: ARM T vs seed on the reserved test, paired
   per-task delta, task-clustered t (n=94), ONE-SIDED (T better), alpha
   0.025, WITH a holdout27 sign gate (T's holdout27 point delta vs the
   fresh seed 0.1898 must also be negative for the win sentence). Joint
   null false-positive ~1.25%. Disclosed: 10th arm on a spent set; the T
   hypothesis was composed after seeing D-arm results; power 43-72% for a
   D-sized effect. Comparators pinned: seed test 0.1633 (per-cell mean of
   2 repeats), holdout27 seed 0.1898, sealed seed 0.1305.
2. SEALED GUARDRAIL (descriptive): point estimate + task-clustered 95% CI;
   TRIPWIRE: sealed delta >= +0.03 AND one-sided p < 0.05 -> declare
   in-distribution regression, no-adopt regardless of OOD result. Sealed
   n=21 resolves only |delta| >= ~0.035-0.05: "fixed the D-arm sealed gap"
   is unanswerable and will not be claimed in any direction. Pre-registered
   secondary (contamination hypothesis test, gate fired at $0: the D-arm
   sealed delta concentrates in the 3 intercode_ctf tasks): sealed deltas
   reported with and without intercode_ctf cells, both descriptive.
3. 2x2 ESTIMATION (no win/loss language): on the reserved test, ask main =
   0.5[(T-seed)+(D-P0)], context main = 0.5[(T-D)+(seed-P0)], interaction =
   (T-D)-(seed-P0), task-clustered 95% CIs, point estimates only.
4. T-vs-D mechanism readouts (where power actually exists): within-model
   AUC of elicited times on reserved-test cells (references: dataset-ME
   0.788, curated-FST 0.648, D-arm times); anchoring-drag diagnostic (if
   T's train Spearman-to-curated-FST rises above D's 0.85 while its test
   AUC falls toward 0.648, the anchors dragged rankings toward the inferior
   ruler - verdict favors bare elicitation); readback diagnostics (share of
   elicited mids within +/-10% of a printed anchor time; train Spearman
   overall and within-bin); level/scale (mean forecast vs realized).
   T-vs-D Brier is expected ~null (plausible gap 0.004-0.006, below this
   venue's resolution) and is estimation-only; it CANNOT promote either
   time arm - any deployment choice needs a fresh holdout.
5. Deployment replay ($0): k=20/50 global level-knob, prequential, on T's
   OOD cells. Pre-committed reading: raw OOD level overshoot is expected
   under the shipped thesis; the knob is the deployed fix.

## Outcome sentences (blanks filled after scoring; nothing else claimed)

P1-WIN: "ARM T beat seed on the reserved test (delta=__, one-sided
task-clustered p<0.025, 10th arm disclosed) and holdout27 agreed in sign
(delta=__): the D-arm OOD win survives the evidence-matched minimal pair;
comparability-grade, adoption still requires a fresh holdout."
P1-NULL: "ARM T did not clear the pre-set threshold or holdout27 disagreed
in sign: the time-ask OOD advantage does not replicate under matched
evidence; we do not adopt T and run no further arms."
P1-INVERSION: "ARM T scored worse than seed OOD (delta=+__): evidence
anchoring removes the time-ask advantage; D remains the best time arm and
rich-evidence framing is implicated."
P2-PASS: "Sealed delta=__ [CI]; below the +0.03 tripwire. Sealed can only
detect |delta|>=~0.035-0.05, so whether T fixed the D-arm +0.012 sealed gap
(itself t=0.70 n.s.) remains unanswerable at this budget."
P2-TRIP: "Sealed delta=__ >= +0.03 (one-sided p<0.05): T regresses
in-distribution; not adoptable regardless of OOD result."
MECH: "On the reserved test the seed-to-D improvement decomposes as ask __
[CI], context __ [CI], interaction __ [CI]; point estimates only."

## Hygiene

Runtime asserts in armT_elicit.py: target never among its evidence
examples; target's own time never printed in its prompt; all evidence tasks
train-family; one rendered evidence block per bin (byte-identical across
targets sharing a bin). 2-task smoke run before each paid launch. The
arvo-22 reservoir is skipped with cause: those tasks have no
estimation_instructions, so no elicitation prompt exists for them.
Total new spend ~$36; session total ~$57 of $100.

2026-09-03. Runners committed with this file: scripts/armT_elicit.py,
configs/noise_study_prompts/p0_bare.txt.
