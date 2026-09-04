# Pre-registration: sabotage-seed positive control (pilot_sabotage_seed)

Committed BEFORE any API call. Requested by Jakub (Slack, quoted below),
owner-approved 2026-09-03 with cost quoted (~$157; +$8 same-session
yardstick arm added on reviewer instruction; session total ~$222, which the
owner's approval supersedes over the audit-mission ledger's ~$200 note).
Design reviewed by two independent adversarial reviewers before this file
was written (both RUN_WITH_CHANGES); every required change is incorporated.

## The question (premise stated correctly)

Jakub: "Maybe we should use a deliberately bad prompt as the starting
point? Then we know there is room for improvement and we can observe
whether GEPA indeed leads to improvements... it would be interesting to see
whether optimisations started with bad prompts can reach the same levels
that we reach currently (when we start with a minimal prompt)."

Correct framing (NOT "GEPA gives no gains" — three good-seed runs produced
replicated 3-pass sealed winners, +0.014..+0.026): good-seed searches
harvest only small sealed gains over a seed already near the in-distribution
statistical ceiling, so small gains cannot distinguish "little headroom
left" from "weak optimizer". This control plants LARGE, known,
instruction-level headroom and asks: (a) does the identical search reliably
harvest it, and (b) do bad starts converge to the good-start winner band
(3-pass sealed 0.1045-0.1153)?

## Arms and machinery (frozen)

- Config `configs/pilot_sabotage_seed.yaml`: byte-equivalent to
  `pilot_baseline_clean.yaml` except run_dir + seed_template_path
  (reviewer-verified by diff). Native acceptance, default reflection
  prompt, manifest seed42, temp 0, parse retry, halt tripwire, 40
  iterations / 4,000 metric calls. Src drift since the clean run (cdf14a2)
  is additive-only model_bin machinery with native defaults — head-to-head
  comparability with pilot_baseline and pilot_baseline_clean holds.
- Sabotage seed `configs/sabotage_seed.txt` (copied byte-identically to
  `configs/noise_study_prompts/sabotage_seed.txt` for measurement): the
  real seed's scaffold and placeholders kept (parseable; the reflection
  hard constraint forces {capability_profile} to remain), with three named
  instruction-level degradations — evidence declared misleading/do-not-use,
  assume-very-likely-solves default anchor, no-deliberation rule. All three
  sit inside GEPA's action space (template text), so the control is fair.
  Disclosures: (i) the default reflection prompt itself names "ignoring the
  pass-rate evidence" as an example failure mode, so the planted defect is
  inside the reflection's advertised hypothesis space — a fast repair shows
  the loop works end-to-end, not that GEPA diagnoses novel defects; (ii)
  deleting the three bad sentences lands essentially at the real seed, so
  full recovery licenses "the loop exploits large instruction-level
  headroom", never "GEPA discovers novel gains". This run makes optimizer-
  behavior claims only; it is distinct from BOTH prior always-99 uses (the
  verify-sign wiring check and the PRIORITY-arms measurement control, which
  showed a single val pass cannot detect sabotage — the reason the pre-gate
  below is placed on sealed).

## Pre-gate (before optimize; abort rule)

`uv run python scripts/val_noise_study.py --config configs/pilot_baseline_clean.yaml
 --prompts sabotage_seed --repeats-val 1 --repeats-sealed 1 --repeats-test 0
 --tag sabotage_pregate` (~$4).
LAUNCH ONLY IF: sealed >= +0.03 worse than the real seed's 0.1305 AND val
>= +0.015 worse than the same-config seed val reference (visibility on the
instrument the search selects on). If sealed trips but val does not: HOLD
and investigate (damage invisible to the optimizer's own instruments makes
a no-recovery outcome uninterpretable). Expected magnitudes: fully
compliant sabotage ~0.21-0.28 sealed / ~0.29-0.41 val (base rates
0.717/0.583); the always-99 precedent (~90% non-compliance when the
evidence table is present) makes instruction-resistance the main gate-fail
risk. One sealed pass suffices (rerun sd ~0.002; the +0.03 bar is >10
sigma).

## Phases (frozen commands; --phase all and --phase test are BANNED for
## this run — the test phase would touch the reserved 1,033-cell set, which
## is closed forever per runs/headtohead_prereg.md)

1. `uv run python -m forecaster_gepa.run --config configs/pilot_sabotage_seed.yaml --phase optimize` (~$110)
2. `... --phase finalist` (~$16; finalist_include_seed rides candidate 0 =
   the sabotage seed)
3. Export top-2 candidates as `configs/noise_study_prompts/sab_cand<N>.txt`.
   SELECTION RULE (frozen): top-2 by finalist-set single-pass Brier among
   the top-5-by-val, EXCLUDING candidate 0; if finalist winner = val winner,
   the second slot goes to finalist rank 2. (Val-only selection misled 3/3
   prior runs; the finalist sealed pass is the established selector.)
4. Readout: `uv run python scripts/val_noise_study.py --config
   configs/pilot_baseline_clean.yaml --prompts
   seed,sabotage_seed,july_cand12,sab_cand_a,sab_cand_b --repeats-val 0
   --repeats-sealed 3 --repeats-test 0 --tag sabotage_rec` (~$40).
   july_cand12 = the same-session champion yardstick (every prior 3-pass
   readout carried it; band claims must not rest on cross-session numbers).

## Pre-committed readout table and outcome sentences

Rows: real seed (S*), sabotage seed (B), july_cand12 (C12), sab_cand_a/b
(W = better of the two by 3-pass mean). Each: 3-pass sealed mean + per-pass
paired edges. Also reported, mirroring prior run rows: n candidates,
accepts/40, val-vs-finalist agreement.
- OUTCOME A (full recovery + convergence): W beats S* in 3/3 paired passes
  AND W is within +0.010 of C12's same-session 3-pass mean -> "from a
  deliberately damaged start the identical search both repaired the damage
  and reached the good-seed winner band: the optimization loop demonstrably
  harvests large headroom when it exists; the small good-seed gains are
  consistent with limited remaining headroom, not a broken optimizer. No
  transfer claim; the reserved test stays closed."
- OUTCOME B (repair only): W beats B in 3/3 paired passes but misses the
  band (fails vs S* or > +0.010 above C12) -> "the loop repairs planted
  damage but does not reach good-seed levels from a bad start within the
  identical 40-iteration budget: bad starts are costlier; n=1 search per
  seed condition, native-run spread (+0.016 vs +0.026) bounds precision —
  band comparison, not point comparison."
- OUTCOME C (no recovery): W fails to beat B in 3/3 paired passes ->
  "conditional on the pre-gate having shown the damage visible on val, the
  identical search failed to harvest large known instruction-level
  headroom: evidence of optimizer/selection weakness in this regime."
No other slicing is reported.

## Ledger and scope

Sealed-set exposure increases by ~19-22 passes (pre-gate 1; finalist ~6
single passes; readout 3x5) on ~100 lifetime — acceptable because every
claim here is an in-distribution optimizer-behavior claim. SCOPING NOTE
reconciling runs/headtohead_prereg.md ("no further arms on any of these
sets"): the terminal teeth are the reserved test and holdout27, which this
run NEVER touches under any phase; sealed remains the GEPA program's
standard in-distribution verdict instrument, per owner approval 2026-09-03.
Launch only after the running head-to-head chain's process exits (rate
limiters are per-process; co-running would double the real API rate and
risk tripping the optimize halt tripwire). Launch git SHA recorded in the
runs-index row at launch.

2026-09-03. Files committed with this prereg: configs/pilot_sabotage_seed.yaml,
configs/sabotage_seed.txt, configs/noise_study_prompts/sabotage_seed.txt.

## Amendment 1 (2026-09-03, pre-launch; gate fired, redesign per the abort rule)

Pre-gate v1 result: sabotage v1 (evidence-declared-misleading +
assume-likely-solves + no-deliberation) scored val 0.1116 / sealed 0.1342 —
only +0.007 val / +0.004 sealed worse than the seed. BOTH bars missed: the
forecaster largely ignored the anti-evidence instructions with the table
present (replicating the always-99 non-compliance precedent at
instruction level; consistent with the audit's sufficiency finding that the
model interpolates the printed table regardless of instructions). Recorded
as a finding; NO optimize launch occurred.

Redesign v2 (this file's committed sabotage_seed.txt as of amendment):
degradations chosen for COMPLIANCE (format-type rules bind where factual
contradictions do not): (1) coarseness rule — p50 restricted to
{0.05, 0.50, 0.95}, nearest value; (2) tie-break-higher bias; (3)
no-deliberation rule; p25=p75=p50 (interval information destroyed). All
single-instruction-repairable; scaffold and placeholders unchanged. Same
gate bars re-run under tag sabotage_pregate2 (~$4). Same abort rule. v1
text preserved in git history (commit 973616f).
