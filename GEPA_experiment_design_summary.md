# GEPA forecaster-optimisation experiment — design summary

Status: design frozen, ready for implementation. This document records the agreed setup and the reasoning behind each choice, so the team has a single reference and CC has an unambiguous spec.

---

## 1. What we're optimising and why

We use GEPA (Genetic-Pareto reflective prompt evolution) to optimise the **forecaster prompt** used in our LLM-based cyber-risk forecasting pipeline. The forecaster reads empirical pass-rate evidence about a model on nearby-difficulty cyber tasks and emits a probabilistic forecast (p25/p50/p75) that a target model solves a target task. The optimisation objective is the **grand Brier score on p50** against per-task binary ground truth.

GEPA is a good fit because forecaster rollouts are expensive and it is sample-efficient (100–500 evaluations vs 10k+ for RL-style methods), works through APIs with no weight access, and produces human-readable prompts. The risk it carries for us is discussed in §4 — our objective is structurally noisier per-instance than the benchmarks GEPA was validated on, which drives several design choices.

## 2. Roles and models

- **Forecaster (the system being optimised):** Claude Sonnet 4.6, `reasoning_effort=off`. Note this disables *extended thinking mode only* — the forecaster still produces a full chain-of-analysis and emits p25/p50/p75 as XML tags at the end.
- **Reflection LLM (the "judge"):** Claude Opus 4.8. Reads forecaster traces + per-cell Brier outcomes and proposes an edited prompt.
- We restrict to the Anthropic API throughout.

## 3. How GEPA actually works (clarifications we had to establish)

Several points were initially misunderstood and are worth recording, because they shaped the design:

- **The candidate pool is built sequentially, one candidate per iteration**, not generated as a batch of 200 up front. The "200–300 candidates" is the *total produced over the whole run* (the budget), which grows the pool; it is not a starting population.
- **Each iteration:** select a parent from the Pareto frontier → score it on a small minibatch (the gate/reflection cells) → reflection LLM reads those traces and produces one child → **acceptance gate**: accept the child only if it beats the parent on the minibatch → *only if accepted*, score it on the validation set and add it to the pool.
- **Acceptance is decided on the minibatch, not the validation set.** This is the load-bearing detail. Native GEPA uses a tiny (default 3-cell) minibatch as the accept/reject gate. Rejected children are discarded at the gate and never reach the validation set. A rejected child costs only the gate evaluation.
- **Parent selection is per-instance Pareto, not greedy.** For each validation cell, find which candidate(s) score best; the frontier is the union of candidates that are best on ≥1 cell; the parent is a stochastic draw weighted by how many cells each candidate wins. This deliberately avoids the local optima of greedy/best-average selection (the paper tests this against greedy and top-k and finds Pareto sampling superior).
- **Archive vs frontier:** the stored pool of accepted candidates is append-only. The *frontier* (the sampleable parent set) is a dynamic subset that can lose members when a later candidate dominates their cells. "Parent" and "child" are roles the same object plays at different times, not separate populations.
- **When a child is rejected:** nothing is retried; the loop simply advances to the next iteration and draws a (possibly different) parent. The parent that produced the rejected child is not penalised and stays on the frontier.
- GEPA maximises its metric, so we optimise **−Brier** (or 1−Brier); Brier is lower-is-better.

## 4. Why our setting is noisier than the GEPA paper — the central concern

The paper's benchmarks (HotpotQA, IFBench, AIME, LiveBench-Math, HoVer, PUPA) share a property ours lacks: **the prompt controls the outcome**, and the per-instance score is often graded and paired with rich textual feedback. Better prompt → better answer → higher score, deterministically.

In our setting the prompt does **not** control the outcome. The binary pass/fail for (model, task) was fixed offline before GEPA ran. The forecaster prompt only controls the *predicted probability*. Two consequences absent from the paper's tasks:

1. **Improper per-cell selection.** On a single binary outcome, "who won this cell" rewards whichever candidate was most confident in the direction the outcome happened to land. A well-calibrated prompt saying 0.6 loses a solved cell to a recklessly overconfident prompt saying 0.95. Being confidently wrong wins the cell whenever the outcome lands on the overconfident side. Aggregate Brier is a proper scoring rule and penalises this correctly; the *per-cell* winner contest is improper. The per-instance Pareto frontier is therefore built on an improper signal.
2. **Irreducible Bernoulli noise floor.** Even a perfectly calibrated forecaster gets a stochastic per-cell Brier, because each outcome is a single Bernoulli draw. The best possible prompt does not reliably "win" cells.

**Position we've reached:** this is an empirical question, not a settled one. Jeff's point stands — GEPA is explicitly designed to tolerate a noisy gate, because the frontier + weighted sampling launders gate noise over many proposals, and a wrongly-rejected child's parent stays available. But our per-cell signal is at the degenerate (purely binary) extreme of what the paper validated. So we **start close to native GEPA and instrument heavily** to detect whether the noise is actually pathological, rather than pre-emptively building complexity to fix a problem we haven't confirmed.

## 5. Cells, tasks, and the train/test split

- A **cell** = (forecasted model, target task, evidence). Evidence uses **`all_except_target` mode**: for a target task in difficulty bin *k*, the forecaster is shown pass-rate evidence from the other bins. We iterate *k* over the 5 FST (functional solve-time) difficulty bins → "5 bin combinations".
- **Cells = tasks × forecasted models.** We frame all set sizes in *task* units; multiply by the model count to get cells. Bins are a design tool for constructing evidence; **bin labels never appear in the forecaster prompt**.
- **Forecasted models during search: 4**, chosen to span the capability spectrum — **GPT-4o, Gemini 2.5 Pro, Opus 4, GPT-5.3 Codex**. Four (not the full 12) for cost. Sonnet 4.6 is excluded as a forecasted model because it is the forecaster (self-forecasting confound).
- **Forecasted models for finalist re-ranking and test: all 12** (the full panel is the real deployment target; affordable because these are one-off end-of-run evaluations).

**The 5/2 benchmark split is mandatory, not optional.** The test set is defined *by benchmark* (the held-out domain-transfer claim), so train/val/finalist tasks must come only from the other 5 benchmarks to avoid leakage. Stratified-by-bin sampling makes benchmark *composition* irrelevant for the train side, but the split itself is required.

- **Train / val / finalist:** drawn from the 5 training benchmarks (CyBashBench, NL2Bash, InterCode-CTF, NYUCTF, CyBench) — 175 tasks available.
- **Test:** CVEBench + CyberGym (116 tasks). **Note: these live only in bins 2–4** (0 tasks in bin 0, 2 in bin 1). The test set is therefore a *hard-bin, new-domain* set. This is a feature — it mirrors realistic cyber-misuse deployment, where risk concerns hard tasks — but it means the transfer claim is "joint domain + hard-difficulty transfer", not disentangled, and difficulty coverage on test is deliberately incomplete. Document as such.

## 6. Final task allocation

The Lyptus difficulty tail is thin: hard bins are sparse. Uniform per-bin reservation is infeasible in bins 3–4. We reserve **5 val + 5 finalist per bin in bins 0–2, and 3 val + 3 finalist per bin in bins 3–4** (option a), taking **everything left over as the train pool**. Rationale: this leaves a *rotatable* train pool in the hard bins (bin 4 would otherwise be a single frozen task, silently overfitting the reflection). The cost — slightly noisier hard-bin val/finalist Brier — is acceptable because those estimates are noise-dominated in the hard bins regardless.

| Bin | Avail (train benchmarks) | Val | Finalist | Train pool (leftover) |
|----|----|----|----|----|
| 0 | 54 | 5 | 5 | 44 |
| 1 | 52 | 5 | 5 | 42 |
| 2 | 40 | 5 | 5 | 30 |
| 3 | 18 | 3 | 3 | 12 |
| 4 | 11 | 3 | 3 | 5 |
| **Total** | **175** | **21** | **21** | **133** |

- **Train pool (133 tasks):** the leftover after reserving val + finalist. Each GEPA iteration **resamples 1 task per bin (5 tasks total)** from this pool → 5 × 4 models = **20 gate/reflection cells per iteration**. Resampling gives rotation variety in the easy bins; in bin 4 (pool of 5) rotation is limited but non-frozen.
- **Val set (21 tasks, fixed):** 21 × 4 models = 84 validation cells. Used for per-instance Pareto frontier scoring of accepted children. Fixed across the whole run.
- **Finalist set (21 tasks, fixed):** scored across **all 12 models** = 252 cells, used only at the end to re-rank the top-k candidates and pick the winner. In-distribution (train benchmarks), disjoint from train pool and val.
- **Test set (CVEBench + CyberGym):** up to 116 tasks × 12 models, concentrated in bins 2–4. The real transfer evaluation, run once on the final winner.

(Constants are configurable; these are the defaults. Selection is seeded for reproducibility. All four sets — train pool, val, finalist — are mutually disjoint on the train-benchmark side; test is disjoint by construction via the benchmark split.)

## 7. Design decisions on the GEPA loop

- **Gate size:** start at the diagnostic setting (N_TRAIN_TASKS = 5 tasks → 20 cells) and measure whether it accepts reliably (§8). This is a deliberate departure to test; native default is 3 cells. We are NOT committing to a widened 100-cell gate — that (and the trainset=valset caching trick that depends on it) is **future work**, explicitly out of scope for this implementation.
- **trainset ≠ valset.** We keep them separate. The caching saving that motivated trainset=valset only materialises with a full-100-cell fixed gate, which we are not doing yet. Keeping them separate also removes the overfitting worry (Jeff's `if task==X report 1%` scenario) for free — relevant because our forecaster prompts *do* contain task identifiers.
- **Pareto frontier inclusion threshold:** `N_CELLS_WON_NEEDED_FOR_PARETO_FRONTIER = 1` (native). Configurable to 3 (Jeff's k-of-n stability suggestion) if the frontier proves noise-dominated.
- **Stratified parent selection:** *not* implemented for the first run. If §8 metrics show the native per-cell frontier is noise-dominated, the fix is to define the frontier over strata (bin × benchmark) so the per-instance score becomes an aggregate Brier over several cells — which restores propriety to the selection criterion. Deferred until the pilot justifies it.
- **Merge/crossover:** off for the first run (the complementarity test is a text-diff and would merge on spurious differences under our noisy signal). (JK -- I don't understand what this refers to	)
- **Fork:** we fork gepa-ai/gepa (pinned commit) only to add logging/checkpointing/metrics and the Anthropic wiring. We do **not** change the algorithm — native parent selection, native acceptance, append-only archive, dynamic frontier all preserved. (The cache-routing fork that would make accepted children's valset pass free is future work, bundled with trainset=valset.)

## 8. Diagnostic metrics (the point of the first run)

The first run is a **diagnostic** to decide whether the cheap native-ish design is adequate or whether our metric's noise forces the expensive gate / stratification. Track:

1. **Gate→val agreement.** For each accepted child, the gate says it beat its parent on the ~20 gate cells; check whether the fixed val set agrees on the direction (child better than parent). >~80% → cheap gate is trustworthy. ~50–60% → gate is accepting coin-flips; strengthen it (k-of-n, or larger N).
2. **Frontier concentration.** Fraction of val cells won by the top-3 candidates vs spread across many one-cell winners; and the share of frontier members winning exactly one cell. Mostly one-cell winners → noise-driven frontier.
3. **Best-val-Brier trajectory** vs candidate count. If improving, the optimiser works regardless of frontier noise (vindicates the native design).
4. **Confidence-vs-wins correlation** — per frontier candidate, cells-won vs mean forecast extremity (mean |p50 − 0.5|), and the partial correlation holding aggregate Brier fixed. If wins track confidence *independent of calibration*, the improper-selection pathology (§4) is biting → argues for stratification. This is the sharpest, most setting-specific test.
5. **Finalist re-ranking stability.** Do the top-k by val Brier keep their ranking on the finalist set (all 12 models)? If the ranking scrambles, val-set selection is too noisy.

Log rejected candidates and their stats too — err toward over-logging for completeness.

## 9. Reproducibility & logging

- Fixed seed for all task selection and sampling.
- W&B for metrics (Brier, frontier stats, acceptance flags, the §8 diagnostics). Heavy rollout text logged as **artifacts / local JSONL**, not as run metrics, to keep the dashboard responsive.
- **Rollout traces stored only for gate/reflection cells** (the ~20/iteration). Validation needs only per-cell scores, not traces. Estimated rollout text over a full run ≈ a few hundred MB — comfortably under W&B's 100GB artifact limit.
- **Checkpoint after every candidate (accepted and rejected):** full state — pool, frontier, cached scores, RNG state, W&B run id — so a broken run resumes exactly.

## 10. Venue framing (context)

Core findings are largely invariance/negative results, which play better at workshop/evaluation-focused venues (AI4GOOD @ ICML 2026, NeurIPS Evaluations & Datasets track) than main-track. Lead with methodology (the inter-benchmark calibration design, the W₁-based agreement metric). Report Yates decomposition as the primary diagnostic (exact for continuous forecasts; slope term interpretable for binary outcomes); Murphy as secondary.
