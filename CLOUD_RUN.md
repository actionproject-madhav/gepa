# Running forecaster-GEPA experiments in a Claude Code cloud sandbox

For running a full optimization (e.g. `configs/pilot_gate100.yaml`, ~2.5 h,
~10k forecaster API calls) without keeping a laptop awake. Works because all
three repos are public and every run checkpoints per iteration (plus W&B
checkpoint artifacts every 10 iterations — see "Cross-machine resume" in
`FORECASTER_GEPA_README.md`).

## One-time setup (browser, ~10 min)

1. Fork this repo to your GitHub account and make sure the branch
   `feat/gepa_on_LLM_estimator` on the fork is up to date.
2. At claude.com/code: connect the fork, branch `feat/gepa_on_LLM_estimator`.
3. Environment secrets (sandbox settings):
   - `ANTHROPIC_API_KEY` — required (the forecaster + reflection calls bill
     this key, ~$60–95 for the gate-100 pilot).
   - `WANDB_API_KEY` — optional but recommended: live panels watchable from
     a phone, and checkpoint artifacts survive the sandbox.
4. Allowed network domains: `api.anthropic.com`, `github.com`, `pypi.org`,
   `files.pythonhosted.org`, `astral.sh`, `api.wandb.ai`, `wandb.ai`.

## Kick-off prompt (paste into the cloud session)

```
Run: bash scripts/cloud_setup.sh   — it must end with "Setup OK".
If WANDB_API_KEY is not set in the environment, change `use_wandb: true` to
`use_wandb: false` in configs/pilot_gate100.yaml first.
Then start the run and let it finish (~2.5 h, ~10k API calls — do not stop it
unless it errors; if interrupted, rerunning the same command resumes exactly):

  uv run python -m forecaster_gepa.run --config configs/pilot_gate100.yaml --phase optimize

When it completes, report:
1. The "Optimisation finished" log line (candidates, best val Brier, metric calls).
2. gate_val_agreement_running and the frontier stats from the last record of
   runs/pilot_gate100/diagnostics.jsonl.
3. The full output of:
   python3 scripts/retrospective_gate_analysis.py --run-dir runs/pilot_gate100 \
       --manifest runs/task_manifest_seed42.json --subsample-tasks-per-bin 1 2 3
4. Commit runs/pilot_gate100/ EXCLUDING gate_traces.jsonl (force-add past the
   runs/ gitignore) to a new branch results/pilot_gate100 and push it to origin,
   so the results survive the sandbox even without W&B.
Do not modify configs, seeds, the manifest, or any code.
```

## Getting results back on your machine

- If W&B was on: metrics/panels are live under your entity; the checkpoint
  artifact allows `--pull-checkpoint` resume anywhere (see the main README).
- Either way: `git fetch fork results/pilot_gate100` after the session
  reports completion.

## Full-ladder autonomous prompt (E1 + gate-100 + E4 in one session)

The July baseline checkpoint (`runs/pilot_baseline/gepa_state.bin` +
`candidates.json`) is committed to this branch, so the finalist phase (E1)
also works in the sandbox. Paste the following into the cloud session:

```
Read CLOUD_RUN.md and FORECASTER_GEPA_RUNS.md first. Then execute this
three-stage experiment ladder autonomously, in order, without stopping for
confirmation. Do not modify any source code, configs, seeds, or the
manifest; do not run anything not listed here.

Stage 0 — setup:
- bash scripts/cloud_setup.sh   (must end with "Setup OK")
- If WANDB_API_KEY is missing from the environment, set use_wandb: false in
  configs/pilot_baseline.yaml, configs/pilot_gate100.yaml and
  configs/pilot_reflection_v2.yaml (the ONLY permitted config edit).
- git checkout -b results/ladder-2026-08-02

Stage 1 — E1, finalist re-ranking of the July baseline (~20 min, ~1,400 calls):
- uv run python -m forecaster_gepa.run --config configs/pilot_baseline.yaml --phase finalist
- git add -f runs/pilot_baseline/finalist_results.json runs/pilot_baseline/finalist_cells.jsonl
- Commit and push the results branch.

Stage 2 — gate-100 optimization (~2.5 h, ~10k calls):
- uv run python -m forecaster_gepa.run --config configs/pilot_gate100.yaml --phase optimize
- If interrupted for any reason, rerun the same command — it resumes
  exactly. Never restart from scratch, never delete the run dir.
- python3 scripts/retrospective_gate_analysis.py --run-dir runs/pilot_gate100 \
    --manifest runs/task_manifest_seed42.json --subsample-tasks-per-bin 1 2 3 \
    > runs/pilot_gate100/retro_gate_analysis.txt
- git add -f everything in runs/pilot_gate100/ EXCEPT gate_traces.jsonl;
  commit and push.

Stage 3 — E4, reflection-prompt v2 (~1 h, ~4k calls):
- uv run python -m forecaster_gepa.run --config configs/pilot_reflection_v2.yaml --phase optimize
- git add -f everything in runs/pilot_reflection_v2/ EXCEPT gate_traces.jsonl;
  commit and push.

Final report, one message:
1. E1: finalist_results.json — per-candidate val vs finalist Brier including
   the seed baseline entry, val<->finalist Spearman, winner; one sentence:
   did the seed->winner gap survive on cells that exerted no selection
   pressure?
2. Gate-100: the "Optimisation finished" log line; gate_val_agreement_running
   and frontier stats from the LAST record of runs/pilot_gate100/diagnostics.jsonl;
   the full retro_gate_analysis.txt output.
3. E4: the "Optimisation finished" line; last diagnostics record; the full
   text of the best candidate's template from runs/pilot_reflection_v2/candidates.json.
4. Total metric calls across all stages.

Rules: never edit source code; never change seeds; if a stage fails twice in
a row, skip it, record the error, and continue with the next stage; before
the session ends for any reason, commit and push whatever exists.
```

## Measurement-study session prompt (2026-08-03: noise study + baseline extension)

Paste into a fresh cloud session (secrets + network as above):

```
Run `git pull origin feat/gepa_on_LLM_estimator` first. Then execute these
stages autonomously, in order, without stopping to ask anything. Never edit
source code, configs (except the single use_wandb flip below), seeds, or the
manifest. Push results before the session ends no matter what.

Stage 0 — setup:
- bash scripts/cloud_setup.sh   (must end "Setup OK")
- If WANDB_API_KEY is missing, set use_wandb: false in
  configs/pilot_baseline_ext80.yaml (the ONLY permitted config edit).
- git checkout -b results/measurement-2026-08-03

Stage 1 — repeated-measurement noise study (~4,450 calls, ~45-60 min):
- uv run python scripts/val_noise_study.py --repeats-val 5 --repeats-sealed 3
- git add -f runs/noise_study/  ; commit; push.

Stage 2 — baseline extension to 80 iterations (~2,700-3,000 calls, ~1 h):
- mkdir -p runs/pilot_baseline_ext80
- cp runs/pilot_baseline/gepa_state.bin runs/pilot_baseline_ext80/
- uv run python -m forecaster_gepa.run --config configs/pilot_baseline_ext80.yaml --phase optimize
- If interrupted, rerun the same command (exact resume). Never restart fresh.
- git add -f everything in runs/pilot_baseline_ext80/ EXCEPT gate_traces.jsonl; commit; push.
- Also commit and push the run dir (excluding gate_traces.jsonl) every ~20
  minutes WHILE stage 2 runs.

Stage 3 — conditional sealed check (~1,400 calls, ~20 min): ONLY IF stage 2
produced any candidate with val Brier better than 0.0937 (July's best):
- uv run python -m forecaster_gepa.run --config configs/pilot_baseline_ext80.yaml --phase finalist
- git add -f the finalist_*.{json,jsonl} files; commit; push.

Final report, one message:
1. The full runs/noise_study/noise_summary.txt table, plus: for each repeat
   r, the paired difference (seed repeat r minus july_cand20 repeat r) on
   the sealed set — is july_cand20 better in every repeat?
2. Stage 2: the "Optimisation finished" line; how many new accepts in
   iterations 41-80; the new best val Brier and which candidate; the last
   diagnostics record.
3. Stage 3 (if run): finalist_results.json content.
4. Total calls across stages.
```

## Caveats

- Research-preview sandboxes may have session/wall-clock limits; a killed
  run loses at most the iterations since the last checkpoint (~$2–8) and
  resumes with the same command (same-machine) or `--pull-checkpoint`
  (cross-machine, W&B required).
- The sandbox bills nothing itself, but the run spends real Anthropic API
  credit from `ANTHROPIC_API_KEY`.
