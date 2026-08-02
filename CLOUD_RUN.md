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

## Caveats

- Research-preview sandboxes may have session/wall-clock limits; a killed
  run loses at most the iterations since the last checkpoint (~$2–8) and
  resumes with the same command (same-machine) or `--pull-checkpoint`
  (cross-machine, W&B required).
- The sandbox bills nothing itself, but the run spends real Anthropic API
  credit from `ANTHROPIC_API_KEY`.
