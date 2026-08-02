#!/usr/bin/env bash
# Bootstrap a fresh machine / Claude Code cloud sandbox for forecaster-GEPA runs.
#
# Assumes this gepa repo is already cloned (any location) on branch
# feat/gepa_on_LLM_estimator. Clones the two sibling data/code repos at
# PINNED commits (reproducibility: these are the exact SHAs the seed-42 task
# manifest and all prior runs were built against), then installs the env.
#
# Requires: ANTHROPIC_API_KEY in the environment (set it as a sandbox
# environment secret — never commit it). Optional: WANDB_API_KEY for live
# dashboards; without it, set `use_wandb: false` in the run config.
#
# Usage (from anywhere):  bash scripts/cloud_setup.sh

set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
PARENT="$(dirname "$REPO_ROOT")"

ESTIMATOR_SHA=1ef1f98   # LLM_elicitation feat/ss2026_intrabenchmark_package (2026-07-17)
LYPTUS_SHA=a514c63feb9eb34df3a013e46cc625338d5beeab  # matches task_manifest_seed42.json

# 1. Sibling estimator repo — MUST be named LLM_elicitation next to this repo
#    (editable path dependency ../LLM_elicitation in pyproject.toml).
if [ ! -d "$PARENT/LLM_elicitation" ]; then
  git clone https://github.com/safer-ai/LLM_elicitation "$PARENT/LLM_elicitation"
fi
git -C "$PARENT/LLM_elicitation" fetch origin
git -C "$PARENT/LLM_elicitation" checkout -B feat/ss2026_intrabenchmark_package "$ESTIMATOR_SHA"

# 2. Lyptus ground-truth data at the exact manifest commit.
LYPTUS="$HOME/gitrepos/cyber-task-horizons-data"
if [ ! -d "$LYPTUS" ]; then
  mkdir -p "$HOME/gitrepos"
  git clone https://github.com/lyptus-research/cyber-task-horizons-data "$LYPTUS"
fi
git -C "$LYPTUS" checkout "$LYPTUS_SHA"

# 3. Environment (uv + all extras, installs llm-estimator editable).
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --extra forecaster

# 4. Sanity: key present, manifest present, estimator importable.
[ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "ERROR: ANTHROPIC_API_KEY not set in environment"; exit 1; }
[ -f "$REPO_ROOT/runs/task_manifest_seed42.json" ] || { echo "ERROR: canonical manifest missing from repo"; exit 1; }
uv run python -c "import intra_benchmark_calibration, forecaster_gepa" || exit 1

echo "Setup OK."
