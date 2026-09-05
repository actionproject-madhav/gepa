#!/usr/bin/env python3
"""Score the pre-registered holdout27 run (step 3 of runs/holdout27_prereg.md).

Inputs: gepa runs/noise_study/noise_cells_holdout27.jsonl (seed, 2 repeats),
frozen_artifacts.json (curves A4/A5, w=0.36, clip), the (model,bin) printed
pass-rate map from the test evidence reconstruction (identical across targets
for a given model x bin), and model_estimate_minutes as the pre-registered
ruler. Outputs every pre-registered readout with task-clustered CIs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GEPA = Path("/Users/madhav/SaferAI/gepa")
WS = Path("/Users/madhav/.claude/projects/-Users-madhav-SaferAI-LLM-elicitation/audit-workspace")
RAW = Path("/Users/madhav/gitrepos/cyber-task-horizons-data")
FZ = json.load(open(WS / "analysis/freeze/frozen_artifacts.json"))
EVID = json.load(open(WS / "analysis/E_incontext_sufficiency/reconstructed_test_evidence.json"))
CLIP = (0.01, 0.99)
W = 0.36


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logit(p):
    return np.log(p / (1 - p))


rows = [json.loads(l) for l in open(GEPA / "runs/noise_study/noise_cells_holdout27.jsonl") if l.strip()]
df = pd.DataFrame([r for r in rows if r["set"] == "holdout27" and r["prompt"] == "seed"])
n_fail = int(df["error"].notna().sum()) if "error" in df else 0
df["brier"] = df["brier"].fillna(1.0)
print(f"rows {len(df)}  cells {df.groupby(['task_id','model']).ngroups}  "
      f"repeats {sorted(df['repeat'].unique())}  failures {n_fail}")

td = pd.read_parquet(RAW / "analysis/figures/data/task_difficulties.parquet").set_index("task_id")
df["minutes"] = df["task_id"].map(td["model_estimate_minutes"])
df["family"] = np.where(df["task_id"].str.startswith("CVE-"), "cvebench", "cybergym")
assert df["minutes"].notna().all()

# (model, bin) -> printed rate of nearest easier bin (constant across targets)
cap_map = {}
for r in EVID:
    key = (r["model"], r["bin"])
    if key not in cap_map:
        cap_map[key] = {p["bin_index"]: p["pass_rate"] for p in r["profiles"]}.get(r["bin"] - 1)


def curve_p(name, sub):
    c = FZ["curves"][name]
    z = sub["model"].map(c["intercepts"]).to_numpy() + c["slope_ln_fst"] * np.log(sub["minutes"].to_numpy())
    return sigmoid(z)


def tstats_cell(delta, tasks):
    per = pd.DataFrame({"t": tasks, "d": delta}).groupby("t")["d"].mean()
    return float(per.mean()), float(per.mean() / (per.std(ddof=1) / np.sqrt(len(per)))), len(per)


# per-repeat predictor rows -> per-cell mean brier
def score(pred_per_row):
    b = (pred_per_row - df["outcome"]) ** 2
    return float(b.groupby([df["task_id"], df["model"]]).mean().mean()), b


out = {}
seed_b = df.groupby(["task_id", "model"])["brier"].mean()
out["seed"] = float(seed_b.mean())
realized = float(df.groupby(["task_id", "model"])["outcome"].first().mean())
out["realized_solve_rate"] = realized
out["seed_mean_p50"] = float(df["p50"].mean())

df["cap"] = [cap_map.get((m, b)) for m, b in zip(df["model"], df["bin"])]
p50c = np.where(df["cap"].notna(), np.minimum(df["p50"], df["cap"].astype(float)), df["p50"])
out["capped_seed"], cap_rows = score(pd.Series(p50c, index=df.index))

for name in ["A4_minus_intercode", "A5_all_train"]:
    cp = pd.Series(np.clip(curve_p(name, df), 1e-9, 1 - 1e-9), index=df.index)
    out[f"curve_{name}"], _ = score(cp)
    out[f"curve_{name}_mean_pred"] = float(cp.groupby([df["task_id"], df["model"]]).first().mean())
    bl = pd.Series(sigmoid((1 - W) * logit(cp) + W * logit(np.clip(df["p50"], *CLIP))), index=df.index)
    out[f"blend_{name}"], bl_rows = score(bl)
    if name == "A4_minus_intercode":
        cell_bl = bl_rows.groupby([df["task_id"], df["model"]]).mean()
        cell_cv = ((cp - df["outcome"]) ** 2).groupby([df["task_id"], df["model"]]).mean()

tasks_of_cell = seed_b.index.get_level_values(0)
for label, delta in [
    ("blendA4_minus_seed", (cell_bl - seed_b)),
    ("cappedseed_minus_seed", cap_rows.groupby([df["task_id"], df["model"]]).mean() - seed_b),
    ("blendA4_minus_curveA4", (cell_bl - cell_cv)),
]:
    m, t, n = tstats_cell(delta.to_numpy(), tasks_of_cell)
    out[f"{label}"] = {"delta": m, "t_task": t, "n_tasks": n}

# level check + family offsets vs frozen LOFO band
for fam in ["cvebench", "cybergym"]:
    sub = df[df["family"] == fam]
    cp = np.clip(curve_p("A4_minus_intercode", sub), 1e-6, 1 - 1e-6)
    y = sub["outcome"].to_numpy()
    from scipy import optimize
    r = optimize.minimize(lambda d: float(np.sum(np.logaddexp(0, logit(cp) + d[0]) - y * (logit(cp) + d[0]))),
                          [0.0], method="L-BFGS-B")
    out[f"offset_{fam}_A4"] = float(r.x[0])
out["frozen_lofo_band"] = FZ["lofo_band_clean4"] | {"note": "mean/sd over 4 clean train families"}

print(json.dumps(out, indent=2))
Path(WS / "analysis/freeze/holdout_scores.json").write_text(json.dumps(out, indent=2))
