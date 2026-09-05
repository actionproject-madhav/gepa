#!/usr/bin/env python3
"""D-arm steps 2-5 (per runs/darm_prereg.md): aggregate elicited times, fit the
audited curve on train families, score head-to-head on identical cells."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

GEPA = Path("/Users/madhav/SaferAI/gepa")
RUNS = GEPA / "runs/noise_study"
WS = Path("/Users/madhav/.claude/projects/-Users-madhav-SaferAI-LLM-elicitation/audit-workspace")
sys.path.insert(0, str(GEPA / "src"))
sys.path.insert(0, "/Users/madhav/SaferAI/LLM_elicitation")
from intra_benchmark_calibration.lyptus_data import load_lyptus_dataset  # noqa: E402

CLEAN_TRAIN = {"cybashbench", "nl2bash", "nyuctf", "cybench"}


def read_jsonl(p):
    return pd.DataFrame([json.loads(l) for l in open(p) if l.strip()])


def sig(z):
    return 1 / (1 + np.exp(-z))


# ---- step 2: aggregate elicited times ------------------------------------
el = pd.concat([read_jsonl(RUNS / "noise_cells_darm_times.jsonl"),
                read_jsonl(RUNS / "noise_cells_darm_times_fix.jsonl")],
               ignore_index=True)
n_err = int(el["error"].notna().sum())
ok = el[el["error"].isna()]
agg = ok.groupby("task_id").agg(lo=("low_minutes", "median"),
                                mid=("mid_minutes", "median"),
                                hi=("high_minutes", "median"),
                                n=("mid_minutes", "size"),
                                sd_logmid=("mid_minutes", lambda v: np.std(np.log(v), ddof=0)))
print(f"elicited: {len(el)} rows, {n_err} errors; tasks with estimates {len(agg)}; "
      f"median repeat sd(log mid) {agg['sd_logmid'].median():.3f}")

ds = load_lyptus_dataset(Path("/Users/madhav/gitrepos/cyber-task-horizons-data"))
frame = ds.outcomes.frame
tasks = pd.DataFrame([{"task_id": t.task_id, "family": t.task_family,
                       "fst": t.fst_minutes} for t in ds.tasks]).set_index("task_id")

# diagnostic: elicited mid vs curated FST on train tasks
tr_ids = [t for t in agg.index if t in tasks.index and tasks.at[t, "family"] in
          CLEAN_TRAIN | {"intercode_ctf"}]
rho = stats.spearmanr(agg.loc[tr_ids, "mid"], tasks.loc[tr_ids, "fst"]).statistic
print(f"train tasks: Spearman(elicited mid, curated FST) = {rho:.2f} (n={len(tr_ids)})")

# ---- step 3: fit on audited train families -------------------------------
test_log = read_jsonl(RUNS / "noise_cells_test_set.jsonl")
seed_test = test_log[(test_log["set"] == "test") & (test_log["prompt"] == "seed")]
PANEL = sorted(seed_test["model"].unique())
MIDX = {m: i for i, m in enumerate(PANEL)}

fit_rows = []
for tid in tasks.index[tasks["family"].isin(CLEAN_TRAIN)]:
    if tid not in agg.index:
        continue
    for m in PANEL:
        v = frame.at[m, tid]
        if not pd.isna(v):
            fit_rows.append({"model": m, "y": float(v),
                             "lm": float(np.log(agg.at[tid, "mid"]))})
fit = pd.DataFrame(fit_rows)
assert len(fit) > 1000
X = np.zeros((len(fit), 12))
X[np.arange(len(fit)), fit["model"].map(MIDX)] = 1.0
X[:, -1] = fit["lm"]
y = fit["y"].to_numpy()
r = optimize.minimize(lambda w: np.sum(np.logaddexp(0, X @ w) - y * (X @ w)),
                      np.zeros(12), jac=lambda w: X.T @ (sig(X @ w) - y),
                      method="L-BFGS-B", options={"maxiter": 2000})
assert r.success
W = r.x
print(f"fit: {len(fit)} clean-train cells; slope {W[-1]:.3f}")


def predict(models, tids, col):
    z = (pd.Series(models).map({m: W[MIDX[m]] for m in PANEL}).to_numpy()
         + W[-1] * np.log(agg.loc[tids, col].to_numpy()))
    return sig(z)


def paired(sub, p, seedcol, label):
    b = (p - sub["y"]) ** 2
    d = b - sub[seedcol].to_numpy()
    per = pd.DataFrame({"t": sub["task_id"].to_numpy(), "d": d}).groupby("t")["d"].mean()
    t_cell = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    t_task = per.mean() / (per.std(ddof=1) / np.sqrt(len(per)))
    print(f"  {label}: Brier {b.mean():.4f}  delta {d.mean():+.4f}  "
          f"t_cell {t_cell:.2f}  t_task {t_task:.2f} (n_task={len(per)})  "
          f"mean_p50 {p.mean():.3f} vs realized {sub['y'].mean():.3f}")
    return {"brier": float(b.mean()), "delta": float(d.mean()),
            "t_cell": float(t_cell), "t_task": float(t_task)}


out = {"n_errors": n_err, "spearman_mid_fst_train": float(rho),
       "slope": float(W[-1]), "sets": {}}

# ---- reserved test -------------------------------------------------------
sc = seed_test.groupby(["task_id", "model"], as_index=False).agg(
    seed_b=("brier", "mean"), y=("outcome", "first"))
sc = sc[sc["task_id"].isin(agg.index)]
print(f"\nRESERVED TEST ({len(sc)} cells):")
p = predict(sc["model"], sc["task_id"], "mid")
out["sets"]["test_vs_seed"] = paired(sc.rename(columns={"seed_b": "cmp"}), p, "cmp", "D-arm vs seed")
for arm, f in [("july_cand12", "noise_cells_test_set.jsonl"),
               ("clean_cand7", "noise_cells_test_set.jsonl"),
               ("joint_cand5", "noise_cells_test_ext_winners.jsonl"),
               ("modelbin_cand18", "noise_cells_test_ext_modelbin.jsonl")]:
    df = read_jsonl(RUNS / f)
    df = df[(df["set"] == "test") & (df["prompt"] == arm)]
    ab = df.groupby(["task_id", "model"], as_index=False).agg(cmp=("brier", "mean"))
    mm = sc.merge(ab, on=["task_id", "model"])
    p2 = predict(mm["model"], mm["task_id"], "mid")
    out["sets"][f"test_vs_{arm}"] = paired(mm, p2, "cmp", f"D-arm vs {arm}")

# ---- sealed --------------------------------------------------------------
sl = pd.concat([read_jsonl(RUNS / f) for f in
                ["noise_cells_temp0.jsonl", "noise_cells_clean_rerun.jsonl",
                 "noise_cells_v2_sealed.jsonl"]])
sl = sl[(sl["set"] == "sealed") & (sl["prompt"] == "seed")]
slc = sl.groupby(["task_id", "model"], as_index=False).agg(
    cmp=("brier", "mean"), y=("outcome", "first"))
slc = slc[slc["task_id"].isin(agg.index)]
print(f"\nSEALED ({len(slc)} cells; seed pooled here = its own mean over passes):")
p = predict(slc["model"], slc["task_id"], "mid")
out["sets"]["sealed_vs_seed"] = paired(slc, p, "cmp", "D-arm vs seed(sealed)")

# ---- holdout27 -----------------------------------------------------------
hd = read_jsonl(RUNS / "noise_cells_holdout27.jsonl")
hd = hd[(hd["set"] == "holdout27") & (hd["prompt"] == "seed")]
hc = hd.groupby(["task_id", "model"], as_index=False).agg(
    cmp=("brier", "mean"), y=("outcome", "first"))
hc = hc[hc["task_id"].isin(agg.index)]
print(f"\nHOLDOUT27 ({len(hc)} cells):")
p = predict(hc["model"], hc["task_id"], "mid")
out["sets"]["holdout_vs_seed"] = paired(hc, p, "cmp", "D-arm vs seed(holdout)")

(WS / "analysis/freeze/darm_scores.json").write_text(json.dumps(out, indent=2))
print("\nsaved -> analysis/freeze/darm_scores.json")
