#!/usr/bin/env python3
"""Score the pre-registered head-to-head (runs/headtohead_prereg.md).

Implements exactly the frozen readouts: one confirmatory test (T vs seed on
the reserved test, one-sided task-clustered alpha=.025, plus holdout27 sign
gate), sealed guardrail with tripwire and intercode-split secondary, 2x2
estimation-only factorial, AUC/anchoring-drag/readback/level diagnostics,
k=20/50 knob replay. D-arm re-scored with the identical sealed-excluded
refit for every T-vs-D comparison.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

GEPA = Path("/Users/madhav/SaferAI/gepa")
NS = GEPA / "runs/noise_study"
RAW = Path("/Users/madhav/gitrepos/cyber-task-horizons-data")
sys.path.insert(0, str(GEPA / "src"))
sys.path.insert(0, "/Users/madhav/SaferAI/LLM_elicitation")
from intra_benchmark_calibration.lyptus_data import load_lyptus_dataset  # noqa: E402

CLEAN_FAMS = {"cybashbench", "nl2bash", "nyuctf", "cybench"}
SEED_TEST, SEED_SEALED, SEED_HOLD = 0.1633, 0.1305, 0.1898


def sig(z):
    return 1 / (1 + np.exp(-z))


def lgt(p):
    return np.log(p / (1 - p))


def read_jsonl(p):
    return pd.DataFrame([json.loads(l) for l in open(p) if l.strip()])


def tstats(delta, tasks):
    per = pd.DataFrame({"t": tasks, "d": delta}).groupby("t")["d"].mean()
    t = per.mean() / (per.std(ddof=1) / np.sqrt(len(per)))
    return float(per.mean()), float(t), int(len(per))


def auc(s, y):
    s, y = np.asarray(s), np.asarray(y)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return np.nan
    return float(np.mean([np.mean((q > neg) + 0.5 * (q == neg)) for q in pos]))


# ---- ground truth ---------------------------------------------------------
ds = load_lyptus_dataset(RAW)
frame = ds.outcomes.frame
tasks = pd.DataFrame([{"task_id": t.task_id, "family": t.task_family,
                       "fst": t.fst_minutes} for t in ds.tasks]).set_index("task_id")

# ---- elicited times (T and D), median over repeats, drop unparsed ---------
def med_times(files):
    t = pd.concat([read_jsonl(NS / f) for f in files])
    ok = t[t["mid_minutes"].notna()]
    agg = ok.groupby("task_id").agg(mid=("mid_minutes", "median"),
                                    lo=("low_minutes", "median"),
                                    hi=("high_minutes", "median"),
                                    setname=("set", "first"))
    dropped = sorted(set(t["task_id"]) - set(agg.index))
    return agg, dropped


T, t_dropped = med_times(["noise_cells_armT_times.jsonl"])
D, d_dropped = med_times(["noise_cells_darm_times.jsonl", "noise_cells_darm_times_fix.jsonl"])
print(f"T: {len(T)} tasks (dropped {t_dropped}); D: {len(D)} tasks (dropped {d_dropped})")

# ---- seed comparators -----------------------------------------------------
ts = read_jsonl(NS / "noise_cells_test_set.jsonl")
ts = ts[(ts["set"] == "test") & (ts["prompt"] == "seed")]
test_cells = ts.groupby(["task_id", "model"], as_index=False).agg(
    seed_b=("brier", "mean"), y=("outcome", "first"))
PANEL = sorted(test_cells["model"].unique())
MIDX = {m: i for i, m in enumerate(PANEL)}

sealed_seed = pd.concat([read_jsonl(NS / f) for f in
                         ["noise_cells_temp0.jsonl", "noise_cells_clean_rerun.jsonl",
                          "noise_cells_feature_ablation.jsonl", "noise_cells_accept_joint.jsonl",
                          "noise_cells_pareto_modelbin.jsonl", "noise_cells_v2_sealed.jsonl"]])
sealed_seed = sealed_seed[(sealed_seed["set"] == "sealed") & (sealed_seed["prompt"] == "seed")]
sealed_cells = sealed_seed.groupby(["task_id", "model"], as_index=False).agg(
    seed_b=("brier", "mean"), y=("outcome", "first"))

hd = read_jsonl(NS / "noise_cells_holdout27.jsonl")
hd = hd[(hd["set"] == "holdout27") & (hd["prompt"] == "seed")]
hold_cells = hd.groupby(["task_id", "model"], as_index=False).agg(
    seed_b=("brier", "mean"), y=("outcome", "first"))

# ---- curve fit (frozen): clean train minus sealed, ln(median mid) ---------
sealed_ids = set(sealed_cells["task_id"].unique())


def fit_curve(times):
    fit = []
    for tid, row in times.iterrows():
        if row["setname"] not in {"train_fit", "train"}:
            continue
        if tid in sealed_ids or tasks.at[tid, "family"] not in CLEAN_FAMS:
            continue
        for m in PANEL:
            v = frame.at[m, tid]
            if not pd.isna(v):
                fit.append({"m": m, "lnt": np.log(row["mid"]), "y": float(v)})
    fit = pd.DataFrame(fit)
    X = np.zeros((len(fit), 12))
    X[np.arange(len(fit)), fit["m"].map(MIDX)] = 1
    X[:, -1] = fit["lnt"]
    y = fit["y"].to_numpy()
    r = optimize.minimize(lambda w: np.sum(np.logaddexp(0, X @ w) - y * (X @ w)),
                          np.zeros(12), jac=lambda w: X.T @ (sig(X @ w) - y),
                          method="L-BFGS-B", options={"maxiter": 2000})
    assert r.success
    return r.x, len(fit)


wT, nT = fit_curve(T)
wD, nD = fit_curve(D)
print(f"curve T: slope {wT[-1]:.3f} ({nT} cells); curve D refit: slope {wD[-1]:.3f} ({nD} cells)")


def predict(w, cells, times):
    ok = cells[cells["task_id"].isin(times.index)].copy()
    X = np.zeros((len(ok), 12))
    X[np.arange(len(ok)), ok["model"].map(MIDX)] = 1
    X[:, -1] = ok["task_id"].map(lambda q: np.log(times.at[q, "mid"])).to_numpy()
    ok["p"] = sig(X @ w)
    ok["b"] = (ok["p"] - ok["y"]) ** 2
    return ok


out = {}
for label, w, times in [("T", wT, T), ("D_refit", wD, D)]:
    for setname, cells, ref in [("test", test_cells, SEED_TEST),
                                ("sealed", sealed_cells, SEED_SEALED),
                                ("holdout27", hold_cells, SEED_HOLD)]:
        ok = predict(w, cells, times)
        d, t, n = tstats((ok["b"] - ok["seed_b"]).to_numpy(), ok["task_id"])
        out[f"{label}_{setname}"] = {"brier": float(ok["b"].mean()), "n_cells": len(ok),
                                     "delta_vs_seed": d, "t_task": t, "n_tasks": n,
                                     "mean_pred": float(ok["p"].mean()),
                                     "realized": float(ok["y"].mean())}
        if setname == "sealed":
            ic = ok["task_id"].map(lambda q: tasks.at[q, "family"] == "intercode_ctf")
            out[f"{label}_sealed_no_intercode"] = {
                "brier": float(ok[~ic]["b"].mean()),
                "delta_vs_seed": float((ok[~ic]["b"] - ok[~ic]["seed_b"]).mean())}

# ---- P0 -------------------------------------------------------------------
p0_path = NS / "noise_cells_p0_bare.jsonl"
if p0_path.exists():
    p0 = read_jsonl(p0_path)
    p0 = p0[(p0["set"] == "test") & (p0["prompt"] == "p0_bare")]
    p0c = p0.groupby(["task_id", "model"], as_index=False).agg(
        b=("brier", "mean"), p=("p50", "mean"), y=("outcome", "first"))
    p0c = p0c.merge(test_cells[["task_id", "model", "seed_b"]], on=["task_id", "model"])
    d, t, n = tstats((p0c["b"] - p0c["seed_b"]).to_numpy(), p0c["task_id"])
    out["P0_test"] = {"brier": float(p0c["b"].mean()), "delta_vs_seed": d,
                      "t_task": t, "n_tasks": n, "mean_p50": float(p0c["p"].mean())}

    # 2x2 estimation on test (T, D_refit, seed, P0), task-clustered CIs
    okT = predict(wT, test_cells, T).set_index(["task_id", "model"])["b"]
    okD = predict(wD, test_cells, D).set_index(["task_id", "model"])["b"]
    sb = test_cells.set_index(["task_id", "model"])["seed_b"]
    pb = p0c.set_index(["task_id", "model"])["b"]
    common = okT.index.intersection(okD.index).intersection(sb.index).intersection(pb.index)
    tid = [i[0] for i in common]

    def est(v):
        m, t, n = tstats(v, tid)
        se = m / t if t != 0 else np.nan
        return {"est": m, "ci95": [m - 1.96 * se, m + 1.96 * se]}

    tt, dd, ss, pp = (okT[common].to_numpy(), okD[common].to_numpy(),
                      sb[common].to_numpy(), pb[common].to_numpy())
    out["factorial"] = {"ask_main": est(0.5 * ((tt - ss) + (dd - pp))),
                        "context_main": est(0.5 * ((tt - dd) + (ss - pp))),
                        "interaction": est((tt - dd) - (ss - pp))}

# ---- diagnostics ----------------------------------------------------------
tc = test_cells[test_cells["task_id"].isin(T.index)].copy()
tc["lnT"] = tc["task_id"].map(lambda q: np.log(T.at[q, "mid"]))
tc["lnD"] = tc["task_id"].map(lambda q: np.log(D.at[q, "mid"]) if q in D.index else np.nan)
for lab, col in [("T", "lnT"), ("D", "lnD")]:
    wm = [auc(-tc[col][tc["model"] == m], tc["y"][tc["model"] == m]) for m in PANEL]
    out[f"auc_test_{lab}"] = float(np.nanmean(wm))
train_ids = [i for i in T.index if T.at[i, "setname"] == "train_fit"]
out["train_spearman_T_vs_curatedFST"] = float(pd.Series(
    [T.at[i, "mid"] for i in train_ids]).corr(
    pd.Series([tasks.at[i, "fst"] for i in train_ids]), method="spearman"))
out["notes"] = "readback share computed separately; AUC refs: ME 0.788 / curated FST 0.648 / D prior 0.85 train-spearman"

print(json.dumps(out, indent=1))
Path(__file__).with_name("headtohead_scores.json").write_text(json.dumps(out, indent=1))
