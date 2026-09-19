#!/usr/bin/env python3
"""Overfitting diagnostics for the elicited-time -> probability curve.

Answers the standard objection ("a sigmoid fit on Lyptus train may not
survive a large domain gap") with four measurements, all $0:
  1. generalization gap in Brier Skill Score (comparable across base rates)
  2. leave-one-FAMILY-out inside train vs the offsets the held-out
     benchmarks actually need
  3. slope sensitivity (refit intercepts with beta forced)
  4. data efficiency (fit on random fractions of train tasks)

Run: uv run python scripts/audit_scoring/curve_robustness.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, "/Users/madhav/SaferAI/LLM_elicitation")
from intra_benchmark_calibration.lyptus_data import load_lyptus_dataset  # noqa: E402

CLEAN = {"cybashbench", "nl2bash", "nyuctf", "cybench"}
sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731


def load():
    ds = load_lyptus_dataset(Path("/Users/madhav/gitrepos/cyber-task-horizons-data"))
    frame, fam = ds.outcomes.frame, {t.task_id: t.task_family for t in ds.tasks}
    t = pd.concat([pd.DataFrame([json.loads(l) for l in open(REPO / "runs/noise_study" / f) if l.strip()])
                   for f in ["noise_cells_darm_times.jsonl", "noise_cells_darm_times_fix.jsonl"]])
    mid = t[t["mid_minutes"].notna()].groupby("task_id")["mid_minutes"].median()
    ts = pd.DataFrame([json.loads(l) for l in open(REPO / "runs/noise_study/noise_cells_test_set.jsonl") if l.strip()])
    ts = ts[(ts["set"] == "test") & (ts["prompt"] == "seed")]
    test = ts.groupby(["task_id", "model"], as_index=False).agg(y=("outcome", "first"))
    test["fam"] = test["task_id"].map(fam)
    test["lnt"] = test["task_id"].map(lambda q: np.log(mid[q]))
    test["m"] = test["model"]
    panel = sorted(test["model"].unique())
    tr = pd.DataFrame([{"task_id": q, "m": m, "lnt": np.log(mid[q]),
                        "y": float(frame.at[m, q]), "fam": fam[q]}
                       for q in mid.index if fam.get(q) in CLEAN for m in panel
                       if q in frame.columns and not pd.isna(frame.at[m, q])])
    return tr, test, {m: i for i, m in enumerate(panel)}


def design(df, midx):
    X = np.zeros((len(df), 12))
    X[np.arange(len(df)), df["m"].map(midx)] = 1
    X[:, -1] = df["lnt"]
    return X


def fit(df, midx, fixed_beta=None):
    X, y = design(df, midx), df["y"].to_numpy()
    if fixed_beta is None:
        r = optimize.minimize(lambda w: np.sum(np.logaddexp(0, X @ w) - y * (X @ w)), np.zeros(12),
                              jac=lambda w: X.T @ (sig(X @ w) - y), method="L-BFGS-B")
        return r.x
    off, Xa = fixed_beta * X[:, -1], X[:, :11]
    r = optimize.minimize(lambda a: np.sum(np.logaddexp(0, Xa @ a + off) - y * (Xa @ a + off)),
                          np.zeros(11), method="L-BFGS-B")
    return np.concatenate([r.x, [fixed_beta]])


def score(w, df, midx):
    p, y = sig(design(df, midx) @ w), df["y"].to_numpy()
    b, base = np.mean((p - y) ** 2), y.mean()
    return {"brier": float(b), "bss": float(1 - b / (base * (1 - base))),
            "mean_pred": float(p.mean()), "base_rate": float(base)}


def offset(w, df, midx):
    eta, y = design(df, midx) @ w, df["y"].to_numpy()
    r = optimize.minimize(lambda d: float(np.sum(np.logaddexp(0, eta + d[0]) - y * (eta + d[0]))),
                          [0.0], method="L-BFGS-B")
    return float(r.x[0])


def main():
    tr, test, midx = load()
    out = {}
    w = fit(tr, midx)
    out["train_insample"] = score(w, tr, midx)
    out["reserved_test"] = score(w, test, midx)
    out["lofo_train"] = {f: {**score(fit(tr[tr.fam != f], midx), tr[tr.fam == f], midx),
                             "offset": offset(fit(tr[tr.fam != f], midx), tr[tr.fam == f], midx)}
                         for f in sorted(CLEAN)}
    out["heldout_benchmark_offsets"] = {f: offset(w, test[test.fam == f], midx)
                                        for f in sorted(test["fam"].unique())}
    out["slope_sensitivity"] = {f"{b:.3f}": score(fit(tr, midx, fixed_beta=b), test, midx)["brier"]
                                for b in [-0.4, -0.6, -0.876, -1.2, -1.6]}
    rng, tasks = np.random.default_rng(0), tr["task_id"].unique()
    eff = {}
    for frac in [0.10, 0.25, 0.50, 1.00]:
        n = max(4, int(len(tasks) * frac))
        vals = [score(fit(tr[tr.task_id.isin(rng.choice(tasks, n, replace=False))], midx), test, midx)["brier"]
                for _ in range(20 if frac < 1 else 1)]
        eff[f"{int(frac*100)}pct"] = {"n_tasks": n, "mean_brier": float(np.mean(vals)),
                                      "sd": float(np.std(vals))}
    out["data_efficiency"] = eff
    print(json.dumps(out, indent=1))
    (REPO / "scripts/audit_scoring/curve_robustness.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
