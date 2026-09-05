#!/usr/bin/env python3
"""Freeze the validation artifacts for the pre-registered holdout run.

Recomputes from scratch (train-family outcomes + logged seed forecasts only):
  - curve A5  = logistic, 11 per-model intercepts + shared slope on ln(fst),
                fit on all 5 train families (1,923 cells)
  - curve A4  = same, fit excluding intercode_ctf only (the judges' repair:
                the one family the data identify as contamination-inflated)
  - curve A3  = same, excluding intercode_ctf + nyuctf (sensitivity)
  - LOFO offset band within A4's four families (fit on 3, offset on held)
  - blends at w = 0.36 (logit scale, p50 clip [0.01, 0.99]) with seed p50s
  - cap rule (p50 <- min(p50, printed pass rate of nearest easier shown bin))
    on test and sealed seed rows, using the reconstructed evidence tables

Halts loudly if any expected number (from two independent judge re-derivations
and five verified analyses) fails to reproduce. Zero test labels enter any fit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

GEPA = Path("/Users/madhav/SaferAI/gepa")
RUNS = GEPA / "runs/noise_study"
LYPTUS = Path("/Users/madhav/gitrepos/cyber-task-horizons-data")
WS = Path("/Users/madhav/.claude/projects/-Users-madhav-SaferAI-LLM-elicitation/audit-workspace")
EVID_TEST = WS / "analysis/E_incontext_sufficiency/reconstructed_test_evidence.json"
EVID_SEALED = WS / "analysis/E_incontext_sufficiency/reconstructed_sealed_evidence.json"
OUT = WS / "analysis/freeze/frozen_artifacts.json"

sys.path.insert(0, str(GEPA / "src"))
sys.path.insert(0, "/Users/madhav/SaferAI/LLM_elicitation")
from intra_benchmark_calibration.lyptus_data import load_lyptus_dataset  # noqa: E402

TRAIN5 = {"cybashbench", "nl2bash", "intercode_ctf", "nyuctf", "cybench"}
TESTF = {"cvebench", "cybergym"}
W_BLEND = 0.36
CLIP = (0.01, 0.99)


def read_jsonl(p):
    return pd.DataFrame([json.loads(l) for l in open(p) if l.strip()])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit(p):
    return np.log(p / (1.0 - p))


# ---- load logs -----------------------------------------------------------
test_log = read_jsonl(RUNS / "noise_cells_test_set.jsonl")
seed_test = test_log[(test_log["set"] == "test") & (test_log["prompt"] == "seed")].copy()
assert seed_test["brier"].notna().all()
seed_cell = (seed_test.groupby(["task_id", "model"], as_index=False)
             .agg(seed_brier=("brier", "mean"), outcome=("outcome", "first"),
                  bin=("bin", "first")))
assert len(seed_cell) == 1033 and seed_cell["task_id"].nunique() == 94
PANEL = sorted(seed_cell["model"].unique())
assert len(PANEL) == 11

sealed_files = ["noise_cells_temp0.jsonl", "noise_cells_clean_rerun.jsonl",
                "noise_cells_feature_ablation.jsonl", "noise_cells_accept_joint.jsonl",
                "noise_cells_pareto_modelbin.jsonl", "noise_cells_v2_sealed.jsonl"]
sealed_seed = pd.concat([read_jsonl(RUNS / f).assign(src=f) for f in sealed_files])
sealed_seed = sealed_seed[(sealed_seed["set"] == "sealed") & (sealed_seed["prompt"] == "seed")]
assert sealed_seed.groupby(["src", "repeat"]).ngroups == 17

ds = load_lyptus_dataset(LYPTUS)
frame = ds.outcomes.frame
tasks = pd.DataFrame([{"task_id": t.task_id, "family": t.task_family,
                       "fst_minutes": t.fst_minutes} for t in ds.tasks])

# ---- reference gates -----------------------------------------------------
refs = {
    "seed_test": (float(seed_cell["seed_brier"].mean()), 0.1633),
    "base_rate": (float(seed_cell["outcome"].mean()), 0.2401),
    "sealed_seed": (float(sealed_seed["brier"].mean()), 0.1305),
}
for k, (got, want) in refs.items():
    assert abs(got - want) < 5e-4, f"reference {k}: {got:.4f} != {want:.4f}"
print("[gate] reference numbers OK:", {k: round(v[0], 4) for k, v in refs.items()})

# ---- fit corpora ---------------------------------------------------------
train_tasks = tasks[tasks["family"].isin(TRAIN5)]
rows = [{"task_id": tid, "model": m, "y": float(frame.at[m, tid])}
        for tid in train_tasks["task_id"] for m in PANEL
        if not pd.isna(frame.at[m, tid])]
fit_all = pd.DataFrame(rows).merge(tasks, on="task_id")
fit_all["log_fst"] = np.log(fit_all["fst_minutes"])
assert len(fit_all) == 1923
assert not set(fit_all["family"]) & TESTF
print(f"[audit] ZERO-LABEL fit corpus: {len(fit_all)} train-family cells only")

MIDX = {m: i for i, m in enumerate(PANEL)}


def design(df):
    X = np.zeros((len(df), len(PANEL) + 1))
    X[np.arange(len(df)), df["model"].map(MIDX).to_numpy()] = 1.0
    X[:, -1] = df["log_fst"].to_numpy()
    return X


def fit_logit(df):
    X, y = design(df), df["y"].to_numpy()

    def nll(w):
        z = X @ w
        return float(np.sum(np.logaddexp(0.0, z) - y * z) + 1e-8 * np.sum(w * w))

    def grad(w):
        return X.T @ (sigmoid(X @ w) - y) + 2e-8 * w

    r = optimize.minimize(nll, np.zeros(X.shape[1]), jac=grad, method="L-BFGS-B",
                          options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-10})
    assert r.success, r.message
    return r.x


def predict(w, df):
    return sigmoid(design(df) @ w)


def tstats(delta_cell, task_ids):
    d = np.asarray(delta_cell)
    t_cell = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    per_task = pd.DataFrame({"t": task_ids, "d": d}).groupby("t")["d"].mean()
    t_task = per_task.mean() / (per_task.std(ddof=1) / np.sqrt(len(per_task)))
    return float(t_cell), float(t_task), int(len(per_task))


test_sc = seed_cell.merge(tasks, on="task_id")
test_sc["log_fst"] = np.log(test_sc["fst_minutes"])

CORPORA = {
    "A5_all_train": fit_all,
    "A4_minus_intercode": fit_all[fit_all["family"] != "intercode_ctf"],
    "A3_minus_intercode_nyuctf": fit_all[~fit_all["family"].isin({"intercode_ctf", "nyuctf"})],
}
frozen = {"panel": PANEL, "w_blend": W_BLEND, "p50_clip": CLIP, "curves": {}}
for name, corpus in CORPORA.items():
    w = fit_logit(corpus)
    p = predict(w, test_sc)
    brier = (p - test_sc["outcome"]) ** 2
    tc, tt, ntask = tstats(brier.to_numpy() - test_sc["seed_brier"].to_numpy(),
                           test_sc["task_id"])
    frozen["curves"][name] = {
        "intercepts": {m: float(w[MIDX[m]]) for m in PANEL},
        "slope_ln_fst": float(w[-1]),
        "n_fit_cells": int(len(corpus)),
        "test_brier": float(brier.mean()),
        "test_mean_pred": float(p.mean()),
        "t_cell_vs_seed": tc, "t_task_vs_seed": tt,
    }
    print(f"[curve {name}] test {brier.mean():.4f}  mean_pred {p.mean():.4f}  "
          f"slope {w[-1]:.4f}  t_task_vs_seed {tt:.2f}")

# ---- LOFO offset band within A4 families ---------------------------------
def lofo_band(families):
    offs = {}
    sub = fit_all[fit_all["family"].isin(families)]
    for fam in sorted(families):
        w = fit_logit(sub[sub["family"] != fam])
        held = sub[sub["family"] == fam]
        eta = design(held) @ w

        def nll_off(d):
            z = eta + d[0]
            return float(np.sum(np.logaddexp(0.0, z) - held["y"].to_numpy() * z))

        r = optimize.minimize(nll_off, [0.0], method="L-BFGS-B")
        offs[fam] = float(r.x[0])
    v = np.array(list(offs.values()))
    return offs, float(v.mean()), float(v.std(ddof=1)), [float(v.min()), float(v.max())]


offs4, m4, s4, r4 = lofo_band(TRAIN5 - {"intercode_ctf"})
offs5, m5, s5, r5 = lofo_band(TRAIN5)
frozen["lofo_band_clean4"] = {"offsets": offs4, "mean": m4, "sd": s4, "range": r4}
frozen["lofo_band_all5"] = {"offsets": offs5, "mean": m5, "sd": s5, "range": r5}
print(f"[lofo clean4] mean {m4:.3f} sd {s4:.3f} range {r4}  offsets {offs4}")

# ---- blends --------------------------------------------------------------
def blend_score(curve_name):
    w = frozen["curves"][curve_name]
    coef = np.array([w["intercepts"][m] for m in PANEL] + [w["slope_ln_fst"]])
    df = seed_test.merge(tasks, on="task_id")
    df["log_fst"] = np.log(df["fst_minutes"])
    curve_p = np.clip(predict(coef, df), 1e-9, 1 - 1e-9)
    p50 = np.clip(df["p50"].to_numpy(), *CLIP)
    bl = sigmoid((1 - W_BLEND) * logit(curve_p) + W_BLEND * logit(p50))
    df["bl_brier"] = (bl - df["outcome"]) ** 2
    cell = df.groupby(["task_id", "model"], as_index=False).agg(
        b=("bl_brier", "mean"), o=("outcome", "first"))
    cell = cell.merge(seed_cell[["task_id", "model", "seed_brier"]], on=["task_id", "model"])
    cw = frozen["curves"][curve_name]
    curve_cell = test_sc.copy()
    ccoef = np.array([cw["intercepts"][m] for m in PANEL] + [cw["slope_ln_fst"]])
    curve_brier = (predict(ccoef, curve_cell) - curve_cell["outcome"]) ** 2
    tc_s, tt_s, _ = tstats(cell["b"].to_numpy() - cell["seed_brier"].to_numpy(), cell["task_id"])
    tc_c, tt_c, _ = tstats(cell["b"].to_numpy() - curve_brier.to_numpy(), cell["task_id"])
    return {"test_brier": float(cell["b"].mean()),
            "t_task_vs_seed": tt_s, "t_task_vs_curve": tt_c}


frozen["blend_A5"] = blend_score("A5_all_train")
frozen["blend_A4"] = blend_score("A4_minus_intercode")
print(f"[blend A5] {frozen['blend_A5']}")
print(f"[blend A4] {frozen['blend_A4']}")

# ---- cap rule ------------------------------------------------------------
def cap_table(evid_path):
    recs = json.load(open(evid_path))
    caps = {}
    for r in recs:
        j = r["bin"]
        easier = {p["bin_index"]: p["pass_rate"] for p in r["profiles"]}.get(j - 1)
        caps[(r["task_id"], r["model"])] = easier
    return caps


caps_t = cap_table(EVID_TEST)
dfc = seed_test.copy()
dfc["cap"] = [caps_t.get((t, m)) for t, m in zip(dfc["task_id"], dfc["model"])]
dfc["p50c"] = np.where(dfc["cap"].notna(), np.minimum(dfc["p50"], dfc["cap"].astype(float)), dfc["p50"])
dfc["bc"] = (dfc["p50c"] - dfc["outcome"]) ** 2
cap_test = float(dfc.groupby(["task_id", "model"])["bc"].mean().mean())

caps_s = cap_table(EVID_SEALED)
sfc = sealed_seed.copy()
sfc["cap"] = [caps_s.get((t, m)) for t, m in zip(sfc["task_id"], sfc["model"])]
sfc["p50c"] = np.where(sfc["cap"].notna(), np.minimum(sfc["p50"], sfc["cap"].astype(float)), sfc["p50"])
cap_sealed = float(((sfc["p50c"] - sfc["outcome"]) ** 2).mean())
frozen["cap_rule"] = {"test_brier": cap_test, "sealed_brier": cap_sealed,
                      "definition": "p50 <- min(p50, printed pass rate of nearest easier shown bin)"}
print(f"[cap] test {cap_test:.4f}  sealed {cap_sealed:.4f}")

# ---- gates vs the two judges' independent numbers ------------------------
gates = [
    ("A5 test", frozen["curves"]["A5_all_train"]["test_brier"], 0.1522, 5e-4),
    ("A4 test", frozen["curves"]["A4_minus_intercode"]["test_brier"], 0.1466, 8e-4),
    ("A4 mean pred", frozen["curves"]["A4_minus_intercode"]["test_mean_pred"], 0.241, 5e-3),
    ("blend A5", frozen["blend_A5"]["test_brier"], 0.1448, 8e-4),
    ("blend A4", frozen["blend_A4"]["test_brier"], 0.1398, 8e-4),
    ("cap test", cap_test, 0.1556, 8e-4),
    ("cap sealed", cap_sealed, 0.1111, 8e-4),
    ("lofo4 mean", m4, -0.062, 0.03),
    ("lofo4 sd", s4, 0.401, 0.05),
]
bad = [(n, g, w) for n, g, w, tol in gates if abs(g - w) > tol]
for n, g, w, tol in gates:
    print(f"[gate] {n}: {g:.4f} vs expected {w:.4f} -> {'OK' if abs(g-w)<=tol else 'FAIL'}")
if bad:
    raise SystemExit(f"FROZEN-ARTIFACT GATES FAILED: {bad}")

OUT.write_text(json.dumps(frozen, indent=2))
print(f"\nAll gates passed. Frozen artifacts -> {OUT}")
