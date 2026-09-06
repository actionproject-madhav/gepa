#!/usr/bin/env python3
"""Score the external-corpus validation exactly per runs/external_validation_prereg.md.

Primary gate: (A) SEC-bench mean within-scaffold AUC over {OpenHands, SWE-agent},
instance-clustered bootstrap CI > 0.5; (B) AIRTBench mean within-model AUC over
the 9 pre-named models, per-run binary, challenge-clustered bootstrap CI > 0.5.
Everything else secondary/descriptive. Baselines computed only now (after the
elicited times are frozen on disk). Run: uv run python scripts/audit_scoring/external_score.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
NS = REPO / "runs/noise_study"
CACHE = REPO / "runs/external_corpora"
SEED = 0
B = 10_000

AIRT_MODELS_9 = ["claude-3-7-sonnet", "gemini-2.5-pro-preview", "gpt-4.5-preview",
                 "o3-mini", "gemini-2.5-flash-preview", "DeepSeek-R1",
                 "gemini-2.0-flash", "gpt-4o", "gemini-1.5-pro"]


def auc(score, y):
    s, y = np.asarray(score, float), np.asarray(y, int)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return np.nan
    return float(np.mean([(np.mean((p > neg) + 0.5 * (p == neg))) for p in pos]))


def med_times(elicitor):
    t = pd.DataFrame([json.loads(l) for l in open(NS / "noise_cells_external_times.jsonl")])
    t = t[(t["elicitor"] == elicitor) & t["mid_minutes"].notna()]
    agg = t.groupby(["corpus", "task_id"])["mid_minutes"].median()
    return agg


def secbench_frame():
    outs = {}
    for s in ["oh", "swea", "aider"]:
        outs[s] = {json.loads(l)["instance_id"]: int(json.loads(l)["success"])
                   for l in open(CACHE / f"secbench_{s}_report.jsonl") if l.strip()}
    return outs


def airt_frame():
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(CACHE / "airtbench.parquet")
    parts = []
    for batch in pf.iter_batches(batch_size=512, columns=["model", "challenge_name",
                                                          "flag_found", "challenge_difficulty"]):
        parts.append(batch.to_pandas())
    return pd.concat(parts, ignore_index=True)


def boot_ci(fn, clusters, rng):
    vals = []
    uc = np.array(sorted(set(clusters)))
    for _ in range(B):
        pick = rng.choice(uc, len(uc), replace=True)
        vals.append(fn(pick))
    vals = np.array([v for v in vals if not np.isnan(v)])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def score_elicitor(elicitor, out):
    times = med_times(elicitor)
    res = {}

    # ---- A. SEC-bench -----------------------------------------------------
    outs = secbench_frame()
    sec = times.loc["secbench"] if "secbench" in times.index.get_level_values(0) else pd.Series(dtype=float)
    ids = [i for i in outs["oh"] if i in sec.index]
    tvec = np.array([sec[i] for i in ids])
    ys = {s: np.array([outs[s][i] for i in ids]) for s in outs}
    a_oh, a_sw = auc(-tvec, ys["oh"]), auc(-tvec, ys["swea"])
    mean_auc = np.mean([a_oh, a_sw])
    rng = np.random.default_rng(SEED)
    idx = {i: k for k, i in enumerate(ids)}

    def sec_stat(pick):
        sel = [idx[i] for i in pick]
        return np.mean([auc(-tvec[sel], ys["oh"][sel]), auc(-tvec[sel], ys["swea"][sel])])

    lo, hi = boot_ci(sec_stat, ids, rng)
    res["secbench"] = {"n_tasks": len(ids), "auc_oh": a_oh, "auc_swea": a_sw,
                       "auc_aider_descriptive": auc(-tvec, ys["aider"]),
                       "primary_mean_auc": float(mean_auc), "ci95": [lo, hi],
                       "gate_pass": bool(lo > 0.5)}
    # baseline: patch length (context, computed after freeze)
    try:
        pt = pd.read_parquet(CACHE / "secbench_eval.parquet", columns=["instance_id", "patch"])
        pl = {r.instance_id: len(str(r.patch)) for r in pt.itertuples()}
        pvec = np.array([pl.get(i, np.nan) for i in ids])
        res["secbench"]["patchlen_auc_oh_context"] = auc(-pvec, ys["oh"])
    except Exception as e:
        res["secbench"]["patchlen_note"] = f"unavailable: {e}"

    # ---- B. AIRTBench -----------------------------------------------------
    air = airt_frame()
    at = times.loc["airtbench"] if "airtbench" in times.index.get_level_values(0) else pd.Series(dtype=float)
    air = air[air["challenge_name"].isin(at.index)]
    air["t"] = air["challenge_name"].map(at)

    def model_key(m):
        for k in AIRT_MODELS_9:
            if k.lower() in str(m).lower():
                return k
        return None

    air["mk"] = air["model"].map(model_key)
    use = air[air["mk"].notna()]
    per_model = {k: auc(-g["t"], g["flag_found"].astype(int))
                 for k, g in use.groupby("mk")}
    mean_air = float(np.nanmean(list(per_model.values())))
    chal = use["challenge_name"].to_numpy()
    rng = np.random.default_rng(SEED)

    def air_stat(pick):
        sub = use[use["challenge_name"].isin(pick)]
        vals = [auc(-g["t"], g["flag_found"].astype(int)) for _k, g in sub.groupby("mk")]
        return float(np.nanmean(vals)) if vals else np.nan

    lo, hi = boot_ci(air_stat, sorted(set(chal)), rng)
    # baseline: their 3-level difficulty label (secondary, paired delta)
    dmap = {"easy": 1, "medium": 2, "hard": 3}
    ddiff = use["challenge_difficulty"].astype(str).str.lower().map(dmap)
    per_model_lab = {k: auc(-g_d, g["flag_found"].astype(int))
                     for (k, g), (_k2, g_d) in zip(use.groupby("mk"), ddiff.groupby(use["mk"]))}
    res["airtbench"] = {"n_challenges": int(use["challenge_name"].nunique()),
                        "n_models_used": len(per_model), "per_model_auc": per_model,
                        "primary_mean_auc": mean_air, "ci95": [lo, hi],
                        "gate_pass": bool(lo > 0.5),
                        "difficulty_label_mean_auc_secondary": float(np.nanmean(list(per_model_lab.values())))}
    out[elicitor] = res
    return res


def main():
    out = {}
    for e in ["claude-sonnet-4-6", "claude-haiku-4-5"]:
        r = score_elicitor(e, out)
        print(f"\n== {e} ==")
        print(f"SEC-bench: mean AUC {r['secbench']['primary_mean_auc']:.3f} "
              f"CI {r['secbench']['ci95']} gate {'PASS' if r['secbench']['gate_pass'] else 'FAIL'} "
              f"(oh {r['secbench']['auc_oh']:.3f}, swea {r['secbench']['auc_swea']:.3f}, "
              f"aider desc {r['secbench']['auc_aider_descriptive']:.3f})")
        print(f"AIRTBench: mean AUC {r['airtbench']['primary_mean_auc']:.3f} "
              f"CI {r['airtbench']['ci95']} gate {'PASS' if r['airtbench']['gate_pass'] else 'FAIL'} "
              f"(label baseline {r['airtbench']['difficulty_label_mean_auc_secondary']:.3f})")

    # recognition probe
    rec = pd.DataFrame([json.loads(l) for l in open(NS / "external_recognition.jsonl")])
    rec["recognized"] = ~rec["reply"].str.strip().str.upper().str.startswith("UNKNOWN")
    rates = rec.groupby("corpus")["recognized"].mean().to_dict()
    out["recognition_rate"] = rates
    print(f"\nrecognition rates: {rates}")
    # robustness: primary AUCs on unrecognized subset (sonnet)
    unrec = set(rec[~rec["recognized"]]["task_id"])
    times = med_times("claude-sonnet-4-6")
    outs = secbench_frame()
    sec = times.loc["secbench"]
    ids = [i for i in outs["oh"] if i in sec.index and i in unrec]
    if len(ids) >= 30:
        tv = np.array([sec[i] for i in ids])
        m = np.mean([auc(-tv, np.array([outs[s][i] for i in ids])) for s in ["oh", "swea"]])
        out["secbench_unrecognized_mean_auc"] = float(m)
        print(f"SEC-bench unrecognized-subset ({len(ids)}) mean AUC: {m:.3f}")

    Path(REPO / "scripts/audit_scoring/external_scores.json").write_text(json.dumps(out, indent=1, default=str))
    print("\nsaved external_scores.json")


if __name__ == "__main__":
    main()
