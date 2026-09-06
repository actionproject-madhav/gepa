#!/usr/bin/env python3
"""Score external v2 (standardized template) per runs/external_v2_prereg.md:
same primary gates as v1 + paired v1-vs-v2 agreement readouts."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/audit_scoring"))
import external_score as es  # noqa: E402


def med_times_file(fname, elicitor):
    t = pd.DataFrame([json.loads(l) for l in open(REPO / "runs/noise_study" / fname)])
    t = t[(t["elicitor"] == elicitor) & t["mid_minutes"].notna()]
    return t.groupby(["corpus", "task_id"])["mid_minutes"].median()


def main():
    # v2 gates: reuse the v1 scorer wholesale by pointing its reader at v2
    es.med_times = lambda e: med_times_file("noise_cells_external_times_v2.jsonl", e)
    out = {}
    for e in ["claude-sonnet-4-6", "claude-haiku-4-5"]:
        r = es.score_elicitor(e, out)
        print(f"\n== v2 {e} ==")
        print(f"SEC-bench: mean AUC {r['secbench']['primary_mean_auc']:.3f} "
              f"CI {[round(x,3) for x in r['secbench']['ci95']]} "
              f"gate {'PASS' if r['secbench']['gate_pass'] else 'FAIL'}")
        print(f"AIRTBench: mean AUC {r['airtbench']['primary_mean_auc']:.3f} "
              f"CI {[round(x,3) for x in r['airtbench']['ci95']]} "
              f"gate {'PASS' if r['airtbench']['gate_pass'] else 'FAIL'}")
    # paired v1-vs-v2 agreement (descriptive)
    for e in ["claude-sonnet-4-6", "claude-haiku-4-5"]:
        v1 = med_times_file("noise_cells_external_times.jsonl", e)
        v2 = med_times_file("noise_cells_external_times_v2.jsonl", e)
        for corpus in ["secbench", "airtbench"]:
            a, b = v1.loc[corpus], v2.loc[corpus]
            common = a.index.intersection(b.index)
            rho = pd.Series(a[common]).corr(pd.Series(b[common]), method="spearman")
            out[f"spearman_v1v2_{e}_{corpus}"] = float(rho)
            print(f"v1-v2 Spearman ({e}, {corpus}): {rho:.3f} (n={len(common)})")
    Path(REPO / "scripts/audit_scoring/external_scores_v2.json").write_text(
        json.dumps(out, indent=1, default=str))
    print("\nsaved external_scores_v2.json")


if __name__ == "__main__":
    main()
