#!/usr/bin/env python3
"""Repeated-measurement study of the val/finalist Brier instruments.

The ladder's central finding is that a single evaluation pass is a noisy
instrument: the identical seed prompt drew val Briers of 0.1144 / 0.0995 /
0.1034 across three runs (forecaster temperature 1.0). This script measures
that noise directly instead of inferring it from three accidental draws:

  For each prompt in --prompts-dir, evaluate the FULL val set (84 cells)
  --repeats-val times and the FULL finalist set (~230 cells)
  --repeats-sealed times, with fresh forecaster calls each repeat.

Pre-declared readouts:
  1. Between-repeat SD of the grand Brier, per prompt per set — the
     instrument's noise floor (expected to shrink ~1/sqrt(K) under
     averaging; the per-repeat numbers let us verify that).
  2. july_cand20 vs seed paired edge per repeat index — is the ladder's one
     verified improvement (+0.0173 sealed) present in EVERY re-measurement,
     or was it partly a lucky draw?
  3. The K needed for future candidate selection to resolve ~0.01 effects.

Cost: (repeats_val x 84 + repeats_sealed x ~230) calls per prompt; defaults
(5 and 3, 4 prompts) ~= 4,440 forecaster calls.

Usage (from the gepa repo root):
    uv run python scripts/val_noise_study.py
    uv run python scripts/val_noise_study.py --limit-cells 2 --repeats-val 1 --repeats-sealed 0  # smoke test
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from forecaster_gepa.adapter import ForecasterAdapter  # noqa: E402
from forecaster_gepa.config import COMPONENT_NAME, load_config  # noqa: E402
from forecaster_gepa.data import ExperimentData  # noqa: E402


def grand_brier(outputs, failure_brier: float = 1.0) -> float:
    vals = [o["brier"] if o["brier"] is not None else failure_brier for o in outputs]
    return sum(vals) / len(vals)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "configs/pilot_baseline.yaml"))
    ap.add_argument("--prompts-dir", default=str(REPO / "configs/noise_study_prompts"))
    ap.add_argument("--repeats-val", type=int, default=5)
    ap.add_argument("--repeats-sealed", type=int, default=3)
    ap.add_argument("--out", default=str(REPO / "runs/noise_study"))
    ap.add_argument("--limit-cells", type=int, default=None, help="smoke-testing only")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cells_path = out / "noise_cells.jsonl"

    prompts = sorted(Path(args.prompts_dir).glob("*.txt"))
    assert prompts, f"no .txt prompts in {args.prompts_dir}"
    data = ExperimentData(cfg)

    batches: list[tuple[str, list, ForecasterAdapter, int]] = []
    if args.repeats_val > 0:
        plans, _tl, val_loader, _tb = data.search_phase()
        val_specs = list(val_loader.fetch(val_loader.all_ids()))
        if args.limit_cells:
            val_specs = val_specs[: args.limit_cells]
        batches.append(("val", val_specs, ForecasterAdapter(plans, cfg), args.repeats_val))
    if args.repeats_sealed > 0:
        fplans, fspecs = data.finalist_phase()
        fspecs = list(fspecs)
        if args.limit_cells:
            fspecs = fspecs[: args.limit_cells]
        batches.append(("sealed", fspecs, ForecasterAdapter(fplans, cfg), args.repeats_sealed))

    summary: dict[str, dict[str, list[float]]] = {}
    for prompt_path in prompts:
        name = prompt_path.stem
        template = prompt_path.read_text(encoding="utf-8")
        summary[name] = {}
        for set_name, specs, adapter, repeats in batches:
            grands = []
            for r in range(repeats):
                eb = adapter.evaluate(specs, {COMPONENT_NAME: template}, capture_traces=False)
                g = grand_brier(eb.outputs)
                grands.append(g)
                n_fail = sum(1 for o in eb.outputs if o["error"])
                with cells_path.open("a", encoding="utf-8") as fh:
                    for o in eb.outputs:
                        fh.write(json.dumps({"prompt": name, "set": set_name, "repeat": r, **o}) + "\n")
                print(f"{name} {set_name} repeat {r + 1}/{repeats}: grand Brier {g:.4f} "
                      f"({len(eb.outputs)} cells, {n_fail} failures)", flush=True)
            summary[name][set_name] = grands

    lines = ["prompt          set     repeats                         mean    sd"]
    for name, sets in summary.items():
        for set_name, gs in sets.items():
            sd = statistics.stdev(gs) if len(gs) > 1 else float("nan")
            lines.append(f"{name:<15} {set_name:<7} {' '.join(f'{g:.4f}' for g in gs):<31} "
                         f"{statistics.mean(gs):.4f}  {sd:.4f}")
    report = "\n".join(lines)
    print("\n" + report)
    (out / "noise_summary.txt").write_text(report + "\n", encoding="utf-8")
    (out / "noise_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
