#!/usr/bin/env python3
"""Closing analysis: level-corrected OOD equivalence of all measured prompts.

Per prompt, remove its forecast level with one global logit shift (ORACLE =
fit on all 1,033 reserved-test cells, labeled; K-LABEL = prequential, 200
draws) and compare corrected Briers. Result (2026-09-06): all 8 prompts --
seed, every GEPA winner, both probes, the numbers-banned v2, the worst
overfit -- land at 0.150-0.155 oracle-corrected with no significant paired
differences (all |t_task| < 1.4). The probability-prompt space is one
forecaster in disguise out-of-family; prompt optimization only ever moved
the one number this correction removes. None beat the time-elicitation
curve (0.1512 zero-label / 0.1395 at k=50).
Run: uv run python scripts/audit_scoring/level_equivalence.py
"""
