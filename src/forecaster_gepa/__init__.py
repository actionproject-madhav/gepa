# Copyright (c) 2026 SaferAI. GEPA forecaster-optimisation experiment.
"""GEPA on the LLM_estimator intra-benchmark forecasting pipeline.

This package holds everything fork-specific: the metric adapter, the
bin-stratified minibatch sampler, the run harness, diagnostics/logging and
checkpoint wiring. The estimation pipeline itself lives in the
``llm-estimator`` package (the LLM_elicitation repo).
"""

from forecaster_gepa.config import COMPONENT_NAME, ForecasterGEPAConfig, load_config

__all__ = ["COMPONENT_NAME", "ForecasterGEPAConfig", "load_config"]
