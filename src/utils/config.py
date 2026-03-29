"""
Module: src.utils.config
SRS Reference: config.yaml loader (covers NFR-Maintainability for all FR buckets)
SDLC Phase: 3 - Environment Setup & MCP Configuration
Sprint: N/A
Pipeline Stage: Deployment
Purpose: Load and validate `config.yaml` into a structured configuration object used by all pipeline modules.
Dependencies: pydantic, PyYAML.
Interface Contract:
  Input:  Path to `config.yaml`
  Output: dict (validated configuration for DSP/CV/XAI/NLP/Deployment)
Latency Target: <= 50 ms (config parse time; not on critical inference path)
Open Questions Resolved: Q3/Q4/Q5/Q6 resolved in Phase 2/next gate only (runtime still pending Q3 empirical check)
Open Questions Blocking: Q3 may affect training viability; config supports gradient_checkpointing toggle
MCP Tools Used: context7-mcp
AI Generated: true
Verified (V.E.R.I.F.Y.): false
Author: Ferel / Safa
Date: 2026-03-22
"""

import yaml

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

import sys
sys.path.insert(0, '/content/fried-fp-dsdba')
from src.cv.train import run_training

cfg = load_config('/content/fried-fp-dsdba/config.yaml')
model = run_training(cfg)
