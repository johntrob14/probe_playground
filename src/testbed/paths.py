from __future__ import annotations
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STORE = Path(os.environ.get("TESTBED_STORE", "/ssd1/john/probe_playground"))
DATA = STORE / "data"
ARTIFACTS = STORE / "artifacts"
RUNS = STORE / "runs"
for _p in (ARTIFACTS, RUNS):
    _p.mkdir(parents=True, exist_ok=True)
