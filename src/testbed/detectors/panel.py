"""Apply the frozen Phase-1 detector panel to a new set of rollouts (checkpoint evaluation)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from testbed.detectors.probes import LinearProbe, MLPProbe, AttentionProbe, YudelsonProbe
from testbed.paths import ARTIFACTS


def apply_frozen(panel_dir: Path, cap: dict, keys: list[str]) -> dict[str, dict[str, float]]:
    """``cap`` from ``activations.load_capture``; ``keys`` = 'family|task_id|sample_idx' for rows to score.
    Returns {detector_name: {key: score}}."""
    man = json.load(open(panel_dir / "frozen_manifest.json"))
    key2i = {f"{ix['family']}|{ix['task_id']}|{ix['sample_idx']}": i for i, ix in enumerate(cap["index"])}
    idx = np.array([key2i[k] for k in keys])
    out = {}
    for name, f in man["frozen"].items():
        if f["type"] == "linear":
            pr = LinearProbe.load(panel_dir / f["file"]); s = pr.score(cap["pooled"][f["pool"]][idx, f["layer"]].astype(np.float32))
        elif f["type"] == "mlp":
            st = torch.load(panel_dir / f["file"], weights_only=False, map_location="cpu"); m = MLPProbe(); m.mu, m.sd = st["mu"], st["sd"]
            X = cap["pooled"][f["pool"]][idx, f["layer"]].astype(np.float32)
            import torch.nn as nn
            m.net = nn.Sequential(nn.Linear(X.shape[1], m.hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(m.hidden, 1)); m.net.load_state_dict(st["state"]); m.net.eval()
            s = m.score(X)
        elif f["type"] == "attention":
            st = torch.load(panel_dir / f["file"], weights_only=False, map_location="cpu"); ap = AttentionProbe(); ap.mu, ap.sd = st["mu"], st["sd"]
            j = list(cap["layers"]).index(f["layer"]); H = cap["tok"][idx, j].astype(np.float32); M = cap["pos"][idx] >= 0
            ap.mod = ap._build(H.shape[-1]); ap.mod.load_state_dict(st["state"]); ap.mod.eval(); s = ap.score(H, M)
        elif f["type"] == "yudelson":
            s = YudelsonProbe(ARTIFACTS / "detectors" / f["path"]).score(cap["followup"][idx].astype(np.float32))
        else:
            continue
        out[name] = dict(zip(keys, map(float, s)))
    return out
