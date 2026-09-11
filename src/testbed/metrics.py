from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auroc(scores, labels) -> float:
    y = np.asarray(labels).astype(int); s = np.asarray(scores, dtype=float)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def tpr_at_fpr(scores, labels, fpr_target: float = 0.01) -> tuple[float, float]:
    """(TPR at the largest threshold whose FPR <= target, threshold). Positive class = hack (1)."""
    y = np.asarray(labels).astype(int); s = np.asarray(scores, dtype=float)
    if y.min() == y.max():
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(y, s)
    ok = fpr <= fpr_target
    if not ok.any():
        return 0.0, float("inf")
    i = np.argmax(np.where(ok, tpr, -1))
    return float(tpr[i]), float(thr[i])


def bootstrap_auroc(scores, labels, n_boot: int = 500, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed); y = np.asarray(labels).astype(int); s = np.asarray(scores, dtype=float)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        vals.append(roc_auc_score(y[idx], s[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def summarize_detector(scores, labels) -> dict:
    a = auroc(scores, labels)
    t1, thr1 = tpr_at_fpr(scores, labels, 0.01)
    t5, _ = tpr_at_fpr(scores, labels, 0.05)
    lo, hi = bootstrap_auroc(scores, labels) if not np.isnan(a) else (float("nan"), float("nan"))
    return {"auroc": a, "auroc_ci95": [lo, hi], "tpr_at_1fpr": t1, "thr_at_1fpr": thr1, "tpr_at_5fpr": t5,
            "n_pos": int(np.sum(np.asarray(labels).astype(int))), "n": int(len(labels))}
