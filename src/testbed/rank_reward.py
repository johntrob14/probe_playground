"""Label-free within-prompt rank penalties; no model or monitor fitting."""
from __future__ import annotations

import numpy as np


def group_rank_penalties(logits, group_size):
    """Average ranks normalized to [0,1], separately within each prompt group.

    Higher monitor logits get higher penalties. Exact ties get average ranks.
    All-tied groups therefore have a constant penalty, canceled by centering.
    Strictly increasing transforms preserve ranks absent floating-point ties.
    """
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or type(group_size) is not int or group_size < 2:
        raise ValueError("Expected a flat logit array and integer group_size >= 2")
    if len(values) == 0 or len(values) % group_size or not np.isfinite(values).all():
        raise ValueError("Require nonempty complete groups with finite logits")
    result = np.empty_like(values)
    for offset in range(0, len(values), group_size):
        group = values[offset:offset + group_size]
        ordered = np.sort(group)
        left = np.searchsorted(ordered, group, side="left")
        right = np.searchsorted(ordered, group, side="right")
        result[offset:offset + group_size] = (left + right - 1) / (2 * (group_size - 1))
    return result
