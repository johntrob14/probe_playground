"""Label-free, group-strength-matched controls for detached monitor rewards."""

from __future__ import annotations

import numpy as np

from testbed.rank_reward import group_rank_penalties


def _affine_sigmoid_values(group):
    """A positive affine transform of ideal sigmoid values, without saturation.

    expm1 preserves small logit differences, including near zero. All exponential
    arguments are nonpositive. An anchor at a group endpoint avoids losing all
    differences when ordinary sigmoid rounds an entire group to zero or one.
    The common scale/offset cancel when the group is standardized below.
    """
    if np.max(group) <= 0:
        # 2 * (sigmoid(z) - sigmoid(max(z))) / sigmoid(max(z)).
        return np.expm1(group - np.max(group)) * (2 / (1 + np.exp(group)))
    if np.min(group) >= 0:
        # 2 * (sigmoid(z) - sigmoid(min(z))) / (1 - sigmoid(min(z))).
        return -np.expm1(np.min(group) - group) * (2 / (1 + np.exp(-group)))
    # 4 * (sigmoid(z) - 1/2), with no cancellation around z=0.
    magnitude = np.abs(group)
    return -np.sign(group) * np.expm1(-magnitude) * (2 / (1 + np.exp(-magnitude)))


def variance_matched_probability_penalties(logits, group_size):
    """Match each group's centered rank-penalty SD, retaining sigmoid spacing.

    For ideal p=sigmoid(logits) and average normalized logit ranks r, return
    0.5 + std(r) * (p - mean(p)) / std(p), separately for each complete group.
    Exact all-logit ties return 0.5. Input validation matches group_rank_penalties:
    a nonempty flat finite float64-compatible array and an integer group_size>=2.

    No labels, epsilon, clipping, or fitted parameters are used. Penalties may
    extend outside [0,1]; clipping would break the variance match. This matches
    scalar monitor-advantage dispersion, not the total reward or gradient norm.
    Numerical affine reparameterization avoids sigmoid-tail underflow and
    saturation; the returned penalties are NOT calibrated detector scores.
    """
    values = np.asarray(logits, dtype=np.float64)
    ranks = group_rank_penalties(values, group_size)
    result = np.empty_like(values)
    for offset in range(0, len(values), group_size):
        group = values[offset:offset + group_size]
        target_sd = ranks[offset:offset + group_size].std()
        if target_sd == 0:
            result[offset:offset + group_size] = 0.5
            continue
        affine = _affine_sigmoid_values(group)
        # Scaling before centering also protects means of subnormal differences.
        affine /= np.max(np.abs(affine))
        centered = affine - affine.mean()
        result[offset:offset + group_size] = 0.5 + target_sd * centered / centered.std()
    if not np.isfinite(result).all():
        raise FloatingPointError("Variance-matched monitor penalty is nonfinite")
    return result
