"""Automatic target-setting rules.

Set each feature's representation target by a rule rather than by hand.
Each rule returns a ``{feature_id: target_amount}`` mapping;
:func:`apply_targets` writes it onto a problem's features.

Mirrors prioritizr's ``add_relative_targets`` / ``add_auto_targets`` /
``add_group_targets`` (Hanson et al. 2024).
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from pymarxan.models.problem import ConservationProblem


def relative_targets(
    problem: ConservationProblem, fraction: float
) -> dict[int, float]:
    """Target = ``fraction`` of each feature's total amount."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    totals = problem.feature_amounts()
    return {
        int(f): fraction * float(totals.get(int(f), 0.0))
        for f in problem.features["id"]
    }


def loglinear_targets(
    problem: ConservationProblem,
    *,
    lower_area: float,
    lower_target: float,
    upper_area: float,
    upper_target: float,
) -> dict[int, float]:
    """IUCN-style range-size targets, interpolated log-linearly.

    Features whose total amount is at or below ``lower_area`` get the
    ``lower_target`` fraction; at or above ``upper_area`` they get
    ``upper_target``; in between, the fraction is interpolated linearly on
    ``log10`` of the total amount. The returned value is the fraction times
    the feature's total amount.
    """
    if lower_area <= 0 or upper_area <= 0:
        raise ValueError("lower_area and upper_area must be positive")
    if upper_area <= lower_area:
        raise ValueError("upper_area must exceed lower_area")

    totals = problem.feature_amounts()
    log_lo = math.log10(lower_area)
    log_hi = math.log10(upper_area)
    out: dict[int, float] = {}
    for f in problem.features["id"]:
        total = float(totals.get(int(f), 0.0))
        if total <= lower_area:
            frac = lower_target
        elif total >= upper_area:
            frac = upper_target
        else:
            t = (math.log10(total) - log_lo) / (log_hi - log_lo)
            frac = lower_target + (upper_target - lower_target) * t
        out[int(f)] = frac * total
    return out


def group_targets(
    problem: ConservationProblem,
    groups: Mapping[int, str],
    fractions: Mapping[str, float],
) -> dict[int, float]:
    """Apply a per-group relative target to each member feature.

    ``groups`` maps feature id to a group label; ``fractions`` maps each
    group label to the fraction of total amount to target. Every group
    referenced in ``groups`` must have an entry in ``fractions``.
    """
    missing = {g for g in groups.values() if g not in fractions}
    if missing:
        raise ValueError(f"no fraction given for group(s): {sorted(missing)}")
    totals = problem.feature_amounts()
    out: dict[int, float] = {}
    for f in problem.features["id"]:
        fid = int(f)
        g = groups.get(fid)
        if g is not None:
            out[fid] = float(fractions[g]) * float(totals.get(fid, 0.0))
    return out


def apply_targets(
    problem: ConservationProblem, targets: Mapping[int, float]
) -> ConservationProblem:
    """Write ``{feature_id: target}`` onto the problem's features in place.

    Features not present in ``targets`` keep their existing target. Returns
    the same problem for chaining.
    """
    fmap = {int(k): float(v) for k, v in targets.items()}
    problem.features["target"] = [
        fmap.get(int(fid), float(t))
        for fid, t in zip(problem.features["id"], problem.features["target"])
    ]
    return problem
