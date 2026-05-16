"""Marxan-faithful Z-score chance-constraint math (PROBMODE 3).

This is the pure-functional layer: no problem objects, no solver state.
Inputs are plain dicts keyed by feature id; outputs are scalars or dicts.
Solver integration lives in ``cache.py`` (SA / iterative improvement),
``heuristic.py`` (inline), and ``mip_solver.py`` (post-hoc evaluation).

The formulation mirrors Marxan v4's ``computation.hpp::computeProbMeasures``:

    Z_j = (T_j − E[T_j]) / sqrt(Var[T_j])    # positive = shortfall
    P_j = probZUT(Z_j) = 1 − Φ(Z_j)          # scipy.stats.norm.sf

When Var[T_j] == 0 the routine returns Marxan's sentinel ``rZ = 4``
(``probZUT(4) ≈ 3e-5``), which short-circuits the upper-tail probability
to "trivially met". The deterministic-target shortfall is handled
elsewhere via the regular penalty path.

Penalty per feature is normalised by ptarget (matching Marxan):

    penalty_j = SPF_j · max(0, (ptarget_j − P_j) / ptarget_j)

Features with ``ptarget_j ≤ 0`` are disabled (Marxan's sentinel) and
contribute 0 regardless of Z.

References
----------
- Game, E. T. et al. (2008). *Ecological Applications* 18:670-680.
- Tulloch, V. J. et al. (2013). *Biological Conservation* 162:41-51.
- Carvalho, S. B. et al. (2011). *Biological Conservation* 144:2020-2030.
"""
from __future__ import annotations

import math

from scipy.stats import norm

# Marxan v4 short-circuit value when variance == 0: rZ = 4 gives
# probZUT(4) ≈ 3.17e-5, i.e. effectively "target is trivially met".
_MARXAN_ZERO_VARIANCE_SENTINEL = 4.0


def compute_zscore_per_feature(
    achieved_mean: dict[int, float],
    achieved_variance: dict[int, float],
    targets: dict[int, float],
) -> dict[int, float]:
    """Z = (T − E[T]) / sqrt(Var[T]) — Marxan sign convention.

    Positive Z means the reserve is on the shortfall side of the target.
    Variance of zero short-circuits to Marxan's sentinel (4.0).

    Parameters
    ----------
    achieved_mean
        Per-feature expected reserve amount E[T_j].
    achieved_variance
        Per-feature reserve variance Var[T_j]. Default 0 when key missing.
    targets
        Per-feature deterministic target T_j. Default 0 when key missing.

    Returns
    -------
    dict[int, float]
        Z-score per feature id.
    """
    z: dict[int, float] = {}
    for fid, mean in achieved_mean.items():
        var = achieved_variance.get(fid, 0.0)
        target = targets.get(fid, 0.0)
        if var <= 0:
            z[fid] = _MARXAN_ZERO_VARIANCE_SENTINEL
        else:
            z[fid] = (target - mean) / math.sqrt(var)
    return z


def compute_zscore_penalty(
    zscore_per_feature: dict[int, float],
    prob_targets: dict[int, float],
    spf: dict[int, float],
    weight: float = 1.0,
) -> float:
    """Marxan-faithful normalised probability shortfall penalty.

    For each feature *j*::

        P_j = norm.sf(Z_j)
        penalty_j = SPF_j · max(0, (ptarget_j − P_j) / ptarget_j)

    Total: ``weight · Σ penalty_j``. Features with ``ptarget_j ≤ 0`` are
    treated as disabled (Marxan's ``-1`` sentinel) and contribute 0.

    Parameters
    ----------
    zscore_per_feature
        Z scores from :func:`compute_zscore_per_feature`.
    prob_targets
        Per-feature probability target. ``-1`` (or any ≤ 0) means disabled.
    spf
        Per-feature species penalty factor.
    weight
        Global ``PROBABILITYWEIGHTING`` scaling factor.

    Returns
    -------
    float
        The total weighted probability penalty.
    """
    total = 0.0
    for fid, z in zscore_per_feature.items():
        ptarget = prob_targets.get(fid, -1.0)
        if ptarget <= 0:
            continue
        # Marxan zero-variance sentinel means "no uncertainty"; the
        # chance constraint is satisfied vacuously (P = 1) and the
        # deterministic shortfall penalty handles any deterministic miss.
        if z == _MARXAN_ZERO_VARIANCE_SENTINEL:
            continue
        prob = float(norm.sf(z))  # upper tail; matches Marxan's probZUT
        spf_j = spf.get(fid, 1.0)
        shortfall = max(0.0, (ptarget - prob) / ptarget)
        total += spf_j * shortfall
    return float(weight * total)
