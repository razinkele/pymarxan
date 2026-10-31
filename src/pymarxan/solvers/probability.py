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

    penalty_j = max(0, (ptarget_j − P_j) / ptarget_j)

Marxan's ``ComputeProbability2D`` scales the probability shortfall only by
``PROBABILITYWEIGHTING`` — **not** by SPF (the SPF weighting applies to the
*representation* penalty, a separate objective term; see
``score_change.cpp:611``). Features with ``ptarget_j ≤ 0`` are disabled
(Marxan's sentinel) and contribute 0 regardless of Z.

Expected amount uses the PROB2D *presence* convention
(``ComputeP_AllPUsSelected_2D``: ``ExpectedAmount2D += amount * prob``), i.e.
``E[T_j] = Σ amount_ij · p_ij`` — the opposite polarity to PROBMODE 2 / Marxan
1D, where the per-PU ``prob`` is a loss probability and held amount is
``amount · (1 − prob)``.

References
----------
- Game, E. T. et al. (2008). *Ecological Applications* 18:670-680.
- Tulloch, V. J. et al. (2013). *Biological Conservation* 162:41-51.
- Carvalho, S. B. et al. (2011). *Biological Conservation* 144:2020-2030.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from scipy.special import ndtr

# Marxan's Z for a zero-variance feature (``probability.c``: ``rZ = 4``);
# probZUT(4) = norm.sf(4) ≈ 3.17e-5.
_MARXAN_ZERO_VARIANCE_SENTINEL = 4.0

ZeroVarRule = Literal["marxan", "met"]
_ZERO_VAR_RULES = ("marxan", "met")
# The rule used when a caller names none. "met" keeps the pre-v0.47 behaviour.
DEFAULT_ZERO_VAR_RULE: ZeroVarRule = "met"

_TERMS = ("2d",)


@dataclass(frozen=True)
class FeatureChance:
    """One feature's chance-term score (spec §3.4).

    ``z`` and ``p`` (= ``norm.sf(z)``) are the Marxan Z-score and probability of meeting
    the target; ``shortfall`` = ``max(0, ptarget − p)`` (absolute, for reports);
    ``penalty`` = ``[ptarget > p]·(ptarget − p)/ptarget``, unweighted. Both are 0 for a
    disabled feature (``ptarget ≤ 0``). Under the ``"met"`` rule a zero-variance feature
    reports ``p = 1`` and no shortfall or penalty.
    """

    z: float
    p: float
    shortfall: float
    penalty: float


def _check_rule(zero_var_rule: str) -> None:
    if zero_var_rule not in _ZERO_VAR_RULES:
        raise ValueError(
            f"zero_var_rule must be one of {_ZERO_VAR_RULES}, got {zero_var_rule!r}"
        )


def _chance_z(
    E: np.ndarray, V: np.ndarray, zero: np.ndarray, target: np.ndarray,
) -> np.ndarray:
    """Z = (target − E)/√V; Marxan's 4 where ``zero`` (no selected cell with variance).

    No ``np.errstate`` here (it costs more than the arithmetic on the hot path): the
    caller guarantees ``V > 0`` wherever ``zero`` is False, or owns the float state.
    """
    z = (target - E) / np.sqrt(np.where(zero, 1.0, V))
    return np.where(zero, _MARXAN_ZERO_VARIANCE_SENTINEL, z)


def _ptarget_den(ptarget: np.ndarray) -> np.ndarray:
    """Penalty denominator: ``ptarget`` where enabled (> 0), else 1 (its gap is 0)."""
    return np.where(ptarget > 0, ptarget, 1.0)


def _chance_from_z(
    z: np.ndarray, zero: np.ndarray, ptarget: np.ndarray, ptarget_den: np.ndarray,
    met: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(p, shortfall, penalty)`` from Z: the one copy of the probability arithmetic.

    ``p ≥ 0``, so ``ptarget > p`` is False for a disabled feature (``ptarget ≤ 0``).
    """
    p = ndtr(-z)  # = norm.sf(z), Marxan's probZUT up to ~1e-9
    gap = np.where(ptarget > p, ptarget - p, 0.0)
    penalty = gap / ptarget_den
    if met:
        p = np.where(zero, 1.0, p)
        gap = np.where(zero, 0.0, gap)
        penalty = np.where(zero, 0.0, penalty)
    return p, gap, penalty


def _chance_penalty(
    E: np.ndarray, V: np.ndarray, zero: np.ndarray, target: np.ndarray,
    ptarget: np.ndarray, ptarget_den: np.ndarray, met: bool,
) -> np.ndarray:
    """Unweighted penalties only (``ChanceState``'s hot path): the same operations, in
    the same order, as the ``penalty`` of :func:`_chance_z` + :func:`_chance_from_z`."""
    p = ndtr(-_chance_z(E, V, zero, target))
    penalty = np.where(ptarget > p, ptarget - p, 0.0) / ptarget_den
    if met:
        penalty = np.where(zero, 0.0, penalty)
    return cast(np.ndarray, penalty)


def score_features(
    E: np.ndarray,
    V: np.ndarray,
    nonzero: np.ndarray,
    target: np.ndarray,
    ptarget: np.ndarray,
    zero_var_rule: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised :func:`score_feature`: arrays ``(z, p, shortfall, penalty)``.

    The evaluator and ``ChanceState`` use this arithmetic; it is the same as
    :func:`score_feature`, element for element.
    """
    _check_rule(zero_var_rule)
    E = np.asarray(E, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    ptarget = np.asarray(ptarget, dtype=np.float64)
    zero = np.asarray(nonzero) == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        z = _chance_z(E, V, zero, target)
    p, shortfall, penalty = _chance_from_z(
        z, zero, ptarget, _ptarget_den(ptarget), zero_var_rule == "met",
    )
    return z, p, shortfall, penalty


def score_feature(
    E: float,
    V: float,
    nonzero: int,
    target: float,
    ptarget: float,
    zero_var_rule: str,
) -> FeatureChance:
    """Score one feature's chance term (spec §3.4).

    Parameters
    ----------
    E, V
        Expected amount and variance of the feature over the selected cells.
    nonzero
        Number of selected cells whose variance term, as computed in float, is > 0.
        **Zero variance is ``nonzero == 0``** (exact; no float threshold on ``V``).
    target
        Deterministic target (the resolved target, without MISSLEVEL).
    ptarget
        Probability target; ``≤ 0`` disables the feature (Marxan's −1).
    zero_var_rule
        ``"marxan"``: zero variance gives Marxan 3's ``Z = 4``, then the normal path
        (so ``P ≈ 3.17e-5`` fails any ``ptarget`` above it). ``"met"``: zero variance
        counts as met (penalty and shortfall 0).
    """
    z, p, shortfall, penalty = score_features(
        np.array([E]), np.array([V]), np.array([nonzero]), np.array([target]),
        np.array([ptarget]), zero_var_rule,
    )
    return FeatureChance(
        z=float(z[0]), p=float(p[0]), shortfall=float(shortfall[0]),
        penalty=float(penalty[0]),
    )


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
    weight: float = 1.0,
    *,
    zero_variance: frozenset[int] = frozenset(),
    zero_var_rule: str = DEFAULT_ZERO_VAR_RULE,
) -> float:
    """Marxan-faithful normalised probability shortfall penalty.

    For each feature *j*::

        P_j = norm.sf(Z_j)
        penalty_j = [ptarget_j > P_j] · (ptarget_j − P_j) / ptarget_j

    Total: ``weight · Σ penalty_j``. Following Marxan's ``ComputeProbability2D``
    the probability shortfall is scaled **only** by ``PROBABILITYWEIGHTING``,
    not by SPF (SPF weights the separate representation penalty). Features with
    ``ptarget_j ≤ 0`` are treated as disabled (Marxan's ``-1`` sentinel) and
    contribute 0.

    Parameters
    ----------
    zscore_per_feature
        Z scores from :func:`compute_zscore_per_feature`.
    prob_targets
        Per-feature probability target. ``-1`` (or any ≤ 0) means disabled.
    weight
        Global ``PROBABILITYWEIGHTING`` scaling factor.
    zero_variance
        Ids of the zero-variance features, so a real ``Z = 4`` is told apart from
        Marxan's zero-variance 4; ``zero_var_rule`` applies to these only.
    zero_var_rule
        ``"marxan"`` or ``"met"`` (see :func:`score_feature`).

    Returns
    -------
    float
        The total weighted probability penalty.
    """
    _check_rule(zero_var_rule)
    fids = list(zscore_per_feature)
    if not fids:
        return 0.0
    z = np.array([zscore_per_feature[f] for f in fids], dtype=np.float64)
    pt = np.array([prob_targets.get(f, -1.0) for f in fids], dtype=np.float64)
    zero = np.array([f in zero_variance for f in fids], dtype=bool)
    _, _, penalty = _chance_from_z(z, zero, pt, _ptarget_den(pt), zero_var_rule == "met")
    return float(weight * float(np.sum(penalty)))


def _term_moments(
    problem,  # ConservationProblem
    selected,  # np.ndarray
    term: str,
    feat_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-feature ``(E, V, nonzero)`` of one chance term over the selected PUs."""
    if term not in _TERMS:
        raise ValueError(f"unknown chance term {term!r}; release A knows {_TERMS}")
    col = {int(f): j for j, f in enumerate(feat_ids)}
    n = len(feat_ids)
    E = np.zeros(n)
    V = np.zeros(n)
    N = np.zeros(n, dtype=np.int64)
    pu_ids = problem.planning_units["id"].values
    selected_pu_ids = {int(pu_ids[i]) for i in range(len(pu_ids)) if selected[i]}
    puvspr = problem.pu_vs_features
    pv_pu = puvspr["pu"].values
    pv_sp = puvspr["species"].values
    pv_am = puvspr["amount"].values
    pv_pr = puvspr["prob"].values if "prob" in puvspr.columns else None
    for k in range(len(pv_pu)):
        if int(pv_pu[k]) not in selected_pu_ids:
            continue
        j = col.get(int(pv_sp[k]))
        if j is None:
            continue
        amt = float(pv_am[k])
        # PROB2D presence convention: q = P(present); default 1.0 (certain presence)
        # when there is no `prob` column. Marxan 3 ComputeProbability2D:
        # E += amount·q, V += amount²·q·(1 − q).
        q = float(pv_pr[k]) if pv_pr is not None else 1.0
        v = amt * amt * q * (1.0 - q)
        E[j] += amt * q
        V[j] += v
        N[j] += int(v > 0)
    return E, V, N


def evaluate_solution_chance(
    problem,  # ConservationProblem — typed loosely to avoid circular import
    selected,  # np.ndarray
    term: str = "2d",
) -> tuple[dict[int, float], float]:
    """Post-hoc evaluation of one chance term for a fully-formed solution.

    Used by ``build_solution`` (every solver's reported objective and the MIP "drop"
    strategy's post-hoc report): computes the per-feature E / V / nonzero over the
    selected PUs and scores them with :func:`score_features`.

    Returns
    -------
    (shortfalls, penalty)
        ``shortfalls`` is a feature_id → ``max(0, ptarget − P)`` dict
        (per-feature absolute gap in probability space, NOT normalised
        by ptarget — easier to interpret in the UI).
        ``penalty`` is the weighted normalised total that the term adds to the
        objective.
    """
    from pymarxan.solvers.utils import _zero_var_rule

    if term not in _TERMS:
        raise ValueError(f"unknown chance term {term!r}; release A knows {_TERMS}")
    if "ptarget" not in problem.features.columns:
        return {}, 0.0
    feat_ids = problem.features["id"].astype(int).values
    feat_target = problem.features["target"].astype(float).values
    feat_ptarget = problem.features["ptarget"].astype(float).values

    if not any(pt > 0 for pt in feat_ptarget):
        return {fid: 0.0 for fid in feat_ids.tolist()}, 0.0

    E, V, N = _term_moments(problem, selected, term, feat_ids)
    _, _, shortfall, penalty = score_features(
        E, V, N, feat_target, feat_ptarget, _zero_var_rule(problem),
    )
    weight = float(problem.parameters.get("PROBABILITYWEIGHTING", 1.0))
    shortfalls = {int(f): float(sf) for f, sf in zip(feat_ids, shortfall)}
    return shortfalls, float(weight * float(np.sum(penalty)))
