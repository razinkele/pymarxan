"""Marxan-faithful Z-score chance-constraint math (Marxan's Probability 1D and 2D).

PROBMODE 3 is Probability 2D, 4 Probability 1D, 5 both (their penalties add, as in
Marxan 3's ``computeReserveValue``).

This is the pure-functional layer: no problem objects, no solver state.
Solver integration lives in ``chance_state.py`` (``ChanceState``: SA / iterative
improvement, incremental per-feature E / V), ``utils.py`` (the evaluator, via
:func:`evaluate_solution_chance`) and ``mip_solver.py`` (post-hoc report).

The reference is Marxan 3 (``marxan3`` branch, ``probability.c``
``ComputeProbability2D`` / ``probZUT``) — not Marxan v4, whose ``probZUT`` has an
integer-division bug that inverts the penalty. Per feature ``j``, over the selected PUs:

    E_j = Σ amount_ij · q_ij,   V_j = Σ amount_ij² · q_ij · (1 − q_ij)
    Z_j = (T_j − E_j) / sqrt(V_j)            # positive = shortfall
    P_j = probZUT(Z_j) = 1 − Φ(Z_j)          # scipy.special.ndtr(−Z) = norm.sf(Z)

``q`` is the per-cell probability of **presence** (puvspr ``prob``; Marxan's PROB2D),
the opposite polarity to PROBMODE 2 / Marxan 1D, where the per-PU ``prob`` is a loss
probability and the held amount is ``amount · (1 − prob)``. Probability 1D (Marxan 3
``ReturnProbabilityAmounts1D`` / ``ComputeProbability1D``) uses the per-PU loss
probability ``p_i`` (pu.dat ``prob``, else prob.dat) and the target ``ptarget1d``::

    E_j = Σ amount_ij · (1 − p_i),   V_j = Σ amount_ij² · p_i · (1 − p_i)

built directly from ``p`` (never ``q = 1 − p`` first: for ``p < 1.1e-16`` ``1 − p``
rounds to 1 and the variance would vanish). ``T_j`` is the resolved
deterministic target (no MISSLEVEL). Marxan's ``probZUT`` is the Ibbetson polynomial
(exact 0 / 1 for |z| ≥ 5, within ~1e-9 of ``norm.sf`` below); pymarxan keeps ``norm.sf``.

**Zero variance** is ``nonzero_j == 0`` — no selected cell whose variance term, as
computed in float, is > 0 — or ``V_j`` not > 0, Marxan 3's own test; never a tolerance
on ``V``. For ``q`` in [0, 1] the two agree (a sum of terms ≥ 0 is > 0 exactly when one
term is, and ``ChanceState`` recomputes a running ``V ≤ 0``); the ``V`` test decides
when a ``q`` outside [0, 1] adds an uncounted negative term. The rule is
``zero_var_rule`` (``PROBZEROVAR`` in input.dat): ``"marxan"`` (``PROBZEROVAR 0``, the
default) gives Marxan 3's ``Z = 4``, so ``P = probZUT(4) ≈ 3.17e-5`` fails any
``ptarget`` above that — nothing selected, all ``q`` in {0, 1}; ``"met"``
(``PROBZEROVAR 1``) counts such a feature as met (pymarxan before v0.47); ``"exact"``
(``PROBZEROVAR 2``, v0.48) scores the certain amount ``E_j`` exactly: met outright
(``Z = −inf``, ``P = 1``) when ``E_j ≥ T_j − TARGET_REL_TOL·|T_j|``
(:func:`_exact_threshold`, the deterministic targets' 1e-7 relative tolerance; a target of
0 gets the noise-scale tolerance ``n_j·eps·S_j``, see :func:`_zero_target_tol`), missed
outright (``Z = +inf``, ``P = 0``: shortfall ``ptarget``, penalty 1) otherwise.

Penalty per feature is the Marxan step function normalised by ptarget:

    penalty_j = [ptarget_j > P_j] · (ptarget_j − P_j) / ptarget_j

Marxan's ``ComputeProbability2D`` scales the summed penalties only by
``PROBABILITYWEIGHTING`` — **not** by SPF (SPF weights the separate representation
penalty). Features with ``ptarget_j ≤ 0`` are disabled (Marxan's −1) and contribute 0.

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

from pymarxan.solvers._tolerance import TARGET_REL_TOL

# Marxan's Z for a zero-variance feature (``probability.c``: ``rZ = 4``);
# probZUT(4) = norm.sf(4) ≈ 3.17e-5.
_MARXAN_ZERO_VARIANCE_SENTINEL = 4.0

ZeroVarRule = Literal["marxan", "met", "exact"]
_ZERO_VAR_RULES = ("marxan", "met", "exact")
# The rule used when a caller names none: Marxan 3's (PROBZEROVAR 0).
DEFAULT_ZERO_VAR_RULE: ZeroVarRule = "marxan"

_TERMS = ("1d", "2d")
# The in-memory probability-target column of each term: the 2D target keeps pymarxan's
# Phase 18 name ``ptarget`` (spec.dat ``ptarget2d``); the 1D target is ``ptarget1d``.
PTARGET_COLUMN = {"1d": "ptarget1d", "2d": "ptarget"}


def _ptarget_array(features, column: str) -> np.ndarray:  # features: pd.DataFrame
    """(n_feat,) float probability targets of ``column`` (``ptarget`` / ``ptarget1d``).

    A missing column or a blank (NaN) value is −1, Marxan's "disabled"; a non-numeric
    value raises ``ValueError`` naming the column and the feature ids (``validate()``
    reports the same values). The one parser of the targets for the evaluator and
    ``ProblemCache``, so both read the same numbers.
    """
    from pymarxan.models.problem import _ptarget_parse

    if column not in features.columns:
        return np.full(len(features), -1.0, dtype=np.float64)
    vals, nonnumeric = _ptarget_parse(features[column])
    if nonnumeric.any():
        bad = sorted(int(f) for f in features["id"].to_numpy()[nonnumeric])
        raise ValueError(f"features {column} has non-numeric values for feature IDs {bad}")
    return cast(np.ndarray, np.asarray(vals, dtype=np.float64))


@dataclass(frozen=True)
class FeatureChance:
    """One feature's chance-term score (spec §3.4).

    ``z`` and ``p`` (= ``norm.sf(z)``) are the Marxan Z-score and probability of meeting
    the target; ``shortfall`` = ``max(0, ptarget − p)`` (absolute, for reports);
    ``penalty`` = ``[ptarget > p]·(ptarget − p)/ptarget``, unweighted. Both are 0 for a
    disabled feature (``ptarget ≤ 0``). Under the ``"met"`` rule a zero-variance feature
    reports ``p = 1`` and no shortfall or penalty. Under the ``"exact"`` rule it reports
    ``z = −inf``, ``p = 1`` when its certain amount meets the target (within
    :func:`_exact_threshold`'s tolerance), else ``z = +inf``, ``p = 0`` (shortfall
    ``ptarget``, penalty 1).
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


_EPS = float(np.finfo(np.float64).eps)


def _zero_target_tol(abs_sum: np.ndarray, n_terms: np.ndarray) -> np.ndarray:
    """The exact rule's tolerance for a target of 0: ``n_j·eps·S_j`` per feature.

    ``S_j`` is the sum of ``|e|`` (``|a·q|`` for 2D, ``|a·(1 − p)|`` for 1D) over the
    feature's puvspr rows and ``n_j`` their count (``ChanceCells.f_abs_e`` / ``f_terms``;
    the evaluator builds the same two arrays from the same rows, :func:`_term_rows`). Any
    float sum of the feature's terms is within ``(n_j − 1)·(eps/2)·S_j`` of the exact sum,
    so an exactly-zero certain amount is met in every summation order, while a genuinely
    negative one (beyond float noise) is short. The one implementation, so every path
    gets the same threshold bit for bit.
    """
    return cast(np.ndarray, np.asarray(n_terms, dtype=np.float64) * _EPS
                * np.asarray(abs_sum, dtype=np.float64))


def _exact_threshold(
    target: np.ndarray, zero_tol: np.ndarray | None = None,
) -> np.ndarray:
    """The exact rule's met threshold: a certain amount meets ``target`` iff it is ``≥``
    this. ``target − TARGET_REL_TOL·|target|`` (``_tolerance.target_tol``, vectorised) for
    a nonzero target; ``−zero_tol`` for a target of 0 (:func:`_zero_target_tol`, a
    noise-scale tolerance; 0 when ``zero_tol`` is not given).

    A prop = 1 target and ``E`` are float sums of the same amounts in different orders, so
    without the tolerance a fully held certain feature can be scored short by an ulp.
    """
    target = np.asarray(target, dtype=np.float64)
    thr = target - TARGET_REL_TOL * np.abs(target)
    if zero_tol is not None:
        thr = np.where(target == 0.0, -np.asarray(zero_tol, dtype=np.float64), thr)
    return cast(np.ndarray, thr)


def _chance_z(
    E: np.ndarray, V: np.ndarray, zero: np.ndarray, target: np.ndarray,
    exact: bool = False, threshold: np.ndarray | None = None,
) -> np.ndarray:
    """Z = (target − E)/√V; where ``zero`` (zero variance) Marxan's 4, or under the exact
    rule ``−inf`` (met: ``E ≥`` :func:`_exact_threshold` of the target) / ``+inf`` (short).
    ``threshold``: that threshold, if the caller has it (``ChanceState`` keeps it per
    feature).

    Both callers, :func:`score_features` and :func:`_chance_penalty`, fold Marxan's own
    test into ``zero`` (``| ~(V > 0)``, also True for a NaN ``V``), so the square root
    only sees ``V > 0`` where ``zero`` is False. No ``np.errstate`` here (it costs more
    than the arithmetic on the hot path).
    """
    z = (target - E) / np.sqrt(np.where(zero, 1.0, V))
    if exact:
        thr = _exact_threshold(target) if threshold is None else threshold
        return cast(np.ndarray, np.where(zero, np.where(E >= thr, -np.inf, np.inf), z))
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
    The exact rule needs no branch here: its ``z = ∓inf`` gives ``p`` = 1 / 0 exactly.
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
    exact: bool = False, threshold: np.ndarray | None = None,
) -> np.ndarray:
    """Unweighted penalties only (``ChanceState``'s hot path): ``zero`` widened by
    Marxan's ``V > 0`` test, then the same operations, in the same order, as the
    ``penalty`` of :func:`score_features` (:func:`_chance_z` + :func:`_chance_from_z`)."""
    zero = zero | ~(V > 0)
    p = ndtr(-_chance_z(E, V, zero, target, exact, threshold))
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
    zero_tol: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised :func:`score_feature`: arrays ``(z, p, shortfall, penalty)``.

    The evaluator and ``ChanceState`` use this arithmetic; it is the same as
    :func:`score_feature`, element for element. Zero variance is ``nonzero == 0`` or
    ``V`` not > 0 (Marxan 3's test; a NaN ``V`` too). ``zero_tol``: the exact rule's
    tolerance for a target of 0 (:func:`_zero_target_tol`); None = none.
    """
    _check_rule(zero_var_rule)
    E = np.asarray(E, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    ptarget = np.asarray(ptarget, dtype=np.float64)
    zero = (np.asarray(nonzero) == 0) | ~(V > 0)
    exact = zero_var_rule == "exact"
    threshold = _exact_threshold(target, zero_tol) if exact else None
    with np.errstate(divide="ignore", invalid="ignore"):
        z = _chance_z(E, V, zero, target, exact, threshold)
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
    zero_tol: float = 0.0,
) -> FeatureChance:
    """Score one feature's chance term (spec §3.4).

    Parameters
    ----------
    E, V
        Expected amount and variance of the feature over the selected cells.
    nonzero
        Number of selected cells whose variance term, as computed in float, is > 0.
        **Zero variance is ``nonzero == 0``, or ``V`` not > 0** (Marxan 3's test, which
        decides when a ``q`` outside [0, 1] makes ``V ≤ 0`` beside counted cells); no
        tolerance on ``V``.
    target
        Deterministic target (the resolved target, without MISSLEVEL).
    ptarget
        Probability target; ``≤ 0`` disables the feature (Marxan's −1).
    zero_var_rule
        ``"marxan"``: zero variance gives Marxan 3's ``Z = 4``, then the normal path
        (so ``P ≈ 3.17e-5`` fails any ``ptarget`` above it). ``"met"``: zero variance
        counts as met (penalty and shortfall 0). ``"exact"``: the amount is certain, so
        ``P = 1`` (``z = −inf``) if ``E ≥ target − 1e-7·|target|``
        (:func:`_exact_threshold`), else ``P = 0`` (``z = +inf``; shortfall ``ptarget``,
        penalty 1).
    zero_tol
        The exact rule's tolerance when ``target == 0``: met iff ``E ≥ −zero_tol``
        (:func:`_zero_target_tol`); default 0.
    """
    z, p, shortfall, penalty = score_features(
        np.array([E]), np.array([V]), np.array([nonzero]), np.array([target]),
        np.array([ptarget]), zero_var_rule, np.array([zero_tol]),
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
    achieved_mean: dict[int, float] | None = None,
    targets: dict[int, float] | None = None,
    target_zero_tol: dict[int, float] | None = None,
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
        ``"marxan"``, ``"met"`` or ``"exact"`` (see :func:`score_feature`).
    achieved_mean, targets
        Per-feature ``E`` and deterministic target, needed by the ``"exact"`` rule only: a
        zero-variance feature's Z then comes from its certain amount, which
        ``zscore_per_feature`` does not carry. ``ValueError`` if the rule is ``"exact"``
        and either is missing, or lacks a zero-variance id.
    target_zero_tol
        Per-feature tolerance for a target of 0 under the ``"exact"`` rule
        (:func:`_zero_target_tol`; missing keys: 0).

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
    if zero_var_rule == "exact":
        if achieved_mean is None or targets is None:
            raise ValueError(
                'zero_var_rule="exact" needs achieved_mean and targets: the certain '
                "amount of a zero-variance feature decides it, and Z does not carry it"
            )
        missing = sorted(
            f for f in fids if f in zero_variance and (f not in achieved_mean or f not in targets)
        )
        if missing:
            raise ValueError(
                'zero_var_rule="exact" needs achieved_mean and targets for every '
                f"zero-variance feature; missing for feature ids {missing}"
            )
        E = np.array([achieved_mean.get(f, 0.0) for f in fids], dtype=np.float64)
        T = np.array([targets.get(f, 0.0) for f in fids], dtype=np.float64)
        tol = target_zero_tol or {}
        zt = np.array([tol.get(f, 0.0) for f in fids], dtype=np.float64)
        z = np.where(zero, np.where(E >= _exact_threshold(T, zt), -np.inf, np.inf), z)
    _, _, penalty = _chance_from_z(z, zero, pt, _ptarget_den(pt), zero_var_rule == "met")
    return float(weight * float(np.sum(penalty)))


def _term_rows(
    problem,  # ConservationProblem
    term: str,
    pu_id_to_idx: dict[int, int],
    feat_id_to_col: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(rows, cols, e, v)`` of one chance term's puvspr rows, in file order.

    Rows naming an unknown PU or feature are dropped; ``rows`` / ``cols`` are planning-
    unit / feature positions; ``e`` / ``v`` each row's terms with the operand order of
    :func:`_term_moments` (bit for bit: 2D ``a·q`` and ``a·a·q·(1 − q)``; 1D ``a·(1 − p)``
    and ``a·a·p·(1 − p)``). Shared by ``ProblemCache``'s chance cells and the evaluator's
    zero-target tolerance, so both see the same ``S_j`` / ``n_j``. A non-finite cell value
    raises ``ValueError`` naming the PU ids.
    """
    from pymarxan.solvers.utils import _check_finite_cells, _loss_probability_vector

    pv = problem.pu_vs_features
    _check_finite_cells(pv, ("prob", "amount") if term == "2d" else ("amount",))
    n_pu = len(problem.planning_units)
    rows = np.fromiter((pu_id_to_idx.get(int(u), -1) for u in pv["pu"].values),
                       np.int64, len(pv))
    cols = np.fromiter((feat_id_to_col.get(int(f), -1) for f in pv["species"].values),
                       np.int64, len(pv))
    ok = (rows >= 0) & (cols >= 0)
    rows, cols = rows[ok], cols[ok]
    a = np.asarray(pv["amount"].values, dtype=np.float64)[ok]
    if term == "2d":
        q = np.asarray(pv["prob"].values, dtype=np.float64)[ok]
        return rows, cols, a * q, a * a * q * (1.0 - q)
    p = _loss_probability_vector(problem, pu_id_to_idx, n_pu)[rows]
    return rows, cols, a * (1.0 - p), a * a * p * (1.0 - p)


def _term_zero_tol(problem, term: str, feat_ids: np.ndarray) -> np.ndarray:
    """Per-feature zero-target tolerance of one term (:func:`_zero_target_tol` of the
    feature's puvspr rows, as ``ChanceCells.f_abs_e`` / ``f_terms`` hold it)."""
    pu_ids = problem.planning_units["id"].values
    pu_index = {int(u): i for i, u in enumerate(pu_ids)}
    col = {int(f): j for j, f in enumerate(feat_ids)}
    _, cols, e, _ = _term_rows(problem, term, pu_index, col)
    n = len(feat_ids)
    return _zero_target_tol(
        np.bincount(cols, weights=np.abs(e), minlength=n), np.bincount(cols, minlength=n),
    )


def _term_moments(
    problem,  # ConservationProblem
    selected,  # np.ndarray
    term: str,
    feat_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-feature ``(E, V, nonzero)`` of one chance term over the selected PUs.

    Row by row, as Marxan 3's ``ReturnProbabilityAmounts1D`` / ``2D`` (a duplicated
    ``(pu, species)`` row adds its own terms). A non-finite cell value raises
    ``ValueError`` naming the PU ids, as building the cache's chance cells does: the 2D
    term's puvspr ``amount`` / ``prob`` in any row, selected or not; the 1D term's
    ``amount`` and its loss probabilities (``_loss_probability_vector``).
    """
    from pymarxan.solvers.utils import _check_finite_cells, _loss_probability_vector

    if term not in _TERMS:
        raise ValueError(f"unknown chance term {term!r}; known terms are {_TERMS}")
    puvspr = problem.pu_vs_features
    _check_finite_cells(puvspr, ("prob", "amount") if term == "2d" else ("amount",))
    col = {int(f): j for j, f in enumerate(feat_ids)}
    n = len(feat_ids)
    E = np.zeros(n)
    V = np.zeros(n)
    N = np.zeros(n, dtype=np.int64)
    pu_ids = problem.planning_units["id"].values
    pu_index = {int(pu_ids[i]): i for i in range(len(pu_ids))}
    loss = (
        _loss_probability_vector(problem, pu_index, len(pu_ids)) if term == "1d" else None
    )
    selected_pu_ids = {int(pu_ids[i]) for i in range(len(pu_ids)) if selected[i]}
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
        if loss is not None:
            # Marxan 3 ReturnProbabilityAmounts1D: p = the PU's loss probability;
            # E += amount·(1 − p), V += amount·amount·p·(1 − p).
            p = float(loss[pu_index[int(pv_pu[k])]])
            e = amt * (1.0 - p)
            v = amt * amt * p * (1.0 - p)
        else:
            # PROB2D presence convention: q = P(present); default 1.0 (certain presence)
            # when there is no `prob` column (only on a direct call: evaluate_solution_chance
            # scores a term without its data as off first). Marxan 3 ComputeProbability2D:
            # E += amount·q, V += amount²·q·(1 − q).
            q = float(pv_pr[k]) if pv_pr is not None else 1.0
            e = amt * q
            v = amt * amt * q * (1.0 - q)
        E[j] += e
        V[j] += v
        N[j] += int(v > 0)
    return E, V, N


def evaluate_solution_chance(
    problem,  # ConservationProblem — typed loosely to avoid circular import
    selected,  # np.ndarray
    term: str = "2d",
) -> tuple[dict[int, float], float]:
    """Post-hoc evaluation of one chance term (``"1d"`` or ``"2d"``) for a solution.

    Used by ``build_solution`` (every solver's reported objective and the MIP "drop"
    strategy's post-hoc report): computes the per-feature E / V / nonzero over the
    selected PUs and scores them with :func:`score_features` against the term's
    probability target (``ptarget1d`` for 1D, ``ptarget`` for 2D).

    The gate is the term's data, not ``PROBMODE``. A term without its data (``"2d"``: no
    ``prob`` column on ``pu_vs_features``; ``"1d"``: neither a pu.dat ``prob`` column nor
    ``problem.probability``) is off, as in every solver (spec §3.1): each feature's
    shortfall is 0 and the penalty is 0 under every ``PROBZEROVAR`` rule, and no cell is
    read (so a non-finite ``amount`` does not raise here). With its data the term is
    scored whatever the mode, so a caller can score a problem as if the term were
    requested. A missing 1D target column counts as all −1 (all shortfalls 0); a missing
    2D one returns ``({}, 0.0)`` as before.

    Returns
    -------
    (shortfalls, penalty)
        ``shortfalls`` is a feature_id → ``max(0, ptarget − P)`` dict
        (per-feature absolute gap in probability space, NOT normalised
        by ptarget — easier to interpret in the UI).
        ``penalty`` is the weighted normalised total that the term adds to the
        objective.
    """
    from pymarxan.solvers.utils import _has_1d_data, _zero_var_rule

    if term not in _TERMS:
        raise ValueError(f"unknown chance term {term!r}; known terms are {_TERMS}")
    pt_col = PTARGET_COLUMN[term]
    feat_ids = problem.features["id"].astype(int).values
    if pt_col not in problem.features.columns:
        if term == "2d":
            return {}, 0.0
        return {fid: 0.0 for fid in feat_ids.tolist()}, 0.0
    feat_target = problem.features["target"].astype(float).values
    feat_ptarget = _ptarget_array(problem.features, pt_col)

    # Off without its data (spec §3.1), as ``utils._active_chance_terms`` makes it in the
    # solvers, or with no ptarget > 0.
    if term == "2d":
        no_data = "prob" not in problem.pu_vs_features.columns
    else:
        no_data = not _has_1d_data(problem)
    if no_data or not any(pt > 0 for pt in feat_ptarget):
        return {fid: 0.0 for fid in feat_ids.tolist()}, 0.0

    E, V, N = _term_moments(problem, selected, term, feat_ids)
    rule = _zero_var_rule(problem)
    zero_tol = _term_zero_tol(problem, term, feat_ids) if rule == "exact" else None
    _, _, shortfall, penalty = score_features(
        E, V, N, feat_target, feat_ptarget, rule, zero_tol,
    )
    weight = float(problem.parameters.get("PROBABILITYWEIGHTING", 1.0))
    shortfalls = {int(f): float(sf) for f, sf in zip(feat_ids, shortfall)}
    return shortfalls, float(weight * float(np.sum(penalty)))
