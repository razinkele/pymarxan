"""``ChanceState`` — incremental chance-term sidecar for SA / iterative improvement.

The ``ClumpState`` / ``SepState`` pattern (spec §3.5): ``ProblemCache`` stays stateless
with respect to the selection and holds the immutable per-cell contributions of each
active chance term (``cache.chance_cells``); this sidecar keeps, per term and per feature,
the running expected amount ``E``, variance ``V`` and ``nonzero`` (selected cells whose
variance term is > 0), the float error bounds of ``E`` and ``V`` since the feature's last
recompute, its unweighted penalty, and its own copy of the selection.

- ``delta_penalty(cache, idx, adding)`` re-scores only the flipped PU's features and
  **never mutates** the state; it caches its result (the preview).
- ``apply_flip(cache, idx, adding)`` commits the cached preview when ``(idx, adding)``
  matches, otherwise computes it through the same code path, so the committed state is
  bit for bit the previewed one. Every ``apply_flip`` clears the preview.

Error-bounded recompute. Running sums lose precision (cancellation when a dominant
cell leaves, drift over long chains). Each update adds ``eps·(|old| + |Δ|)`` to the
feature's bound (``BE`` for ``E``, ``BV`` for ``V``). A touched feature is recomputed from
its cells (the CSC column) over the selection *after* the previewed flip when
``V ≤ 0`` while ``nonzero > 0``, or ``BV > τ·V``, or ``BE > max(τ·√V, 64·eps·|E|)``
(τ = 1e-12); the ``64·eps·|E|`` floor keeps the E trigger from firing on every touch when
the coefficient of variation is tiny (mostly-certain cells beside a few risky ones). After
a recompute both bounds reset to ``4·eps·|value|``; ``from_selection`` counts as one. The
reset is starting slack, not a bound on the recompute's own summation error, which grows
with the number of selected cells (it can exceed ``4·eps·|value|`` by a factor that also
grows with that number: about 2x at 10^3 cells, over 10x at 10^5); the per-update
increments are true bounds. When
``nonzero`` reaches 0, ``E`` and ``V`` snap to the exact sums of the remaining
zero-variance cells (``V = 0``). Zero variance itself is the exact count, never a
tolerance on ``V``; the scorer also applies Marxan's ``V > 0`` test, which after the
recompute only decides when a ``q`` outside [0, 1] leaves ``V ≤ 0`` beside counted cells.

Exact rule's knife edge (``PROBZEROVAR 2``). Under Marxan's rule (0) and "met" (1) a
zero-variance feature's penalty does not depend on ``E``. Under the exact rule it steps at
the met threshold ``thr_j`` (``probability._exact_threshold``, kept per term in
``_TermState.thr``): ``T_j − 1e-7·|T_j|``, or ``−n_j·eps·S_j`` for a target of 0
(``probability._zero_target_tol``); met iff ``E ≥ thr_j``. The decision must be the
evaluator's (``probability._term_moments``), whose ``E_ev`` is its own float sum of the
selected puvspr rows in **file order**. So whenever this state takes a zero-variance
feature's decision from a fresh sum — at ``from_selection`` and at every recompute of a
zero-variance feature — it sums the selected rows in that same order (``_row_sum`` over
``ChanceCells.r_e``), which equals ``E_ev`` bit for bit, whatever the planning-unit order,
the puvspr row order or repeated ``(pu, species)`` rows. (B0 review M-1: the column sum,
in planning-unit order, disagreed with the evaluator at ``from_selection`` with descending
PU ids.)

Between fresh sums the running ``E`` drifts. Let ``Ê`` be the exact sum of the feature's
terms over the selection, ``S_j`` the sum of ``|e|`` over the feature's whole column and
``n_j`` its number of terms (``ChanceCells.f_abs_e`` / ``f_terms``; ``S_j`` bounds
``Σ|terms|`` over every selection, negative amounts included). With
``γ = γ_{n_j−1} ≈ (n_j − 1)·u`` (u = eps/2), any float sum of the feature's terms is within
``γ·S_j`` of the exact sum:

- the evaluator: ``|E_ev − Ê| ≤ γ·S_j``;
- the running ``E``: its last fresh sum (row-order or column sum) was within ``γ·S_j`` of
  the exact sum of the terms then selected (adding and removing the same cell float
  cancels exactly, so the cells' own rounding contributes at most another ``γ·S_j`` over
  the current selection), and every update since added at most its increment to ``BE``
  (the reset ``4·eps·|E|`` is extra slack). So ``|E − Ê| ≤ BE + 2·γ·S_j``.

Hence ``|E − E_ev| ≤ BE + 3·γ·S_j < BE + 1.5·n_j·eps·S_j``. A touched zero-variance
feature is recomputed when

    |E − thr_j| < W_j = (1 + 8·eps)·BE + 3·F_j·eps·S_j,     F_j = max(64, n_j)

(strict; ``_knife_hit``): ``W_j`` is twice the evaluator-error term, and the ``8·eps`` and
that factor 2 absorb the rounding of computing ``|E − thr_j|`` and ``W_j`` themselves.
Outside the window ``E`` and ``E_ev`` are strictly on the same side of ``thr_j``; inside
it the row-order sum, i.e. ``E_ev`` itself, decides. The window is a property of the exact
rule only: the shipped ``64·eps·|E|`` redo floor and every bound above are unchanged, so
rules 0 / 1 are bit for bit as before. A feature whose cells all have ``e = 0`` has
``E = BE = S_j = 0`` and an empty window. For a nonzero target the threshold is
``1e-7·|T_j|`` below it, so an ``E`` landing exactly on its target (grid amounts) is
outside the window unless ``3·F_j·eps·S_j`` reaches ``1e-7·|T_j|``
(``n_j·S_j/|T_j| ≳ 5e7``) or ``BE`` grows by about 10^8 updates without a fresh sum.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from pymarxan.solvers.cache import ChanceCells
from pymarxan.solvers.probability import (
    PTARGET_COLUMN,
    _chance_penalty,
    _exact_threshold,
    _ptarget_den,
    _zero_target_tol,
    score_features,
)

EPS = float(np.finfo(np.float64).eps)
TAU = 1e-12
_E_FLOOR = 64.0 * EPS
_RESET = 4.0 * EPS
# The exact rule's knife-edge window (module docstring): (1 + _KNIFE_BE)·BE + knife_j,
# knife_j = 3·max(_KNIFE_MIN_TERMS, n_j)·eps·S_j.
_KNIFE_BE = 8.0 * EPS
_KNIFE_MIN_TERMS = 64


@dataclass
class _TermState:
    """Running per-feature state of one chance term.

    ``EV`` (2, n_feat): running ``E`` (row 0) and ``V`` (row 1); ``B`` (2, n_feat): their
    float error bounds since the feature's last recompute; ``N``: nonzero counts; ``pen``:
    unweighted penalties; ``ptarget`` / ``ptarget_den``: the term's probability targets
    (2D ``ptarget``, 1D ``ptarget1d``) and penalty denominators; ``knife``: under the exact
    rule, each feature's evaluator-error term of the knife-edge window,
    ``3·max(64, n_j)·eps·S_j``, and ``thr`` its met thresholds
    (``probability._exact_threshold`` with the term's zero-target tolerance) (else None).
    """

    cells: ChanceCells
    EV: np.ndarray
    B: np.ndarray
    N: np.ndarray
    pen: np.ndarray
    ptarget: np.ndarray
    ptarget_den: np.ndarray
    knife: np.ndarray | None = None
    thr: np.ndarray | None = None

    @property
    def E(self) -> np.ndarray:
        return cast(np.ndarray, self.EV[0])

    @property
    def V(self) -> np.ndarray:
        return cast(np.ndarray, self.EV[1])

    @property
    def BE(self) -> np.ndarray:
        return cast(np.ndarray, self.B[0])

    @property
    def BV(self) -> np.ndarray:
        return cast(np.ndarray, self.B[1])


def term_ptarget(cache, name: str) -> np.ndarray:
    """The probability targets of chance term ``name`` on ``cache`` (float64): the 2D
    term's ``feat_ptarget`` (spec ``ptarget``), the 1D term's ``feat_ptarget1d``."""
    attr = {"ptarget": "feat_ptarget", "ptarget1d": "feat_ptarget1d"}[PTARGET_COLUMN[name]]
    return np.asarray(getattr(cache, attr), dtype=np.float64)


# One term's preview for the flipped PU's features: (cols, EV2, B2, N2, pen2).
_Payload = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _row_sum(
    c: ChanceCells, j: int, selected: np.ndarray, idx: int = -1, adding: bool = False,
) -> float:
    """Feature ``j``'s ``E`` over ``selected``, summed as the evaluator sums it.

    ``probability._term_moments`` adds the selected puvspr rows' terms to ``E[j]`` one by
    one in file order, from 0.0; ``np.cumsum`` is sequential and the row terms are the same
    floats (``ChanceCells.r_e``), so the result equals the evaluator's ``E`` bit for bit
    (up to the sign of a zero). A planning unit named by several rows contributes each.
    ``idx`` / ``adding``: read PU ``idx`` as flipped to ``adding`` (a previewed flip).
    """
    a, b = int(c.r_indptr[j]), int(c.r_indptr[j + 1])
    rows = c.r_rows[a:b]
    m = selected[rows]  # fancy index: a copy
    if idx >= 0:
        m[rows == idx] = adding
    vals = c.r_e[a:b][m]
    return float(np.cumsum(vals)[-1]) if len(vals) else 0.0


class ChanceState:
    """Mutable chance-term companion to ``ProblemCache`` (see the module docstring)."""

    def __init__(
        self,
        selected: np.ndarray,
        terms: dict[str, _TermState],
        weight: float,
        zero_var_rule: str,
        targets: np.ndarray,
    ) -> None:
        self.selected = np.asarray(selected, dtype=bool)  # a mask, whatever the caller's dtype
        self.terms = terms
        self._term_list = list(terms.values())
        self.weight = weight
        self.zero_var_rule = zero_var_rule
        self._met = zero_var_rule == "met"
        self._exact = zero_var_rule == "exact"
        self.targets = targets
        self.n_recompute = 0
        self._preview: tuple[int, bool, float, list[_Payload]] | None = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_selection(cls, cache, selected: np.ndarray) -> ChanceState:
        """Fresh state for ``selected`` (one per SA run / II pass); a recompute of all."""
        sel = np.asarray(selected, dtype=bool).copy()
        n_feat = cache.n_feat
        terms: dict[str, _TermState] = {}
        for name in sorted(cache.chance_terms):
            cells = cache.chance_cells[name]
            mask = sel[cells.c_rows]
            col = np.repeat(np.arange(n_feat), np.diff(cells.c_indptr))[mask]
            # bincount of an empty index array is int64 even with float weights: a selection
            # without a cell of the term would make EV int, and every apply_flip would then
            # truncate the committed E and V (v0.47.0 review). B and pen follow EV; N is int.
            EV = np.vstack([
                np.bincount(col, weights=cells.c_e[mask], minlength=n_feat),
                np.bincount(col, weights=cells.c_v[mask], minlength=n_feat),
            ]).astype(np.float64, copy=False)
            N = np.bincount(col, weights=(cells.c_v[mask] > 0), minlength=n_feat)
            N = N.astype(np.int64)
            ptarget = term_ptarget(cache, name)
            knife = thr = zero_tol = None
            if cache.zero_var_rule == "exact":
                n_terms = np.maximum(_KNIFE_MIN_TERMS, cells.f_terms)
                knife = 3.0 * n_terms * EPS * cells.f_abs_e
                zero_tol = _zero_target_tol(cells.f_abs_e, cells.f_terms)
                thr = _exact_threshold(cache.feat_targets, zero_tol)
                # The decision of a zero-variance feature comes from the evaluator's own
                # float sum (its puvspr rows in file order), not the column sum.
                for j in np.flatnonzero((N == 0) | ~(EV[1] > 0)):
                    EV[0, j] = _row_sum(cells, int(j), sel)
            pen = score_features(
                EV[0], EV[1], N, cache.feat_targets, ptarget, cache.zero_var_rule, zero_tol,
            )[3]
            terms[name] = _TermState(
                cells=cells, EV=EV, B=_RESET * np.abs(EV), N=N, pen=pen,
                ptarget=ptarget, ptarget_den=_ptarget_den(ptarget), knife=knife, thr=thr,
            )
        return cls(
            selected=sel, terms=terms, weight=float(cache.prob_weight),
            zero_var_rule=cache.zero_var_rule,
            targets=np.asarray(cache.feat_targets, dtype=np.float64),
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def penalty_total(self) -> float:
        """Weighted chance penalty of the current selection (all active terms)."""
        return self.weight * float(sum(float(np.sum(t.pen)) for t in self._term_list))

    def delta_penalty(self, cache, idx: int, adding: bool) -> float:
        """Δ(chance penalty) for flipping PU ``idx``. Never mutates; caches the preview."""
        delta = 0.0
        payloads: list[_Payload] = []
        for t in self._term_list:
            pay = self._preview_term(t, idx, adding)
            payloads.append(pay)
            if len(pay[0]):
                delta += float(np.sum(pay[4] - t.pen[pay[0]]))
        delta *= self.weight
        self._preview = (idx, adding, delta, payloads)
        return delta

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def apply_flip(self, cache, idx: int, adding: bool) -> None:
        """Commit flipping PU ``idx``: the cached preview if it matches, else recomputed."""
        prev = self._preview
        self._preview = None
        if prev is not None and prev[0] == idx and prev[1] == adding:
            payloads = prev[3]
        else:
            payloads = [self._preview_term(t, idx, adding) for t in self._term_list]
        for t, (cols, EV2, B2, N2, pen2) in zip(self._term_list, payloads):
            if len(cols):
                t.EV[:, cols] = EV2
                t.B[:, cols] = B2
                t.N[cols] = N2
                t.pen[cols] = pen2
        self.selected[idx] = adding

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _preview_term(self, t: _TermState, idx: int, adding: bool) -> _Payload:
        """The touched features' post-flip state of one term (fresh arrays).

        Mutates nothing but the diagnostic ``n_recompute`` counter.
        """
        c = t.cells
        s, e = int(c.indptr[idx]), int(c.indptr[idx + 1])
        cols = c.indices[s:e]
        if s == e:
            return cols, np.zeros((2, 0)), np.zeros((2, 0)), np.zeros(0, np.int64), np.zeros(0)
        d = c.ev[:, s:e]
        N0 = t.N[cols]
        if adding:
            N2 = N0 + c.pos[s:e]
        else:
            d = -d
            N2 = N0 - c.pos[s:e]
        EV0 = t.EV[:, cols]
        EV2 = EV0 + d
        B2 = t.B[:, cols] + EPS * (np.abs(EV0) + c.abs_ev[:, s:e])
        E2 = EV2[0]
        V2 = EV2[1]
        live = N2 > 0
        redo = live & (
            (V2 <= 0)
            | (B2[1] > TAU * V2)
            | ((B2[0] > TAU * np.sqrt(np.maximum(V2, 0.0))) & (B2[0] > _E_FLOOR * np.abs(E2)))
        )
        # nonzero reached 0: snap to the exact sums of the remaining zero-variance cells.
        snap = ~live & (N0 > 0)
        fix = redo | snap
        thr = None
        if self._exact:
            # Knife edge: a zero-variance feature this close to its met threshold may sit
            # on the other side from the evaluator's sum; recompute it (module docstring).
            assert t.thr is not None
            thr = t.thr[cols]
            fix = fix | (~live & self._knife_hit(t, cols, E2, thr, B2[0]))
        if fix.any():
            for k in np.flatnonzero(fix):
                ee, vv = self._column_sums(c, int(cols[k]), idx, adding)
                if snap[k]:
                    vv = 0.0
                if self._exact and not live[k]:
                    # The exact rule decides from the evaluator's float sum (row order).
                    ee = _row_sum(c, int(cols[k]), self.selected, idx, adding)
                EV2[0, k] = ee
                EV2[1, k] = vv
                B2[0, k] = _RESET * abs(ee)
                B2[1, k] = _RESET * abs(vv)
                self.n_recompute += 1
        pen2 = _chance_penalty(
            E2, V2, ~live, self.targets[cols], t.ptarget[cols],
            t.ptarget_den[cols], self._met, self._exact, thr,
        )
        return cols, EV2, B2, N2, pen2

    @staticmethod
    def _knife_hit(
        t: _TermState, cols: np.ndarray, E: np.ndarray, thr: np.ndarray, BE: np.ndarray,
    ) -> np.ndarray:
        """True where ``|E − thr|`` is strictly inside the knife-edge window."""
        return cast(np.ndarray, np.abs(E - thr) < ChanceState._knife_window(t, cols, BE))

    @staticmethod
    def _knife_window(t: _TermState, cols: np.ndarray, BE: np.ndarray) -> np.ndarray:
        """The exact rule's knife-edge window ``(1 + 8·eps)·BE + 3·max(64, n_j)·eps·S_j``
        of features ``cols``, ``BE`` their bounds (module docstring)."""
        assert t.knife is not None
        return cast(np.ndarray, (1.0 + _KNIFE_BE) * BE + t.knife[cols])

    def _column_sums(
        self, c: ChanceCells, j: int, idx: int, adding: bool,
    ) -> tuple[float, float]:
        """Exact (fresh) ``E``, ``V`` of feature ``j`` over the selection after the flip.

        The post-flip selection mask is the ``bincount`` bin index: bin 1 receives exactly
        the selected cells, added in column order as ``from_selection``'s per-column
        ``bincount`` adds them, so a recompute equals a fresh state bit for bit; bin 0
        absorbs the unselected cells (a non-finite one cannot poison the sum, as ``NaN · 0``
        did in a masked matvec). There is no BLAS call (a threaded ``dgemv`` on the strided
        slice was ~50× slower past ~800 cells and depended on the thread count), and no
        gather: ``c_e`` / ``c_v`` are contiguous rows, read in place.
        """
        a, b = int(c.c_indptr[j]), int(c.c_indptr[j + 1])
        rows = c.c_rows[a:b]
        m = self.selected[rows]  # fancy index: a copy (bool, see __init__)
        k = int(np.searchsorted(rows, idx))
        if k < len(rows) and rows[k] == idx:
            m[k] = adding
        ee = np.bincount(m, weights=c.c_e[a:b], minlength=2)[1]
        vv = np.bincount(m, weights=c.c_v[a:b], minlength=2)[1]
        return float(ee), float(vv)
