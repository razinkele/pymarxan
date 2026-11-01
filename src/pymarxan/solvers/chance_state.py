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
a recompute both bounds reset to ``4·eps·|value|``; ``from_selection`` counts as one. When
``nonzero`` reaches 0, ``E`` and ``V`` snap to the exact sums of the remaining
zero-variance cells (``V = 0``). Zero variance itself is the exact count, never a
tolerance on ``V``; the scorer also applies Marxan's ``V > 0`` test, which after the
recompute only decides when a ``q`` outside [0, 1] leaves ``V ≤ 0`` beside counted cells.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from pymarxan.solvers.cache import ChanceCells
from pymarxan.solvers.probability import _chance_penalty, _ptarget_den, score_features

EPS = float(np.finfo(np.float64).eps)
TAU = 1e-12
_E_FLOOR = 64.0 * EPS
_RESET = 4.0 * EPS


@dataclass
class _TermState:
    """Running per-feature state of one chance term.

    ``EV`` (2, n_feat): running ``E`` (row 0) and ``V`` (row 1); ``B`` (2, n_feat): their
    float error bounds since the feature's last recompute; ``N``: nonzero counts; ``pen``:
    unweighted penalties.
    """

    cells: ChanceCells
    EV: np.ndarray
    B: np.ndarray
    N: np.ndarray
    pen: np.ndarray

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


# One term's preview for the flipped PU's features: (cols, EV2, B2, N2, pen2).
_Payload = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class ChanceState:
    """Mutable chance-term companion to ``ProblemCache`` (see the module docstring)."""

    def __init__(
        self,
        selected: np.ndarray,
        terms: dict[str, _TermState],
        weight: float,
        zero_var_rule: str,
        targets: np.ndarray,
        ptargets: np.ndarray,
    ) -> None:
        self.selected = np.asarray(selected, dtype=bool)  # a mask, whatever the caller's dtype
        self.terms = terms
        self._term_list = list(terms.values())
        self.weight = weight
        self.zero_var_rule = zero_var_rule
        self._met = zero_var_rule == "met"
        self.targets = targets
        self.ptargets = ptargets
        self._ptarget_den = _ptarget_den(ptargets)
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
            pen = score_features(
                EV[0], EV[1], N, cache.feat_targets, cache.feat_ptarget,
                cache.zero_var_rule,
            )[3]
            terms[name] = _TermState(cells=cells, EV=EV, B=_RESET * np.abs(EV), N=N, pen=pen)
        return cls(
            selected=sel, terms=terms, weight=float(cache.prob_weight),
            zero_var_rule=cache.zero_var_rule,
            targets=np.asarray(cache.feat_targets, dtype=np.float64),
            ptargets=np.asarray(cache.feat_ptarget, dtype=np.float64),
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
        if fix.any():
            for k in np.flatnonzero(fix):
                ee, vv = self._column_sums(c, int(cols[k]), idx, adding)
                if snap[k]:
                    vv = 0.0
                EV2[0, k] = ee
                EV2[1, k] = vv
                B2[0, k] = _RESET * abs(ee)
                B2[1, k] = _RESET * abs(vv)
                self.n_recompute += 1
        pen2 = _chance_penalty(
            E2, V2, ~live, self.targets[cols], self.ptargets[cols],
            self._ptarget_den[cols], self._met,
        )
        return cols, EV2, B2, N2, pen2

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
