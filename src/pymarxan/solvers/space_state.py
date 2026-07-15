"""SpaceState — incremental space/adequacy penalty companion for the SA / greedy loops.

Stateful and self-contained: precomputes the attribute-space kernel (positions, per-feature
demand points/weights/TSS, and a PU→feature inverse index) once, then recomputes ``space_held``
only for the features whose occupied set contains a flipped PU — the documented v1 recompute
delta, mirroring :class:`~pymarxan.solvers.separation.SepState`. Unlike ``SepState`` it does NOT
take the :class:`ProblemCache` (the attribute space may be non-geographic ``attribute_columns``
the cache doesn't hold), so it holds its own precompute — a deliberate, documented departure.

Space penalty is a soft shortfall ``Σ_f space_spf_f · max(0, space_target_f − space_held_f)`` —
ADDITIVE on top of the amount penalty (a feature may carry both). NOT thread-safe; lives within
one solver call frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymarxan.adequacy.model import SpaceSpec, pu_attribute_space
from pymarxan.models.problem import ConservationProblem


@dataclass
class _SpaceFeat:
    """Precomputed per-active-feature record (selection-independent)."""

    fid: int
    occ: np.ndarray  # (n_dp,) int — occupied-PU indices = demand points
    w: np.ndarray  # (n_dp,) float — demand weights (amounts)
    tss: float  # total sum of squares about the demand centroid (constant)
    target: float  # space_target proportion in (0, 1]
    spf: float  # space penalty weight (space_spf, defaults to the amount spf)


class SpaceState:
    """Mutable companion tracking the incremental space penalty over PU flips."""

    def __init__(
        self, pos: np.ndarray, feats: list[_SpaceFeat], selected: np.ndarray
    ) -> None:
        self.pos = pos
        self.feats = feats
        self.pu_to_space_feats = self._build_inverse(pos.shape[0], feats)
        self.selected = selected.copy()
        self.held = np.array(
            [self._held_for(f, self.selected) for f in feats], dtype=float
        )
        self.space_penalty_total = float(self._penalty(self.held))

    @property
    def active(self) -> bool:
        return len(self.feats) > 0

    @staticmethod
    def _build_inverse(n_pu: int, feats: list[_SpaceFeat]) -> list[np.ndarray]:
        inv: list[list[int]] = [[] for _ in range(n_pu)]
        for fi, f in enumerate(feats):
            for idx in f.occ:
                inv[int(idx)].append(fi)
        return [np.array(x, dtype=int) for x in inv]

    def _held_for(self, f: _SpaceFeat, selected: np.ndarray) -> float:
        occ, w, tss = f.occ, f.w, f.tss
        if len(occ) == 0:
            return 0.0
        sel_occ = occ[selected[occ]]
        if len(sel_occ) == 0:
            return 0.0
        if tss == 0.0:
            return 1.0
        d2 = ((self.pos[occ][:, None, :] - self.pos[sel_occ][None, :, :]) ** 2).sum(axis=2)
        wss = float(np.sum(w * d2.min(axis=1)))
        return float(np.clip(1.0 - wss / tss, 0.0, 1.0))

    def _penalty(self, held: np.ndarray) -> float:
        tot = 0.0
        for fi, f in enumerate(self.feats):
            tot += f.spf * max(0.0, f.target - float(held[fi]))
        return tot

    @classmethod
    def from_problem(
        cls, problem: ConservationProblem, spec: SpaceSpec, selected: np.ndarray
    ) -> SpaceState:
        """One-time build. Reads the ``space_target`` / ``space_spf`` feature columns; features
        with ``space_target <= 0`` (or the column absent) are inactive."""
        feats_df = problem.features
        n_pu = problem.n_planning_units
        if "space_target" not in feats_df.columns:
            return cls(np.zeros((n_pu, 1)), [], selected)

        pos = pu_attribute_space(problem, spec)
        idx_map = problem.pu_id_to_index
        pv = problem.pu_vs_features
        species = np.asarray(pv["species"].to_numpy(), dtype=np.int64)
        pu_all = np.asarray(pv["pu"].to_numpy(), dtype=np.int64)
        amt_all = np.asarray(pv["amount"].to_numpy(), dtype=float)
        spf_col = "space_spf" if "space_spf" in feats_df.columns else "spf"

        feats: list[_SpaceFeat] = []
        for _, row in feats_df.iterrows():
            tgt = float(row.get("space_target", 0.0) or 0.0)
            if tgt <= 0.0:
                continue
            fid = int(row["id"])
            fmask = (species == fid) & (amt_all > 0)
            pu_ids = pu_all[fmask]
            # Keep-mask keeps weights aligned to demand points (adequacy BUG-B).
            keep = np.array([int(p) in idx_map for p in pu_ids], dtype=bool)
            occ = np.array([idx_map[int(p)] for p in pu_ids[keep]], dtype=int)
            w = amt_all[fmask][keep]
            if len(occ) == 0:
                tss = 0.0
            else:
                p_d = pos[occ]
                c = p_d.mean(axis=0)
                tss = float(np.sum(w * np.sum((p_d - c) ** 2, axis=1)))
            feats.append(
                _SpaceFeat(fid=fid, occ=occ, w=w, tss=tss, target=tgt,
                           spf=float(row.get(spf_col, 1.0)))
            )
        return cls(pos, feats, selected)

    def penalty_total(self) -> float:
        """Current total space penalty."""
        return self.space_penalty_total

    def delta_penalty(self, idx: int, adding: bool) -> float:
        """Δ(space penalty) for flipping PU ``idx``. Does NOT mutate state.

        Only ``pu_to_space_feats[idx]`` features change — ``space_held_f`` depends on the
        selected∩occupied set of ``f``, which a flip of ``idx`` alters iff ``idx`` is occupied
        in ``f``.
        """
        affected = self.pu_to_space_feats[idx]
        if len(affected) == 0:
            return 0.0
        sel_after = self.selected.copy()
        sel_after[idx] = adding
        delta = 0.0
        for fi in affected:
            f = self.feats[int(fi)]
            new_h = self._held_for(f, sel_after)
            delta += f.spf * (
                max(0.0, f.target - new_h) - max(0.0, f.target - float(self.held[int(fi)]))
            )
        return delta

    def apply_flip(self, idx: int, adding: bool) -> None:
        """Commit the flip: update ``selected``, per-feature ``held``, and the running total."""
        affected = self.pu_to_space_feats[idx]
        self.selected[idx] = adding
        for fi in affected:
            fi_int = int(fi)
            f = self.feats[fi_int]
            new_h = self._held_for(f, self.selected)
            self.space_penalty_total += f.spf * (
                max(0.0, f.target - new_h)
                - max(0.0, f.target - float(self.held[fi_int]))
            )
            self.held[fi_int] = new_h

    def all_targets_met(self) -> bool:
        """True when every active feature's ``space_held`` meets its ``space_target``."""
        return all(
            float(self.held[fi]) >= f.target for fi, f in enumerate(self.feats)
        )

    def held_by_id(self) -> dict[int, float]:
        """``{feature_id: space_held}`` for the current selection (active features only)."""
        return {f.fid: float(self.held[fi]) for fi, f in enumerate(self.feats)}
