"""Marxan-faithful separation-distance constraints (Phase 20).

Implements per-feature SEPDISTANCE / SEPNUM ("type-4 separation") matching
Marxan v4's reference C++ source:

- ``computeSepPenalty`` (``computation.hpp:15-27``) — hyperbolic curve
  ``1/(7·fval + 0.2) − 1/7.2`` with ``fval = max(count, 1)/sepnum``.
- ``CountSeparation2`` + ``makelist`` + ``SepDealList`` (``clumping.cpp:1075-1279``) —
  greedy admission in PU-id insertion order, capped at ``sepnum``.
- ``CheckDistance`` (``clumping.cpp:1006-1012``) — Euclidean-squared distance
  on raw ``pu.xloc`` / ``pu.yloc``.

A feature is "separation-active" iff ``sepdistance > 0 AND sepnum > 1``.
The ``sepnum=1`` default matches Marxan's trivially-satisfied disabled state.

Public surface (module ``__all__``):
- :func:`compute_sep_penalty` — hyperbolic per-feature penalty curve.
- :func:`count_separation` — greedy MIS-like admission count.
- :func:`compute_sep_penalty_from_scratch` — reference per-feature evaluator.
- :func:`evaluate_solution_separation` — post-hoc Solution attr populator.
- :func:`get_pu_coordinates` — three-tier coordinate resolution helper.
- :class:`SepState` — mutable companion to ``ProblemCache`` for SA inner loop.
- :exc:`PUCoordinatesUnavailableError` — raised when separation-active problem
  has no derivable PU coordinates.

This module is NOT re-exported at the ``pymarxan.solvers`` package level
(matches the Phase 19 ``clumping`` precedent). Import via
``from pymarxan.solvers.separation import ...``.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from pymarxan.models.problem import ConservationProblem, has_geometry

__all__ = [
    "PUCoordinatesUnavailableError",
    "SepState",
    "compute_sep_penalty",
    "compute_sep_penalty_from_scratch",
    "count_separation",
    "evaluate_solution_separation",
    "get_pu_coordinates",
    "is_separation_active",
    "raise_if_separation_active",
]


def is_separation_active(problem: ConservationProblem) -> bool:
    """Return True iff any feature has ``sepdistance > 0 AND sepnum > 1``.

    Used by zone solvers (round-3 H1) to refuse separation-active problems
    rather than silently producing wrong-because-incomplete results.
    """
    feats = problem.features
    if "sepnum" not in feats.columns or "sepdistance" not in feats.columns:
        return False
    return bool(((feats["sepnum"] > 1) & (feats["sepdistance"] > 0)).any())


def raise_if_separation_active(
    problem: ConservationProblem, solver_name: str,
) -> None:
    """Raise ``NotImplementedError`` if the problem is separation-active.

    Used by zone solvers (round-3 H1) — per-zone SEPDISTANCE / SEPNUM
    isn't implemented in Phase 20; v0.3 will close that gap if demand
    materialises. For now this guard prevents silent no-ops.
    """
    if is_separation_active(problem):
        raise NotImplementedError(
            f"{solver_name} does not honour SEPDISTANCE/SEPNUM; per-zone "
            "separation is deferred to v0.3. Use the non-zone solvers "
            "(SASolver, IterativeImprovementSolver, MIPSolver, "
            "HeuristicSolver) or set sepnum<=1 on all features."
        )


def compute_sep_penalty(count: int, sepnum: int) -> float:
    """Marxan's hyperbolic separation-penalty curve.

    Verbatim from ``computation.hpp::computeSepPenalty:15-27``::

        fval = ival / itarget
        if !ival: fval = 1.0 / itarget           # count==0 bump
        return 1 / (7·fval + 0.2) - 1 / 7.2

    Properties:
    - ``count >= sepnum`` (target met): returns exactly 0.
    - ``count == 0``: bumped to ``fval = 1/sepnum`` so the hyperbola
      doesn't blow up. Same value as ``count == 1`` with this bump.
    - ``sepnum <= 0`` (disabled): returns 0 unconditionally.

    The curve is NOT the linear ``(sepnum-count)/sepnum`` Marxan's user
    manual implies — it's steeper near the target and flatter when badly
    missed. Round-1 scientific-accuracy review caught this.

    Parameters
    ----------
    count
        Achieved separation count (from :func:`count_separation`).
    sepnum
        Per-feature SEPNUM target.

    Returns
    -------
    float
        Non-negative scalar penalty. Multiply by ``baseline_penalty · SPF``
        to get the contribution to the total objective.
    """
    if sepnum <= 0:
        return 0.0
    # Clamp to sepnum — Marxan's CountSeparation2 caps at sepnum and so do
    # we (round-2 C3). Above sepnum the raw curve goes negative, which is
    # meaningless (target is met). Defensive guard against out-of-contract
    # callers; in normal use count_separation already returns <= sepnum.
    if count >= sepnum:
        return 0.0
    fval = float(count) / float(sepnum) if count > 0 else 1.0 / float(sepnum)
    return 1.0 / (7.0 * fval + 0.2) - 1.0 / 7.2


def count_separation(
    selected: np.ndarray,
    feat_amounts: np.ndarray,
    pu_coords: np.ndarray,
    sepdistance: float,
    sepnum: int,
) -> int:
    """Greedy Marxan-faithful separation count for one feature.

    Mirrors ``clumping.cpp::CountSeparation2 + makelist + SepDealList``:
    iterate candidate PUs (selected AND amount > 0) in **ascending PU-id
    order** (NOT sorted by amount — round-1 C2 fix), admit each that is at
    least ``sepdistance`` away from every already-admitted PU; stop as
    soon as ``len(kept) == sepnum``.

    Implementation note (round-2 CR1): the candidate pairwise-distance
    matrix is allocated on the **candidate sub-array only** (k×k where
    ``k = len(candidates)``), never the full n_pu×n_pu. Per-flip memory
    is bounded by selection footprint, not problem size.

    Parameters
    ----------
    selected
        (n_pu,) bool — current selection vector.
    feat_amounts
        (n_pu,) float — per-PU amount of the feature (for the candidate
        filter ``amount > 0``).
    pu_coords
        (n_pu, 2) float — PU coordinates in CRS-native units.
    sepdistance
        Minimum pairwise distance for separation. ``0`` makes every pair
        trivially separated.
    sepnum
        Target separation count. Function returns ``min(actual, sepnum)``
        — values above ``sepnum`` are meaningless (penalty plateaus at 0).

    Returns
    -------
    int
        Number of pairwise-separated PUs admitted, clamped at ``sepnum``.
    """
    # Build candidate index — already in ascending PU-id order since
    # np.where returns sorted indices.
    candidates = np.where(selected & (feat_amounts > 0))[0]
    k = len(candidates)
    if k == 0:
        return 0

    # Cheap early exit when the constraint is trivially satisfied.
    if sepdistance <= 0:
        return min(k, sepnum)

    # k×k squared-distance matrix on the candidate sub-array only.
    sq_thresh = float(sepdistance) * float(sepdistance)
    if k == 1:
        return min(1, sepnum)
    dist_sq = squareform(pdist(pu_coords[candidates], "sqeuclidean"))

    # Greedy admission in candidate order (== ascending PU-id order).
    # `kept_local` is a small bool mask over the candidate indices.
    kept_local = np.zeros(k, dtype=bool)
    n_kept = 0
    for i in range(k):
        if n_kept == 0:
            kept_local[i] = True
            n_kept = 1
            if n_kept >= sepnum:
                return n_kept
            continue
        # Admit if at least sepdistance from every already-admitted PU.
        if (dist_sq[i, kept_local] >= sq_thresh).all():
            kept_local[i] = True
            n_kept += 1
            if n_kept >= sepnum:
                return n_kept

    return n_kept


def compute_sep_penalty_from_scratch(
    problem: ConservationProblem,
    selected: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Reference per-feature separation penalty evaluator.

    For each separation-active feature (``sepdistance > 0 AND sepnum > 1``):

    1. Compute ``count_j = count_separation(...)``.
    2. Compute ``pen_j = compute_sep_penalty(count_j, sepnum_j) ·
       baseline_penalty_j · SPF_j`` reusing Phase 19's
       :func:`pymarxan.solvers.clumping.compute_baseline_penalty`.
    3. Sum into the total.

    Inactive features contribute count=0 and penalty=0 (not summed; they
    show up as 0 in the returned counts array but don't affect total).

    Parameters
    ----------
    problem
        Conservation problem with ``features['sepdistance']`` and
        ``features['sepnum']`` columns.
    selected
        (n_pu,) bool — current selection vector.

    Returns
    -------
    tuple[np.ndarray, float]
        ``(counts, total_penalty)`` where ``counts[j]`` is the separation
        count for feature column ``j`` (0 for inactive features).
    """
    from pymarxan.solvers.clumping import compute_baseline_penalty

    features = problem.features
    n_feat = len(features)
    counts = np.zeros(n_feat, dtype=np.int64)

    if "sepdistance" not in features.columns or "sepnum" not in features.columns:
        return counts, 0.0

    sepdistance = features["sepdistance"].astype(float).to_numpy()
    sepnum = features["sepnum"].astype(int).to_numpy()
    spf = features["spf"].astype(float).to_numpy()
    feat_ids = features["id"].astype(int).to_numpy()

    # A feature is sep-active iff BOTH sepdistance > 0 AND sepnum > 1.
    active_mask = (sepdistance > 0) & (sepnum > 1)
    if not active_mask.any():
        return counts, 0.0

    # PU coordinates — raises PUCoordinatesUnavailableError when missing.
    pu_coords = get_pu_coordinates(problem)

    # Per-feature amount vectors indexed by PU position.
    pu_ids = problem.planning_units["id"].astype(int).to_numpy()
    pu_id_to_idx = {int(pid): i for i, pid in enumerate(pu_ids)}
    n_pu = len(pu_ids)
    puvspr = problem.pu_vs_features

    baselines = compute_baseline_penalty(problem)

    total = 0.0
    feat_id_to_col = {int(fid): j for j, fid in enumerate(feat_ids)}

    # Group puvspr by species once.
    groups = puvspr.groupby("species")
    for j in np.where(active_mask)[0]:
        fid = int(feat_ids[j])
        amounts_vec = np.zeros(n_pu, dtype=np.float64)
        if int(fid) in groups.groups:
            grp = groups.get_group(int(fid))
            for pu_id, amt in zip(grp["pu"].astype(int),
                                  grp["amount"].astype(float)):
                idx = pu_id_to_idx.get(int(pu_id))
                if idx is not None:
                    amounts_vec[idx] = float(amt)
        c = count_separation(
            selected=selected,
            feat_amounts=amounts_vec,
            pu_coords=pu_coords,
            sepdistance=float(sepdistance[j]),
            sepnum=int(sepnum[j]),
        )
        counts[j] = c
        total += (
            compute_sep_penalty(c, int(sepnum[j]))
            * float(baselines[j])
            * float(spf[j])
        )

    # Silence "unused" hint (feat_id_to_col may be used by future callers).
    del feat_id_to_col
    return counts, total


def evaluate_solution_separation(
    problem: ConservationProblem,
    selected: np.ndarray,
) -> tuple[dict[int, int], float]:
    """Post-hoc Solution-attribute populator.

    Wraps :func:`compute_sep_penalty_from_scratch` to return a
    ``{feature_id: shortfall}`` dict (shortfall = ``max(0, sepnum - count)``)
    plus the total penalty scalar.

    Called by :func:`pymarxan.solvers.utils.build_solution` (Phase 20
    Task 9) to populate ``Solution.sep_shortfalls`` and
    ``Solution.sep_penalty`` post-hoc for every solver path.

    Parameters
    ----------
    problem
        Conservation problem with separation-active features.
    selected
        (n_pu,) bool — solver's final selection.

    Returns
    -------
    tuple[dict[int, int], float]
        ``(shortfalls, total_penalty)``. Dict keys are feature ids that
        are sep-active; values are integer shortfalls (0 if target met).

    Raises
    ------
    PUCoordinatesUnavailableError
        Propagated from :func:`get_pu_coordinates` when no coordinates
        are derivable. ``build_solution`` catches this specific subclass
        so heuristic-only users on no-geometry problems still get a
        valid Solution (round-3 M8).
    """
    counts, total = compute_sep_penalty_from_scratch(problem, selected)
    features = problem.features
    if "sepdistance" not in features.columns or "sepnum" not in features.columns:
        return {}, 0.0

    sepdistance = features["sepdistance"].astype(float).to_numpy()
    sepnum = features["sepnum"].astype(int).to_numpy()
    feat_ids = features["id"].astype(int).to_numpy()

    shortfalls: dict[int, int] = {}
    for j in range(len(feat_ids)):
        if sepdistance[j] > 0 and sepnum[j] > 1:
            shortfalls[int(feat_ids[j])] = max(0, int(sepnum[j]) - int(counts[j]))
    return shortfalls, float(total)


class SepState:
    """Mutable companion to :class:`ProblemCache` for the SA / iterative-improvement
    inner loop. Maintains per-feature separation counts and the total
    separation penalty incrementally.

    NOT thread-safe (round-3 F9). Lives entirely within a single solver's
    call frame. Do NOT expose to progress observers without explicit
    snapshotting.

    v1 implementation: per-affected-feature full recompute via vectorised
    :func:`count_separation`. The candidate sub-array pdist allocation is
    bounded by selection footprint, not problem size (round-2 CR1).

    Iteration uses ``cache.pu_to_sep_feats[idx]`` — the precomputed inverse
    PU→feature index (round-3 H15) — so the outer loop is O(features-at-PU)
    rather than O(n_feat) per flip.
    """

    def __init__(
        self,
        selected: np.ndarray,
        sep_counts: np.ndarray,
        sep_penalty_total: float,
    ) -> None:
        self.selected = selected
        self.sep_counts = sep_counts  # (n_feat,) int — count per feature
        self.sep_penalty_total = sep_penalty_total

    @classmethod
    def from_selection(cls, cache, selected: np.ndarray) -> SepState:
        """One-time full build. Called once before the SA loop starts.

        Computes the initial per-feature separation count and the total
        penalty (baseline · SPF · hyperbolic-curve summed across active
        features).
        """
        n_feat = cache.n_feat
        sep_counts = np.zeros(n_feat, dtype=np.int64)
        total = 0.0
        sep_active = (cache.feat_sepdistance > 0) & (cache.feat_sepnum > 1)
        for j in np.where(sep_active)[0]:
            amounts = np.asarray(cache.pu_feat_matrix[:, j], dtype=np.float64)
            c = count_separation(
                selected=selected,
                feat_amounts=amounts,
                pu_coords=cache.pu_coords,
                sepdistance=float(cache.feat_sepdistance[j]),
                sepnum=int(cache.feat_sepnum[j]),
            )
            sep_counts[j] = c
            total += (
                compute_sep_penalty(int(c), int(cache.feat_sepnum[j]))
                * float(cache.feat_baseline_penalty[j])
                * float(cache.feat_spf[j])
            )
        return cls(
            selected=selected.copy(),
            sep_counts=sep_counts,
            sep_penalty_total=float(total),
        )

    def penalty_total(self) -> float:
        """Current total separation penalty."""
        return self.sep_penalty_total

    def delta_penalty(self, cache, idx: int, adding: bool) -> float:
        """Δ(separation penalty) for flipping PU ``idx``. Does NOT mutate state.

        Iterates only ``cache.pu_to_sep_feats[idx]`` — features sep-active
        AND containing PU ``idx`` (round-3 H15 inverse index). Features
        whose count would not be affected by the flip are skipped.
        """
        affected_features = cache.pu_to_sep_feats[idx]
        if len(affected_features) == 0:
            return 0.0

        # Synthesise the post-flip selection without mutating self.selected
        sel_after = self.selected.copy()
        sel_after[idx] = adding

        total_delta = 0.0
        for j in affected_features:
            j_int = int(j)
            amounts = np.asarray(cache.pu_feat_matrix[:, j_int], dtype=np.float64)
            new_count = count_separation(
                selected=sel_after,
                feat_amounts=amounts,
                pu_coords=cache.pu_coords,
                sepdistance=float(cache.feat_sepdistance[j_int]),
                sepnum=int(cache.feat_sepnum[j_int]),
            )
            old_count = int(self.sep_counts[j_int])
            sepnum = int(cache.feat_sepnum[j_int])
            old_pen = compute_sep_penalty(old_count, sepnum)
            new_pen = compute_sep_penalty(int(new_count), sepnum)
            total_delta += (
                (new_pen - old_pen)
                * float(cache.feat_baseline_penalty[j_int])
                * float(cache.feat_spf[j_int])
            )
        return total_delta

    def apply_flip(self, cache, idx: int, adding: bool) -> None:
        """Commit the flip: update selected mask, per-feature counts, and
        the running penalty total. Mirrors :meth:`delta_penalty`'s scan."""
        affected_features = cache.pu_to_sep_feats[idx]
        self.selected[idx] = adding
        if len(affected_features) == 0:
            return

        for j in affected_features:
            j_int = int(j)
            amounts = np.asarray(cache.pu_feat_matrix[:, j_int], dtype=np.float64)
            new_count = count_separation(
                selected=self.selected,
                feat_amounts=amounts,
                pu_coords=cache.pu_coords,
                sepdistance=float(cache.feat_sepdistance[j_int]),
                sepnum=int(cache.feat_sepnum[j_int]),
            )
            old_count = int(self.sep_counts[j_int])
            sepnum = int(cache.feat_sepnum[j_int])
            old_pen = compute_sep_penalty(old_count, sepnum)
            new_pen = compute_sep_penalty(int(new_count), sepnum)
            self.sep_penalty_total += (
                (new_pen - old_pen)
                * float(cache.feat_baseline_penalty[j_int])
                * float(cache.feat_spf[j_int])
            )
            self.sep_counts[j_int] = int(new_count)


class PUCoordinatesUnavailableError(ValueError):
    """PU coordinates required for separation evaluation but not derivable
    from the problem (no geometry, no ``xloc``/``yloc``, or NaN/invalid
    centroids).

    Subclasses ``ValueError`` so legacy ``except ValueError`` blocks still
    catch it. ``build_solution`` (Phase 20 Task 9) catches this specific
    class so genuine internal ``ValueError`` bugs in ``count_separation``
    propagate instead of being silently swallowed (round-3 M8).
    """


def get_pu_coordinates(problem: ConservationProblem) -> np.ndarray:
    """Resolve per-PU 2D coordinates for separation distance calculation.

    Four-tier fallback:

    1. If ``problem.planning_units`` is a GeoDataFrame with a non-empty
       geometry column, use ``geometry.centroid``.
    2. Else if ``planning_units`` has both ``xloc`` and ``yloc`` columns
       (Marxan's classic ``pu.dat`` convention), use those.
    3. Else if ``problem.grid`` is set (a raster-grid problem), use
       ``grid.cell_centroids()`` (S4a).
    4. Else raise :exc:`PUCoordinatesUnavailableError`.

    NaN guard (round-2 H3): if any resolved coordinate is NaN — from an
    empty/invalid geometry or a missing ``xloc``/``yloc`` value — raise
    rather than silently corrupting downstream distance comparisons (NaN
    comparisons always evaluate False, so a NaN-centroid PU would be
    perpetually rejected from every admitted set without any signal).

    Parameters
    ----------
    problem
        Conservation problem whose planning_units carry coordinates.

    Returns
    -------
    np.ndarray
        Shape ``(n_pu, 2)`` float64 array of ``(x, y)`` coordinates in
        the planning_units' native CRS units.

    Raises
    ------
    PUCoordinatesUnavailableError
        When neither geometry nor ``xloc``/``yloc`` is available, or when
        any resolved coordinate is NaN.
    """
    pu = problem.planning_units

    if has_geometry(problem):
        centroids = pu.geometry.centroid  # type: ignore[union-attr]
        coords = np.column_stack([
            np.asarray(centroids.x, dtype=np.float64),
            np.asarray(centroids.y, dtype=np.float64),
        ])
        if np.isnan(coords).any():
            bad_idx = np.where(np.isnan(coords).any(axis=1))[0].tolist()
            raise PUCoordinatesUnavailableError(
                f"PU geometry contains {len(bad_idx)} empty or invalid rows "
                f"at indices {bad_idx[:10]}{'...' if len(bad_idx) > 10 else ''}; "
                "cannot compute centroids for separation."
            )
        return coords

    if "xloc" in pu.columns and "yloc" in pu.columns:
        coords = np.column_stack([
            np.asarray(pu["xloc"], dtype=np.float64),
            np.asarray(pu["yloc"], dtype=np.float64),
        ])
        if np.isnan(coords).any():
            bad_idx = np.where(np.isnan(coords).any(axis=1))[0].tolist()
            raise PUCoordinatesUnavailableError(
                f"planning_units has NaN xloc/yloc at "
                f"{len(bad_idx)} rows {bad_idx[:10]}{'...' if len(bad_idx) > 10 else ''}."
            )
        return coords

    # Tier 3 (S4a): a raster-grid problem carries no geometry/xloc, but its
    # GridGeometry cell centroids are the per-PU coordinates (in PU order).
    if problem.grid is not None:
        coords = np.asarray(problem.grid.cell_centroids(), dtype=np.float64)
        if np.isnan(coords).any():
            raise PUCoordinatesUnavailableError(
                "grid.cell_centroids() produced NaN coordinates "
                "(non-finite grid origin/cell size)."
            )
        return coords

    raise PUCoordinatesUnavailableError(
        "PU coordinates required for separation-active problems. Either pass "
        "a GeoDataFrame planning_units with a geometry column, or include "
        "xloc/yloc columns. See pymarxan.spatial.importers.import_planning_units "
        "for converting Marxan-format pu.dat with coordinates."
    )
