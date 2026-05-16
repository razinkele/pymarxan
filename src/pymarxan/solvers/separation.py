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

from pymarxan.models.problem import ConservationProblem, has_geometry

__all__ = [
    "PUCoordinatesUnavailableError",
    "get_pu_coordinates",
]


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

    Three-tier fallback:

    1. If ``problem.planning_units`` is a GeoDataFrame with a non-empty
       geometry column, use ``geometry.centroid``.
    2. Else if ``planning_units`` has both ``xloc`` and ``yloc`` columns
       (Marxan's classic ``pu.dat`` convention), use those.
    3. Else raise :exc:`PUCoordinatesUnavailableError`.

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

    raise PUCoordinatesUnavailableError(
        "PU coordinates required for separation-active problems. Either pass "
        "a GeoDataFrame planning_units with a geometry column, or include "
        "xloc/yloc columns. See pymarxan.spatial.importers.import_planning_units "
        "for converting Marxan-format pu.dat with coordinates."
    )
