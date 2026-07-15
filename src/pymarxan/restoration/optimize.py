"""greedy_mesh_restore — benefit–cost greedy MESH maximization under a restoration budget.

A fast heuristic: MESH = Σarea²/A_total is *supermodular* (a cell bridging two patches yields an
outsized, non-diminishing gain), so no greedy variant carries an approximation guarantee — restoptr
uses an exact constraint-programming solver. Module function, not a Marxan ``Solver`` — mirrors
``zonation.rank_removal``.

Scale: ``O(iterations × candidates × label)`` with a full ``compute_mesh`` per candidate every
iteration (gains cannot be cached — restoring a cell changes its neighbours' merge gains). Suited
to moderate grids (hundreds–low-thousands of restorable cells); a national-scale raster with
thousands of restorable cells and a large budget will be slow (a union-find delta is future work).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pymarxan.restoration.mesh import compute_mesh
from pymarxan.restoration.problem import RestorationProblem

_CRITERIA = ("gain_per_cost", "gain")


@dataclass(eq=False)  # numpy fields break the auto __eq__ (repo convention)
class MeshRestorationResult:
    """A greedy MESH restoration plan + its budget–MESH frontier.

    ``mesh_curve`` / ``cost_curve`` are aligned (length ``n_restored + 1``, index 0 = baseline /
    zero cost); ``order`` lists the PU indices in pick sequence, so any prefix reconstructs a
    sub-budget plan (``restored`` at cumulative cost ``cost_curve[k]``).
    """

    restored: np.ndarray      # (n_pu,) bool — the chosen restoration plan
    mesh: float               # final post-restoration MESH
    baseline_mesh: float      # pre-restoration MESH
    total_cost: float
    n_restored: int
    mesh_curve: np.ndarray    # MESH after each step (index 0 = baseline)
    cost_curve: np.ndarray    # cumulative cost after each step (index 0 = 0.0)
    order: list[int] = field(default_factory=list)  # PU indices in pick sequence


def greedy_mesh_restore(
    problem: RestorationProblem,
    budget: float,
    *,
    criterion: str = "gain_per_cost",
    connectivity: str = "rook",
    cell_area: float | None = None,
) -> MeshRestorationResult:
    """Greedily restore cells to maximize effective mesh size under a cost ``budget``.

    Each step restores the affordable restorable cell with the best marginal MESH gain
    (``criterion="gain_per_cost"``, default — gain per unit cost, budget-aware; a zero-cost cell
    scores ``+inf``) or the best raw gain (``criterion="gain"``), stopping when no affordable
    positive-gain cell remains. See the module docstring for the performance profile.
    """
    if budget < 0:
        raise ValueError(f"budget must be >= 0, got {budget}")
    if criterion not in _CRITERIA:
        raise ValueError(f"criterion must be one of {_CRITERIA}, got {criterion!r}")

    grid = problem.grid
    existing = problem.existing_habitat
    cost = problem.cost
    assert cost is not None  # set in RestorationProblem.__post_init__

    restored = np.zeros(problem.n_pu, dtype=bool)
    spent = 0.0
    current = float(compute_mesh(grid, existing, connectivity=connectivity,
                                 cell_area=cell_area).mesh)
    baseline = current
    mesh_curve = [current]
    cost_curve = [0.0]
    order: list[int] = []

    while True:
        best_idx = -1
        best_score = -np.inf
        best_mesh = current
        for raw in problem.restorable_indices:
            c = int(raw)
            if restored[c]:
                continue
            cc = float(cost[c])
            if spent + cc > budget:  # unaffordable
                continue
            trial = existing | restored
            trial[c] = True
            m = float(compute_mesh(grid, trial, connectivity=connectivity,
                                   cell_area=cell_area).mesh)
            gain = m - current
            if gain <= 0:
                continue  # no-op cell (e.g. an already-habitat overlap)
            score = gain if criterion == "gain" else (np.inf if cc == 0.0 else gain / cc)
            if score > best_score:
                best_score = score
                best_idx = c
                best_mesh = m
        if best_idx < 0:
            break
        restored[best_idx] = True
        spent += float(cost[best_idx])
        current = best_mesh
        mesh_curve.append(current)
        cost_curve.append(spent)
        order.append(best_idx)

    return MeshRestorationResult(
        restored=restored,
        mesh=current,
        baseline_mesh=baseline,
        total_cost=float(spent),
        n_restored=int(restored.sum()),
        mesh_curve=np.asarray(mesh_curve, dtype=float),
        cost_curve=np.asarray(cost_curve, dtype=float),
        order=order,
    )
