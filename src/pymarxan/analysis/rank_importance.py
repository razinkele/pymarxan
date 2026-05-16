"""Jung et al. (2021) sequential-removal rank importance (Phase 22).

Starting from a candidate solution, repeatedly remove the PU whose
removal least increases the objective until only one selected PU
remains. The order in which PUs are removed becomes their inverse
rank: PUs removed earlier are easier to replace (lower importance);
PUs removed later are harder to replace (higher importance).

Reference: Jung, Renaud, Tobón-Niedfeldt & Jetz (2021). A user-friendly
toolset for systematic conservation planning. *Methods in Ecology and
Evolution* 12(5): 869-877. https://doi.org/10.1111/2041-210X.13578
"""
from __future__ import annotations

import numpy as np

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution
from pymarxan.solvers.utils import compute_objective_terms


def compute_rank_importance(
    problem: ConservationProblem,
    solution: Solution,
) -> dict[int, float]:
    """Rank each selected PU by sequential removal order.

    Removed-last PUs get the highest rank (most important). Unselected
    PUs get rank 0 — they are not part of the candidate reserve and the
    method does not score them.

    Algorithm
    ---------
    1. Start with ``selected = solution.selected.copy()``.
    2. Among currently-selected PUs, find the one whose removal increases
       the objective the least (greedy "least damaging" removal).
    3. Remove it; record the removal order.
    4. Repeat until only one selected PU remains (or none).
    5. Score each removed PU by its removal order: rank ``1`` → first
       removed (least important); rank ``n_selected`` → last remaining
       (most important).

    Parameters
    ----------
    problem
        The conservation problem (needed for objective re-evaluation).
    solution
        A candidate solution. Only ``solution.selected`` is read.

    Returns
    -------
    dict[int, float]
        ``planning_unit_id -> rank``. Unselected PUs map to 0.0.
        Selected PUs map to 1..n_selected (n_selected most important).
    """
    pu_ids = problem.planning_units["id"].astype(int).to_numpy()
    blm = float(problem.parameters.get("BLM", 0.0))

    selected = solution.selected.copy()
    pu_index = {int(pid): i for i, pid in enumerate(pu_ids)}

    ranks: dict[int, float] = {int(pid): 0.0 for pid in pu_ids}
    n_selected = int(selected.sum())
    if n_selected == 0:
        return ranks

    # Sequential removal. Each step is O(n_selected) objective evals;
    # total O(n_selected² · objective_cost). Acceptable for typical
    # solution sizes; users with thousands of selected PUs should run
    # this offline.
    removal_order: list[int] = []
    current = selected.copy()
    # Track which indices remain selected for efficient iteration.
    remaining = np.where(current)[0].tolist()

    while len(remaining) > 1:
        best_idx: int | None = None
        best_delta_obj = float("inf")
        for i in remaining:
            # Synthesise the post-removal selection.
            trial = current.copy()
            trial[i] = False
            terms = compute_objective_terms(problem, trial, pu_index, blm)
            obj_after = float(terms["objective"])
            # Lower objective-after = less damage from removing this PU.
            if obj_after < best_delta_obj:
                best_delta_obj = obj_after
                best_idx = i
        # Remove the least-damaging PU.
        if best_idx is None:
            break
        current[best_idx] = False
        removal_order.append(best_idx)
        remaining.remove(best_idx)

    # The single surviving PU (if any) is the most important.
    last_survivor = remaining[0] if remaining else None

    # Score: removed first → rank 1; removed last → rank n_selected.
    for order, idx in enumerate(removal_order, start=1):
        pid = int(pu_ids[idx])
        ranks[pid] = float(order)
    if last_survivor is not None:
        pid = int(pu_ids[last_survivor])
        ranks[pid] = float(n_selected)

    return ranks
