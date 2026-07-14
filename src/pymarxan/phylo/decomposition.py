"""Branch-as-feature decomposition: PD as a standard ConservationProblem."""
from __future__ import annotations

import pandas as pd

from pymarxan.models.problem import ConservationProblem
from pymarxan.phylo.diversity import _resolve_tips
from pymarxan.phylo.tree import PhylogeneticTree


def phylogenetic_branch_problem(
    problem: ConservationProblem,
    tree: PhylogeneticTree,
    *,
    target: float = 1.0,
    tip_feature_map: dict[str, int] | None = None,
) -> ConservationProblem:
    """Return a ConservationProblem whose features are the tree's branches.

    Each branch becomes a synthetic feature: presence amount (1.0) in every PU
    where any descendant tip's feature occurs, ``target`` occurrences required,
    and ``spf`` = branch length. Solving ``min_set`` on the result yields the
    cheapest reserve capturing 100% of representable PD; ``max_weighted_features``
    with a ``COSTBUDGET`` maximizes PD under budget. Branches whose descendant
    tips occur in no PU are dropped (they would make ``min_set`` infeasible).
    The original species features are entirely replaced. A branch's provenance
    is recoverable from its feature ``name`` (``"branch:<child_node>"``).

    ``parameters`` carry through, so ``BLM``/``boundary`` (if set) yield a
    compact PD reserve and a budget solve reads the caller's ``COSTBUDGET``. The
    ``probability`` frame is dropped (it references now-deleted feature ids and
    branch features carry no probability semantics).
    """
    resolved = _resolve_tips(problem, tree, tip_feature_map)

    present = problem.pu_vs_features[problem.pu_vs_features["amount"] > 0]
    pus_by_feature: dict[int, set[int]] = {}
    for fid, pu in zip(present["species"], present["pu"]):
        pus_by_feature.setdefault(int(fid), set()).add(int(pu))

    feature_rows: list[dict] = []
    puvspr_rows: list[dict] = []
    next_id = 0
    for child, length, desc in tree.branches:
        pus: set[int] = set()
        for tip in desc:
            fid = resolved.get(tip)
            if fid is not None:
                pus |= pus_by_feature.get(fid, set())
        if not pus:
            continue  # unrepresentable branch — drop (keeps min_set feasible)
        bfid = next_id
        next_id += 1
        feature_rows.append(
            {
                "id": bfid,
                "name": f"branch:{child}",
                "target": float(target),
                "spf": float(length),
            }
        )
        for pu in sorted(pus):
            puvspr_rows.append({"species": bfid, "pu": pu, "amount": 1.0})

    branch_features = pd.DataFrame(
        feature_rows, columns=["id", "name", "target", "spf"]
    ).astype({"id": int, "target": float, "spf": float})
    branch_puvspr = pd.DataFrame(
        puvspr_rows, columns=["species", "pu", "amount"]
    ).astype({"species": int, "pu": int, "amount": float})
    return problem.copy_with(
        features=branch_features,
        pu_vs_features=branch_puvspr,
        probability=None,  # source prob frame references deleted feature ids
    )
