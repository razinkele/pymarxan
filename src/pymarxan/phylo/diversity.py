"""Faith (1992) phylogenetic-diversity scoring of a solution."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import pandas as pd

from pymarxan.models.problem import ConservationProblem
from pymarxan.phylo.tree import NodeId, PhylogeneticTree
from pymarxan.solvers.base import Solution


@dataclass
class PDResult:
    """Phylogenetic diversity a solution represents, with per-branch detail."""

    pd_represented: float
    pd_total: float
    pd_representable: float
    fraction_pd_total: float
    fraction_pd_representable: float
    n_tips: int
    n_tips_represented: int
    n_tips_unresolved: int
    branch_child: list[NodeId]
    branch_length: list[float]
    branch_represented: list[bool]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "child_node": self.branch_child,
                "length": self.branch_length,
                "represented": self.branch_represented,
            }
        )


def _resolve_tips(
    problem: ConservationProblem,
    tree: PhylogeneticTree,
    tip_feature_map: dict[str, int] | None,
) -> dict[NodeId, int | None]:
    """Map each tree tip to a feature id (or None if it matches nothing)."""
    name_to_id = {
        str(n): int(i)
        for n, i in zip(problem.features["name"], problem.features["id"])
    }
    id_to_id = {str(i): int(i) for i in problem.features["id"]}
    resolved: dict[NodeId, int | None] = {}
    for tip in tree.tips:
        key = str(tip)
        if tip_feature_map is not None:
            resolved[tip] = tip_feature_map.get(key)
        elif key in name_to_id:
            resolved[tip] = name_to_id[key]
        elif key in id_to_id:
            resolved[tip] = id_to_id[key]
        else:
            resolved[tip] = None
    if resolved and all(fid is None for fid in resolved.values()):
        warnings.warn(
            "no tree tip matched any feature (by name or id) — check the "
            "phylogeny labels or pass tip_feature_map; PD will be 0. Did you "
            "score against the branch problem instead of the original?",
            stacklevel=2,
        )
    return resolved


def compute_phylogenetic_diversity(
    problem: ConservationProblem,
    solution: Solution,
    tree: PhylogeneticTree,
    *,
    tip_feature_map: dict[str, int] | None = None,
) -> PDResult:
    """Report the Faith PD a ``solution`` represents against ``tree``.

    A tip is *represented* if its feature occurs (amount > 0) in at least one
    selected PU. ``pd_represented`` sums branch lengths whose descendant tips
    intersect the represented set (rooted PD). ``pd_representable`` is the
    ceiling reachable given which tips occur in any PU at all.
    """
    resolved = _resolve_tips(problem, tree, tip_feature_map)

    selected_ids = {
        int(pid)
        for pid, sel in zip(problem.planning_units["id"], solution.selected)
        if sel
    }
    puvspr = problem.pu_vs_features
    present = puvspr[puvspr["amount"] > 0]
    features_in_reserve = {
        int(s) for s in present.loc[present["pu"].isin(selected_ids), "species"]
    }
    features_anywhere = {int(s) for s in present["species"]}

    represented_tips = {
        tip
        for tip, fid in resolved.items()
        if fid is not None and fid in features_in_reserve
    }
    representable_tips = {
        tip
        for tip, fid in resolved.items()
        if fid is not None and fid in features_anywhere
    }

    branch_child: list[NodeId] = []
    branch_length: list[float] = []
    branch_represented: list[bool] = []
    pd_represented = 0.0
    pd_representable = 0.0
    for child, length, desc in tree.branches:
        is_repr = bool(desc & represented_tips)
        branch_child.append(child)
        branch_length.append(length)
        branch_represented.append(is_repr)
        if is_repr:
            pd_represented += length
        if desc & representable_tips:
            pd_representable += length

    pd_total = tree.total_pd
    return PDResult(
        pd_represented=pd_represented,
        pd_total=pd_total,
        pd_representable=pd_representable,
        fraction_pd_total=(pd_represented / pd_total if pd_total > 0 else 0.0),
        fraction_pd_representable=(
            pd_represented / pd_representable if pd_representable > 0 else 0.0
        ),
        n_tips=tree.n_tips,
        n_tips_represented=len(represented_tips),
        n_tips_unresolved=sum(1 for fid in resolved.values() if fid is None),
        branch_child=branch_child,
        branch_length=branch_length,
        branch_represented=branch_represented,
    )
