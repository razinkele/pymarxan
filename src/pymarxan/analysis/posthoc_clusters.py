"""Post-hoc cluster diagnostics on a Solution (Phase 24).

Partitions the selected PUs of a Solution into connected components
using the problem's boundary graph. Useful for "is this reserve
fragmented?" diagnostics — a single big cluster typically scores well
under BLM; many tiny clusters indicate the solver gave up on
contiguity.

For problems without ``problem.boundary`` (no edges), every selected
PU is its own cluster (no adjacency information to merge them).
"""
from __future__ import annotations

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution


def compute_solution_clusters(
    problem: ConservationProblem,
    solution: Solution,
) -> dict:
    """Partition the selected PUs of ``solution`` into connected components
    using the problem's boundary edges.

    Parameters
    ----------
    problem
        Conservation problem with optional ``boundary`` DataFrame.
    solution
        Solution whose ``selected`` array is partitioned.

    Returns
    -------
    dict
        ``{"n_clusters", "cluster_sizes", "max_cluster_fraction",
        "pu_to_cluster"}``. ``pu_to_cluster`` maps each selected PU id to
        an integer cluster index in ``range(n_clusters)``; unselected
        PUs are absent. ``cluster_sizes`` is a list (one per cluster);
        ``max_cluster_fraction`` is ``max(sizes) / n_selected`` (or 0
        if nothing selected).
    """
    import networkx as nx

    pu_ids = problem.planning_units["id"].astype(int).to_numpy()
    pu_id_to_idx = {int(pid): i for i, pid in enumerate(pu_ids)}
    selected_ids = [
        int(pid) for i, pid in enumerate(pu_ids) if solution.selected[i]
    ]
    n_selected = len(selected_ids)

    if n_selected == 0:
        return {
            "n_clusters": 0,
            "cluster_sizes": [],
            "max_cluster_fraction": 0.0,
            "pu_to_cluster": {},
        }

    G = nx.Graph()
    G.add_nodes_from(selected_ids)

    if problem.boundary is not None:
        b_id1 = problem.boundary["id1"].astype(int).to_numpy()
        b_id2 = problem.boundary["id2"].astype(int).to_numpy()
        selected_set = set(selected_ids)
        for k in range(len(b_id1)):
            i1, i2 = int(b_id1[k]), int(b_id2[k])
            if i1 == i2:
                continue  # self-boundary, no edge
            # Only edges between selected PUs matter for cluster identity.
            if i1 in selected_set and i2 in selected_set:
                G.add_edge(i1, i2)

    components = list(nx.connected_components(G))
    cluster_sizes = sorted(
        (len(c) for c in components), reverse=True,
    )

    # Assign cluster indices in deterministic (size-descending) order.
    pu_to_cluster: dict[int, int] = {}
    for idx, comp in enumerate(
        sorted(components, key=lambda c: (-len(c), min(c)))
    ):
        for pid in comp:
            pu_to_cluster[int(pid)] = idx

    max_frac = float(cluster_sizes[0]) / float(n_selected) if cluster_sizes else 0.0

    # Suppress "unused" hint on pu_id_to_idx (kept for future callers
    # that may want PU-index mappings as well as id mappings).
    del pu_id_to_idx

    return {
        "n_clusters": len(components),
        "cluster_sizes": cluster_sizes,
        "max_cluster_fraction": max_frac,
        "pu_to_cluster": pu_to_cluster,
    }
