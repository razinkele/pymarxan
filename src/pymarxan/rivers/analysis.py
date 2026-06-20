"""River barrier-restoration analysis helpers (Phase E).

``budget_dci_frontier`` solves the barrier problem across a budget sweep to
produce the DCI-gain-vs-budget efficiency frontier practitioners ask for.
``barrier_selection_frequency`` runs many SA solves and ranks barriers by how
often they appear in good portfolios — a robust no-regret diagnostic, mirroring
the pattern of ``analysis.selection_freq`` (a membership count over each
``BarrierSolution.removed``, not an API call into that PU-shaped module).
"""
from __future__ import annotations

import pandas as pd

from pymarxan.rivers.barriers import BarrierProblem
from pymarxan.rivers.network import RiverNetwork
from pymarxan.rivers.optimize import (
    optimize_barriers_greedy,
    optimize_barriers_mip,
    optimize_barriers_sa,
)


def budget_dci_frontier(
    network: RiverNetwork,
    budgets,
    *,
    form: str = "diadromous",
    optimizer: str = "greedy",
    **kwargs,
) -> pd.DataFrame:
    """Solve the barrier problem at each budget and return the frontier.

    Parameters
    ----------
    network
        The river network.
    budgets
        Iterable of budget caps to evaluate.
    form
        ``"diadromous"`` or ``"potamodromous"``.
    optimizer
        ``"greedy"`` (default), ``"sa"``, or ``"mip"`` (exact; binary
        diadromous only). Extra ``kwargs`` are forwarded to the SA / MIP solver.

    Returns
    -------
    pandas.DataFrame
        One row per budget with columns ``budget``, ``dci_before``,
        ``dci_after``, ``gain``, ``cost``, ``n_removed``.
    """
    solvers = {
        "greedy": lambda p: optimize_barriers_greedy(p),
        "sa": lambda p: optimize_barriers_sa(p, **kwargs),
        "mip": lambda p: optimize_barriers_mip(p, **kwargs),
    }
    if optimizer not in solvers:
        raise ValueError(
            f"unknown optimizer {optimizer!r}; choose 'greedy', 'sa', or 'mip'"
        )
    run = solvers[optimizer]

    rows = []
    for b in budgets:
        sol = run(BarrierProblem(network, budget=b, form=form))
        rows.append(
            {
                "budget": b,
                "dci_before": sol.dci_before,
                "dci_after": sol.dci_after,
                "gain": sol.gain,
                "cost": sol.cost,
                "n_removed": len(sol.removed),
            }
        )
    return pd.DataFrame(rows)


def barrier_selection_frequency(
    network: RiverNetwork,
    budget: float | None,
    *,
    n_runs: int = 50,
    form: str = "diadromous",
    base_seed: int = 0,
    **kwargs,
) -> dict[int, float]:
    """Fraction of SA runs in which each barrier is selected for removal.

    Runs ``optimize_barriers_sa`` ``n_runs`` times with seeds
    ``base_seed + i`` and counts, per barrier, how often it appears in
    ``removed``. Returns ``{barrier_id: frequency}`` over **all** barriers
    (frequency in [0, 1]); deterministic for a given ``base_seed``.
    """
    counts = {bid: 0 for bid in network.barrier_passabilities()}
    problem = BarrierProblem(network, budget=budget, form=form)
    for i in range(n_runs):
        sol = optimize_barriers_sa(problem, seed=base_seed + i, **kwargs)
        for bid in sol.removed:
            counts[bid] += 1
    return {bid: counts[bid] / n_runs for bid in counts}
