"""Runnable copies of every code block in ``docs/TUTORIAL.md``.

This file exists so the tutorial doesn't rot. Every code block in
``docs/TUTORIAL.md`` corresponds to a function here; the tests below
exercise each one. If the public API changes in a way that breaks the
tutorial, this test suite fails — forcing the doc to be updated in the
same commit as the code change.

Convention: the function names match the section anchors used in the
tutorial markdown. Keep the bodies copy-pasteable so doc and test
stay in sync.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# --------------------------------------------------------------------
# Section: "Building a minimal problem"
# --------------------------------------------------------------------


def build_minimal_problem():
    from pymarxan.models.problem import ConservationProblem

    planning_units = pd.DataFrame({
        "id": list(range(1, 13)),       # 12 PUs
        "cost": [1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 1.0, 2.0, 1.0],
        "status": [0] * 12,
    })
    features = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["coral", "seagrass", "mangrove"],
        "target": [5.0, 4.0, 3.0],
        "spf": [1.0, 1.0, 1.0],
    })
    pu_vs_features = pd.DataFrame({
        # coral on PUs 1-6
        "species": [1] * 6 + [2] * 4 + [3] * 3,
        "pu":      [1, 2, 3, 4, 5, 6,
                    3, 4, 7, 8,
                    9, 10, 11],
        "amount":  [1.0] * 13,
    })
    return ConservationProblem(
        planning_units=planning_units,
        features=features,
        pu_vs_features=pu_vs_features,
    )


def test_section_minimal_problem():
    problem = build_minimal_problem()
    assert len(problem.planning_units) == 12
    assert len(problem.features) == 3


# --------------------------------------------------------------------
# Section: "Solving with the default min-set objective"
# --------------------------------------------------------------------


def test_section_default_solve():
    from pymarxan.solvers.base import SolverConfig
    from pymarxan.solvers.mip_solver import MIPSolver

    problem = build_minimal_problem()
    solver = MIPSolver()
    solutions = solver.solve(problem, SolverConfig(num_solutions=1))

    best = solutions[0]
    assert all(best.targets_met.values())
    # CBC is the auto-resolved default on machines without HiGHS.
    assert best.metadata["mip_backend"] in ("cbc", "highs")
    assert best.metadata["objective"] == "min_set"


# --------------------------------------------------------------------
# Section: "Choosing the MIP backend (Phase 21)"
# --------------------------------------------------------------------


def test_section_mip_backend():
    from pymarxan.solvers.mip_solver import (
        MIPSolver,
        _available_backends,
    )

    available = _available_backends()
    assert "cbc" in available
    # CBC always available; pick it explicitly when scripts must be
    # reproducible across machines.
    solver = MIPSolver(mip_backend="cbc")
    assert solver.mip_backend == "cbc"


# --------------------------------------------------------------------
# Section: "Importance scores (Phase 22)"
# --------------------------------------------------------------------


def test_section_importance_scores():
    from pymarxan.analysis.ferrier_importance import compute_ferrier_importance
    from pymarxan.analysis.rank_importance import compute_rank_importance
    from pymarxan.analysis.replacement_cost import compute_replacement_cost
    from pymarxan.solvers.base import SolverConfig
    from pymarxan.solvers.mip_solver import MIPSolver

    problem = build_minimal_problem()
    best = MIPSolver().solve(problem, SolverConfig(num_solutions=1))[0]

    ferrier = compute_ferrier_importance(problem)
    ranks = compute_rank_importance(problem, best)
    repl_cost = compute_replacement_cost(problem)

    # Every PU has a Ferrier score (closed form).
    assert set(ferrier.keys()) == set(range(1, 13))
    # Only selected PUs get a non-zero rank; unselected ones score 0.
    selected_ids = [int(pid) for pid, sel in zip(
        problem.planning_units["id"], best.selected,
    ) if sel]
    for pid in problem.planning_units["id"]:
        if pid in selected_ids:
            assert ranks[int(pid)] > 0
        else:
            assert ranks[int(pid)] == 0
    # Replacement cost finite (positive or zero).
    for v in repl_cost.values():
        assert v >= 0 or np.isinf(v)


# --------------------------------------------------------------------
# Section: "Alternative MIP objectives (Phase 23)"
# --------------------------------------------------------------------


def test_section_alternative_objectives():
    from pymarxan.solvers.base import SolverConfig
    from pymarxan.solvers.mip_solver import MIPSolver

    problem = build_minimal_problem()
    problem.parameters["COSTBUDGET"] = 5.0  # tight budget

    max_feat = MIPSolver(objective="max_features").solve(
        problem, SolverConfig(num_solutions=1),
    )[0]
    min_short = MIPSolver(objective="min_largest_shortfall").solve(
        problem, SolverConfig(num_solutions=1),
    )[0]
    min_pen = MIPSolver(objective="min_penalties").solve(
        problem, SolverConfig(num_solutions=1),
    )[0]

    # max_features respects the cost budget.
    assert max_feat.cost <= 5.0 + 1e-6
    # min_largest_shortfall respects the cost budget too.
    assert min_short.cost <= 5.0 + 1e-6
    # min_penalties prefers meeting targets over saving cost.
    assert min_pen.metadata["objective"] == "min_penalties"


# --------------------------------------------------------------------
# Section: "Connectivity metrics (Phase 24)"
# --------------------------------------------------------------------


def test_section_connectivity_metrics():
    from pymarxan.connectivity.io import (
        connectivity_to_boundary,
        connectivity_to_matrix,
    )
    from pymarxan.connectivity.metrics import (
        compute_donors,
        compute_pagerank_centrality,
        compute_recipients,
    )

    # Connectivity edge list — a chain 1 → 2 → 3 → 4 plus a side branch.
    edges = pd.DataFrame({
        "id1": [1, 2, 3, 2],
        "id2": [2, 3, 4, 5],
        "value": [1.0, 1.0, 1.0, 0.5],
    })
    # symmetric=False keeps the directed donor → recipient structure
    # (the default `symmetric=True` would make every donor also a
    # recipient of the same flow and the asymmetry tests below would
    # collapse).
    matrix = connectivity_to_matrix(
        edges, pu_ids=[1, 2, 3, 4, 5], symmetric=False,
    )
    pagerank = compute_pagerank_centrality(matrix)
    donors = compute_donors(matrix)
    recipients = compute_recipients(matrix)

    assert pagerank.sum() == pytest.approx(1.0, abs=1e-6)
    # Node 1 sends flow but receives none → donor; node 4 receives only.
    assert donors[0]
    assert recipients[3]

    # Convert to boundary so the BLM penalty can use it.
    boundary = connectivity_to_boundary(edges, scale=-1.0)
    assert list(boundary.columns) == ["id1", "id2", "boundary"]


# --------------------------------------------------------------------
# Section: "Solution portfolios via no-good cuts (Phase 25)"
# --------------------------------------------------------------------


def test_section_portfolio_cuts():
    from pymarxan.analysis.portfolio_cuts import generate_portfolio_cuts
    from pymarxan.solvers.base import SolverConfig
    from pymarxan.solvers.mip_solver import MIPSolver

    problem = build_minimal_problem()
    portfolio = generate_portfolio_cuts(
        problem,
        solver=MIPSolver(),
        k=3,
        config=SolverConfig(num_solutions=1),
    )

    assert 1 <= len(portfolio) <= 3
    # All distinct selections.
    seen = {tuple(bool(b) for b in s.selected) for s in portfolio}
    assert len(seen) == len(portfolio)


# --------------------------------------------------------------------
# Section: "BLM calibration with Pareto filter (Phase 25)"
# --------------------------------------------------------------------


def test_section_pareto_blm():
    from pymarxan.calibration.blm import calibrate_blm
    from pymarxan.calibration.pareto import pareto_frontier
    from pymarxan.solvers.base import SolverConfig
    from pymarxan.solvers.mip_solver import MIPSolver

    problem = build_minimal_problem()
    # Boundary edges between neighbouring PUs so BLM actually does something.
    boundary_rows = [
        {"id1": i, "id2": i + 1, "boundary": 1.0}
        for i in range(1, 12)
    ]
    problem.boundary = pd.DataFrame(boundary_rows)

    result = calibrate_blm(
        problem,
        solver=MIPSolver(),
        blm_values=[0.0, 0.5, 1.0, 2.0, 5.0],
        config=SolverConfig(num_solutions=1),
    )
    pareto = pareto_frontier(result)

    # Pareto frontier never has more points than the raw sweep.
    assert len(pareto.blm_values) <= len(result.blm_values)
    # Every Pareto point comes from the raw sweep.
    for blm in pareto.blm_values:
        assert blm in result.blm_values
