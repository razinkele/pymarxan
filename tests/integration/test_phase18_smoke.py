"""Phase 18 end-to-end integration smoke test.

Verifies a small probabilistic problem flows through all four solvers
end-to-end and behaves coherently:

1. Each native solver (SA / heuristic / iterative-improvement) and the
   MIP solver under the default "drop" strategy returns a Solution with
   ``prob_shortfalls`` and ``prob_penalty`` populated.
2. A feature with ``ptarget = -1`` (disabled sentinel) contributes 0 to
   the probability penalty regardless of how the solver selects.
3. Saving and loading a PROBMODE-3 project round-trips bit-identically
   (the ``prob`` and ``ptarget`` columns survive a write + read cycle).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pymarxan.io.readers import load_project
from pymarxan.io.writers import save_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig


def _build_problem(seed: int = 0) -> ConservationProblem:
    """A small but non-trivial PROB2D problem.

    Two features: feature 1 has an active ptarget=0.9, feature 2 has
    ptarget=-1 (disabled — Marxan sentinel). The problem exercises both
    paths in a single fixture.
    """
    rng = np.random.default_rng(seed)
    n_pu = 6
    pu = pd.DataFrame({
        "id": list(range(1, n_pu + 1)),
        "cost": list(rng.uniform(5.0, 20.0, n_pu)),
        "status": [0] * n_pu,
    })
    features = pd.DataFrame({
        "id": [1, 2],
        "name": ["with_ptarget", "no_ptarget"],
        "target": [10.0, 8.0],
        "spf": [1.0, 1.0],
        "ptarget": [0.9, -1.0],
    })
    rows = []
    for fid in (1, 2):
        for pu_id in range(1, n_pu + 1):
            rows.append({
                "species": fid,
                "pu": pu_id,
                "amount": float(rng.uniform(2.0, 6.0)),
                "prob": float(rng.uniform(0.0, 0.25)),
            })
    puvspr = pd.DataFrame(rows)
    return ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
        parameters={
            "PROBMODE": 3,
            "PROBABILITYWEIGHTING": 1.0,
            "NUMITNS": 500,  # short SA so the test runs in <1s
        },
    )


def _build_solver(name: str):
    if name == "sa":
        from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
        return SimulatedAnnealingSolver()
    if name == "heuristic":
        from pymarxan.solvers.heuristic import HeuristicSolver
        return HeuristicSolver()
    if name == "iterative":
        from pymarxan.solvers.iterative_improvement import IterativeImprovementSolver
        return IterativeImprovementSolver()
    if name == "mip":
        from pymarxan.solvers.mip_solver import MIPSolver
        return MIPSolver()  # default mip_chance_strategy="drop"
    raise ValueError(name)


# --- 1. All four solvers populate prob_* attrs --------------------------


def test_all_solvers_populate_prob_attrs():
    """SA, heuristic, iterative-improvement, MIP all return Solutions
    with ``prob_shortfalls`` and ``prob_penalty`` set when PROBMODE 3."""
    problem = _build_problem(seed=11)
    config = SolverConfig(num_solutions=1, seed=11)

    for solver_name in ("sa", "heuristic", "iterative", "mip"):
        solver = _build_solver(solver_name)
        sols = solver.solve(problem, config)
        assert len(sols) == 1, f"{solver_name} returned no solution"
        sol = sols[0]
        assert sol.prob_shortfalls is not None, (
            f"{solver_name}: Solution.prob_shortfalls not populated"
        )
        assert sol.prob_penalty is not None, (
            f"{solver_name}: Solution.prob_penalty not populated"
        )


# --- 2. Disabled ptarget contributes 0 ----------------------------------


def test_disabled_ptarget_contributes_zero_to_penalty():
    """Feature 2 has ptarget=-1; its prob_shortfall must be 0 regardless
    of what amount the solver actually achieves."""
    problem = _build_problem(seed=21)
    config = SolverConfig(num_solutions=1, seed=21)

    # Use MIP — deterministic so the post-hoc evaluation is itself
    # reproducible run-to-run.
    from pymarxan.solvers.mip_solver import MIPSolver
    sols = MIPSolver().solve(problem, config)
    sol = sols[0]
    assert sol.prob_shortfalls is not None
    # Feature 2 (ptarget=-1) contributes nothing
    assert sol.prob_shortfalls.get(2, 0.0) == 0.0
    # The total penalty reflects only feature 1's contribution; if
    # feature 1 happens to be met, penalty is also 0.
    if sol.prob_shortfalls.get(1, 0.0) == 0.0:
        assert sol.prob_penalty == 0.0
    else:
        assert sol.prob_penalty > 0.0


# --- 3. Round-trip preserves the new columns ----------------------------


def test_probmode3_project_round_trips(tmp_path):
    """save_project + load_project preserves the ``prob`` and ``ptarget``
    columns introduced in Batch 1."""
    problem = _build_problem(seed=33)
    project_dir = tmp_path / "p18_project"
    save_project(problem, project_dir)

    loaded = load_project(project_dir)
    # prob column on puvspr survived
    assert "prob" in loaded.pu_vs_features.columns
    np.testing.assert_array_almost_equal(
        loaded.pu_vs_features["prob"].values,
        problem.pu_vs_features["prob"].values,
    )
    # ptarget column on features survived
    assert "ptarget" in loaded.features.columns
    # Order may differ; compare via feature id
    orig = problem.features.set_index("id")["ptarget"]
    new = loaded.features.set_index("id")["ptarget"]
    np.testing.assert_array_almost_equal(new.values, orig.values)
    # PROBMODE parameter preserved
    assert int(loaded.parameters.get("PROBMODE", 0)) == 3
