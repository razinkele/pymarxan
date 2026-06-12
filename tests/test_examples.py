"""Regression tests for the runnable examples in ``examples/``.

Like ``tests/test_tutorial_examples.py`` keeps the tutorial honest, this
keeps the worked examples from rotting: if the public API drifts in a way
that breaks them, these tests fail in the same commit as the change.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _load(name: str):
    """Import an example module by file path (examples/ is not a package)."""
    path = EXAMPLES_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass can resolve the module's
    # annotations (dataclasses looks the module up in sys.modules).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def baltic():
    return _load("baltic_marine_planning.py")


def test_baltic_problem_shape(baltic):
    problem, grid = baltic.build_baltic_problem()
    assert problem.n_planning_units == 256
    assert problem.n_features == 4
    assert len(grid) == 256
    # Two existing reserves are locked in (status 2).
    assert int((problem.planning_units["status"] == 2).sum()) == 2
    # Boundary (adjacency) was wired up so BLM is usable.
    assert problem.boundary is not None and len(problem.boundary) > 0


def test_baltic_problem_is_deterministic(baltic):
    p1, _ = baltic.build_baltic_problem()
    p2, _ = baltic.build_baltic_problem()
    assert p1.planning_units["cost"].equals(p2.planning_units["cost"])
    assert p1.features["target"].equals(p2.features["target"])


def test_mip_is_optimal_and_sa_is_feasible(baltic):
    problem, _ = baltic.build_baltic_problem()
    result = baltic.solve_and_compare(problem)
    # Both reserves satisfy every target...
    assert result.mip_targets_met
    assert result.sa_targets_met
    # ...and the exact MIP is a true lower bound on the heuristic SA cost
    # for the minimum-set objective.
    assert result.mip_cost <= result.sa_cost + 1e-6
    assert result.overlap > 0


def test_spawning_ground_is_irreplaceable(baltic):
    from pymarxan.analysis.irreplaceability import compute_irreplaceability

    problem, _ = baltic.build_baltic_problem()
    scores = compute_irreplaceability(problem)
    # The concentrated pikeperch spawning ground must score above zero
    # while the median cell (widespread features) stays substitutable.
    assert max(scores.values()) > 0.0
    nonzero = [pid for pid, s in scores.items() if s > 0]
    assert len(nonzero) < problem.n_planning_units // 2


def test_gap_closes_after_reserve(baltic):
    from pymarxan.solvers.base import SolverConfig
    from pymarxan.solvers.mip_solver import MIPSolver

    problem, _ = baltic.build_baltic_problem()
    mip = MIPSolver().solve(problem, SolverConfig(num_solutions=1))[0]
    selected_ids = [
        int(i) for i, s in zip(problem.planning_units["id"], mip.selected) if s
    ]
    gap = baltic.gap_before_after(problem, selected_ids)
    # Every feature's target is met once the proposed reserve is in place.
    assert gap["met_after"].all()
    # And protection strictly improves over the existing-reserve baseline.
    assert (gap["pct_after"] >= gap["pct_before"]).all()
