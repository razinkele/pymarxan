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


@pytest.fixture(scope="module")
def reedbeds():
    return _load("curonian_spit_reedbeds.py")


@pytest.fixture(scope="module")
def validation():
    return _load("validate_marxan_parity.py")


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


# --------------------------------------------------------------------
# validate_marxan_parity.py
# --------------------------------------------------------------------


def test_validation_exact_optimum_is_ground_truth(validation):
    """MIP finds the known optimum (35.0 on the six-unit project) and the
    heuristics meet targets without ever beating it."""
    problem = validation.load_simple_problem()
    reports = validation.solve_all(problem)
    by_name = {r.name: r for r in reports}

    mip = by_name["MIP (exact)"]
    assert mip.cost == pytest.approx(35.0)
    assert mip.targets_met

    for r in reports:
        assert r.targets_met, f"{r.name} left a target unmet"
        # No heuristic may come in below the exact optimum.
        assert r.cost >= mip.cost - 1e-6, f"{r.name} beat the exact optimum"


def test_validation_marxan_format_roundtrips(validation):
    problem = validation.load_simple_problem()
    assert validation.roundtrips_through_marxan_format(problem)


# --------------------------------------------------------------------
# curonian_spit_reedbeds.py
# --------------------------------------------------------------------


def test_reedbed_problem_is_a_linear_two_zone_chain(reedbeds):
    problem = reedbeds.build_reedbed_problem()
    n = reedbeds.N_SEGMENTS
    assert problem.n_planning_units == n
    assert dict(zip(problem.zones["id"], problem.zones["name"], strict=True)) == {
        reedbeds.CUT: "cut", reedbeds.LEAVE: "leave",
    }
    # Every segment must be either cut or left (no third "unmanaged" state).
    assert problem.parameters["MANDATORYALLOCATION"] == 1
    bnd = problem.boundary
    # A shoreline: each segment touches only the next one, no self-boundary rows.
    assert sorted(zip(bnd["id1"], bnd["id2"], strict=True)) == [(i, i + 1) for i in range(1, n)]
    # Unit edges: the zone MIP scales zone boundary costs by length, the
    # evaluator does not, so the example relies on every length being 1.
    assert (bnd["boundary"] == 1.0).all()
    assert problem.validate() == []


def test_reedbed_problem_is_deterministic(reedbeds):
    p1 = reedbeds.build_reedbed_problem()
    p2 = reedbeds.build_reedbed_problem()
    assert p1.zone_costs.equals(p2.zone_costs)
    assert p1.pu_vs_features.equals(p2.pu_vs_features)
    assert p1.zone_targets.equals(p2.zone_targets)


def test_edge_reward_is_a_negative_symmetric_zone_boundary_cost(reedbeds):
    zbc = reedbeds.build_reedbed_problem(edge_reward=1.5).zone_boundary_costs
    pairs = {(int(a), int(b)): float(c)
             for a, b, c in zip(zbc["zone1"], zbc["zone2"], zbc["cost"], strict=True)}
    assert pairs == {(reedbeds.CUT, reedbeds.LEAVE): -1.5, (reedbeds.LEAVE, reedbeds.CUT): -1.5}
    assert reedbeds.build_reedbed_problem(edge_reward=0.0).zone_boundary_costs is None


@pytest.mark.parametrize("edge_reward", [0.0, 0.4, 3.0])
def test_mosaic_mip_matches_brute_force_on_a_short_shore(reedbeds, edge_reward):
    """Exact optimum of a 10-segment shore by enumerating all 2^10 cut/leave plans."""
    import itertools

    import numpy as np

    from pymarxan.zones.objective import compute_zone_objective

    problem = reedbeds.build_reedbed_problem(n_segments=10, edge_reward=edge_reward)
    best = min(
        compute_zone_objective(problem, np.array(plan), problem.parameters["BLM"])
        for plan in itertools.product((reedbeds.CUT, reedbeds.LEAVE), repeat=10)
    )
    solution = reedbeds.solve_mosaic(problem)
    assert solution.objective == pytest.approx(best)


@pytest.mark.parametrize("same_zone", [(0.3, -0.7), (-0.5, 0.2)])
def test_mip_matches_brute_force_with_same_zone_costs(reedbeds, same_zone):
    """Same-zone pair costs (cut-cut, leave-leave) in the MIP match enumeration (#10)."""
    import itertools

    import numpy as np
    import pandas as pd

    from pymarxan.zones.objective import compute_zone_objective

    base = reedbeds.build_reedbed_problem(n_segments=10, edge_reward=0.4)
    cut_cut, leave_leave = same_zone
    zbc = pd.concat([base.zone_boundary_costs, pd.DataFrame({
        "zone1": [reedbeds.CUT, reedbeds.LEAVE], "zone2": [reedbeds.CUT, reedbeds.LEAVE],
        "cost": [cut_cut, leave_leave]})], ignore_index=True)
    problem = base.copy_with(zone_boundary_costs=zbc)
    best = min(
        compute_zone_objective(problem, np.array(plan), problem.parameters["BLM"])
        for plan in itertools.product((reedbeds.CUT, reedbeds.LEAVE), repeat=10)
    )
    assert reedbeds.solve_mosaic(problem).objective == pytest.approx(best)


def test_edge_reward_fragments_the_cutting(reedbeds):
    plain = reedbeds.summarize_mosaic(
        *_solved(reedbeds, reedbeds.build_reedbed_problem(edge_reward=0.0))
    )
    mosaic = reedbeds.summarize_mosaic(*_solved(reedbeds, reedbeds.build_reedbed_problem()))
    assert plain.zone_targets_met and mosaic.zone_targets_met
    # Without the reward the cheapest plan cuts a few solid blocks; with it the
    # shore becomes an interleaved cut/uncut mosaic.
    assert mosaic.edges > 2 * plain.edges
    assert mosaic.longest_uncut_km < plain.longest_uncut_km
    # Fragmentation is bought, not free: the reward-free plan is the cheapest.
    assert mosaic.cutting_cost >= plain.cutting_cost - 1e-9


def test_mip_objective_matches_the_zone_evaluator(reedbeds):
    problem, solution = _solved(reedbeds, reedbeds.build_reedbed_problem())
    assert solution.metadata["mip_objective_value"] == pytest.approx(solution.objective)


def test_reward_sweep_trades_cost_for_edges_monotonically(reedbeds):
    rows = reedbeds.reward_sweep([0.0, 0.5, 1.0, 2.0, 4.0])
    assert [r.edge_reward for r in rows] == [0.0, 0.5, 1.0, 2.0, 4.0]
    # Exact optima of cost - R * edges: edges and cost can only rise with R.
    for lo, hi in zip(rows, rows[1:]):
        assert hi.edges >= lo.edges
        assert hi.cutting_cost >= lo.cutting_cost - 1e-9
    assert rows[-1].edges > rows[0].edges


def test_reedbed_main_runs(reedbeds, capsys):
    reedbeds.main()
    out = capsys.readouterr().out
    assert "Curonian Spit" in out
    assert "edges" in out


def _solved(reedbeds, problem):
    return problem, reedbeds.solve_mosaic(problem)
