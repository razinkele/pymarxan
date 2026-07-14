"""Tests for the ZonationSolver adapter (Phase B)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.zonation_solver import ZonationSolver


def _problem(q_rows, cost=None, status=None, feat_ids=(1, 2)) -> ConservationProblem:
    n_pu = len(q_rows)
    n_feat = len(q_rows[0])
    pu_ids = list(range(1, n_pu + 1))
    planning_units = pd.DataFrame(
        {
            "id": pu_ids,
            "cost": [1.0] * n_pu if cost is None else list(cost),
            "status": [0] * n_pu if status is None else list(status),
        }
    )
    features = pd.DataFrame(
        {
            "id": list(feat_ids),
            "name": [f"f{j}" for j in feat_ids],
            "target": [1.0] * n_feat,
            "spf": [1.0] * n_feat,
        }
    )
    rows = []
    for pu, row in zip(pu_ids, q_rows):
        for fid, amt in zip(feat_ids, row):
            if amt:
                rows.append({"species": fid, "pu": pu, "amount": float(amt)})
    pu_vs_features = pd.DataFrame(rows, columns=["species", "pu", "amount"])
    return ConservationProblem(planning_units, features, pu_vs_features)


P1 = [[10, 0], [0, 10], [5, 5]]  # CAZ ranks: PU3=1/3, PU1=2/3, PU2=1.0


def test_thresholded_reserve_hand_known():
    sol = ZonationSolver(rule="caz", top_fraction=2 / 3).solve(_problem(P1))[0]
    assert sol.selected.tolist() == [True, True, False]  # {PU1, PU2}
    assert sol.cost == pytest.approx(2.0)


def test_metadata_carries_ranking():
    sol = ZonationSolver(top_fraction=2 / 3).solve(_problem(P1))[0]
    assert sol.metadata["solver"] == "zonation"
    assert sol.metadata["rule"] == "caz"
    assert sol.metadata["priority_rank"][2] == pytest.approx(1.0)
    assert "prop_landscape_remaining" in sol.metadata["performance_curves"].columns


def test_build_solution_populated():
    sol = ZonationSolver(top_fraction=2 / 3).solve(_problem(P1))[0]
    assert len(sol.targets_met) == 2
    assert sol.all_targets_met  # {PU1, PU2} covers both features
    assert np.isfinite(sol.objective)


def test_top_fraction_controls_size_monotone():
    small = ZonationSolver(top_fraction=1 / 3).solve(_problem(P1))[0]
    big = ZonationSolver(top_fraction=1.0).solve(_problem(P1))[0]
    assert small.n_selected <= big.n_selected
    assert big.n_selected == 3  # top_fraction=1.0 selects all (no locks here)


def test_locks_enforced_at_high_top_fraction():
    # PU1 locked-in (2), PU2 locked-out (3), PU3 normal (0). Even at
    # top_fraction=1.0 (which by rank would select all), the reserve must
    # exclude the locked-out PU and include the locked-in one.
    sol = ZonationSolver(top_fraction=1.0).solve(_problem(P1, status=[2, 3, 0]))[0]
    assert bool(sol.selected[0]) is True    # PU1 locked-in → always selected
    assert bool(sol.selected[1]) is False   # PU2 locked-out → never selected


def test_deterministic_single_solution():
    sols = ZonationSolver().solve(_problem(P1), SolverConfig(num_solutions=5))
    assert len(sols) == 1


def test_abc_surface():
    s = ZonationSolver()
    assert s.name() == "Zonation (rank-removal)"
    assert s.supports_zones() is False
    assert s.available() is True
    # capability flags inherit True (build_solution reports gaps post-hoc);
    # pinned so a future change doesn't silently flip them to False.
    assert s.supports_probmode3() is True
    assert s.supports_clumping() is True


def test_invalid_rule_raises():
    with pytest.raises(ValueError, match="rule"):
        ZonationSolver(rule="bogus")


def test_invalid_top_fraction_raises():
    with pytest.raises(ValueError, match="top_fraction"):
        ZonationSolver(top_fraction=0.0)
    with pytest.raises(ValueError, match="top_fraction"):
        ZonationSolver(top_fraction=1.5)


def test_registered_in_default_registry():
    from pymarxan.solvers.registry import get_default_registry

    solver = get_default_registry().create("zonation")
    assert isinstance(solver, ZonationSolver)


def test_smoothing_passthrough_matches_engine():
    import numpy as np

    from pymarxan.zonation.rank_removal import rank_removal
    from pymarxan.zonation.smoothing import SmoothingSpec

    spec = SmoothingSpec(alpha=1.0, coords=np.array([[0.0], [1.0], [2.0]]))
    problem = _problem([[10], [0], [0]], feat_ids=(1,))
    sol = ZonationSolver(smoothing=spec, top_fraction=2 / 3).solve(problem)[0]
    engine_top = rank_removal(problem, smoothing=spec).top_fraction(2 / 3)
    selected_ids = {
        int(pid)
        for pid, s in zip(problem.planning_units["id"], sol.selected)
        if s
    }
    assert selected_ids == engine_top
    # provenance marker recorded in metadata
    assert sol.metadata["smoothed"] is True
    assert sol.metadata["smoothing_alpha"] == pytest.approx(1.0)
