"""Tests for SpaceState — the incremental space/adequacy penalty companion (Phase B)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.adequacy import SpaceSpec, compute_space_held
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.space_state import SpaceState


def _line_problem(n=5, space_target=0.5, space_spf=None):
    ids = np.arange(1, n + 1)
    pu = pd.DataFrame({"id": ids, "cost": 1.0, "status": 0,
                       "xloc": np.arange(n, dtype=float), "yloc": 0.0})
    fcols: dict = {"id": [1], "name": ["a"], "target": [1.0], "spf": [1.0]}
    if space_target is not None:
        fcols["space_target"] = [space_target]
    if space_spf is not None:
        fcols["space_spf"] = [space_spf]
    feats = pd.DataFrame(fcols)
    pvf = pd.DataFrame({"species": 1, "pu": ids, "amount": 1.0})
    return ConservationProblem(pu, feats, pvf)


def _oracle_penalty(problem, selected, spec):
    held = compute_space_held(problem, selected, spec)
    pen = 0.0
    for _, r in problem.features.iterrows():
        t = float(r.get("space_target", 0.0) or 0.0)
        if t <= 0:
            continue
        spf = float(r.get("space_spf", r["spf"]))
        pen += spf * max(0.0, t - held[int(r["id"])])
    return pen


def test_penalty_total_matches_oracle():
    p = _line_problem(5, space_target=0.6)
    spec = SpaceSpec()
    sel = np.array([True, False, False, False, True])
    ss = SpaceState.from_problem(p, spec, sel)
    assert ss.active
    assert ss.penalty_total() == pytest.approx(_oracle_penalty(p, sel, spec))


def test_penalty_total_respects_space_spf():
    p = _line_problem(5, space_target=0.9, space_spf=7.0)
    spec = SpaceSpec()
    sel = np.array([True, False, False, False, False])
    ss = SpaceState.from_problem(p, spec, sel)
    assert ss.penalty_total() == pytest.approx(_oracle_penalty(p, sel, spec))


def test_delta_equals_recompute():
    p = _line_problem(6, space_target=0.7)
    spec = SpaceSpec()
    sel = np.array([True, False, True, False, False, True])
    ss = SpaceState.from_problem(p, spec, sel)
    for idx in range(6):
        adding = not sel[idx]
        after = sel.copy()
        after[idx] = adding
        expected = SpaceState.from_problem(p, spec, after).penalty_total() - ss.penalty_total()
        assert ss.delta_penalty(idx, adding) == pytest.approx(expected)


def test_apply_flip_consistent():
    p = _line_problem(6, space_target=0.7)
    spec = SpaceSpec()
    sel = np.array([True, False, False, False, False, True])
    ss = SpaceState.from_problem(p, spec, sel)
    cur = sel.copy()
    for idx in (1, 3, 5):
        adding = not cur[idx]
        ss.apply_flip(idx, adding)
        cur[idx] = adding
        fresh = SpaceState.from_problem(p, spec, cur)
        assert ss.penalty_total() == pytest.approx(fresh.penalty_total())
        assert ss.held_by_id()[1] == pytest.approx(fresh.held_by_id()[1])


def test_held_by_id_matches_measure():
    p = _line_problem(5, space_target=0.5)
    spec = SpaceSpec()
    sel = np.array([True, False, True, False, True])
    ss = SpaceState.from_problem(p, spec, sel)
    assert ss.held_by_id()[1] == pytest.approx(compute_space_held(p, sel, spec)[1])


def test_all_targets_met():
    p = _line_problem(3, space_target=0.5)
    spec = SpaceSpec()
    ss_full = SpaceState.from_problem(p, spec, np.array([True, True, True]))
    assert ss_full.all_targets_met()  # held 1.0 >= 0.5
    ss_centre = SpaceState.from_problem(p, spec, np.array([False, True, False]))
    assert not ss_centre.all_targets_met()  # held 0.0 < 0.5


def test_inactive_without_space_target():
    p = _line_problem(5, space_target=None)
    spec = SpaceSpec()
    sel = np.array([True, False, True, False, True])
    ss = SpaceState.from_problem(p, spec, sel)
    assert ss.active is False
    assert ss.penalty_total() == 0.0
    assert ss.delta_penalty(0, True) == 0.0
    assert ss.all_targets_met()  # vacuously true


# --- B2: SA integration + Solution reporting + supports_space ---

def test_sa_supports_space():
    from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
    assert SimulatedAnnealingSolver().supports_space() is True


def test_build_solution_reports_space():
    from pymarxan.solvers.utils import build_solution
    p = _line_problem(5, space_target=0.6)
    sol = build_solution(p, np.array([True, False, False, False, True]), blm=0.0)
    assert sol.space_held is not None and 1 in sol.space_held
    assert sol.space_penalty is not None


def test_build_solution_no_space_leaves_none():
    from pymarxan.solvers.utils import build_solution
    p = _line_problem(5, space_target=None)
    sol = build_solution(p, np.ones(5, bool), blm=0.0)
    assert sol.space_held is None
    assert sol.space_penalty is None


def test_sa_space_penalty_spreads_reserve():
    # amount target (1.0) is met by any single PU; a large space_spf should push SA to a
    # spread reserve (higher space_held) than the same solver on a no-space problem.
    from pymarxan.solvers.base import SolverConfig
    from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
    base = _line_problem(6, space_target=None)
    withspace = _line_problem(6, space_target=0.8, space_spf=50.0)
    cfg = SolverConfig(num_solutions=1, seed=1)
    solver = SimulatedAnnealingSolver(num_iterations=20_000, num_temp_steps=200)
    s_base = solver.solve(base, cfg)[0]
    s_space = solver.solve(withspace, cfg)[0]
    spec = SpaceSpec()
    h_base = compute_space_held(base, s_base.selected, spec)[1]
    h_space = compute_space_held(withspace, s_space.selected, spec)[1]
    assert h_space > h_base
    assert s_space.space_penalty is not None
    assert s_base.space_held is None
