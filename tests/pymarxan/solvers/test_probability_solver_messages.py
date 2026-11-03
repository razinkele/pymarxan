"""Per-solver probability messages (v0.48.0, spec §3.1 / §3.5).

Greedy, Zonation and the zone solvers ignore the chance terms during optimisation; the
Marxan binary switches its own 1D / 2D terms on from the files; RunModePipeline says
"ignored" for RUNMODE 1 (heuristic only) and nothing for the RUNMODEs where SA or II
optimise the terms. Each message goes through the guarded emitter, so a nested solve (the
pipeline's inner heuristic) stays silent.
"""
from __future__ import annotations

import warnings

import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.heuristic import HeuristicSolver
from pymarxan.solvers.marxan_binary import MarxanBinarySolver, _binary_project
from pymarxan.solvers.run_mode import RunModePipeline
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
from pymarxan.solvers.zonation_solver import ZonationSolver
from pymarxan.zones.heuristic import ZoneHeuristicSolver
from pymarxan.zones.iterative_improvement import ZoneIISolver
from pymarxan.zones.mip_solver import ZoneMIPSolver
from pymarxan.zones.solver import ZoneSASolver

IGNORED = "ignores the chance terms"


def _problem(mode: int | None = 4, *, pu_prob=True, table=None) -> ConservationProblem:
    pu = pd.DataFrame({"id": [1, 2, 3, 4], "cost": [1.0, 2.0, 1.5, 3.0], "status": 0})
    if pu_prob:
        pu["prob"] = [0.1, 0.2, 0.0, 0.3]
    params = {"NUMITNS": 2000, "NUMTEMP": 20}
    if mode is not None:
        params["PROBMODE"] = mode
    return ConservationProblem(
        planning_units=pu,
        features=pd.DataFrame({"id": [1, 2], "name": ["a", "b"], "target": [4.0, 3.0],
                               "spf": 5.0, "ptarget1d": [0.9, -1.0]}),
        pu_vs_features=pd.DataFrame({"species": [1, 1, 2, 2], "pu": [1, 2, 3, 4],
                                     "amount": [5.0, 4.0, 3.0, 6.0]}),
        probability=table, parameters=params,
    )


def _record(fn) -> list[str]:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fn()
    return [str(w.message) for w in rec if issubclass(w.category, UserWarning)]


CFG = SolverConfig(num_solutions=3, seed=1)


@pytest.mark.parametrize("solver", [HeuristicSolver(), ZonationSolver()])
def test_greedy_and_zonation_say_they_ignore_the_terms(solver):
    msgs = _record(lambda: solver.solve(_problem(), CFG))
    assert sum(IGNORED in m for m in msgs) == 1


@pytest.mark.parametrize("solver", [HeuristicSolver(), ZonationSolver()])
def test_no_message_without_chance_terms(solver):
    assert not any(IGNORED in m for m in _record(lambda: solver.solve(_problem(0), CFG)))


def test_sa_says_nothing_extra():
    assert not any(IGNORED in m for m in _record(
        lambda: SimulatedAnnealingSolver().solve(_problem(), CFG)))


@pytest.mark.parametrize("cls", [ZoneSASolver, ZoneIISolver, ZoneHeuristicSolver,
                                 ZoneMIPSolver])
def test_zone_solvers_say_they_ignore_the_terms(cls):
    (m,) = cls()._probability_messages(_problem())
    assert IGNORED in m and "zone" in m.lower()
    assert cls()._probability_messages(_problem(0)) == []


def test_runmode_1_says_ignored_once_and_runmode_4_never():
    m1 = _record(lambda: RunModePipeline(runmode=1).solve(
        _problem(), SolverConfig(num_solutions=10, seed=1)))
    assert sum(IGNORED in m for m in m1) == 1
    m4 = _record(lambda: RunModePipeline(runmode=4).solve(
        _problem(), SolverConfig(num_solutions=10, seed=1)))
    assert not any(IGNORED in m for m in m4)  # the inner heuristic step stays silent


@pytest.mark.parametrize("runmode", [0, 2, 3, 4, 5, 6])
def test_other_runmodes_say_nothing(runmode):
    assert RunModePipeline(runmode=runmode)._probability_messages(_problem()) == []


def test_binary_says_it_switches_its_own_terms_on():
    (m,) = MarxanBinarySolver(binary_path="x")._probability_messages(_problem())
    assert "Marxan" in m and "pu.dat prob" in m
    assert MarxanBinarySolver(binary_path="x")._probability_messages(
        _problem(0, pu_prob=False)) == []


def test_binary_project_writes_prob_dat_loss_into_pu_dat():
    table = pd.DataFrame({"pu": [1, 2, 4], "probability": [0.1, 0.2, 0.3]})
    out = _binary_project(_problem(4, pu_prob=False, table=table))
    assert out.planning_units["prob"].tolist() == [0.1, 0.2, 0.0, 0.3]
    # pu.dat prob already present, or 1D not active: unchanged
    p = _problem(4)
    assert _binary_project(p).planning_units["prob"].tolist() == [0.1, 0.2, 0.0, 0.3]
    q = _problem(1, pu_prob=False, table=table)
    assert "prob" not in _binary_project(q).planning_units.columns
