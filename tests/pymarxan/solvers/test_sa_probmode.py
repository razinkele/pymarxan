"""Simulated annealing optimises PROBMODE 1 and 2 (v0.45 §6).

Pinned SA parameters (probed; do not change without re-probing): ``NUMITNS = 10_000``,
``NUMTEMP = 100``; best-of-10 tests run ``SolverConfig(num_solutions=10, seed=42)``,
consistency tests ``num_solutions=1, seed=42``.
"""
from __future__ import annotations

import itertools
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, SolverConfig
from pymarxan.solvers.cache import ProblemCache
from pymarxan.solvers.clumping import ClumpState, compute_clump_penalty_from_scratch
from pymarxan.solvers.iterative_improvement import IterativeImprovementSolver
from pymarxan.solvers.mip_solver import MIPSolver
from pymarxan.solvers.separation import SepState, compute_sep_penalty_from_scratch
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
from pymarxan.solvers.utils import build_solution, compute_feature_shortfalls

DATA = Path(__file__).parent.parent.parent / "data"
SA_PARAMS = {"NUMITNS": 10_000, "NUMTEMP": 100}
LINSPACE = pd.DataFrame({"pu": [1, 2, 3, 4, 5, 6], "probability": np.linspace(0.1, 0.6, 6)})
HALF_456 = pd.DataFrame({"pu": [4, 5, 6], "probability": [0.5, 0.5, 0.5]})


def _problem(
    probability: pd.DataFrame | None,
    mode: int | None,
    *,
    big_spf: bool = False,
    **params: Any,
) -> ConservationProblem:
    p = load_project(DATA / "simple")
    merged = {**p.parameters, **SA_PARAMS, **params}
    if mode is not None:
        merged["PROBMODE"] = mode
    features = p.features.copy()
    if big_spf:  # every spf >= 10 × Σ cost: the soft optimum meets every target
        features["spf"] = 10.0 * float(p.planning_units["cost"].sum())
    return p.copy_with(features=features, probability=probability, parameters=merged)


def _sa(problem: ConservationProblem, n: int) -> list[Solution]:
    return SimulatedAnnealingSolver().solve(problem, SolverConfig(num_solutions=n, seed=42))


def _tracked(sol: Solution) -> float:
    return float(sol.metadata["history"]["best_objective"][-1])


def _plans(n: int) -> list[np.ndarray]:
    return [np.array(bits, dtype=bool) for bits in itertools.product([0, 1], repeat=n)]


class TestConsistency:
    @pytest.mark.parametrize("mode", [0, 1, 2, None])
    def test_reported_objective_equals_tracked(self, mode: int | None) -> None:
        p = _problem(LINSPACE, mode, PROBABILITYWEIGHTING=2.0)
        (sol,) = _sa(p, 1)
        assert sol.objective == pytest.approx(_tracked(sol), rel=1e-9, abs=1e-9)


class TestMode2Oracle:
    def test_raw_amounts_meet_a_target_expected_do_not(self) -> None:
        """The fixture's premise: the 35.0 anchor {2, 4, 6} misses under expected amounts."""
        anchor = np.array([False, True, False, True, False, True])
        assert all(build_solution(_problem(None, None), anchor, 0.0).targets_met.values())
        assert not all(build_solution(_problem(HALF_456, 2), anchor, 0.0).targets_met.values())

    def test_feasible_optimum_is_50(self) -> None:
        p = _problem(HALF_456, 2, big_spf=True)
        c = ProblemCache.from_problem(p)
        blm = float(p.parameters["BLM"])
        best = min(
            (c.compute_full_objective(s, c.compute_held(s), blm), tuple(s.astype(int)))
            for s in _plans(c.n_pu)
        )
        assert best[0] == pytest.approx(50.0, abs=1e-9)
        assert best[1] == (1, 1, 1, 0, 0, 0)

    @pytest.mark.slow
    def test_sa_best_of_10_reaches_feasible_optimum(self) -> None:
        p = _problem(HALF_456, 2, big_spf=True)
        sols = _sa(p, 10)
        best = min(sols, key=lambda s: s.objective)
        assert best.objective == pytest.approx(50.0, abs=1e-9)
        assert best.selected.astype(int).tolist() == [1, 1, 1, 0, 0, 0]
        assert all(best.targets_met.values())


CROSS = [pytest.param(1, 69.2, id="mode1"), pytest.param(2, 74.0, id="mode2")]
# LINSPACE with PU 2 listed twice: the first row (0.9) must lose to the last (0.2) in the
# cache, the evaluator and the MIP alike (presolve's P0 only reports it).
LINSPACE_DUP = pd.concat(
    [pd.DataFrame({"pu": [2], "probability": [0.9]}), LINSPACE], ignore_index=True,
)


def _mip_value(p: ConservationProblem) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # PuLP 3.3.2
        (s,) = MIPSolver(mip_backend="cbc").solve(p, SolverConfig(num_solutions=1))
    return float(s.metadata["mip_objective_value"])


class TestCrossSolver:
    @pytest.mark.parametrize(("mode", "optimum"), CROSS)
    def test_cache_evaluator_and_mip_agree(self, mode: int, optimum: float) -> None:
        p = _problem(LINSPACE, mode, big_spf=True, PROBABILITYWEIGHTING=2.0)
        c = ProblemCache.from_problem(p)
        blm = float(p.parameters["BLM"])
        cache_best = min(
            (c.compute_full_objective(s, c.compute_held(s), blm), tuple(s)) for s in _plans(c.n_pu)
        )
        soft = build_solution(p, np.array(cache_best[1]), blm)
        assert all(soft.targets_met.values())  # the soft optimum is target-feasible
        feasible = [
            sol.objective
            for sol in (build_solution(p, s, blm) for s in _plans(c.n_pu))
            if all(sol.targets_met.values())
        ]
        assert cache_best[0] == pytest.approx(optimum, abs=1e-9)
        assert min(feasible) == pytest.approx(optimum, abs=1e-9)
        assert _mip_value(p) == pytest.approx(optimum, abs=1e-9)

    @pytest.mark.parametrize(("mode", "optimum"), CROSS)
    def test_duplicate_pu_last_row_wins_across_solvers(self, mode: int, optimum: float) -> None:
        p = _problem(LINSPACE_DUP, mode, big_spf=True, PROBABILITYWEIGHTING=2.0)
        c = ProblemCache.from_problem(p)
        blm = float(p.parameters["BLM"])
        cache_best = min(
            c.compute_full_objective(s, c.compute_held(s), blm) for s in _plans(c.n_pu)
        )
        feasible = [
            sol.objective
            for sol in (build_solution(p, s, blm) for s in _plans(c.n_pu))
            if all(sol.targets_met.values())
        ]
        assert cache_best == pytest.approx(optimum, abs=1e-9)  # as without the duplicate
        assert min(feasible) == pytest.approx(optimum, abs=1e-9)
        assert _mip_value(p) == pytest.approx(optimum, abs=1e-9)

    @pytest.mark.slow
    @pytest.mark.parametrize(("mode", "optimum"), CROSS)
    def test_sa_best_of_10_reaches_optimum(self, mode: int, optimum: float) -> None:
        p = _problem(LINSPACE, mode, big_spf=True, PROBABILITYWEIGHTING=2.0)
        sols = _sa(p, 10)
        assert min(s.objective for s in sols) == pytest.approx(optimum, abs=1e-9)


def _with_clumping(p: ConservationProblem) -> ConservationProblem:
    f = p.features.copy()
    f["target2"] = [10.0, 0.0, 0.0]
    f["clumptype"] = [0, 0, 0]
    return p.copy_with(features=f)


def _with_separation(p: ConservationProblem) -> ConservationProblem:
    pu = p.planning_units.copy()
    pu["xloc"] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    pu["yloc"] = 0.0
    f = p.features.copy()
    f["sepdistance"] = [0.0, 0.0, 2.0]
    f["sepnum"] = [1, 1, 2]
    return p.copy_with(planning_units=pu, features=f)


COMBOS = [
    pytest.param(_with_clumping, id="clumping", marks=pytest.mark.slow),  # ~2.3 s
    pytest.param(_with_separation, id="separation"),
]


_RAW_WARNING = "PROBMODE 2 expected amounts are not applied to clumping/separation"
pytestmark = pytest.mark.filterwarnings(f"ignore:{_RAW_WARNING}:UserWarning")


class TestCombinations:
    """PROBMODE 2 with clumping / separation: their amounts stay raw (documented; the cache
    build warns)."""

    def test_clump_state_matches_from_scratch(self) -> None:
        p = _with_clumping(_problem(HALF_456, 2))
        c = ProblemCache.from_problem(p)
        rng = np.random.default_rng(3)
        sel = rng.random(c.n_pu) < 0.5
        state = ClumpState.from_selection(c, sel)
        pen = compute_clump_penalty_from_scratch(p, sel)[1]
        for idx in rng.integers(c.n_pu, size=40):
            adding = not sel[idx]
            pen += state.delta_penalty(c, int(idx), adding)
            state.apply_flip(c, int(idx), adding)
            sel[idx] = adding
        assert pen == pytest.approx(compute_clump_penalty_from_scratch(p, sel)[1], abs=1e-9)

    def test_sep_state_matches_from_scratch(self) -> None:
        p = _with_separation(_problem(HALF_456, 2))
        c = ProblemCache.from_problem(p)
        rng = np.random.default_rng(4)
        sel = rng.random(c.n_pu) < 0.5
        state = SepState.from_selection(c, sel)
        for idx in rng.integers(c.n_pu, size=40):
            adding = not sel[idx]
            state.apply_flip(c, int(idx), adding)
            sel[idx] = adding
        assert state.penalty_total() == pytest.approx(
            compute_sep_penalty_from_scratch(p, sel)[1], abs=1e-9,
        )

    @pytest.mark.parametrize("combine", COMBOS)
    def test_sa_tracked_objective_identity(self, combine: Any) -> None:
        p = combine(_problem(HALF_456, 2))
        (sol,) = _sa(p, 1)
        pu_index = {int(u): i for i, u in enumerate(p.planning_units["id"])}
        shortfalls = compute_feature_shortfalls(p, sol.selected, pu_index)
        f = p.features
        active = (f.get("target2", 0) > 0) | (
            (f.get("sepnum", 1) > 1) & (f.get("sepdistance", 0) > 0)
        )
        removed = sum(
            float(spf) * shortfalls[int(fid)]
            for fid, spf, on in zip(f["id"], f["spf"], active)
            if on
        )
        want = sol.objective - removed + (sol.clump_penalty or 0.0) + (sol.sep_penalty or 0.0)
        assert _tracked(sol) == pytest.approx(want, rel=1e-9, abs=1e-9)


def _ii_premium_problem(mode: int | None) -> ConservationProblem:
    """PU 1 (cost 4, p 0.9) alone carries the feature (target 5, spf 5). Its removal saves
    the premium 10·0.9·4 = 36 at a penalty of 25: worth it only when the premium counts."""
    pu = pd.DataFrame({"id": [1, 2], "cost": [4.0, 1.0], "status": [0, 0]})
    feat = pd.DataFrame({"id": [1], "name": ["f"], "target": [5.0], "spf": [5.0]})
    puvspr = pd.DataFrame({"species": [1], "pu": [1], "amount": [5.0]})
    prob = pd.DataFrame({"pu": [1, 2], "probability": [0.9, 0.0]})
    params: dict[str, Any] = {"PROBABILITYWEIGHTING": 10.0, "BLM": 0.0}
    if mode is not None:
        params["PROBMODE"] = mode
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=puvspr, probability=prob,
        parameters=params,
    )


class TestIterativeImprovement:
    @pytest.mark.parametrize("mode", [1, None])
    @pytest.mark.parametrize("itimptype", [1, 2])
    def test_ii_optimises_the_premium(self, mode: int | None, itimptype: int) -> None:
        p = _ii_premium_problem(mode)
        (sol,) = IterativeImprovementSolver(itimptype=itimptype).solve(
            p, SolverConfig(num_solutions=1, seed=1),
        )
        assert sol.selected.tolist() == [False, False]  # premium 36 > penalty 25
        assert sol.objective == pytest.approx(25.0)
        assert sol.objective == pytest.approx(build_solution(p, sol.selected, 0.0).objective)

    @pytest.mark.parametrize("mode", [1, None])
    def test_swap_pass_optimises_the_premium(self, mode: int | None) -> None:
        """ITIMPTYPE 3 (swaps only) from {PU 1}: swapping in PU 2 gains 36 + 4 − 25 − 1."""
        p = _ii_premium_problem(mode)
        start = build_solution(p, np.array([True, False]), 0.0)
        sol = IterativeImprovementSolver(itimptype=3).improve(p, start)
        assert sol.selected.tolist() == [False, True]
        assert sol.objective == pytest.approx(26.0)
