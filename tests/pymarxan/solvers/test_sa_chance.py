"""SA and iterative improvement optimise the 2D chance term through ``ChanceState``.

v0.47.0 (spec §3.5): the chance term left ``ProblemCache``'s full objective and delta; SA
builds a ``ChanceState`` per run, II per removal / addition pass, and the swap pass adds a
from-scratch chance term at both of its full-objective calls. The reported objective (the
evaluator's, via ``build_solution``) must equal what the solvers tracked.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.cache import ProblemCache
from pymarxan.solvers.chance_state import ChanceState
from pymarxan.solvers.iterative_improvement import IterativeImprovementSolver
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
from pymarxan.solvers.utils import build_solution


def _random_problem(seed: int, n_pu: int = 14, n_feat: int = 3) -> ConservationProblem:
    rng = np.random.default_rng(seed)
    pu = pd.DataFrame({
        "id": rng.permutation(np.arange(1, n_pu + 1) * 7),
        "cost": rng.uniform(1, 5, n_pu), "status": 0,
    })
    rows = [(f, int(pid)) for f in range(1, n_feat + 1) for pid in pu.id if rng.random() < 0.6]
    pv = pd.DataFrame(rows, columns=["species", "pu"])
    pv["amount"] = rng.uniform(1, 6, len(pv))
    q = rng.uniform(0.3, 0.95, len(pv))
    q[rng.random(len(pv)) < 0.2] = 1.0
    pv["prob"] = q
    tot = (pv.amount * pv.prob).groupby(pv.species).sum().values
    feat = pd.DataFrame({
        "id": range(1, n_feat + 1), "name": [f"f{j}" for j in range(n_feat)],
        "target": tot * 0.35, "spf": 3.0, "ptarget": [0.95, 0.7, 0.85][:n_feat],
    })
    b = [(int(a), int(c), 1.0) for a, c in zip(pu.id[:-1], pu.id[1:])]
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pv,
        boundary=pd.DataFrame(b, columns=["id1", "id2", "boundary"]),
        parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 20.0, "BLM": 0.3,
                    "NUMITNS": 20_000, "NUMTEMP": 100},
    )


def _keeps_needed_pu_problem() -> ConservationProblem:
    """4 PUs x 1 feature, each PU alone meets the deterministic target (amount 10, target 10)
    but the chance target needs all four (q = 0.6, ptarget 0.9): k = 4 PUs cost 4 with no
    chance penalty; k = 3 cost 3 + 50·(0.9 − P3)/0.9 ≈ 7.06 (SPF 100 rules out k = 0)."""
    pu = pd.DataFrame({"id": [1, 2, 3, 4], "cost": [1.0] * 4, "status": [0] * 4})
    feat = pd.DataFrame({"id": [1], "name": ["f"], "target": [10.0], "spf": [100.0],
                         "ptarget": [0.9]})
    pv = pd.DataFrame({"species": [1] * 4, "pu": [1, 2, 3, 4], "amount": [10.0] * 4,
                       "prob": [0.6] * 4})
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pv,
        parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 50.0, "BLM": 0.0},
    )


def _duplicate_rows_problem() -> ConservationProblem:
    """PU 1 (cost 1) lists the feature twice, (a = 3, q = 1) and (a = 2, q = 0.5); PU 2
    (cost 3) once, (a = 5, q = 0.95). Target 4, ptarget 0.6, weighting 10.

    Row by row (Marxan 3 and the evaluator) PU 1 alone has E = 4, V = 1, P = 0.5, objective
    1 + 10·(0.6 − 0.5)/0.6 ≈ 2.667: the optimum. PU 2 alone costs 3.0 (P ≈ 0.75 ≥ 0.6). With
    the rows merged (summed amount, last q) PU 1 read as E = 2.5, V = 6.25 (≈ 6.43), so SA
    returned PU 2.
    """
    pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 3.0], "status": [0, 0]})
    feat = pd.DataFrame({"id": [1], "name": ["f"], "target": [4.0], "spf": [10.0],
                         "ptarget": [0.6]})
    pv = pd.DataFrame({"species": [1, 1, 1], "pu": [1, 1, 2], "amount": [3.0, 2.0, 5.0],
                       "prob": [1.0, 0.5, 0.95]})
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pv,
        parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 10.0, "BLM": 0.0,
                    "NUMITNS": 2000, "NUMTEMP": 20},
    )


def _obj(problem: ConservationProblem, sel: np.ndarray) -> float:
    return build_solution(problem, sel, float(problem.parameters.get("BLM", 0.0))).objective


class TestSimulatedAnnealing:
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_reported_objective_equals_tracked(self, seed: int) -> None:
        problem = _random_problem(seed)
        (sol,) = SimulatedAnnealingSolver().solve(
            problem, SolverConfig(num_solutions=1, seed=seed),
        )
        assert sol.prob_penalty is not None
        assert sol.objective == pytest.approx(
            sol.metadata["history"]["best_objective"][-1], abs=1e-9,
        )

    def test_chance_term_is_optimised(self) -> None:
        problem = _keeps_needed_pu_problem().copy_with(
            parameters={**_keeps_needed_pu_problem().parameters,
                        "NUMITNS": 5000, "NUMTEMP": 50},
        )
        sols = SimulatedAnnealingSolver().solve(problem, SolverConfig(num_solutions=3, seed=4))
        for sol in sols:
            assert sol.selected.all()
            assert sol.objective == pytest.approx(4.0)

    def test_duplicate_rows_tracked_as_reported(self) -> None:
        """Duplicate (pu, species) rows: what SA tracks is what it reports, for every
        selection, so SA finds the reported optimum (PU 1 alone, ≈ 2.667)."""
        problem = _duplicate_rows_problem()
        cache = ProblemCache.from_problem(problem)
        for sel in itertools.product([False, True], repeat=2):
            x = np.array(sel)
            tracked = cache.compute_full_objective(x, cache.compute_held(x), blm=0.0)
            tracked += ChanceState.from_selection(cache, x).penalty_total()
            assert tracked == pytest.approx(_obj(problem, x), abs=1e-12)
        sols = SimulatedAnnealingSolver().solve(problem, SolverConfig(num_solutions=3, seed=7))
        for sol in sols:
            assert sol.selected.tolist() == [True, False]
            assert sol.objective == pytest.approx(1.0 + 10.0 * (0.6 - 0.5) / 0.6)

    def test_term_off_without_data_matches_mode_0(self) -> None:
        """Mode 3 with no puvspr `prob`: the term is off (silently, in this release) and SA
        runs exactly as mode 0."""
        p3 = _random_problem(5)
        p3 = p3.copy_with(pu_vs_features=p3.pu_vs_features.drop(columns=["prob"]))
        p0 = p3.copy_with(parameters={**p3.parameters, "PROBMODE": 0})
        assert ProblemCache.from_problem(p3).chance_terms == frozenset()
        cfg = SolverConfig(num_solutions=2, seed=9)
        s3 = SimulatedAnnealingSolver().solve(p3, cfg)
        s0 = SimulatedAnnealingSolver().solve(p0, cfg)
        for a, b in zip(s3, s0):
            assert a.selected.tolist() == b.selected.tolist()
            assert a.objective == b.objective
            assert a.prob_penalty is None


def _single_moves_do_not_improve(problem, sol, *, removals: bool, additions: bool) -> None:
    base = sol.objective
    for i in range(len(sol.selected)):
        if (sol.selected[i] and removals) or (not sol.selected[i] and additions):
            s2 = sol.selected.copy()
            s2[i] = not s2[i]
            assert _obj(problem, s2) >= base - 1e-9


class TestIterativeImprovement:
    @pytest.mark.parametrize("itimptype", [1, 2])
    def test_removal_pass_keeps_a_pu_the_chance_term_needs(self, itimptype: int) -> None:
        problem = _keeps_needed_pu_problem()
        (sol,) = IterativeImprovementSolver(itimptype=itimptype).solve(
            problem, SolverConfig(num_solutions=1, seed=1),
        )
        assert sol.selected.all()
        assert sol.objective == pytest.approx(4.0)

    @pytest.mark.parametrize("seed", range(4))
    @pytest.mark.parametrize("itimptype", [1, 2])
    def test_passes_stop_at_an_evaluator_local_optimum(self, itimptype: int, seed: int) -> None:
        """Reported == tracked: every move the pass rejected also fails under the
        evaluator's objective (the passes saw the evaluator's chance term)."""
        problem = _random_problem(10 + seed)
        (sol,) = IterativeImprovementSolver(itimptype=itimptype).solve(
            problem, SolverConfig(num_solutions=1),
        )
        assert sol.objective == pytest.approx(_obj(problem, sol.selected), abs=1e-12)
        _single_moves_do_not_improve(
            problem, sol, removals=True, additions=itimptype == 2,
        )

    @pytest.mark.parametrize("seed", range(3))
    def test_swap_pass_stops_at_an_evaluator_local_optimum(self, seed: int) -> None:
        problem = _random_problem(20 + seed, n_pu=10)
        rng = np.random.default_rng(seed)
        start = build_solution(problem, rng.random(10) < 0.5, problem.parameters["BLM"])
        sol = IterativeImprovementSolver(itimptype=3).improve(problem, start)
        assert sol.prob_penalty is not None
        base = sol.objective
        assert base <= start.objective + 1e-12
        sel = sol.selected
        for r, a in itertools.product(np.flatnonzero(sel), np.flatnonzero(~sel)):
            s2 = sel.copy()
            s2[r], s2[a] = False, True
            assert _obj(problem, s2) >= base - 1e-9

    def test_swap_pass_optimises_the_chance_term(self) -> None:
        """Swap a risky PU for a safe one that costs more: worth it only with the term.

        PU 1 (cost 1, q = 0.5) vs PU 2 (cost 2, q = 1.0); amount 10, target 5, ptarget 0.9,
        weight 50. Holding PU 1: P = sf((5 − 5)/5) = 0.5 → penalty 50·0.4/0.9 ≈ 22.2;
        holding PU 2: zero variance, "met" in this release → 0, objective 2.
        """
        pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 2.0], "status": [0, 0]})
        feat = pd.DataFrame({"id": [1], "name": ["f"], "target": [5.0], "spf": [1.0],
                             "ptarget": [0.9]})
        pv = pd.DataFrame({"species": [1, 1], "pu": [1, 2], "amount": [10.0, 10.0],
                           "prob": [0.5, 1.0]})
        problem = ConservationProblem(
            planning_units=pu, features=feat, pu_vs_features=pv,
            parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 50.0, "BLM": 0.0,
                        "PROBZEROVAR": 1},
        )
        start = build_solution(problem, np.array([True, False]), 0.0)
        sol = IterativeImprovementSolver(itimptype=3).improve(problem, start)
        assert sol.selected.tolist() == [False, True]
        assert sol.objective == pytest.approx(2.0)
