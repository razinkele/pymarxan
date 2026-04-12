"""Tests for probability support in MIP and heuristic solvers."""
import pandas as pd

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.heuristic import HeuristicSolver
from pymarxan.solvers.mip_solver import MIPSolver


def _make_problem(prob_data=None, params=None):
    """3 PUs, 1 feature. PU 1 cheap, PU 2 cheap, PU 3 expensive.

    All PUs contribute 5.0 to feature 1, target=5.0 so only one PU needed.
    PU 1 and PU 2 have cost 1.0; PU 3 has cost 100.0.
    """
    pu = pd.DataFrame({
        "id": [1, 2, 3],
        "cost": [1.0, 1.0, 100.0],
        "status": [0, 0, 0],
    })
    feat = pd.DataFrame({
        "id": [1],
        "name": ["f1"],
        "target": [5.0],
        "spf": [1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 1],
        "pu": [1, 2, 3],
        "amount": [5.0, 5.0, 5.0],
    })
    probability = None
    if prob_data is not None:
        probability = pd.DataFrame(prob_data)
    problem = ConservationProblem(
        planning_units=pu,
        features=feat,
        pu_vs_features=puvspr,
        probability=probability,
    )
    if params:
        problem.parameters.update(params)
    return problem


def _solve_mip(problem):
    solver = MIPSolver()
    sols = solver.solve(problem, SolverConfig(num_solutions=1))
    assert len(sols) == 1
    return sols[0]


def _pu_selected(sol, problem, pu_id):
    """Check if a specific PU ID is in the solution."""
    ids = problem.planning_units["id"].tolist()
    idx = ids.index(pu_id)
    return bool(sol.selected[idx])


# ---- MIP Mode 1: risk premium ----

class TestMIPProbMode1:
    def test_high_prob_avoided(self):
        """With high PROBABILITYWEIGHTING, PU 1 (high prob) is avoided
        in favour of PU 2 (zero prob), even though both cost the same."""
        problem = _make_problem(
            prob_data={"pu": [1, 2, 3], "probability": [0.9, 0.0, 0.0]},
            params={"PROBMODE": 1, "PROBABILITYWEIGHTING": 100.0},
        )
        sol = _solve_mip(problem)
        assert not _pu_selected(sol, problem, 1), "PU 1 (high prob) should be avoided"
        assert _pu_selected(sol, problem, 2), "PU 2 (zero prob) should be selected"

    def test_low_weight_no_effect(self):
        """With zero PROBABILITYWEIGHTING, probability has no effect."""
        problem = _make_problem(
            prob_data={"pu": [1, 2, 3], "probability": [0.9, 0.0, 0.0]},
            params={"PROBMODE": 1, "PROBABILITYWEIGHTING": 0.0},
        )
        sol = _solve_mip(problem)
        # Either PU 1 or PU 2 is fine (both cost 1.0), just check targets met
        assert sol.all_targets_met


# ---- MIP Mode 2: persistence-adjusted amounts ----

class TestMIPProbMode2:
    def test_needs_more_pus(self):
        """Mode 2 discounts contributions, so one PU may not suffice.

        PU 1: amount=5, prob=0.9 → effective=0.5  (not enough for target=5)
        PU 2: amount=5, prob=0.0 → effective=5.0
        PU 3: amount=5, prob=0.0 → effective=5.0
        With discount, PU 2 alone meets target, PU 1 alone does not.
        """
        problem = _make_problem(
            prob_data={"pu": [1, 2, 3], "probability": [0.9, 0.0, 0.0]},
            params={"PROBMODE": 2},
        )
        sol = _solve_mip(problem)
        assert sol.all_targets_met
        # PU 2 (cheap, zero prob) must be selected
        assert _pu_selected(sol, problem, 2)

    def test_high_prob_compensated(self):
        """When all PUs have moderate probability, more PUs are needed."""
        # Each PU: amount=5, prob=0.7 → effective=1.5; target=5 → need ≥4 PUs
        # But we only have 3, each effective=1.5 → sum=4.5 < 5, so we need all 3
        # Actually 3*1.5=4.5 < 5. Let's use 4 PUs.
        pu = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "cost": [1.0, 1.0, 1.0, 1.0],
            "status": [0, 0, 0, 0],
        })
        feat = pd.DataFrame({
            "id": [1], "name": ["f1"], "target": [5.0], "spf": [1.0],
        })
        puvspr = pd.DataFrame({
            "species": [1, 1, 1, 1],
            "pu": [1, 2, 3, 4],
            "amount": [5.0, 5.0, 5.0, 5.0],
        })
        probability = pd.DataFrame({
            "pu": [1, 2, 3, 4],
            "probability": [0.7, 0.7, 0.7, 0.7],
        })
        problem = ConservationProblem(
            planning_units=pu, features=feat, pu_vs_features=puvspr,
            probability=probability,
        )
        problem.parameters["PROBMODE"] = 2
        sol = _solve_mip(problem)
        # effective per PU = 5 * 0.3 = 1.5; need ceil(5/1.5)=4 PUs
        n_selected = int(sol.selected.sum())
        assert n_selected >= 4, f"Expected ≥4 PUs selected, got {n_selected}"


# ---- MIP no probability ----

class TestMIPNoProb:
    def test_no_probability_data(self):
        """Without probability data, solver works normally."""
        problem = _make_problem()
        sol = _solve_mip(problem)
        assert sol.all_targets_met
        # Should pick one of the cheap PUs
        assert sol.cost <= 2.0


# ---- Heuristic with probability ----

class TestHeuristicProb:
    def test_mode1_prefers_low_prob(self):
        """Heuristic with Mode 1 avoids high-probability PU."""
        problem = _make_problem(
            prob_data={"pu": [1, 2, 3], "probability": [0.9, 0.0, 0.0]},
            params={"PROBMODE": 1, "PROBABILITYWEIGHTING": 100.0,
                    "HEURTYPE": 1},  # greedy cheapest
        )
        solver = HeuristicSolver(heurtype=1)
        sols = solver.solve(problem, SolverConfig(num_solutions=1, seed=42))
        sol = sols[0]
        assert not _pu_selected(sol, problem, 1), "PU 1 (high prob) should be avoided"
        assert _pu_selected(sol, problem, 2), "PU 2 (zero prob) should be selected"

    def test_mode2_uses_effective_amounts(self):
        """Heuristic with Mode 2 uses discounted amounts for target tracking."""
        problem = _make_problem(
            prob_data={"pu": [1, 2, 3], "probability": [0.9, 0.0, 0.0]},
            params={"PROBMODE": 2, "HEURTYPE": 0},  # richness
        )
        solver = HeuristicSolver(heurtype=0)
        sols = solver.solve(problem, SolverConfig(num_solutions=1, seed=42))
        sol = sols[0]
        # PU 2 alone has effective amount=5.0, meeting target
        assert _pu_selected(sol, problem, 2)

    def test_no_prob_unchanged(self):
        """Without probability, heuristic works normally."""
        problem = _make_problem()
        solver = HeuristicSolver(heurtype=1)
        sols = solver.solve(problem, SolverConfig(num_solutions=1, seed=42))
        sol = sols[0]
        assert sol.all_targets_met
