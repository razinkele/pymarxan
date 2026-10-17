"""Stage 4 (spec §6.1, §6.2, §8): analysis readers with fractional solutions."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from pymarxan.analysis.equity import compute_equity
from pymarxan.analysis.portfolio import selection_frequency, solution_diversity
from pymarxan.analysis.portfolio_cuts import generate_portfolio_cuts
from pymarxan.analysis.posthoc_clusters import compute_solution_clusters
from pymarxan.analysis.rank_importance import compute_rank_importance
from pymarxan.analysis.replacement_cost import compute_replacement_cost
from pymarxan.analysis.representation import compute_representation
from pymarxan.analysis.robustness import evaluate_plans_across_scenarios
from pymarxan.analysis.scenarios import ScenarioSet
from pymarxan.analysis.selection_freq import compute_selection_frequency
from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.phylo.diversity import compute_phylogenetic_diversity
from pymarxan.phylo.tree import PhylogeneticTree
from pymarxan.solvers import FractionalSolutionError, IterativeImprovementSolver, MIPSolver
from pymarxan.solvers.base import Solution, Solver, SolverConfig
from pymarxan.solvers.heuristic import HeuristicSolver
from pymarxan.solvers.run_mode import RunModePipeline
from pymarxan.solvers.utils import build_solution, compute_objective
from tests.pymarxan.fractional_fixtures import X, three_pu

DATA = Path(__file__).parent.parent.parent / "data"


@pytest.fixture
def simple() -> ConservationProblem:
    return load_project(DATA / "simple")


@pytest.fixture
def frac(simple: ConservationProblem) -> Solution:
    return build_solution(simple, np.array([0.3, 1.0, 0.0, 0.5, 0.1, 1.0]), 1.0)


# --- hand calculation (spec §8) -----------------------------------------------------


def test_representation_hand_calculation() -> None:
    p = three_pu()
    sol = build_solution(p, X, 1.0)
    rep = compute_representation(p, sol, threshold=0.3)
    # totals: f1 = 55, f2 = 12; held: f1 = 19, f2 = 8.8
    assert rep.represented_amount == pytest.approx({1: 19.0, 2: 8.8})
    assert rep.pct_represented == pytest.approx({1: 1900.0 / 55.0, 2: 880.0 / 12.0})
    assert rep.meets_threshold == {1: True, 2: True}


def test_equity_hand_calculation() -> None:
    p = three_pu()
    sol = build_solution(p, X, 1.0)
    eq = compute_equity(p, sol, {1: "a", 2: "a", 3: "b"})  # cost: a 0.6·10 + 0.2·20, b 30
    assert eq.group_values == pytest.approx({"a": 10.0, "b": 30.0})
    assert eq.group_shares == pytest.approx({"a": 0.25, "b": 0.75})
    count = compute_equity(p, sol, {1: "a", 2: "a", 3: "b"}, value="count")
    assert count.group_values == pytest.approx({"a": 0.8, "b": 1.0})


def test_binary_representation_and_equity_unchanged() -> None:
    p = three_pu()
    b = build_solution(p, X >= 0.5, 1.0)
    assert compute_representation(p, b).represented_amount == {1: 15.0, 2: 12.0}
    assert compute_equity(p, b, {1: "a", 2: "a", 3: "b"}).group_values == {"a": 10.0, "b": 30.0}


# --- frequencies --------------------------------------------------------------------


def test_selection_frequencies(simple: ConservationProblem, frac: Solution) -> None:
    b = build_solution(simple, frac.selected >= 0.5, 1.0)
    sf = compute_selection_frequency([frac, b])
    assert sf.counts.dtype == np.float64
    assert sf.counts.tolist() == pytest.approx([0.3, 2.0, 0.0, 1.5, 0.1, 2.0])
    assert sf.frequencies.tolist() == pytest.approx([0.15, 1.0, 0.0, 0.75, 0.05, 1.0])
    assert compute_selection_frequency([b, b]).counts.dtype.kind == "i"
    assert selection_frequency([frac, b]).tolist() == pytest.approx(sf.frequencies.tolist())


# --- robustness ---------------------------------------------------------------------


def test_robustness_no_silent_rounding(simple: ConservationProblem, frac: Solution) -> None:
    m, _, _ = evaluate_plans_across_scenarios({"s": simple}, {"p": frac}, blm=1.0)
    pu_index = {int(q): i for i, q in enumerate(simple.planning_units["id"])}
    assert m[0, 0] == pytest.approx(compute_objective(simple, frac.selected, pu_index, 1.0))
    assert m[0, 0] == pytest.approx(frac.objective)


def test_robustness_probmode3_scenario_rejects_fractional(
    simple: ConservationProblem, frac: Solution,
) -> None:
    pm3 = simple.copy_with(parameters={**simple.parameters, "PROBMODE": 3})
    with pytest.raises(ValueError, match="PROBMODE 3"):
        evaluate_plans_across_scenarios({"pm3": pm3}, {"p": frac})


# --- guards (spec §6.2) -------------------------------------------------------------


def test_guards_on_solutions(simple: ConservationProblem, frac: Solution) -> None:
    b = build_solution(simple, frac.selected >= 0.5, 1.0)
    with pytest.raises(FractionalSolutionError, match="solution_diversity"):
        solution_diversity([b, frac])
    scen = ScenarioSet()
    scen.add("a", b)
    scen.add("b", frac)
    with pytest.raises(FractionalSolutionError, match="overlap_matrix"):
        scen.overlap_matrix()
    with pytest.raises(FractionalSolutionError, match="compute_solution_clusters"):
        compute_solution_clusters(simple, frac)
    with pytest.raises(FractionalSolutionError, match="compute_rank_importance"):
        compute_rank_importance(simple, frac)
    tree = PhylogeneticTree.from_newick("((species_a:1,species_b:1):1,species_c:2);")
    with pytest.raises(FractionalSolutionError, match="compute_phylogenetic_diversity"):
        compute_phylogenetic_diversity(simple, frac, tree)
    with pytest.raises(FractionalSolutionError, match="improve"):
        IterativeImprovementSolver(itimptype=1).improve(simple, frac)
    with pytest.raises(FractionalSolutionError, match="improve"):
        IterativeImprovementSolver(itimptype=0).improve(simple, frac)


def test_binary_guarded_functions_still_work(simple: ConservationProblem, frac: Solution) -> None:
    b = build_solution(simple, frac.selected >= 0.5, 1.0)
    assert solution_diversity([b, b]) == 0.0
    scen = ScenarioSet()
    scen.add("a", b)
    assert scen.overlap_matrix().tolist() == [[1.0]]
    assert compute_solution_clusters(simple, b)["n_clusters"] >= 1


class _FractionalSA(Solver):
    """Stands in for a solver whose output feeds an improve() chain."""

    def solve(self, problem: ConservationProblem, config: SolverConfig | None = None
              ) -> list[Solution]:
        n = problem.n_planning_units
        return [build_solution(problem, np.full(n, 0.5), 0.0)]

    def name(self) -> str:
        return "fractional"

    def supports_zones(self) -> bool:
        return False


def test_run_mode_chain_propagates_guard(simple: ConservationProblem) -> None:
    with pytest.raises(FractionalSolutionError, match="improve"):
        RunModePipeline._run_pipeline(
            simple, 2, SolverConfig(num_solutions=1, seed=1), HeuristicSolver, _FractionalSA,
            IterativeImprovementSolver,
        )


@pytest.mark.parametrize("decisions", ["proportion", "semicontinuous"])
def test_solver_guards(simple: ConservationProblem, decisions: str) -> None:
    kw = {"lower": 0.3} if decisions == "semicontinuous" else {}
    solver = MIPSolver(decisions=decisions, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(FractionalSolutionError, match="generate_portfolio_cuts"):
            generate_portfolio_cuts(simple, solver=solver, k=2)
        with pytest.raises(FractionalSolutionError, match="compute_replacement_cost"):
            compute_replacement_cost(simple, solver=solver)


def test_require_binary_solver_public() -> None:
    """Review R7: the solver guard is public and re-exported."""
    from pymarxan.solvers import require_binary_solver

    require_binary_solver(MIPSolver(), "x")
    with pytest.raises(FractionalSolutionError, match="decisions='proportion'"):
        require_binary_solver(MIPSolver(decisions="proportion"), "x")

