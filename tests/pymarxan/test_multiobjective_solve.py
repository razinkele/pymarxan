"""MultiObjectiveMIP.solve (Stage 3 spec §3, §5, §6, §7.4 failure paths, §7.6)."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import pulp
import pytest

import pymarxan.multiobjective as mo
import pymarxan.solvers.mip_solver as ms
from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.multiobjective import (
    Hierarchical,
    MaxFeatures,
    MaxWeightedFeatures,
    MaxWtdSum,
    MinBoundary,
    MinCost,
    MinLargestShortfall,
    MinShortfall,
    MultiObjectiveMIP,
    ObjectiveSpec,
    WeightedSum,
)
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver
from pymarxan.zones import load_zone_project
from tests.pymarxan.scripted_solver import (
    Scripted,
    fail,
    not_solved_with_incumbent,
    optimal_without_solution,
)

DATA = Path(__file__).parent.parent / "data"


def simple(**params: Any) -> ConservationProblem:
    p = load_project(DATA / "simple")
    return p.copy_with(parameters={**p.parameters, **params})


def run(objectives: list[Any], approach: Any, problem: ConservationProblem,
        config: SolverConfig | None = None, **kw: Any) -> list[Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # PuLP 3.3.2
        return MultiObjectiveMIP(objectives, approach, mip_backend="cbc", **kw).solve(
            problem, config,
        )


def test_public_names() -> None:
    assert set(mo.__all__) == {
        "MultiObjectiveMIP", "Hierarchical", "WeightedSum", "ObjectiveSpec", "MinCost",
        "MinBoundary", "MinShortfall", "MinLargestShortfall", "MaxFeatures",
        "MaxWeightedFeatures", "MaxWtdSum", "weights_matrix", "rel_tol_matrix", "nondominated",
    }


# --- construction errors (spec §6) ---------------------------------------------------


def test_construction_errors() -> None:
    with pytest.raises(ValueError, match="objectives"):
        MultiObjectiveMIP([], Hierarchical(0.0))
    with pytest.raises(ValueError, match="duplicate.*min_shortfall"):
        MultiObjectiveMIP([MinShortfall(), MinShortfall(label="min_shortfall")], Hierarchical(0.0))
    with pytest.raises(TypeError, match="approach"):
        MultiObjectiveMIP([MinCost()], "hierarchical")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ObjectiveSpec"):
        MultiObjectiveMIP(["min_cost"], Hierarchical(0.0))  # type: ignore[list-item]
    with pytest.raises(TypeError, match="builder"):
        MultiObjectiveMIP([ObjectiveSpec()], Hierarchical(0.0))  # base class: no builder
    with pytest.raises(ValueError, match="rel_tol row 0 has 2"):
        MultiObjectiveMIP([MinCost(), MinShortfall()], Hierarchical([0.1, 0.1]))
    with pytest.raises(ValueError, match="weights row 0 has 3"):
        MultiObjectiveMIP([MinCost(), MinShortfall()], WeightedSum([1, 1, 1]))
    with pytest.raises(ValueError, match="mip_backend"):
        MultiObjectiveMIP([MinCost()], Hierarchical(0.0), mip_backend="nope")
    with pytest.raises(ValueError, match="mip_sep_strategy"):
        MultiObjectiveMIP([MinCost()], Hierarchical(0.0), mip_sep_strategy="socp")


def test_zonal_problem_is_rejected() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zp = load_zone_project(DATA / "zones")
    with pytest.raises(TypeError, match="ZonalProblem"):
        MultiObjectiveMIP([MinCost()], Hierarchical(0.0)).solve(zp)


def test_min_boundary_needs_a_boundary_table() -> None:
    p = simple()
    p = p.copy_with(boundary=None)
    with pytest.raises(ValueError, match="MinBoundary.*boundary"):
        MultiObjectiveMIP([MinBoundary()], Hierarchical(0.0)).solve(p)


def test_max_wtd_sum_weights_must_cover_the_features() -> None:
    with pytest.raises(ValueError, match=r"weights.*missing \[3\]"):
        run([MaxWtdSum(weights={1: 1, 2: 1})], Hierarchical(0.0), simple(COSTBUDGET=30))
    with pytest.raises(ValueError, match=r"weights.*unknown \[9\]"):
        run([MaxWtdSum(weights={1: 1, 2: 1, 3: 1, 9: 1})], Hierarchical(0.0),
            simple(COSTBUDGET=30))


# --- outputs (spec §5) ---------------------------------------------------------------


def test_hierarchical_metadata() -> None:
    sols = run([MinShortfall(), MinCost(label="cost")], Hierarchical([[0.0], [0.5]]),
               simple(COSTBUDGET=30.0), config=SolverConfig(num_solutions=7))
    assert len(sols) == 2  # one per rel_tol row; num_solutions is ignored
    meta = sols[1].metadata
    assert set(meta) == {
        "solver", "approach", "row", "rel_tol", "stage_reached", "status", "sol_status",
        "objective_values", "objective_senses", "mip_backend",
    }
    assert meta["solver"] == "MultiObjectiveMIP" and meta["approach"] == "hierarchical"
    assert (meta["row"], meta["rel_tol"], meta["stage_reached"]) == (1, [0.5], 2)
    assert (meta["status"], meta["sol_status"], meta["mip_backend"]) == ("Optimal", 1, "cbc")
    assert meta["objective_senses"] == {"min_shortfall": "min", "cost": "min"}
    assert set(meta["objective_values"]) == {"min_shortfall", "cost"}


def test_weighted_sum_metadata_and_solution_fields() -> None:
    p = simple(COSTBUDGET=30.0)
    sols = run([MinCost(), MaxFeatures()], WeightedSum([[1, 0], [1, 1]]), p)
    meta = sols[1].metadata
    assert set(meta) == {
        "solver", "approach", "row", "weights", "status", "sol_status",
        "objective_values", "objective_senses", "mip_backend",
    }
    assert (meta["approach"], meta["row"], meta["weights"]) == ("weighted_sum", 1, [1.0, 1.0])
    # Marxan-style fields exactly as MIPSolver fills them.
    ref = MIPSolver(mip_backend="cbc").solve(p, SolverConfig(num_solutions=1))[0]
    assert ref.metadata.keys() >= {"objective"} and "objective" not in meta
    s = sols[0]
    assert sols[0].metadata["objective_values"] == {"min_cost": 0.0, "max_features": 0.0}
    assert int(s.selected.sum()) == 0 and s.cost == 0.0 and s.boundary == 0.0


def test_costbudget_optional_and_honoured() -> None:
    unbounded = run([MinShortfall()], Hierarchical(0.0), simple())[0]
    assert unbounded.metadata["objective_values"]["min_shortfall"] == 0.0
    budgeted = run([MinShortfall(), MinCost()], Hierarchical(0.0), simple(COSTBUDGET=30.0))[0]
    assert budgeted.cost <= 30.0
    # Soft targets: at the best shortfall under the budget some target is unmet.
    assert budgeted.metadata["objective_values"]["min_shortfall"] > 0
    assert not all(budgeted.targets_met.values())


def test_min_boundary_is_non_trivial_at_blm_zero() -> None:
    p = simple(BLM=0.0, COSTBUDGET=45.0)
    alone = run([MinShortfall(), MinCost()], Hierarchical(0.0), p)[0]
    compact = run([MinShortfall(), MinBoundary(), MinCost()], Hierarchical(0.0), p)[0]
    vals = compact.metadata["objective_values"]
    assert vals["min_shortfall"] == alone.metadata["objective_values"]["min_shortfall"]
    assert vals["min_boundary"] == pytest.approx(compact.boundary)
    assert compact.boundary < alone.boundary


def test_two_min_largest_shortfall_labels_build() -> None:
    sols = run(
        [MinLargestShortfall(label="rel"), MinLargestShortfall("absolute", label="abs")],
        WeightedSum([1, 1]), simple(COSTBUDGET=30.0),
    )
    vals = sols[0].metadata["objective_values"]
    assert set(vals) == {"rel", "abs"} and vals["abs"] >= vals["rel"]


def test_presolve_runs_once_on_the_first_rows_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, float]] = []

    def spy(flag: Any, data: Any, model: pulp.LpProblem, name: str, *, stacklevel: int) -> None:
        seen.append({v.name: a for v, a in model.objective.items()})

    monkeypatch.setattr(mo, "_emit_presolve", spy)
    run([MinCost(), MaxFeatures()], WeightedSum([[3, 1], [1, 3]]), simple(COSTBUDGET=30.0),
        presolve_check=True)
    assert len(seen) == 1
    assert seen[0]["x_6"] == pytest.approx(0.75 * (8.0 + 2.0))  # (3/4)·(cost + BLM·self-edge)
    assert seen[0]["y_f1"] == pytest.approx(-0.25)  # row 0: −(1/4)·met


# --- failures (spec §4.1, §7.4, §7.5) -----------------------------------------------


def _scripted(monkeypatch: pytest.MonkeyPatch, solver: Scripted) -> None:
    # MultiObjectiveMIP builds its solver through mip_solver._mip_solver_for.
    monkeypatch.setattr(ms, "_make_pulp_solver", lambda *a, **k: solver)


def test_stage_1_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _scripted(monkeypatch, Scripted({1: fail}))
    with pytest.raises(RuntimeError, match=r"stage 1 \(min_shortfall\).*presolve_check"):
        run([MinShortfall(), MinCost()], Hierarchical(0.0), simple(COSTBUDGET=30.0))


def test_later_stage_failure_warns_and_returns_the_previous_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = Scripted({2: fail})
    _scripted(monkeypatch, solver)
    with pytest.warns(UserWarning, match="stage 2"):
        sols = run([MinShortfall(), MinCost()], Hierarchical([[0.0], [0.0]]),
                   simple(COSTBUDGET=30.0))
    first, second = sols
    assert first.metadata["stage_reached"] == 1
    # The restored stage-1 plan: best relative shortfall under COSTBUDGET 30 (the failed
    # stage left every x at 1, which would read as shortfall 0 at cost 83).
    assert first.metadata["objective_values"]["min_shortfall"] == pytest.approx(0.38333, abs=1e-4)
    assert first.cost <= 30.0
    assert (first.metadata["status"], first.metadata["sol_status"]) == ("Optimal", 1)
    # Row 2 is a fresh run: no cap from row 1 remains, and it reaches stage 2.
    assert second.metadata["stage_reached"] == 2
    assert not any(n.startswith("hier_cap_") for n in solver.models[2])
    assert "hier_cap_1" in solver.models[3]


def test_weighted_sum_row_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _scripted(monkeypatch, Scripted({2: fail}))
    with pytest.raises(RuntimeError, match="weighted-sum row 1"):
        run([MinCost(), MaxFeatures()], WeightedSum([[1, 1], [1, 2]]), simple(COSTBUDGET=30.0))


def test_infeasible_base_raises_for_hierarchical() -> None:
    p = simple(COSTBUDGET=30.0)
    pu = p.planning_units.copy()
    pu["status"] = 2  # every PU locked in: cost 83 > budget 30
    with pytest.raises(RuntimeError, match="stage 1"):
        run([MinCost()], Hierarchical(0.0), p.copy_with(planning_units=pu))


def test_ghost_feature_is_reported() -> None:
    p = simple(COSTBUDGET=30.0)
    ghost = pd.DataFrame({"id": [4], "name": ["g"], "target": [5.0], "spf": [1.0]})
    p = p.copy_with(features=pd.concat([p.features, ghost], ignore_index=True))
    sol = run([MaxWtdSum(weights={1: 1, 2: 1, 3: 1, 4: 5}), MinCost()], Hierarchical(0.0), p)[0]
    assert sol.metadata["objective_values"]["max_wtd_sum"] > 0


def test_unbounded_stage_says_so() -> None:
    p = simple()
    features = p.features.copy()
    features.loc[0, "spf"] = -1.0  # minimising −s/R: s grows without limit
    p = p.copy_with(features=features)
    with pytest.raises(RuntimeError, match="stage 1 .* is unbounded"):
        run([MinShortfall()], Hierarchical(0.0), p)
    with pytest.raises(RuntimeError, match="weighted-sum row 0 .* is unbounded"):
        run([MinShortfall()], WeightedSum([1]), p)


@pytest.mark.parametrize("approach", [Hierarchical(0.0), WeightedSum([1, 1])])
def test_acceptance_follows_sol_status(
    approach: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = [MinShortfall(), MinCost()]
    last = 2 if isinstance(approach, Hierarchical) else 1  # the last solve of row 0
    _scripted(monkeypatch, Scripted({last: not_solved_with_incumbent}))
    sol = run(specs, approach, simple(COSTBUDGET=30.0))[0]
    assert (sol.metadata["status"], sol.metadata["sol_status"]) == ("Not Solved", 2)
    _scripted(monkeypatch, Scripted({1: optimal_without_solution}))
    with pytest.raises(RuntimeError, match="found no solution"):
        run(specs, approach, simple(COSTBUDGET=30.0))


def test_weighted_sum_resets_values_before_each_row(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[set[Any]] = []

    class Blank(Scripted):
        def actualSolve(self, lp: pulp.LpProblem, **kwargs: Any) -> int:  # noqa: N802
            if self.calls == 0:
                return super().actualSolve(lp)
            self.calls += 1
            seen.append({v.varValue for v in lp.variables()})  # row 1, no inner solve
            lp.status, lp.sol_status = pulp.LpStatusNotSolved, pulp.LpSolutionNoSolutionFound
            return int(lp.status)

    _scripted(monkeypatch, Blank())
    with pytest.raises(RuntimeError, match="weighted-sum row 1"):
        run([MinCost(), MinShortfall()], WeightedSum([[1, 1], [1, 2]]), simple(COSTBUDGET=30.0))
    assert seen == [{None}]


def test_near_integral_x_is_rounded(monkeypatch: pytest.MonkeyPatch) -> None:
    specs = [MinShortfall(), MinCost()]
    p = simple(COSTBUDGET=30.0)
    exact = run(specs, WeightedSum([20, 1]), p)[0]
    assert exact.selected.any()  # fuzz a selected PU: PUs 2 and 6 (cost 23; MinCost 29 at BLM 1)

    def fuzz(lp: pulp.LpProblem) -> None:
        for v in lp.variables():
            if v.name.startswith("x_") and v.varValue is not None:
                v.varValue = 0.9999999 if v.varValue > 0.5 else 1e-7

    _scripted(monkeypatch, Scripted({1: fuzz}))
    fuzzy = run(specs, WeightedSum([20, 1]), p)[0]
    assert fuzzy.selected.tolist() == exact.selected.tolist()
    assert fuzzy.metadata["objective_values"] == exact.metadata["objective_values"]


def test_strategy_metadata_keys() -> None:
    p = simple()
    features = p.features.assign(target2=[1.0, 0, 0], sepnum=[2, 0, 0], sepdistance=[1.0, 0, 0])
    p = p.copy_with(features=features, parameters={**p.parameters, "PROBMODE": 3})
    meta = run([MinCost()], Hierarchical(0.0), p)[0].metadata
    assert {k: meta[k] for k in meta if k.endswith("_strategy")} == {
        "mip_chance_strategy": "drop", "mip_clump_strategy": "drop", "mip_sep_strategy": "drop",
    }
    plain = run([MinCost()], Hierarchical(0.0), simple())[0].metadata
    assert not any(k.endswith("_strategy") for k in plain)


def test_stage_1_failure_in_a_later_row_names_the_row(monkeypatch: pytest.MonkeyPatch) -> None:
    _scripted(monkeypatch, Scripted({3: fail}))  # row 1, stage 1
    with pytest.raises(RuntimeError, match=r"hierarchical row 1: stage 1 \(min_shortfall\)"):
        run([MinShortfall(), MinCost()], Hierarchical([[0.0], [0.0]]), simple(COSTBUDGET=30.0))


def test_single_objective_equals_mip_solver() -> None:
    """n = 1 is one solve; the Solution is filled exactly as MIPSolver fills it."""
    p = simple(COSTBUDGET=30.0)
    ref = MIPSolver(objective="min_shortfall", mip_backend="cbc").solve(
        p, SolverConfig(num_solutions=1),
    )[0]
    for approach in (Hierarchical(0.0), WeightedSum([1])):
        sol = run([MinShortfall()], approach, p)[0]
        assert sol.selected.any() and sol.selected.tolist() == ref.selected.tolist()
        assert (sol.cost, sol.boundary, sol.objective) == (ref.cost, ref.boundary, ref.objective)
        assert sol.targets_met == ref.targets_met
        assert sol.metadata["objective_values"]["min_shortfall"] == pytest.approx(
            ref.metadata["mip_objective_value"], abs=1e-9,
        )


def P(pu: dict[str, list[Any]], feat: dict[str, list[Any]], pvf: dict[str, list[Any]],
      **kw: Any) -> ConservationProblem:
    params = kw.pop("params", {})
    return ConservationProblem(
        planning_units=pd.DataFrame(pu), features=pd.DataFrame(feat),
        pu_vs_features=pd.DataFrame(pvf), parameters=params, **kw,
    )


def test_near_integral_budget_falls_back_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """Round 6 through MultiObjectiveMIP: v_rec's implied point breaks COSTBUDGET, so the
    cap must use v_model and stage 2 solves."""
    c1, eps = 1e7, 5e-8
    p = P({"id": [1, 2], "cost": [c1, 1.0], "status": [0, 0]},
          {"id": [1], "name": ["a"], "target": [10.0]},
          {"species": [1], "pu": [1], "amount": [10.0]},
          params={"COSTBUDGET": c1 * (1 - eps)})

    def near(lp: pulp.LpProblem) -> None:
        lp.sol_status = pulp.LpSolutionIntegerFeasible
        values = {"x_1": 1 - eps, "x_2": 0.0, "y_f1": 0.0}
        for v in lp.variables():
            if v.name in values:
                v.varValue = values[v.name]

    _scripted(monkeypatch, Scripted({1: near}))
    sol = run([MaxFeatures(), MinCost()], Hierarchical(0.0), p)[0]
    assert sol.metadata["stage_reached"] == 2


def test_earlier_caps_are_checked_end_to_end() -> None:
    """Round 7 through MultiObjectiveMIP: negative SPF on tolerance-boundary amounts."""
    p = P({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]},
          {"id": [1], "name": ["a"], "target": [0.8], "spf": [-1.0]},
          {"species": [1, 1], "pu": [1, 2], "amount": [0.7, 0.1]})
    sol = run([MaxFeatures(), MaxWeightedFeatures(), MinCost()], Hierarchical(0.0), p)[0]
    assert sol.metadata["stage_reached"] == 3


def test_hierarchical_passes_x_and_the_registry_as_tracked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[set[str]] = []
    real = mo.solve_hierarchical

    def spy(*args: Any, tracked: Any, **kwargs: Any) -> Any:
        tracked = list(tracked)
        seen.append({v.name for v in tracked})
        return real(*args, tracked=tracked, **kwargs)

    monkeypatch.setattr(mo, "solve_hierarchical", spy)
    p = simple(COSTBUDGET=30.0).copy_with(boundary=None)  # x_1 … x_6, no y_* auxiliaries
    run([MinShortfall(), MinCost()], Hierarchical(0.0), p)
    assert seen == [{f"x_{i}" for i in range(1, 7)} | {"s_f1", "s_f2", "s_f3"}]
