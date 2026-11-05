"""Solving recovery problems: targets / budget modes, heuristic fallback, scaled spf,
registry (R3, spec §3.5 / §3.9)."""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.recovery import (
    SPF_SCALE,
    PassiveAction,
    RecoveryInputs,
    RecoverySizeWarning,
    build_recovery_problem,
    scaled_spf,
    solve_recovery,
)
from pymarxan.recovery import solve as solve_module
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.registry import get_default_registry
from pymarxan.zones import ZonalProblem
from pymarxan.zones.cache import ZoneProblemCache
from tests.pymarxan.recovery.recovery_anchor import anchor_inputs
from tests.pymarxan.recovery.test_recovery_builder import PASSIVE_CASES

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")
FAST = {"NUMITNS": 20_000, "NUMTEMP": 200}


def _anchor_problems() -> list[tuple[str, ZonalProblem, float]]:
    """(label, problem, MIP optimum) for every anchor."""
    out = []
    p, _ = build_recovery_problem(anchor_inputs())
    out.append(("anchor", p, 6.0))
    for cf, target, _, cost in PASSIVE_CASES:
        p, _ = build_recovery_problem(anchor_inputs(
            passive=PassiveAction(cost_factor=cf), restore_target={1: target},
        ))
        out.append((f"passive-{cf}-{target}", p, cost))
    return out


def test_scaled_spf_is_scale_times_dearest_cost_per_unit_supplied() -> None:
    problem, mapping = build_recovery_problem(anchor_inputs())
    assert mapping.spf_scaled
    # existing (feature 1): dearest supply = restore on PU 2 (surcharged 3.5 for 6) -> 0.5833
    # potential (feature 2): dearest = active on PU 3, 2.5 for 4.0 expected -> 0.625
    assert scaled_spf(problem) == pytest.approx({1: SPF_SCALE * 3.5 / 6, 2: SPF_SCALE * 0.625})
    assert problem.features["spf"].tolist() == pytest.approx(
        [SPF_SCALE * 3.5 / 6, SPF_SCALE * 0.625]
    )
    assert SPF_SCALE == 10.0


def test_explicit_spf_is_used_as_given() -> None:
    problem, mapping = build_recovery_problem(anchor_inputs(spf=3.0))
    assert problem.features["spf"].tolist() == [3.0, 3.0]
    assert not mapping.spf_scaled


@pytest.mark.parametrize("label,problem,optimum", _anchor_problems(), ids=lambda x: x
                         if isinstance(x, str) else "")
def test_targets_mode_mip_hits_the_anchor_optimum(label: str, problem: ZonalProblem,
                                                  optimum: float) -> None:
    sol = solve_recovery(problem, mode="targets", method="mip")
    assert sol.cost == pytest.approx(optimum)
    assert sol.all_targets_met
    assert sol.metadata["recovery_mode"] == "targets"
    assert sol.metadata["recovery_method"] == "mip"
    assert sol.metadata["boundary_dropped"] is False


@pytest.mark.slow
@pytest.mark.parametrize("label,problem,optimum", _anchor_problems(), ids=lambda x: x
                         if isinstance(x, str) else "")
def test_heuristic_on_the_anchors(label: str, problem: ZonalProblem, optimum: float) -> None:
    # spec §3.5 / review H7: heuristic >= the MIP optimum, or the shortfall is reported
    p = problem.copy_with(parameters={**problem.parameters, **FAST})
    sol = solve_recovery(p, method="heuristic", config=SolverConfig(num_solutions=5, seed=1))
    assert sol.metadata["recovery_method"] == "heuristic"
    if label == "passive-0.2-9.0":  # the one anchor whose penalised minimum is infeasible
        assert not sol.all_targets_met and sol.shortfall > 0 and sol.cost < optimum
    else:  # scaled spf: SA + II reach the exact optimum
        assert sol.all_targets_met and sol.cost == pytest.approx(optimum)


@pytest.mark.slow
def test_b1_penalised_minimum_is_infeasible_and_reported() -> None:
    # review H7 / b1 at passive cost factor 0.2: [1, 0, 2, 3] delivers 8.8 of 9.0 expected
    # for 3.9; its objective 3.9 + spf 6.25 x 0.2 = 5.15 is below the feasible optimum 6.0,
    # so SA + II rightly stop there. No finite proportional spf removes such traps
    # (integrality), hence the shortfall is reported, never hidden.
    _, problem, optimum = _anchor_problems()[1]
    p = problem.copy_with(parameters={**problem.parameters, **FAST})
    sol = solve_recovery(p, method="heuristic", config=SolverConfig(num_solutions=5, seed=1))
    assert sol.zone_assignment.tolist() == [1, 0, 2, 3]
    assert sol.cost == pytest.approx(3.9) and sol.cost < optimum
    assert not sol.all_targets_met
    assert sol.shortfall == pytest.approx(0.2)
    assert sol.objective == pytest.approx(3.9 + SPF_SCALE * 0.625 * 0.2)


def test_heuristic_is_pipeline_runmode_2_itimptype_3(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}
    real = solve_module.RunModePipeline.solve

    def spy(self, problem, config=None):  # type: ignore[no-untyped-def]
        seen.update(problem.parameters)
        return real(self, problem, config)

    monkeypatch.setattr(solve_module.RunModePipeline, "solve", spy)
    problem, _ = build_recovery_problem(anchor_inputs())
    solve_recovery(problem.copy_with(parameters={**problem.parameters, **FAST}),
                   method="heuristic", config=SolverConfig(num_solutions=1, seed=1))
    assert seen["RUNMODE"] == 2 and seen["ITIMPTYPE"] == 3


def test_budget_mode_best_plan_within_budget() -> None:
    # passive cf 1.0, restore target 8.5: optimum 5.5 meets both. Budget 4.0 cannot.
    problem, _ = build_recovery_problem(anchor_inputs(
        passive=PassiveAction(), restore_target={1: 8.5},
    ))
    sol = solve_recovery(problem, mode="budget", budget=4.0)
    assert sol.cost <= 4.0 + 1e-9
    assert sol.metadata["recovery_mode"] == "budget"
    assert sol.metadata["objective"] == "min_shortfall"
    assert not sol.all_targets_met
    full = solve_recovery(problem, mode="budget", budget=5.5)
    assert full.all_targets_met and full.cost == pytest.approx(5.5)


def test_budget_mode_flags_dropped_boundary() -> None:
    inputs = anchor_inputs()
    inputs.base = inputs.base.copy_with(
        boundary=pd.DataFrame({"id1": [1, 2, 3], "id2": [2, 3, 4], "boundary": [1.0] * 3}),
        parameters={"BLM": 0.5},
    )
    problem, _ = build_recovery_problem(inputs)
    with pytest.warns(UserWarning, match="ignores boundary penalties"):
        sol = solve_recovery(problem, mode="budget", budget=10.0)
    assert sol.metadata["boundary_dropped"] is True
    tgt = solve_recovery(problem, mode="targets")
    assert tgt.metadata["boundary_dropped"] is False


def test_budget_mode_needs_a_budget_and_the_mip() -> None:
    problem, _ = build_recovery_problem(anchor_inputs())
    with pytest.raises(ValueError, match="budget mode needs a budget"):
        solve_recovery(problem, mode="budget")
    with pytest.raises(ValueError, match="budget mode is solved by the zone MIP only"):
        solve_recovery(problem, mode="budget", budget=5.0, method="heuristic")
    with pytest.raises(ValueError, match="mode must be"):
        solve_recovery(problem, mode="cheapest")


def test_targets_mode_raises_when_the_mip_finds_no_plan() -> None:
    problem, _ = build_recovery_problem(anchor_inputs())
    # tamper: a target above what any plan reaches (the builder's check would refuse it)
    feats = problem.features.assign(target=[8.0, 50.0])
    with pytest.raises(ValueError, match="no plan meets every target"):
        solve_recovery(problem.copy_with(features=feats))


def test_size_guard_warns_above_the_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    problem, _ = build_recovery_problem(anchor_inputs())
    monkeypatch.setattr(solve_module, "RECOVERY_MIP_SIZE_WARNING", 7)  # 4 units x 2 zones = 8
    with pytest.warns(RecoverySizeWarning, match="8 unit-zone decisions"):
        solve_recovery(problem)
    assert solve_module.RECOVERY_MIP_SIZE_WARNING == 7
    monkeypatch.undo()
    assert solve_module.RECOVERY_MIP_SIZE_WARNING == 50_000


def test_registry_offers_zone_mip_and_pipeline_on_zone_problems() -> None:
    reg = get_default_registry()
    assert {"zone_mip", "zone_sa", "pipeline"} <= set(reg.list_solvers())
    problem, _ = build_recovery_problem(anchor_inputs())
    problem = problem.copy_with(parameters={**problem.parameters, "MIP_GAP": 0.0})
    sol = reg.create("zone_mip").solve(problem, SolverConfig(num_solutions=1))[0]
    assert sol.zone_assignment.tolist() == [1, 0, 0, 2]
    p2 = problem.copy_with(parameters={**problem.parameters, "RUNMODE": 2, "ITIMPTYPE": 3,
                                       **FAST})
    sol2 = reg.create("pipeline").solve(p2, SolverConfig(num_solutions=1, seed=1))[0]
    assert sol2.zone_assignment is not None


def _random_inputs(seed: int) -> RecoveryInputs:
    rng = np.random.default_rng(seed)
    n, n_hab = 7, 2
    ids = list(range(1, n + 1))
    base = ConservationProblem(
        planning_units=pd.DataFrame({"id": ids, "cost": rng.uniform(1, 5, n).round(2),
                                     "status": 0}),
        features=pd.DataFrame({"id": [1, 2], "name": ["a", "b"], "target": 0.0, "spf": 1.0}),
        pu_vs_features=pd.DataFrame(
            [{"species": h, "pu": i, "amount": round(float(rng.uniform(1, 10)), 2)}
             for i in ids for h in (1, 2) if rng.random() < 0.4],
            columns=["species", "pu", "amount"],
        ),
        parameters={},
    )
    pot = pd.DataFrame(
        [{"pu": i, "feature": h, "amount": round(float(rng.uniform(1, 10)), 2)}
         for i in ids for h in range(1, n_hab + 1) if rng.random() < 0.4],
        columns=["pu", "feature", "amount"],
    )
    return RecoveryInputs(
        base=base, potential=pot,
        p_success=pd.Series(rng.uniform(0.2, 0.9, n).round(2), index=ids),
        protect_prop=0.4, restore_prop=0.5, passive=PassiveAction(cost_factor=0.5),
    )


def _exact_optimum(problem: ZonalProblem) -> float:
    """Vectorised brute force (all units free): the cheapest assignment meeting every
    overall target. Used instead of the MIP: PuLP's bundled CBC 2.10.3 returns a
    suboptimal 'Optimal' on seed 2 (preprocess bug; CBC 2.10.13 is right)."""
    cache = ZoneProblemCache.from_zone_problem(problem)
    cols = np.array(list(itertools.product(range(cache.n_zones + 1), repeat=cache.n_pu)))
    rows = np.arange(cache.n_pu)
    cost = cache.zone_cost_matrix[rows, cols].sum(axis=1)
    held = np.einsum("if,kif->kf", cache.pu_feat_matrix, cache.contribution_matrix[cols])
    ok = (held >= cache.overall_target_vector * (1 - 1e-9)).all(axis=1)
    return float(cost[ok].min())


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(6))
def test_random_instances_heuristic_is_honest(seed: int) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem, _ = build_recovery_problem(_random_inputs(seed))
    optimum = _exact_optimum(problem)
    p = problem.copy_with(parameters={**problem.parameters, **FAST})
    heur = solve_recovery(p, method="heuristic", config=SolverConfig(num_solutions=3, seed=seed))
    assert heur.all_targets_met == (heur.shortfall == 0)
    if heur.all_targets_met:
        assert heur.cost >= optimum - 1e-9
    else:
        assert heur.shortfall > 0


def test_budget_below_the_locked_in_cost_names_the_floor() -> None:
    problem, _ = build_recovery_problem(anchor_inputs(status=(0, 2, 0, 0), restore_prop=0.0))
    with pytest.raises(ValueError, match=r"least cost any plan must pay \(1: locked-in"):
        solve_recovery(problem, mode="budget", budget=0.5)


def test_heuristic_returns_a_feasible_run_over_a_cheaper_infeasible_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The cheaper run misses the target; the dearer run meets it. Feasible-first must win.
    problem, _ = build_recovery_problem(anchor_inputs())
    n = len(problem.planning_units)
    infeasible = solve_module.Solution(
        selected=np.zeros(n, dtype=bool), cost=1.0, boundary=0.0, objective=1.0,
        targets_met={1: False, 2: True}, shortfall=0.5,
    )
    feasible = solve_module.Solution(
        selected=np.ones(n, dtype=bool), cost=5.0, boundary=0.0, objective=5.0,
        targets_met={1: True, 2: True}, shortfall=0.0,
    )

    def fake_runs(self, problem, config=None):  # type: ignore[no-untyped-def]
        return [infeasible, feasible]

    monkeypatch.setattr(solve_module.RunModePipeline, "solve", fake_runs)
    sol = solve_recovery(problem, method="heuristic")
    assert sol.all_targets_met
    assert sol.objective == pytest.approx(5.0)


@pytest.mark.parametrize("mode", ["budget", "targets"])
def test_budget_mode_pins_equal_weight_spf_and_targets_mode_keeps_scaled(
    monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    from pymarxan.zones.mip_solver import ZoneMIPSolver

    seen: list[ZonalProblem] = []
    real = ZoneMIPSolver.solve

    def spy(self, problem, config=None):  # type: ignore[no-untyped-def]
        seen.append(problem)
        return real(self, problem, config)

    monkeypatch.setattr(ZoneMIPSolver, "solve", spy)
    problem, _ = build_recovery_problem(anchor_inputs(
        passive=PassiveAction(), restore_target={1: 8.5},
    ))
    if mode == "budget":
        solve_recovery(problem, mode="budget", budget=5.5)
        assert seen[-1].features["spf"].tolist() == [1.0, 1.0]
    else:
        solve_recovery(problem)
        assert seen[-1].features["spf"].tolist() == pytest.approx(
            [SPF_SCALE * 3.5 / 6, SPF_SCALE * 0.625]
        )


def test_budget_without_budget_mode_is_rejected() -> None:
    problem, _ = build_recovery_problem(anchor_inputs())
    with pytest.raises(ValueError, match='mode="budget"'):
        solve_recovery(problem, budget=10)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_budget_is_rejected(bad: float) -> None:
    problem, _ = build_recovery_problem(anchor_inputs())
    with pytest.raises(ValueError, match="finite"):
        solve_recovery(problem, mode="budget", budget=bad)
