"""build_recovery_problem: tables, anchors (brute-force verified), status rules (R2)."""
from __future__ import annotations

import warnings

import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.recovery import (
    PassiveAction,
    RecoveryInputs,
    build_recovery_problem,
    existing_feature_id,
    potential_feature_id,
)
from pymarxan.solvers.base import SolverConfig
from pymarxan.zones import ZonalProblem, ZoneMIPSolver
from tests.pymarxan.recovery.recovery_anchor import anchor_inputs, brute, fingerprint


def _mip(problem: ZonalProblem) -> tuple[list[int], float]:
    p = problem.copy_with(parameters={**problem.parameters, "MIP_GAP": 0.0})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        sol = ZoneMIPSolver(mip_backend="cbc").solve(p, SolverConfig(num_solutions=1))[0]
    return sol.zone_assignment.tolist(), round(sol.cost, 9)


def test_feature_id_scheme() -> None:
    assert (existing_feature_id(1), potential_feature_id(1)) == (1, 2)
    assert (existing_feature_id(4), potential_feature_id(4)) == (7, 8)


def test_anchor_tables() -> None:
    problem, mapping = build_recovery_problem(anchor_inputs())
    assert problem.validate() == []
    assert problem.zones["id"].tolist() == [1, 2]
    assert problem.zones["name"].tolist() == ["protect", "restore_active"]
    assert problem.parameters["MANDATORYALLOCATION"] == 0
    assert problem.features["id"].tolist() == [1, 2]
    assert problem.features["name"].tolist() == ["wetland_existing", "wetland_potential"]
    assert problem.features["target"].tolist() == pytest.approx([8.0, 0.45 * 13.6])
    pv = problem.pu_vs_features.set_index(["species", "pu"])["amount"]
    assert pv.to_dict() == pytest.approx({(1, 1): 10.0, (1, 2): 6.0, (2, 3): 4.0, (2, 4): 9.6})
    con = problem.zone_contributions.set_index(["feature", "zone"])["contribution"].to_dict()
    assert con == {(1, 1): 1.0, (1, 2): 1.0, (2, 1): 0.0, (2, 2): 1.0}
    zc = problem.zone_costs.set_index(["pu", "zone"])["cost"].to_dict()
    # PU 1/2 have no potential: restoring them costs protect + restore (never chosen)
    assert zc == pytest.approx({(1, 1): 1.0, (1, 2): 3.5, (2, 1): 1.0, (2, 2): 3.5,
                                (3, 1): 1.0, (3, 2): 2.5, (4, 1): 2.0, (4, 2): 5.0})
    assert problem.pu_locks is None and problem.pu_zones is None
    assert mapping.zones == {0: "no_action", 1: "protect", 2: "restore_active"}
    h = mapping.habitats.iloc[0]
    assert (h["existing_feature"], h["potential_feature"]) == (1, 2)
    assert h["restore_max"] == pytest.approx(13.6)
    assert mapping.units["restorable"].tolist() == [False, False, True, True]


@pytest.mark.parametrize("kw", [{}, {"restore_target": {1: 9.0}}])
def test_rower_anchor_unique_optimum(kw: dict[str, object]) -> None:
    # rev-2 semantics: restore target 0.45 x 13.6 = 6.12 expected units (or 9.0 absolute);
    # PU 3 alone delivers 4.0, so PU 4 restored is still the cheapest plan: unchanged anchor.
    problem, _ = build_recovery_problem(anchor_inputs(**kw))
    ranked = brute(problem)
    assert ranked[0] == (6.0, (1, 0, 0, 2)) and ranked[1][0] > 6.0
    assert _mip(problem) == ([1, 0, 0, 2], 6.0)


# (passive cost factor, absolute restore target, optimum, cost) — brute-forced on the
# semantics independently of the builder. cost_factor 0.2 reproduces review cases b1-b3;
# 1.0 is the D2 default.
PASSIVE_CASES = [
    (0.2, 9.0, [1, 0, 0, 2], 6.0),
    (0.2, 6.5, [1, 0, 3, 3], 1.6),
    (0.2, 8.5, [1, 0, 2, 3], 3.9),
    (1.0, 9.0, [1, 0, 0, 2], 6.0),
    (1.0, 6.5, [1, 0, 3, 3], 4.0),
    (1.0, 8.5, [1, 0, 2, 3], 5.5),
]


@pytest.mark.parametrize("cost_factor,target,optimum,cost", PASSIVE_CASES)
def test_passive_cases(cost_factor: float, target: float, optimum: list[int],
                       cost: float) -> None:
    problem, mapping = build_recovery_problem(anchor_inputs(
        passive=PassiveAction(efficacy=0.5, cost_factor=cost_factor),
        restore_target={1: target},
    ))
    assert problem.zones["id"].tolist() == [1, 2, 3]
    con = problem.zone_contributions.set_index(["feature", "zone"])["contribution"].to_dict()
    assert con[(1, 3)] == 1.0 and con[(2, 3)] == 0.5
    ranked = brute(problem)
    assert ranked[0] == (cost, tuple(optimum)) and ranked[1][0] > cost
    assert _mip(problem) == (optimum, cost)
    assert mapping.zones[3] == "restore_passive"


def _shared_unit_inputs() -> RecoveryInputs:
    """U1 holds existing 10 AND potential 8; U2 existing 10; U3 potential 8. p 0.5."""
    base = ConservationProblem(
        planning_units=pd.DataFrame({"id": [1, 2, 3], "cost": [1.0] * 3, "status": [0] * 3}),
        features=pd.DataFrame({"id": [1], "name": ["meadow"], "target": [0.0], "spf": [1.0]}),
        pu_vs_features=pd.DataFrame({"species": [1, 1], "pu": [1, 2], "amount": [10.0, 10.0]}),
        parameters={},
    )
    return RecoveryInputs(
        base=base,
        potential=pd.DataFrame({"pu": [1, 3], "feature": [1, 1], "amount": [8.0, 8.0]}),
        p_success=0.5, protect_prop=0.5, restore_target={1: 4.0},
    )


def test_existing_habitat_is_kept_when_restored() -> None:
    # Restoring U1 keeps its 10 existing (review H2) and adds 4 expected: both targets at
    # 2.5. If restore wiped existing habitat, the best plan would cost 3.5.
    problem, _ = build_recovery_problem(_shared_unit_inputs())
    assert brute(problem)[0] == (2.5, (2, 0, 0))
    assert _mip(problem) == ([2, 0, 0], 2.5)


def test_status_2_without_potential_is_locked_to_protect() -> None:
    problem, mapping = build_recovery_problem(anchor_inputs(status=(0, 2, 0, 0), restore_prop=0.0))
    assert problem.pu_locks.values.tolist() == [[2, 1]]
    assert problem.pu_zones is None and problem.validate() == []
    assert _mip(problem) == ([1, 1, 0, 0], 2.0)
    assert mapping.units["status"].tolist() == [0, 2, 0, 0]


def test_status_2_with_potential_is_kept_protected_or_restored() -> None:
    problem, _ = build_recovery_problem(anchor_inputs(
        status=(0, 0, 2, 0), restore_prop=0.0, passive=PassiveAction(),
    ))
    assert problem.pu_zones.values.tolist() == [[3, 1], [3, 2], [3, 3]]
    rule = problem.pu_zone_rules()[2]
    assert rule.allowed == frozenset({1, 2, 3}) and not rule.may_be_unassigned
    cost = brute(problem)[0][0]
    assert cost == 2.0  # PU 3 protected (or passive at the same cost), never dropped
    assigned, mip_cost = _mip(problem)
    assert mip_cost == 2.0 and assigned[2] in (1, 3) and assigned[0] == 1


def test_status_3_units_are_excluded_from_restore_max() -> None:
    problem, mapping = build_recovery_problem(anchor_inputs(status=(0, 0, 0, 3), restore_prop=1.0))
    assert mapping.habitats.loc[0, "restore_max"] == pytest.approx(4.0)
    assert brute(problem)[0] == (3.5, (1, 0, 2, 0))
    assert _mip(problem) == ([1, 0, 2, 0], 3.5)


def test_status_1_is_treated_as_0() -> None:
    problem, _ = build_recovery_problem(anchor_inputs(status=(0, 1, 0, 0)))
    assert problem.planning_units["status"].tolist() == [0, 0, 0, 0]
    assert _mip(problem) == ([1, 0, 0, 2], 6.0)


def test_unit_without_potential_is_never_restored() -> None:
    # PU 3 loses its potential: only PU 4 can restore; PU 1/2 restore costs carry a surcharge
    inputs = anchor_inputs(
        potential=pd.DataFrame({"pu": [4], "feature": [1], "amount": [12.0]}),
        passive=PassiveAction(cost_factor=0.2), restore_target={1: 4.0},
    )
    problem, _ = build_recovery_problem(inputs)
    ranked = brute(problem)
    assert ranked[0] == (1.4, (1, 0, 0, 3))
    zc = problem.zone_costs.set_index(["pu", "zone"])["cost"]
    # restore on PU 3 now costs more than protecting it, so it is strictly dominated
    assert zc[(3, 2)] == pytest.approx(3.5) and zc[(3, 3)] == pytest.approx(1.2)


def test_non_contiguous_habitat_ids() -> None:
    base = ConservationProblem(
        planning_units=pd.DataFrame({"id": [10, 20], "cost": [1.0, 2.0], "status": [0, 0]}),
        features=pd.DataFrame({"id": [10, 3], "name": ["fen", "bog"], "target": 0.0, "spf": 1.0}),
        pu_vs_features=pd.DataFrame({"species": [3, 10], "pu": [10, 20], "amount": [5.0, 7.0]}),
        parameters={},
    )
    inputs = RecoveryInputs(
        base=base,
        potential=pd.DataFrame({"pu": [20, 10], "feature": [3, 10], "amount": [4.0, 6.0]}),
        p_success=0.5,
    )
    problem, mapping = build_recovery_problem(inputs)
    assert problem.features["id"].tolist() == [5, 6, 19, 20]
    assert problem.features["name"].tolist() == [
        "bog_existing", "bog_potential", "fen_existing", "fen_potential",
    ]
    assert mapping.habitats["habitat"].tolist() == [3, 10]
    assert problem.validate() == []


def test_build_is_deterministic_and_row_order_free() -> None:
    a, _ = build_recovery_problem(anchor_inputs(passive=PassiveAction()))
    b, _ = build_recovery_problem(anchor_inputs(passive=PassiveAction()))
    shuffled = anchor_inputs(
        passive=PassiveAction(),
        potential=pd.DataFrame({"pu": [4, 3], "feature": [1, 1], "amount": [12.0, 8.0]}),
    )
    c, _ = build_recovery_problem(shuffled)
    assert fingerprint(a) == fingerprint(b) == fingerprint(c)
