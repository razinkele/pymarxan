"""RecoveryResult / evaluate_recovery / success_sensitivity / caveat (R4, spec §3.7)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pymarxan.recovery import (
    NO_TIME_LAG,
    RECOVERY_CAVEAT,
    PassiveAction,
    build_recovery_problem,
    evaluate_recovery,
    recovery_caveat,
    solve_recovery,
    success_sensitivity,
)
from pymarxan.solvers.base import Solution
from pymarxan.zones import build_zone_solution
from tests.pymarxan.recovery.recovery_anchor import anchor_inputs
from tests.pymarxan.recovery.test_recovery_builder import _shared_unit_inputs

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

SPEC_CAVEAT = (
    "Restored amounts are expected values: potential × probability that restoration "
    "succeeds (passive: × efficacy η), judged at horizon H — often only 1–2-year "
    "establishment survival. Targets are met on average, not with any stated probability: a "
    "target met by many low-probability units may well fall short in practice. Restored "
    "habitat usually stays below the quality of existing habitat for decades; "
    "[no time lag is modelled | a maturity factor of … is applied]. Costs, η and the "
    "maturity factor are placeholders unless replaced with local evidence. This plan is a "
    "decision aid for stakeholder discussion, not a prescription."
)


def _plan(assignment: list[int], **kw: object):  # type: ignore[no-untyped-def]
    problem, mapping = build_recovery_problem(anchor_inputs(**kw))
    sol = build_zone_solution(problem, np.array(assignment), 0.0, solver_name="fixed")
    return evaluate_recovery(sol, problem, mapping)


def test_caveat_is_the_spec_text() -> None:
    assert RECOVERY_CAVEAT.replace("{maturity}", "[no time lag is modelled | a maturity "
                                   "factor of … is applied]") == SPEC_CAVEAT


def test_anchor_result() -> None:
    problem, mapping = build_recovery_problem(anchor_inputs())
    result = evaluate_recovery(solve_recovery(problem), problem, mapping)
    assert result.actions["action"].tolist() == [
        "protect", "no_action", "no_action", "restore_active",
    ]
    assert result.cost_by_action == {"protect": 1.0, "restore_active": 5.0}
    assert result.total_cost == 6.0 and result.targets_met
    assert (result.mode, result.method, result.solver) == ("targets", "mip", "Zone MIP (PuLP)")
    assert result.boundary_dropped is False
    row = result.habitats.iloc[0].to_dict()
    assert row["existing_kept"] == 10.0 and row["existing_target"] == 8.0
    assert row["restored_active"] == pytest.approx(9.6) and row["restored_passive"] == 0.0
    assert row["restore_target"] == pytest.approx(6.12)
    assert row["restore_shortfall"] == 0.0 and row["protect_shortfall"] == 0.0
    assert result.caveat == RECOVERY_CAVEAT.format(maturity=NO_TIME_LAG)


def test_table_leads_with_shortfall_and_never_sums_existing_and_restored() -> None:
    result = _plan([1, 0, 2, 3], passive=PassiveAction(cost_factor=0.2),
                   restore_target={1: 9.0})
    cols = result.habitats.columns.tolist()
    assert cols[:4] == ["habitat", "name", "restore_shortfall", "protect_shortfall"]
    assert not any("total" in c or "secured" in c for c in cols)
    row = result.habitats.iloc[0]
    assert row["restore_shortfall"] == pytest.approx(0.2) and not row["restore_met"]
    assert not result.targets_met


@pytest.mark.filterwarnings("ignore:habitat 'fen' has no restoration potential")
def test_rows_with_the_largest_relative_shortfall_come_first() -> None:
    import pandas as pd

    from pymarxan.models.problem import ConservationProblem
    from pymarxan.recovery import RecoveryInputs

    base = ConservationProblem(
        planning_units=pd.DataFrame({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]}),
        features=pd.DataFrame({"id": [1, 2], "name": ["fen", "bog"], "target": 0.0,
                               "spf": 1.0}),
        pu_vs_features=pd.DataFrame({"species": [1, 2], "pu": [1, 2], "amount": [4.0, 4.0]}),
        parameters={},
    )
    inputs = RecoveryInputs(
        base=base, potential=pd.DataFrame({"pu": [2], "feature": [2], "amount": [2.0]}),
        p_success=1.0, protect_prop=0.5, restore_prop=1.0,
    )
    problem, mapping = build_recovery_problem(inputs)
    sol = build_zone_solution(problem, np.array([1, 1]), 0.0, solver_name="fixed")
    result = evaluate_recovery(sol, problem, mapping)
    assert result.habitats["name"].tolist() == ["bog", "fen"]  # bog: restore target unmet


def test_existing_habitat_counts_when_restored() -> None:
    problem, mapping = build_recovery_problem(_shared_unit_inputs())
    result = evaluate_recovery(solve_recovery(problem), problem, mapping)
    row = result.habitats.iloc[0]
    assert result.actions["action"].tolist() == ["restore_active", "no_action", "no_action"]
    assert row["existing_kept"] == 10.0 and row["restored_active"] == pytest.approx(4.0)


def test_passive_amounts_and_cost_by_action() -> None:
    result = _plan([1, 0, 2, 3], passive=PassiveAction(), restore_target={1: 8.5})
    row = result.habitats.iloc[0]
    assert row["restored_active"] == pytest.approx(4.0)
    assert row["restored_passive"] == pytest.approx(4.8)
    assert result.cost_by_action == pytest.approx(
        {"protect": 1.0, "restore_active": 2.5, "restore_passive": 2.0}
    )
    assert result.targets_met


def test_success_sensitivity_rescored_without_resolving() -> None:
    result = _plan([1, 0, 2, 3], passive=PassiveAction(), restore_target={1: 8.5})
    sens = success_sensitivity(result)
    row = sens.iloc[0].to_dict()
    # PU 3 active (p 0.5, potential 8); PU 4 passive (p 0.8, potential 12, eta 0.5)
    assert row["restored_at_0.5"] == pytest.approx(2.0 + 2.4)
    assert row["restored_at_0.75"] == pytest.approx(3.0 + 3.6)
    assert row["restored_at_1"] == pytest.approx(4.0 + 4.8)
    assert row["restored_at_1.25"] == pytest.approx(5.0 + 6.0)  # PU 4: p capped at 1
    assert row["lowest_multiplier_met"] == 1.0
    h = result.habitats.iloc[0]
    assert row["restored_at_1"] == pytest.approx(h["restored_active"] + h["restored_passive"])


def test_success_sensitivity_caps_p_at_one_and_reports_nan_when_never_met() -> None:
    result = _plan([1, 0, 2, 3], passive=PassiveAction(), restore_target={1: 8.5})
    row = success_sensitivity(result, multipliers=(2.0, 0.5)).iloc[0].to_dict()
    assert row["restored_at_2"] == pytest.approx(8.0 + 6.0)
    assert row["lowest_multiplier_met"] == 2.0
    low = success_sensitivity(result, multipliers=(0.5, 0.75)).iloc[0]
    assert math.isnan(low["lowest_multiplier_met"])
    with pytest.raises(ValueError, match="multipliers"):
        success_sensitivity(result, multipliers=())


def test_maturity_factor_scales_restored_amounts_and_fills_the_caveat() -> None:
    result = _plan([1, 0, 0, 2], maturity={1: 0.5}, horizon_years=10.0)
    row = result.habitats.iloc[0]
    assert row["restored_active"] == pytest.approx(4.8)
    assert row["restore_target"] == pytest.approx(0.45 * 6.8)
    assert "a maturity factor of 0.5 is applied" in result.caveat
    assert NO_TIME_LAG not in result.caveat
    assert result.caveat == recovery_caveat(result.mapping)


def test_evaluate_needs_a_zone_solution() -> None:
    problem, mapping = build_recovery_problem(anchor_inputs())
    sol = Solution(selected=np.zeros(4, dtype=bool), cost=0.0, boundary=0.0, objective=0.0,
                   targets_met={})
    with pytest.raises(ValueError, match="zone solution"):
        evaluate_recovery(sol, problem, mapping)
