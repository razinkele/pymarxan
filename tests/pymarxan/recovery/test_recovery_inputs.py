"""RecoveryInputs.validate / check_recovery_inputs (spec §3.1) — R2."""
from __future__ import annotations

import warnings

import pandas as pd
import pytest

from pymarxan.recovery import (
    PassiveAction,
    RecoveryInputError,
    RecoveryInputWarning,
    build_recovery_problem,
    check_recovery_inputs,
)
from tests.pymarxan.recovery.recovery_anchor import anchor_base, anchor_inputs


def _errors(**kw: object) -> list[str]:
    return anchor_inputs(**kw).validate()


def test_anchor_is_valid_and_has_no_warnings() -> None:
    assert anchor_inputs().validate() == []
    assert check_recovery_inputs(anchor_inputs()) == ([], [])


@pytest.mark.parametrize("kw,needle", [
    ({"potential": pd.DataFrame({"pu": [9], "feature": [1], "amount": [1.0]})},
     "potential names unknown units 9"),
    ({"potential": pd.DataFrame({"pu": [3], "feature": [7], "amount": [1.0]})},
     "potential names unknown habitats 7"),
    ({"potential": pd.DataFrame({"pu": [3], "feature": [1], "amount": [-1.0]})},
     "potential amounts must be finite and >= 0"),
    ({"potential": pd.DataFrame({"pu": [3, 3], "feature": [1, 1], "amount": [1.0, 2.0]})},
     "duplicate (pu, feature)"),
    ({"p_success": 1.2}, "p_success must be in [0, 1]"),
    ({"p_success": pd.Series({1: 0.5, 2: 0.5, 3: 0.5})}, "p_success has no value for units 4"),
    ({"passive": PassiveAction(efficacy=0.0)}, "passive efficacy must be in (0, 1]"),
    ({"passive": PassiveAction(efficacy_by_habitat={1: 1.5})},
     "passive efficacy_by_habitat[1] = 1.5 must be in (0, 1]"),
    ({"maturity": {1: 0.0}, "horizon_years": 10.0}, "maturity[1] = 0.0 must be in (0, 1]"),
    ({"maturity": {1: 0.6}}, "horizon_years must be stated"),
    ({"restore_cost": pd.Series({1: 1.0, 2: 1.0, 3: 0.0, 4: 1.0})},
     "restore cost must be > 0 in every unit"),
    ({"passive": PassiveAction(cost_factor=2.5)}, "passive recovery is never chosen there"),
    ({"protect_prop": 1.5}, "protect_prop must be in [0, 1]"),
    ({"restore_target": {2: 1.0}}, "restore_target names unknown habitats 2"),
    ({"spf": 0.0}, "spf must be > 0"),
])
def test_validate_reports(kw: dict[str, object], needle: str) -> None:
    errors = _errors(**kw)
    assert any(needle in e for e in errors), errors


def test_zero_cost_unit_makes_factor_restore_cost_zero() -> None:
    base = anchor_base()
    base = base.copy_with(planning_units=base.planning_units.assign(cost=[0.0, 1.0, 1.0, 2.0]))
    inputs = anchor_inputs()
    inputs.base = base
    assert any("restore cost must be > 0 in every unit" in e and "units 1" in e
               for e in inputs.validate())


def test_habitat_ids_must_be_positive_integers() -> None:
    inputs = anchor_inputs()
    inputs.base = inputs.base.copy_with(features=inputs.base.features.assign(id=[0]))
    assert any("must be a positive integer" in e for e in inputs.validate())


def test_infeasible_restore_target_is_an_error_naming_the_habitat() -> None:
    check = check_recovery_inputs(anchor_inputs(restore_target={1: 14.0}))
    assert len(check.errors) == 1
    msg = check.errors[0]
    assert msg.startswith("restore target for habitat 'wetland' (expected units)")
    assert "is 14 but at most 13.6 is reachable" in msg
    with pytest.raises(RecoveryInputError) as exc:
        build_recovery_problem(anchor_inputs(restore_target={1: 14.0}))
    assert exc.value.errors == check.errors


def test_restore_prop_one_is_feasible_by_construction() -> None:
    assert check_recovery_inputs(anchor_inputs(restore_prop=1.0)).errors == []


def test_habitat_without_potential_warns() -> None:
    inputs = anchor_inputs(potential=pd.DataFrame(columns=["pu", "feature", "amount"]))
    check = check_recovery_inputs(inputs)
    assert check.errors == []
    assert check.warnings == [
        "habitat 'wetland' has no restoration potential (with p_success > 0 in a unit that "
        "may act); its restore target is 0"
    ]
    with pytest.warns(RecoveryInputWarning, match="no restoration potential"):
        build_recovery_problem(inputs)


def test_restore_cheaper_than_protect_on_existing_habitat_warns() -> None:
    inputs = anchor_inputs(
        potential=pd.DataFrame(
            {"pu": [1, 3, 4], "feature": [1, 1, 1], "amount": [2.0, 8.0, 12.0]}
        ),
        restore_cost=pd.Series({1: 0.5, 2: 2.5, 3: 2.5, 4: 5.0}),
    )
    check = check_recovery_inputs(inputs)
    assert len(check.warnings) == 1 and "units 1" in check.warnings[0]


def test_provenance_marks_defaults_as_placeholders() -> None:
    prov = anchor_inputs(passive=PassiveAction()).provenance()
    assert prov["restore_cost"] == "placeholder"  # factor 2.5 left at the default
    assert prov["passive_efficacy"] == "placeholder"
    assert prov["passive_cost_factor"] == "placeholder"
    assert prov["maturity"] == "placeholder"
    assert prov["protect_prop"] == "user"  # the anchor sets 0.5


def test_build_has_no_warnings_on_the_anchor() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RecoveryInputWarning)
        build_recovery_problem(anchor_inputs(passive=PassiveAction()))


def test_non_integral_habitat_id_is_rejected() -> None:
    inputs = anchor_inputs()
    inputs.base = inputs.base.copy_with(features=inputs.base.features.assign(id=[2.5]))
    assert any("must be a positive integer" in e for e in inputs.validate())


def test_potential_only_in_locked_out_units_warns() -> None:
    check = check_recovery_inputs(anchor_inputs(status=(0, 0, 3, 3)))
    assert check.errors == []
    assert len(check.warnings) == 1 and "no restoration potential" in check.warnings[0]
