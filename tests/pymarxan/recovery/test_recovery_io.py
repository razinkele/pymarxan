"""Recovery sidecar read/write pair and save/load of a whole recovery project (R5)."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymarxan.recovery import (
    PassiveAction,
    RecoveryInputs,
    build_recovery_problem,
    evaluate_recovery,
    load_recovery_project,
    read_recovery_inputs,
    save_recovery_project,
    solve_recovery,
    write_recovery_inputs,
)
from tests.pymarxan.recovery.recovery_anchor import anchor_inputs, fingerprint

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _rich_inputs() -> RecoveryInputs:
    """Every optional input set: statuses 1/2/2/3 (2 with and without potential), per-unit
    p and restore cost, implementation cost, passive with a per-habitat η, maturity +
    horizon, per-habitat props, boundary with BLM."""
    inputs = anchor_inputs(
        status=(1, 2, 2, 3),
        restore_cost=pd.Series({1: 2.0, 2: 2.5, 3: 3.0, 4: 6.0}),
        restore_cost_by_habitat={1: 0.05},
        passive=PassiveAction(efficacy=0.6, efficacy_by_habitat={1: 0.4}, cost_factor=0.5),
        maturity={1: 0.8}, horizon_years=15.0,
        protect_prop={1: 0.4}, restore_prop={1: 0.3},
    )
    inputs.base = inputs.base.copy_with(
        boundary=pd.DataFrame({"id1": [1, 2, 3], "id2": [2, 3, 4], "boundary": [1.0] * 3}),
        parameters={"BLM": 0.25},
    )
    return inputs


@pytest.mark.parametrize("make", [
    lambda: anchor_inputs(),
    lambda: anchor_inputs(p_success=0.6, restore_target={1: 3.0}),
    _rich_inputs,
])
def test_inputs_round_trip(tmp_path: Path, make) -> None:  # type: ignore[no-untyped-def]
    inputs = make()
    write_recovery_inputs(inputs, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "potential.csv", "recovery.json", "recovery_units.csv",
    ]
    again = read_recovery_inputs(tmp_path, inputs.base)
    assert again.validate() == []
    assert fingerprint(build_recovery_problem(again)[0]) == fingerprint(
        build_recovery_problem(inputs)[0]
    )
    assert again.provenance() == inputs.provenance()


def test_sidecar_contents(tmp_path: Path) -> None:
    write_recovery_inputs(_rich_inputs(), tmp_path)
    side = json.loads((tmp_path / "recovery.json").read_text())
    assert side["format"] == "pymarxan-recovery" and side["version"] == 1
    assert side["habitats"] == [
        {"habitat": 1, "name": "wetland", "existing_feature": 1, "potential_feature": 2}
    ]
    assert side["zones"] == {"0": "no_action", "1": "protect", "2": "restore_active",
                             "3": "restore_passive"}
    assert side["passive"] == {"efficacy": 0.6, "efficacy_by_habitat": {"1": 0.4},
                               "cost_factor": 0.5}
    assert side["maturity"] == {"1": 0.8} and side["horizon_years"] == 15.0
    units = pd.read_csv(tmp_path / "recovery_units.csv")
    assert units.columns.tolist() == ["pu", "status", "p_success", "restore_cost"]
    assert units["status"].tolist() == [1, 2, 2, 3]  # as given, not as built


def test_project_round_trip_rebuilds_problem_mapping_and_results(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        problem, mapping = save_recovery_project(_rich_inputs(), tmp_path)
        project = load_recovery_project(tmp_path)
    assert project.problem.validate() == []
    assert fingerprint(project.problem) == fingerprint(problem)
    pd.testing.assert_frame_equal(project.mapping.habitats, mapping.habitats)
    pd.testing.assert_frame_equal(project.mapping.units, mapping.units)
    first = evaluate_recovery(solve_recovery(problem), problem, mapping)
    second = evaluate_recovery(solve_recovery(project.problem), project.problem,
                               project.mapping)
    assert np.array_equal(first.assignment, second.assignment)
    pd.testing.assert_frame_equal(first.habitats, second.habitats)


def test_saved_zone_project_reproduces_in_pymarxan(tmp_path: Path) -> None:
    from pymarxan.zones import load_zone_project

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        problem, _ = save_recovery_project(_rich_inputs(), tmp_path)
        zonal = load_zone_project(tmp_path)
    assert zonal.validate() == []
    assert solve_recovery(zonal).cost == pytest.approx(solve_recovery(problem).cost)


def test_rejects_a_foreign_sidecar(tmp_path: Path) -> None:
    write_recovery_inputs(anchor_inputs(), tmp_path)
    (tmp_path / "recovery.json").write_text(json.dumps({"format": "other", "version": 1}))
    with pytest.raises(ValueError, match="not a pymarxan-recovery v1 file"):
        read_recovery_inputs(tmp_path, anchor_inputs().base)


def test_write_refuses_invalid_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid recovery inputs"):
        write_recovery_inputs(anchor_inputs(p_success=2.0), tmp_path)


@pytest.mark.parametrize("seed", range(20))
def test_sidecar_floats_round_trip_bit_exact(tmp_path: Path, seed: int) -> None:
    """read_csv's default float parser drops the last bit on ~13% of values; the sidecar
    must reload every amount, p_success and restore cost exactly (==, not approx)."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.01, 1.0, 4)
    rc = rng.uniform(1.0, 100.0, 4)
    amounts = rng.uniform(1.0, 100.0, 2)
    inputs = anchor_inputs(
        p_success=pd.Series(dict(zip([1, 2, 3, 4], p.tolist()))),
        restore_cost=pd.Series(dict(zip([1, 2, 3, 4], rc.tolist()))),
        potential=pd.DataFrame({"pu": [3, 4], "feature": [1, 1], "amount": amounts.tolist()}),
    )
    write_recovery_inputs(inputs, tmp_path)
    back = read_recovery_inputs(tmp_path, inputs.base)
    assert back.p_by_unit().tolist() == inputs.p_by_unit().tolist()
    assert back.restore_cost.loc[[1, 2, 3, 4]].tolist() == rc.tolist()
    assert back.potential["amount"].tolist() == amounts.tolist()
