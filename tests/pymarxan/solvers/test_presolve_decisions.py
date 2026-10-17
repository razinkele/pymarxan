"""Stage 4 (spec §3.7): presolve data checks scale with the decision bounds."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest

from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers import MIPSolver, presolve_check
from pymarxan.solvers.presolve import _presolve_data

DATA = Path(__file__).parent.parent.parent / "data"


def _locked(**params: Any) -> ConservationProblem:
    """simple with PU 3 (cost 20) locked in and PU 6 (cost 8) locked out."""
    p = load_project(DATA / "simple")
    pu = p.planning_units.copy()
    pu.loc[pu["id"] == 3, "status"] = 2
    pu.loc[pu["id"] == 6, "status"] = 3
    return p.copy_with(planning_units=pu, parameters={**p.parameters, **params})


def _codes(msgs: list[str]) -> list[str]:
    return [m.split(":")[0] for m in msgs]


def test_binary_defaults_unchanged() -> None:
    p = _locked(COSTBUDGET=15.0)
    assert presolve_check(p) == presolve_check(p, decisions="binary", upper=1.0, lower=0.0)
    assert "P9a" in _codes(presolve_check(p))


def test_p9a_floor_is_unscaled() -> None:
    """Locked-in units are whole in every mode (review R3/R4): the forced cost is 20."""
    msgs = presolve_check(_locked(COSTBUDGET=15.0), decisions="proportion", upper=0.5)
    assert any(m.startswith("P9a") and "(20)" in m for m in msgs), msgs
    assert "P9a" not in _codes(
        presolve_check(_locked(COSTBUDGET=25.0), decisions="proportion", upper=0.5)
    )


def test_p9b_skipped_for_proportion_scaled_for_semicontinuous() -> None:
    p = load_project(DATA / "simple")
    p = p.copy_with(parameters={**p.parameters, "COSTBUDGET": 5.0})  # min free cost 8
    assert "P9b" in _codes(presolve_check(p))
    assert "P9b" not in _codes(presolve_check(p, decisions="proportion"))
    # cheapest option 0.5 · 8 = 4 <= 5
    assert "P9b" not in _codes(presolve_check(p, decisions="semicontinuous", lower=0.5))
    msgs = presolve_check(p, decisions="semicontinuous", lower=0.8)  # 6.4 > 5
    assert any(m.startswith("P9b") and "(6.4)" in m for m in msgs), msgs


def test_p11_ceiling_locked_whole_free_scaled() -> None:
    p = _locked(COSTBUDGET=48.0)  # binary ceiling 75; at upper 0.5: 20 + 0.5 · 55 = 47.5
    assert "P11" not in _codes(presolve_check(p))
    msgs = presolve_check(p, decisions="proportion", upper=0.5)
    assert any(m.startswith("P11") and "47.5" in m for m in msgs), msgs


def test_p6_locked_whole_free_scaled() -> None:
    p = _locked()  # f1: target 30; locked-in PU 3 holds 5, free units 45
    assert "P6" not in _codes(presolve_check(p))
    msgs = presolve_check(p, decisions="proportion", upper=0.5)  # 5 + 22.5 = 27.5 < 30
    p6 = [m for m in msgs if m.startswith("P6")]
    assert p6 and "feature 1 target 30" in p6[0] and "27.5" in p6[0], msgs
    assert "P6" not in _codes(presolve_check(p, decisions="proportion", upper=0.6))  # 32


@pytest.mark.parametrize(
    "kw", [{"decisions": "x"}, {"decisions": "proportion", "upper": 2.0},
           {"decisions": "binary", "lower": 0.5}],
)
def test_presolve_check_validates_decisions(kw: dict[str, Any]) -> None:
    """Review R6: a bad argument is a caller error, raised outside the never-raises path."""
    with pytest.raises(ValueError):
        presolve_check(_locked(), **kw)


def test_presolve_data_passes_bounds() -> None:
    p = _locked(COSTBUDGET=48.0)
    assert "P11" not in _codes(_presolve_data(True, p, "MIP"))
    assert "P11" in _codes(
        _presolve_data(True, p, "MIP", decisions="proportion", upper=0.5)
    )


def test_strict_mip_uses_its_bounds() -> None:
    p = _locked(COSTBUDGET=48.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert MIPSolver(objective="max_features", presolve_check="strict").solve(p)
        with pytest.raises(ValueError, match="P11"):
            MIPSolver(
                objective="max_features", presolve_check="strict", decisions="proportion",
                upper=0.5,
            ).solve(p)
        sols = MIPSolver(objective="max_features", decisions="proportion", upper=0.5).solve(p)
    assert sols and sols[0].selected[2] == 1.0
    assert sols[0].cost <= 48.0 + 1e-6
