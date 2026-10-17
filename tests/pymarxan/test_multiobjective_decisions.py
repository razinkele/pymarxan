"""Stage 4 (spec §2, §4, §8): MultiObjectiveMIP with proportion / semicontinuous decisions."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.multiobjective import (
    Hierarchical,
    MaxFeatures,
    MinCost,
    MinShortfall,
    MultiObjectiveMIP,
    WeightedSum,
)
from pymarxan.solvers.base import Solution, SolverConfig
from pymarxan.solvers.utils import check_targets
from tests.pymarxan.fractional_fixtures import rejected_cases
from tests.pymarxan.lp_canon import captured_models

DATA = Path(__file__).parent.parent / "data"


def _simple(**params: Any) -> ConservationProblem:
    p = load_project(DATA / "simple")
    return p.copy_with(parameters={**p.parameters, **params})


def _solve(mo: MultiObjectiveMIP, problem: ConservationProblem,
           config: SolverConfig | None = None) -> list[Solution]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # PuLP 3.3.2
        return mo.solve(problem, config)


def test_attributes_and_validation() -> None:
    mo = MultiObjectiveMIP([MinCost()], Hierarchical(0.0))
    assert (mo.decisions, mo.upper, mo.lower) == ("binary", 1.0, 0.0)
    mo = MultiObjectiveMIP([MinCost()], Hierarchical(0.0), decisions="semicontinuous",
                           upper=0.9, lower=0.2)
    assert (mo.decisions, mo.upper, mo.lower) == ("semicontinuous", 0.9, 0.2)
    with pytest.raises(ValueError, match="upper must be in"):
        MultiObjectiveMIP([MinCost()], Hierarchical(0.0), decisions="proportion", upper=2.0)
    with pytest.raises(ValueError, match="decisions must be one of"):
        MultiObjectiveMIP([MinCost()], Hierarchical(0.0), decisions="fraction")


@pytest.mark.parametrize(
    "cause",
    ["PROBMODE 3", "clumping", "separation", "space", "forbidden_selections",
     "negative connectivity"],
)
def test_rejected_combinations(cause: str) -> None:
    problem, config = rejected_cases(_simple())[cause]
    problem = problem.copy_with(parameters={**problem.parameters, "COSTBUDGET": 30.0})
    mo = MultiObjectiveMIP([MaxFeatures(), MinCost()], Hierarchical(0.0),
                           decisions="proportion", mip_backend="cbc")
    with pytest.raises(ValueError, match=cause):
        _solve(mo, problem, config)


def test_hierarchical_max_features_then_min_cost_proportion() -> None:
    """COSTBUDGET 30: the LP optimum meets every target on active rows (held ≈ R)."""
    p = _simple(COSTBUDGET=30.0)
    mo = MultiObjectiveMIP([MaxFeatures(), MinCost()], Hierarchical(0.0),
                           decisions="proportion", mip_backend="cbc")
    with captured_models() as models:
        (s,) = _solve(mo, p)
    assert s.is_fractional
    assert s.selected.dtype == np.float64
    assert not set(s.selected.tolist()) <= {0.0, 1.0}
    assert s.metadata["stage_reached"] == 2
    count = s.metadata["objective_values"]["max_features"]
    pu_index = {int(q): i for i, q in enumerate(p.planning_units["id"])}
    met = check_targets(p, s.selected, pu_index)
    assert count > 0
    assert count == sum(met.values())
    y = sum(v.varValue or 0.0 for v in models[-1].variables() if v.name.startswith("y_f"))
    assert count >= y - 1e-9
    assert s.cost <= 30.0 + 1e-6


def test_semicontinuous_weighted_sum_floor() -> None:
    p = _simple(COSTBUDGET=40.0)
    mo = MultiObjectiveMIP([MinShortfall(), MinCost()], WeightedSum([0.9, 0.1]),
                           decisions="semicontinuous", lower=0.4, mip_backend="cbc")
    (s,) = _solve(mo, p)
    x = s.selected
    assert s.is_fractional
    assert ((x == 0.0) | (x >= 0.4)).all(), x


def test_binary_selection_stays_bool() -> None:
    mo = MultiObjectiveMIP([MaxFeatures(), MinCost()], Hierarchical(0.0), mip_backend="cbc")
    (s,) = _solve(mo, _simple(COSTBUDGET=30.0))
    assert s.selected.dtype == bool


def test_presolve_uses_bounds() -> None:
    """Locked-in PU 3 (cost 20) is whole: at upper 0.5 the ceiling is 20 + 0.5 · 63 = 51.5."""
    p = _simple(COSTBUDGET=52.0)
    pu = p.planning_units.copy()
    pu.loc[pu["id"] == 3, "status"] = 2
    p = p.copy_with(planning_units=pu)
    strict = {"presolve_check": "strict", "mip_backend": "cbc"}
    assert _solve(MultiObjectiveMIP([MaxFeatures(), MinCost()], Hierarchical(0.0), **strict), p)
    with pytest.raises(ValueError, match="P11"):
        _solve(MultiObjectiveMIP([MaxFeatures(), MinCost()], Hierarchical(0.0),
                                 decisions="proportion", upper=0.5, **strict), p)
    (s,) = _solve(MultiObjectiveMIP([MaxFeatures(), MinCost()], Hierarchical(0.0),
                                    decisions="proportion", upper=0.5, mip_backend="cbc"), p)
    assert s.selected[2] == 1.0
    assert (np.delete(s.selected, 2) <= 0.5).all()
