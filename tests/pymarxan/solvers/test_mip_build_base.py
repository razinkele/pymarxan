"""MIPSolver._build_base (Stage 3 spec §2): the objective-free base model."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver

DATA = Path(__file__).parent.parent.parent / "data"


def _simple(**params: object) -> ConservationProblem:
    p = load_project(DATA / "simple")
    return p.copy_with(parameters={**p.parameters, **params})


def test_base_has_no_targets_budget_or_objective() -> None:
    base = MIPSolver()._build_base(_simple(COSTBUDGET=30.0), SolverConfig())
    names = set(base.model.constraints)
    assert not any(n.startswith(("target_", "cost_budget", "short_", "met_")) for n in names)
    assert base.model.objective is None or not base.model.objective.items()
    assert [t.key for t in base.terms] == ["f1", "f2", "f3"]
    assert set(base.amount_exprs) == {1, 2, 3}


def test_context_shares_the_registry() -> None:
    base = MIPSolver()._build_base(_simple(COSTBUDGET=30.0), SolverConfig())
    ctx = base.context()
    ctx.slack(base.terms[0])
    assert ctx.implied is base.implied and base.implied[-1][0].name == "s_f1"


def test_need_boundary_builds_the_unweighted_boundary_at_blm_zero() -> None:
    p = _simple(BLM=0.0)
    without = MIPSolver()._build_base(p, SolverConfig())
    assert not without.boundary_expr.items()
    assert not any(n.startswith("abs") for n in without.model.constraints)
    base = MIPSolver()._build_base(p, SolverConfig(), need_boundary=True)
    assert {"abs1_1_2", "abs2_5_6"} <= set(base.model.constraints)
    coefs = {v.name: a for v, a in base.boundary_expr.items()}
    assert coefs["x_1"] == 2.0 and coefs["y_1_2"] == 1.0  # self-boundary, shared edge


def test_base_registers_boundary_and_connectivity_auxiliaries() -> None:
    conn = pd.DataFrame({"id1": [1, 2], "id2": [2, 3], "value": [1.0, 2.0]})
    for asym in (0, 1):
        p = _simple(CONNECTIVITY_WEIGHT=1.0, ASYMMETRIC_CONNECTIVITY=asym).copy_with(
            connectivity=conn,
        )
        base = MIPSolver()._build_base(p, SolverConfig())
        names = [v.name for v, _ in base.implied]
        assert names[:5] == ["y_1_2", "y_2_3", "y_3_4", "y_4_5", "y_5_6"]
        assert names[5:] == ["zc_1_2", "zc_2_3"]
        base.x[1].varValue, base.x[2].varValue, base.x[3].varValue = 1.0, 0.0, 1.0
        rules = dict((v.name, rule) for v, rule in base.implied)
        assert rules["y_1_2"]() == 1.0
        # symmetric: x1·x2 = 0; asymmetric: x2·(1 − x3) = 0 and x1·(1 − x2) = 1
        assert rules["zc_1_2"]() == (1.0 if asym else 0.0)
        assert rules["zc_2_3"]() == 0.0


def test_min_penalties_primary_is_absolute_min_shortfall_builder() -> None:
    """Ruling (plan Task 3): min_penalties' primary is build_min_shortfall(ctx, "absolute"),
    so a target with R = target × MISSLEVEL <= 0 gets no slack row."""
    model = MIPSolver(objective="min_penalties").build_model(_simple(MISSLEVEL=0.0))
    assert not any(n.startswith("short_") for n in model.constraints)
    model = MIPSolver(objective="min_penalties").build_model(_simple())
    assert {"short_f1", "short_f2", "short_f3"} <= set(model.constraints)
