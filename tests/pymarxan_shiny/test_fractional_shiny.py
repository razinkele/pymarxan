"""Stage 4 (spec §6.1, §6.2, §6.5): Shiny helpers with fractional solutions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers import FractionalSolutionError, MIPSolver
from pymarxan.solvers.base import Solution
from pymarxan.solvers.utils import build_solution
from pymarxan_shiny.modules.mapping.comparison_map import comparison_counts, comparison_summary
from pymarxan_shiny.modules.mapping.ocean_palette import MAP_NOT_SEL, MAP_SELECTED
from pymarxan_shiny.modules.mapping.solution_map import solution_colors, solution_listing
from pymarxan_shiny.modules.results.summary_table import _target_rows
from pymarxan_shiny.modules.solver_config.solver_picker import (
    mip_decision_kwargs,
    solver_picker_ui,
)

DATA = Path(__file__).parent.parent / "data"
X = np.array([0.25, 1.0, 0.0, 0.5, 0.1, 1.0])


@pytest.fixture
def simple() -> ConservationProblem:
    return load_project(DATA / "simple")


@pytest.fixture
def frac(simple: ConservationProblem) -> Solution:
    return build_solution(simple, X, 1.0)


def test_target_rows_fractional(simple: ConservationProblem, frac: Solution) -> None:
    rows = _target_rows(simple, frac)
    held1 = 10 * 0.25 + 15 + 12 * 0.5 + 8 * 0.1 + 10
    assert [r["id"] for r in rows] == [1, 2, 3]
    assert rows[0]["achieved"] == pytest.approx(held1)
    assert rows[0]["met"] == frac.targets_met[1]
    assert [r["met"] for r in rows] == [frac.targets_met[f] for f in (1, 2, 3)]
    assert rows[0]["pct"] == pytest.approx(held1 / 30.0 * 100)


def test_target_rows_binary_honours_misslevel(simple: ConservationProblem) -> None:
    """Spec §6.5: the table took met as held >= target, ignoring MISSLEVEL."""
    sel = np.array([False, True, False, True, False, False])  # f1 held 27 of 30
    p = simple.copy_with(parameters={**simple.parameters, "MISSLEVEL": 0.9})
    rows = _target_rows(p, build_solution(p, sel, 1.0))
    assert rows[0]["achieved"] == 27.0
    assert rows[0]["met"] is True


def test_solution_colors(frac: Solution, simple: ConservationProblem) -> None:
    colors = solution_colors(frac)
    assert colors[1] == MAP_SELECTED.lower() and colors[2] == MAP_NOT_SEL.lower()
    assert colors[0] not in (MAP_SELECTED.lower(), MAP_NOT_SEL.lower())
    b = build_solution(simple, X >= 0.5, 1.0)
    assert solution_colors(b) == [
        MAP_SELECTED if v else MAP_NOT_SEL for v in (X >= 0.5)
    ]


def test_solution_listing(frac: Solution, simple: ConservationProblem) -> None:
    assert solution_listing(simple, frac) == [
        "PU 1 (0.25)", "PU 2 (1)", "PU 4 (0.5)", "PU 5 (0.1)", "PU 6 (1)",
    ]
    b = build_solution(simple, X >= 0.5, 1.0)
    assert solution_listing(simple, b) == ["PU 2", "PU 4", "PU 6"]


def test_comparison_guarded(frac: Solution, simple: ConservationProblem) -> None:
    b = build_solution(simple, X >= 0.5, 1.0)
    assert comparison_counts(b, b) == (3, 0, 0)
    with pytest.raises(FractionalSolutionError, match="comparison map"):
        comparison_counts(b, frac)


def test_comparison_summary_explains_fractional(
    frac: Solution, simple: ConservationProblem,
) -> None:
    """Review focus: the comparison renders a message, not an uncaught exception."""
    b = build_solution(simple, X >= 0.5, 1.0)
    assert comparison_summary(b, b) == "Both: 3 | A only: 0 | B only: 0"
    for pair in ((b, frac), (frac, b), (frac, frac)):
        msg = comparison_summary(*pair)
        assert msg.startswith("Comparison unavailable")
        assert "binarize_solution" in msg


def test_solver_picker_has_decision_inputs() -> None:
    html = str(solver_picker_ui("sp"))
    for name in ("mip_decisions", "mip_upper", "mip_lower"):
        assert name in html


def test_mip_decision_kwargs() -> None:
    assert mip_decision_kwargs({}) == {"decisions": "binary"}
    kw = mip_decision_kwargs(
        {"mip_decisions": "proportion", "mip_upper": 0.5, "mip_lower": 0.2},
    )
    assert kw == {"decisions": "proportion", "upper": 0.5}
    kw = mip_decision_kwargs(
        {"mip_decisions": "semicontinuous", "mip_upper": 0.9, "mip_lower": 0.2},
    )
    assert kw == {"decisions": "semicontinuous", "upper": 0.9, "lower": 0.2}
    solver = MIPSolver(**kw)
    assert (solver.decisions, solver.upper, solver.lower) == ("semicontinuous", 0.9, 0.2)
