"""``_chance_terms`` / ``_active_chance_terms`` (spec §3.1, final review I-4).

Release A (v0.47.0) knows one chance term, ``"2d"`` (PROBMODE 3, from the puvspr
``prob`` column). *Requested* terms follow the mode alone; *active* terms are the
requested ones whose data exists, so a term without its data is off (penalty 0).
"""
from __future__ import annotations

import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.utils import _active_chance_terms, _chance_terms


def _problem(params: dict, *, prob_column: bool = True, table: bool = False):
    pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 2.0], "status": [0, 0]})
    feat = pd.DataFrame({
        "id": [1], "name": ["a"], "target": [5.0], "spf": [1.0], "ptarget": [0.9],
    })
    pv = {"species": [1, 1], "pu": [1, 2], "amount": [3.0, 4.0]}
    if prob_column:
        pv["prob"] = [0.5, 0.8]
    probability = (
        pd.DataFrame({"pu": [1, 2], "probability": [0.1, 0.2]}) if table else None
    )
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pd.DataFrame(pv),
        probability=probability, parameters=params,
    )


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_modes_without_a_chance_term_request_none(mode: int) -> None:
    p = _problem({"PROBMODE": mode})
    assert _chance_terms(p) == frozenset()
    assert _active_chance_terms(p) == frozenset()


def test_probmode_3_requests_the_2d_term() -> None:
    p = _problem({"PROBMODE": 3})
    assert _chance_terms(p) == frozenset({"2d"})
    assert _active_chance_terms(p) == frozenset({"2d"})


def test_2d_term_without_puvspr_prob_is_requested_but_not_active() -> None:
    p = _problem({"PROBMODE": 3}, prob_column=False)
    assert _chance_terms(p) == frozenset({"2d"})
    assert _active_chance_terms(p) == frozenset()


@pytest.mark.parametrize("mode", [4, 5, 7, -1])
def test_other_modes_request_nothing_in_release_a(mode: int) -> None:
    """Modes 4 / 5 (Probability 1D) arrive in release B; until then they request nothing."""
    p = _problem({"PROBMODE": mode})
    assert _chance_terms(p) == frozenset()
    assert _active_chance_terms(p) == frozenset()


def test_default_mode_follows_probmode_rule() -> None:
    """No PROBMODE: the v0.45 default (1 with a table, else 0) — no chance term."""
    assert _chance_terms(_problem({})) == frozenset()
    assert _chance_terms(_problem({}, table=True)) == frozenset()


def test_helpers_return_frozensets() -> None:
    p = _problem({"PROBMODE": 3})
    assert isinstance(_chance_terms(p), frozenset)
    assert isinstance(_active_chance_terms(p), frozenset)


def test_cache_exposes_active_terms() -> None:
    from pymarxan.solvers.cache import ProblemCache

    assert ProblemCache.from_problem(_problem({"PROBMODE": 3})).chance_terms == {"2d"}
    no_data = _problem({"PROBMODE": 3}, prob_column=False)
    assert ProblemCache.from_problem(no_data).chance_terms == frozenset()
    assert ProblemCache.from_problem(_problem({"PROBMODE": 0})).chance_terms == frozenset()


def test_solution_fields_follow_active_terms() -> None:
    """``Solution.prob_shortfalls`` / ``prob_penalty`` are None unless 2D is active."""
    import numpy as np

    from pymarxan.solvers.utils import build_solution

    sel = np.array([True, True])
    on = build_solution(_problem({"PROBMODE": 3}), sel, 0.0)
    assert on.prob_shortfalls is not None and on.prob_penalty is not None
    off = build_solution(_problem({"PROBMODE": 3}, prob_column=False), sel, 0.0)
    assert off.prob_shortfalls is None and off.prob_penalty is None
