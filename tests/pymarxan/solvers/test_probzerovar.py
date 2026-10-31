"""Marxan 3's zero-variance rule (Z = 4) is the default; ``PROBZEROVAR 1`` opts out.

v0.47.0 (spec §1 "Zero-variance quirk", §3.4). Marxan 3 ``ComputeProbability2D`` sets
``Z = 4`` when a feature's variance over the selection is 0, so ``P = probZUT(4) ≈ 3.17e-5``
fails any ``ptarget`` above it. ``PROBZEROVAR 1`` restores pymarxan's pre-v0.47 rule
("zero variance = met"). Includes a hand example against a test-local Python port of
Marxan 3's ``ReturnProbabilityAmounts2D`` / ``ComputeProbability2D`` / ``probZUT``
(``marxan3`` branch, ``probability.c``).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from pymarxan.io.readers import load_project
from pymarxan.io.writers import save_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.cache import ProblemCache
from pymarxan.solvers.chance_state import ChanceState
from pymarxan.solvers.iterative_improvement import IterativeImprovementSolver
from pymarxan.solvers.probability import _term_moments, evaluate_solution_chance, score_features
from pymarxan.solvers.utils import _zero_var_rule, build_solution

SF4 = float(norm.sf(4.0))


# ------------------------------------------------- Marxan 3 transcription (probability.c)


def probZUT(z: float) -> float:
    """Ibbetson (CACM Alg. 209) upper tail, Z_MAX = 5 — probability.c ``probZUT``."""
    Z_MAX = 5.0
    if z == 0.0:
        x = 0.0
    else:
        y = 0.5 * abs(z)
        if y >= Z_MAX * 0.5:
            x = 1.0
        elif y < 1.0:
            w = y * y
            x = ((((((((0.000124818987 * w
                        - 0.001075204047) * w + 0.005198775019) * w
                      - 0.019198292004) * w + 0.059054035642) * w
                    - 0.151968751364) * w + 0.319152932694) * w
                  - 0.531923007300) * w + 0.797884560593) * y * 2.0
        else:
            y -= 2.0
            x = (((((((((((((-0.000045255659 * y
                             + 0.000152529290) * y - 0.000019538132) * y
                           - 0.000676904986) * y + 0.001390604284) * y
                         - 0.000794620820) * y - 0.002034254874) * y
                       + 0.006549791214) * y - 0.010557625006) * y
                     + 0.011630447319) * y - 0.009279453341) * y
                   + 0.005353579108) * y - 0.002141268741) * y
                 + 0.000535310849) * y + 0.999936657524
    return (x + 1.0) * 0.5 if z < 0.0 else (1.0 - x) * 0.5


def marxan3_probability2d(problem: ConservationProblem, selected: np.ndarray):
    """``ReturnProbabilityAmounts2D`` over the selected PUs, then ``ComputeProbability2D``.

    Returns (per-feature P, weighted probability2D)."""
    feats = problem.features
    idx = {int(f): j for j, f in enumerate(feats["id"])}
    E = [0.0] * len(feats)
    V = [0.0] * len(feats)
    chosen = {int(p) for p, s in zip(problem.planning_units["id"], selected) if s}
    for row in problem.pu_vs_features.itertuples():
        if int(row.pu) in chosen and row.amount:
            j = idx[int(row.species)]
            E[j] += row.amount * row.prob
            V[j] += row.amount * row.amount * row.prob * (1 - row.prob)
    P, total = [], 0.0
    for j, (target, ptarget2d) in enumerate(zip(feats["target"], feats["ptarget"])):
        z = (target - E[j]) / math.sqrt(V[j]) if V[j] > 0 else 4
        p = probZUT(z) if z >= 0 else 1 - probZUT(-1 * z)
        heaviside = 1 if ptarget2d > p else 0
        shortfall = (ptarget2d - p) / ptarget2d if ptarget2d > 0 else 0
        P.append(p)
        total += heaviside * shortfall
    weight = float(problem.parameters.get("PROBABILITYWEIGHTING", 1.0))
    return P, total * weight


def hand_example(**params) -> tuple[ConservationProblem, np.ndarray]:
    """PUs 1-3 selected, PU 4 not.

    f1: q ≠ 0.5, E = 7.4 > target 5 (negative Z), ptarget 0.95.
    f2: only q = 1 cells selected → zero variance, ptarget 0.6 > probZUT(4).
    f3: unselected (its only cell is in PU 4) → zero variance, ptarget 0.3.
    f4: E = 3.4 < target 6 (positive Z), ptarget 0.8.
    """
    pu = pd.DataFrame({"id": [1, 2, 3, 4], "cost": [1.0, 2.0, 3.0, 4.0], "status": [0] * 4})
    feat = pd.DataFrame({
        "id": [1, 2, 3, 4], "name": list("abcd"), "target": [5.0, 4.0, 2.0, 6.0],
        "spf": [1.0] * 4, "ptarget": [0.95, 0.6, 0.3, 0.8],
    })
    pv = pd.DataFrame({
        "species": [1, 1, 1, 2, 2, 3, 4, 4],
        "pu": [1, 2, 3, 1, 2, 4, 2, 3],
        "amount": [4.0, 3.0, 2.0, 5.0, 1.0, 9.0, 2.0, 3.0],
        "prob": [0.8, 0.7, 0.8, 1.0, 1.0, 0.4, 0.2, 1.0],
    })
    problem = ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pv,
        parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 7.0, **params},
    )
    return problem, np.array([True, True, True, False])


class TestHandExampleAgainstMarxan3:
    def test_per_feature_probability(self) -> None:
        problem, sel = hand_example()
        P_mx, _ = marxan3_probability2d(problem, sel)
        E, V, N = _term_moments(problem, sel, "2d", problem.features["id"].values)
        z, p, _, _ = score_features(
            E, V, N, problem.features["target"].values, problem.features["ptarget"].values,
            _zero_var_rule(problem),
        )
        assert z[0] < 0 and z[3] > 0  # both signs of Z exercised
        assert N[1] == 0 and N[2] == 0  # f2 (q = 1) and f3 (unselected): zero variance
        assert np.all(np.abs(z) < 5)  # Marxan's probZUT is exact 0/1 beyond |z| = 5
        np.testing.assert_allclose(p, P_mx, rtol=0, atol=1e-6)

    def test_total_penalty_evaluator_and_chance_state(self) -> None:
        problem, sel = hand_example()
        _, total_mx = marxan3_probability2d(problem, sel)
        _, total = evaluate_solution_chance(problem, sel)
        # penalty differences are P differences / ptarget (min ptarget 0.3), times weight 7
        assert total == pytest.approx(total_mx, abs=7 * 1e-6 / 0.3)
        assert total > 7 * 0.9  # the zero-variance f2 alone costs ~ weight·1
        cache = ProblemCache.from_problem(problem)
        assert ChanceState.from_selection(cache, sel).penalty_total() == pytest.approx(
            total, abs=1e-12,
        )
        assert build_solution(problem, sel, 0.0).prob_penalty == pytest.approx(total, abs=1e-12)

    def test_duplicate_rows_follow_marxan3(self) -> None:
        """Marxan 3 keeps one sparse-matrix entry per puvspr row (``input.c``
        ``readSparseMatrix``) and adds them up one by one: a cell named twice is two rows,
        not the summed amount with the last row's q. Adds f4 at PU 2 again with q = 1 (after
        its q = 0.2 row) and f1 at PU 1 again with the same q."""
        problem, sel = hand_example()
        extra = pd.DataFrame({"species": [4, 1], "pu": [2, 1], "amount": [1.0, 4.0],
                              "prob": [1.0, 0.8]})
        problem = problem.copy_with(
            pu_vs_features=pd.concat([problem.pu_vs_features, extra], ignore_index=True),
        )
        _, total_mx = marxan3_probability2d(problem, sel)
        _, total = evaluate_solution_chance(problem, sel)
        assert total == pytest.approx(total_mx, abs=7 * 1e-6 / 0.3)
        cache = ProblemCache.from_problem(problem)
        assert ChanceState.from_selection(cache, sel).penalty_total() == pytest.approx(
            total, abs=1e-12,
        )

    def test_zero_variance_features_pay_marxans_penalty(self) -> None:
        problem, sel = hand_example()
        shortfalls, _ = evaluate_solution_chance(problem, sel)
        assert shortfalls[2] == pytest.approx(0.6 - SF4)
        assert shortfalls[3] == pytest.approx(0.3 - SF4)

    def test_opt_out_drops_only_the_zero_variance_features(self) -> None:
        problem, sel = hand_example()
        opt_out, _ = hand_example(PROBZEROVAR=1)
        _, marxan = evaluate_solution_chance(problem, sel)
        shortfalls, met = evaluate_solution_chance(opt_out, sel)
        assert shortfalls[2] == 0.0 and shortfalls[3] == 0.0
        expected_gap = 7.0 * ((0.6 - SF4) / 0.6 + (0.3 - SF4) / 0.3)
        assert marxan - met == pytest.approx(expected_gap, abs=1e-12)


# ------------------------------------------------------------- zero-variance rule cases


def _zero_var_problem(qs: list[float], **params) -> ConservationProblem:
    pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]})
    feat = pd.DataFrame({"id": [1, 2], "name": ["a", "b"], "target": [3.0, 3.0],
                         "spf": [1.0, 1.0], "ptarget": [0.9, 0.5]})
    pv = pd.DataFrame({"species": [1, 1, 2, 2], "pu": [1, 2, 1, 2],
                       "amount": [2.0, 2.0, 2.0, 2.0], "prob": qs})
    return ConservationProblem(planning_units=pu, features=feat, pu_vs_features=pv,
                               parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 2.0,
                                           **params})


ZERO_VAR_CASES = {
    "nothing selected": ([0.5, 0.5, 0.5, 0.5], [False, False]),
    "all q = 1": ([1.0, 1.0, 1.0, 1.0], [True, True]),
    "all q = 0": ([0.0, 0.0, 0.0, 0.0], [True, True]),
    "q in {0, 1}": ([0.0, 1.0, 1.0, 0.0], [True, True]),
}


@pytest.mark.parametrize("case", sorted(ZERO_VAR_CASES))
def test_zero_variance_default_is_marxans_rule(case: str) -> None:
    qs, sel = ZERO_VAR_CASES[case]
    problem = _zero_var_problem(qs)
    sel_arr = np.array(sel)
    expected = 2.0 * ((0.9 - SF4) / 0.9 + (0.5 - SF4) / 0.5)
    assert evaluate_solution_chance(problem, sel_arr)[1] == pytest.approx(expected)
    cache = ProblemCache.from_problem(problem)
    assert cache.zero_var_rule == "marxan"
    assert ChanceState.from_selection(cache, sel_arr).penalty_total() == pytest.approx(expected)


@pytest.mark.parametrize("case", sorted(ZERO_VAR_CASES))
def test_zero_variance_opt_out_is_met(case: str) -> None:
    qs, sel = ZERO_VAR_CASES[case]
    problem = _zero_var_problem(qs, PROBZEROVAR=1)
    sel_arr = np.array(sel)
    shortfalls, penalty = evaluate_solution_chance(problem, sel_arr)
    assert penalty == 0.0 and shortfalls == {1: 0.0, 2: 0.0}
    cache = ProblemCache.from_problem(problem)
    assert cache.zero_var_rule == "met"
    assert ChanceState.from_selection(cache, sel_arr).penalty_total() == 0.0


def test_the_documented_safe_vs_risky_inversion() -> None:
    """Under Marxan's rule a safe (q = 1) PU scores worse than a risky one (CHANGELOG).

    PU 1 risky (cost 1, q = 0.5), PU 2 safe (cost 2, q = 1); amount 10, target 5,
    ptarget 0.9, weight 50. Holding PU 1: 1 + 50·(0.9 − 0.5)/0.9 ≈ 23.2; holding PU 2:
    2 + 50·(0.9 − sf(4))/0.9 ≈ 52.0 by default, 2 under PROBZEROVAR 1.
    """
    def problem(**params):
        pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 2.0], "status": [0, 0]})
        feat = pd.DataFrame({"id": [1], "name": ["f"], "target": [5.0], "spf": [1.0],
                             "ptarget": [0.9]})
        pv = pd.DataFrame({"species": [1, 1], "pu": [1, 2], "amount": [10.0, 10.0],
                           "prob": [0.5, 1.0]})
        return ConservationProblem(
            planning_units=pu, features=feat, pu_vs_features=pv,
            parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 50.0, "BLM": 0.0, **params},
        )

    risky, safe = np.array([True, False]), np.array([False, True])
    p = problem()
    assert build_solution(p, safe, 0.0).objective == pytest.approx(
        2 + 50 * (0.9 - SF4) / 0.9,
    )
    assert build_solution(p, risky, 0.0).objective < build_solution(p, safe, 0.0).objective
    start = build_solution(p, risky, 0.0)
    assert IterativeImprovementSolver(itimptype=3).improve(p, start).selected.tolist() == [
        True, False,
    ]
    q = problem(PROBZEROVAR=1)
    assert build_solution(q, safe, 0.0).objective == pytest.approx(2.0)
    sol = IterativeImprovementSolver(itimptype=3).improve(q, build_solution(q, risky, 0.0))
    assert sol.selected.tolist() == [False, True]


# ------------------------------------------------------------------ parameter handling


@pytest.mark.parametrize("value, rule", [(None, "marxan"), (0, "marxan"), (1, "met"),
                                         (1.0, "met"), ("1", "met"), (0.0, "marxan")])
def test_rule_from_parameter(value, rule: str) -> None:
    params = {} if value is None else {"PROBZEROVAR": value}
    problem, _ = hand_example(**params)
    assert _zero_var_rule(problem) == rule


@pytest.mark.parametrize("value", [2, -1, 0.5, "yes"])
def test_validate_reports_bad_probzerovar(value) -> None:
    problem, _ = hand_example(PROBZEROVAR=value)
    errors = problem.validate()
    assert any("PROBZEROVAR" in e for e in errors)
    assert _zero_var_rule(problem) == "marxan"  # pure: never raises; validate() reports


@pytest.mark.parametrize("value", [None, 0, 1])
def test_validate_accepts_probzerovar(value) -> None:
    params = {} if value is None else {"PROBZEROVAR": value}
    problem, _ = hand_example(**params)
    assert not any("PROBZEROVAR" in e for e in problem.validate())


def test_probzerovar_round_trips_through_input_dat(tmp_path) -> None:
    problem, _ = hand_example(PROBZEROVAR=1)
    save_project(problem, tmp_path)
    loaded = load_project(tmp_path)
    assert loaded.parameters["PROBZEROVAR"] == 1
    assert _zero_var_rule(loaded) == "met"
