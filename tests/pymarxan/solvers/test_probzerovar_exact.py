"""``PROBZEROVAR 2``: the exact zero-variance rule (v0.48.0, B0; spec §3.4 decision).

A zero-variance feature (no selected cell carrying variance) holds a certain amount ``E``.
Under ``PROBZEROVAR 2`` ("exact") it is met outright (z = −inf, P = 1, no shortfall or
penalty) when ``E ≥ target − TARGET_REL_TOL·|target|`` (1e-7, relative: the deterministic
targets' tolerance, since a target such as a prop = 1 total is summed in another order than
``E``), missed outright (z = +inf, P = 0, shortfall ``ptarget``, penalty 1) otherwise. A
target of 0 has no tolerance. ``PROBZEROVAR 0`` (Marxan's Z = 4, the default) and ``1``
("met") are unchanged, bit for bit (``TestRulesZeroAndOneAreUnchanged``: numbers recorded
from v0.47.1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from pymarxan.io.readers import _resolve_prop_targets, load_project
from pymarxan.io.writers import save_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers._tolerance import TARGET_REL_TOL
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.cache import ProblemCache
from pymarxan.solvers.chance_state import ChanceState
from pymarxan.solvers.probability import _term_moments, evaluate_solution_chance
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
from pymarxan.solvers.utils import _zero_var_rule, build_solution

SF4 = float(norm.sf(4.0))


def hand_example(**params) -> tuple[ConservationProblem, np.ndarray]:
    """``test_probzerovar.hand_example``: PUs 1-3 selected, PU 4 not.

    f1: E = 6.9 > target 5 with variance; f2: only q = 1 cells selected (zero variance,
    E = 6 ≥ target 4: met under the exact rule); f3: unselected (zero variance, E = 0 <
    target 2: short); f4: E = 3.4 < target 6 with variance.
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


def _random_problem(**params) -> ConservationProblem:
    """30 PUs × 4 features, about half the cells certain (q = 1); rows species-major."""
    rng = np.random.default_rng(3)
    n, f = 30, 4
    pu = pd.DataFrame({"id": np.arange(1, n + 1), "cost": rng.uniform(1, 5, n).round(2),
                       "status": [0] * n})
    feat = pd.DataFrame({"id": np.arange(1, f + 1), "name": list("abcd"),
                         "target": [20.0, 15.0, 10.0, 25.0], "spf": [5.0] * f,
                         "ptarget": [0.9, 0.7, 0.8, 0.6]})
    rows = [(j + 1, i + 1) for j in range(f) for i in range(n) if rng.random() < 0.5]
    pv = pd.DataFrame({
        "species": [r[0] for r in rows], "pu": [r[1] for r in rows],
        "amount": rng.uniform(0.5, 4, len(rows)).round(2),
        "prob": np.where(rng.random(len(rows)) < 0.6, 1.0,
                         rng.uniform(0.3, 1, len(rows)).round(2)),
    })
    pv = pv.sort_values(["species", "pu"]).reset_index(drop=True)
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pv,
        parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 10.0, "BLM": 0.0, **params},
    )


# ------------------------------------------- PROBZEROVAR 0 / 1: unchanged since v0.47.1


class TestRulesZeroAndOneAreUnchanged:
    """Numbers recorded from v0.47.1 (``75aeab1``), compared with ``==``: adding the exact
    rule must not move a bit of the Marxan (0) or "met" (1) results — evaluator, the
    ``ChanceState`` total and deltas, and seeded simulated annealing."""

    HAND = {
        0: (22.097994138995148, [0.149849145928745, 0.5999683287581669,
                                 0.29996832875816687, 0.7994229749576093],
            [4.3714867881710475, 2.5443807454512632, 1.8000441111011602,
             -6.999261004357227]),
        1: (8.099102632459308, [0.149849145928745, 0.0, 0.0, 0.7994229749576093],
            [4.3714867881710475, -4.455342131182697, 1.8000441111011602, 0.0]),
    }
    SA_HAND = {0: 26.098733134637918, 1: 12.64376050127661}
    SA_RANDOM = {0: [41.409708007619386, 40.10863846084933],
                 1: [39.42, 41.202451793803064]}

    @pytest.mark.parametrize("zv", [0, 1])
    def test_evaluator_and_chance_state(self, zv: int) -> None:
        total, shortfalls, deltas = self.HAND[zv]
        problem, sel = hand_example(PROBZEROVAR=zv)
        sf, tot = evaluate_solution_chance(problem, sel)
        assert tot == total
        assert [sf[f] for f in (1, 2, 3, 4)] == shortfalls
        cache = ProblemCache.from_problem(problem)
        cs = ChanceState.from_selection(cache, sel)
        assert cs.penalty_total() == total
        assert [cs.delta_penalty(cache, i, not sel[i]) for i in range(4)] == deltas

    @pytest.mark.parametrize("zv", [0, 1])
    def test_seeded_annealing(self, zv: int) -> None:
        problem, _ = hand_example(PROBZEROVAR=zv)
        sols = SimulatedAnnealingSolver(num_iterations=2000, num_temp_steps=20).solve(
            problem, SolverConfig(num_solutions=2, seed=42),
        )
        assert [s.objective for s in sols] == [self.SA_HAND[zv]] * 2
        sols = SimulatedAnnealingSolver(num_iterations=5000, num_temp_steps=50).solve(
            _random_problem(PROBZEROVAR=zv), SolverConfig(num_solutions=2, seed=7),
        )
        assert [s.objective for s in sols] == self.SA_RANDOM[zv]


# ---------------------------------------------------------------- the exact rule


class TestExactRuleEvaluator:
    def test_hand_example(self) -> None:
        """f2 (certain 6 ≥ 4) is met, f3 (certain 0 < 2) is missed outright; f1 / f4 (with
        variance) score as under every rule."""
        problem, sel = hand_example(PROBZEROVAR=2)
        marxan, _ = hand_example()
        sf, total = evaluate_solution_chance(problem, sel)
        sf0, _ = evaluate_solution_chance(marxan, sel)
        assert sf[1] == sf0[1] and sf[4] == sf0[4]
        assert sf[2] == 0.0 and sf[3] == 0.3
        f1 = sf0[1] / 0.95
        f4 = sf0[4] / 0.8
        assert total == pytest.approx(7.0 * (f1 + 1.0 + f4), abs=1e-12)
        sol = build_solution(problem, sel, 0.0)
        assert sol.prob_penalty == total
        assert sol.prob_shortfalls == sf

    @pytest.mark.parametrize("qs, sel, expected", [
        # target 3, amount 2 per cell; weighting 2, ptarget 0.9 / 0.5
        ([0.5, 0.5, 0.5, 0.5], [False, False], 2.0 * 2),    # nothing selected: E = 0 short
        ([1.0, 1.0, 1.0, 1.0], [True, True], 0.0),          # E = 4 ≥ 3: met
        ([1.0, 1.0, 1.0, 1.0], [True, False], 2.0 * 2),     # E = 2 < 3: short
        ([0.0, 0.0, 0.0, 0.0], [True, True], 2.0 * 2),      # q = 0: E = 0
        ([0.0, 1.0, 1.0, 0.0], [True, True], 2.0 * 2),      # E = 2 each: short
    ])
    def test_zero_variance_cases(self, qs, sel, expected) -> None:
        pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]})
        feat = pd.DataFrame({"id": [1, 2], "name": ["a", "b"], "target": [3.0, 3.0],
                             "spf": [1.0, 1.0], "ptarget": [0.9, 0.5]})
        pv = pd.DataFrame({"species": [1, 1, 2, 2], "pu": [1, 2, 1, 2],
                           "amount": [2.0] * 4, "prob": qs})
        problem = ConservationProblem(
            planning_units=pu, features=feat, pu_vs_features=pv,
            parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 2.0, "PROBZEROVAR": 2},
        )
        s = np.array(sel)
        assert evaluate_solution_chance(problem, s)[1] == expected
        cache = ProblemCache.from_problem(problem)
        assert cache.zero_var_rule == "exact"
        assert ChanceState.from_selection(cache, s).penalty_total() == expected


def _prop_one_problem(seed: int, **params) -> ConservationProblem:
    """300 PUs, 10 features each in about half of them, every cell certain (q = 1) with an
    amount in U(0.01, 10), and prop = 1 resolved as ``load_project`` does
    (``_resolve_prop_targets``: a pandas groupby sum); species-major rows for odd seeds."""
    rng = np.random.default_rng(seed)
    n_pu, n_feat = 300, 10
    r, c = np.nonzero(rng.random((n_pu, n_feat)) < 0.5)
    pv = pd.DataFrame({"species": c + 1, "pu": r + 1,
                       "amount": rng.uniform(0.01, 10, len(r)), "prob": 1.0})
    if seed % 2:
        pv = pv.sort_values(["species", "pu"]).reset_index(drop=True)
    feat = _resolve_prop_targets(pd.DataFrame({
        "id": np.arange(1, n_feat + 1), "name": [f"f{j}" for j in range(n_feat)],
        "target": 0.0, "prop": 1.0, "spf": 1.0, "ptarget": 0.9,
    }), pv)
    pu = pd.DataFrame({"id": np.arange(1, n_pu + 1), "cost": 1.0, "status": 0})
    return ConservationProblem(planning_units=pu, features=feat, pu_vs_features=pv,
                               parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 10.0,
                                           **params})


@pytest.mark.parametrize("seed", [0, 1])
def test_prop_one_targets_of_fully_held_certain_features_are_met(seed: int) -> None:
    """A prop = 1 target is the feature's total as a pandas groupby sum; the evaluator's
    E is a sequential sum, so the two can differ by an ulp or two. Here at least one
    feature's E is below its target in float. With every PU selected and every cell
    certain, the exact rule finds every feature met (the tolerance), as the deterministic
    target check does, in the evaluator, the reported Solution and ChanceState."""
    problem = _prop_one_problem(seed, PROBZEROVAR=2)
    sel = np.ones(len(problem.planning_units), dtype=bool)
    fids = problem.features["id"].to_numpy()
    E, _, N = _term_moments(problem, sel, "2d", fids)
    assert (N == 0).all()
    assert (E < problem.features["target"].to_numpy()).any()  # the layout needs the tolerance
    shortfalls, penalty = evaluate_solution_chance(problem, sel)
    assert shortfalls == dict.fromkeys(fids.tolist(), 0.0) and penalty == 0.0
    sol = build_solution(problem, sel, 0.0)
    assert sol.prob_penalty == 0.0 and all(sol.targets_met.values())
    cache = ProblemCache.from_problem(problem)
    assert cache.zero_var_rule == "exact"
    assert ChanceState.from_selection(cache, sel).penalty_total() == 0.0


def test_a_certain_amount_short_by_more_than_the_tolerance_is_missed() -> None:
    """The tolerance is relative (1e-7 of the target): leaving out one unit leaves every
    feature it holds short by that unit's amount (≥ 0.01, far more than 1e-7 of a total of
    about 750), so each is missed outright (shortfall ptarget, penalty the weighting)."""
    problem = _prop_one_problem(0, PROBZEROVAR=2)
    pv = problem.pu_vs_features
    sel = np.ones(len(problem.planning_units), dtype=bool)
    pu = int(pv.loc[pv["species"] == 1, "pu"].iloc[0])
    sel[pu - 1] = False
    held = set(pv.loc[pv["pu"] == pu, "species"].astype(int).tolist())
    shortfalls, penalty = evaluate_solution_chance(problem, sel)
    assert shortfalls == {f: (0.9 if f in held else 0.0)
                          for f in problem.features["id"].astype(int).tolist()}
    assert penalty == 10.0 * len(held)
    cache = ProblemCache.from_problem(problem)
    assert ChanceState.from_selection(cache, sel).penalty_total() == penalty


def _one_feature(amounts: list[float], target: float, **params) -> ConservationProblem:
    n = len(amounts)
    pu = pd.DataFrame({"id": np.arange(1, n + 1), "cost": 1.0, "status": 0})
    pv = pd.DataFrame({"species": 1, "pu": np.arange(1, n + 1), "amount": amounts,
                       "prob": 1.0})
    feat = pd.DataFrame({"id": [1], "name": ["f"], "target": [target], "spf": [1.0],
                         "ptarget": [0.9]})
    return ConservationProblem(planning_units=pu, features=feat, pu_vs_features=pv,
                               parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 10.0,
                                           "PROBZEROVAR": 2, **params})


def test_a_zero_target_with_a_negative_certain_amount_is_short() -> None:
    """Target 0 has no tolerance: a negative certain amount is short (P = 0), not met;
    nothing selected (E = 0) and a positive amount are met."""
    problem = _one_feature([-2.0, 3.0], 0.0)
    cache = ProblemCache.from_problem(problem)
    for sel, expected in (([True, False], 10.0), ([False, False], 0.0),
                          ([False, True], 0.0), ([True, True], 0.0)):
        s = np.array(sel)
        sf, tot = evaluate_solution_chance(problem, s)
        assert tot == expected and sf[1] == (0.9 if expected else 0.0)
        assert ChanceState.from_selection(cache, s).penalty_total() == expected


def test_a_negative_target_uses_its_absolute_tolerance() -> None:
    """threshold = target − 1e-7·|target|: for target −4 that is −4.0000004."""
    tol = TARGET_REL_TOL * 4.0
    for amount, expected in ((-4.0, 0.0), (-4.0 - 0.5 * tol, 0.0), (-4.0 - 2 * tol, 10.0)):
        problem = _one_feature([amount], -4.0)
        assert evaluate_solution_chance(problem, np.array([True]))[1] == expected


# ------------------------------------------------------------------ parameter handling


@pytest.mark.parametrize("value", [2, 2.0, "2"])
def test_rule_from_parameter(value) -> None:
    problem, _ = hand_example(PROBZEROVAR=value)
    assert _zero_var_rule(problem) == "exact"
    assert not any("PROBZEROVAR" in e for e in problem.validate())


@pytest.mark.parametrize("value", [3, -1, 1.5, "two"])
def test_validate_reports_other_values(value) -> None:
    problem, _ = hand_example(PROBZEROVAR=value)
    errors = [e for e in problem.validate() if "PROBZEROVAR" in e]
    assert errors and "0" in errors[0] and "1" in errors[0] and "2" in errors[0]
    assert _zero_var_rule(problem) == "marxan"


def test_round_trips_through_input_dat(tmp_path) -> None:
    problem, _ = hand_example(PROBZEROVAR=2)
    save_project(problem, tmp_path)
    loaded = load_project(tmp_path)
    assert loaded.parameters["PROBZEROVAR"] == 2
    assert _zero_var_rule(loaded) == "exact"
