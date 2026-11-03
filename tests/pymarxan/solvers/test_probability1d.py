"""Probability 1D: explicit PROBMODE 4 (1D) and 5 (1D + 2D) — evaluator, reports, MIP.

Reference: Marxan 3 (``marxan3`` branch) ``probability.c``. Per feature, over the
selected PUs, with ``p_i`` the per-PU probability of **loss** (pu.dat ``prob``):
``ReturnProbabilityAmounts1D`` adds ``E += a·(1 − p)``, ``V += a·a·p·(1 − p)`` per puvspr
row; ``ComputeProbability1D`` gives ``Z = (target − E)/√V`` (``V > 0``, else 4),
``P = probZUT(Z)``, penalty ``[ptarget1d > P]·(ptarget1d − P)/ptarget1d`` (0 when
``ptarget1d ≤ 0``), times ``PROBABILITYWEIGHTING``. Marxan's total adds both terms.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pymarxan.io.writers import write_mvbest
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver
from pymarxan.solvers.probability import evaluate_solution_chance
from pymarxan.solvers.utils import (
    _active_chance_terms,
    _chance_terms,
    build_solution,
    compute_held,
    compute_objective_terms,
)


def probZUT(z: float) -> float:
    """Ibbetson (CACM Alg. 209) upper tail, Z_MAX = 5 — Marxan 3 probability.c ``probZUT``."""
    z_max = 5.0
    if z == 0.0:
        x = 0.0
    else:
        y = 0.5 * abs(z)
        if y >= z_max * 0.5:
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


def marxan3_probability1d(problem: ConservationProblem, selected: np.ndarray):
    """Test-local port of ``ReturnProbabilityAmounts1D`` + ``ComputeProbability1D``.

    Returns per-feature ``(Z, P, penalty)`` lists and the weighted total."""
    pu = problem.planning_units
    p_of = dict(zip(pu["id"].astype(int), pu["prob"].astype(float)))
    sel_ids = {int(i) for i, s in zip(pu["id"], selected) if s}
    feats = problem.features
    fidx = {int(f): k for k, f in enumerate(feats["id"])}
    E = [0.0] * len(feats)
    V = [0.0] * len(feats)
    for sp, ipu, a in zip(problem.pu_vs_features["species"], problem.pu_vs_features["pu"],
                          problem.pu_vs_features["amount"]):
        if int(ipu) in sel_ids and a:
            p = p_of[int(ipu)]
            E[fidx[int(sp)]] += a * (1 - p)
            V[fidx[int(sp)]] += a * a * p * (1 - p)
    out = []
    total = 0.0
    for k, (t, pt) in enumerate(zip(feats["target"], feats["ptarget1d"])):
        z = (t - E[k]) / math.sqrt(V[k]) if V[k] > 0 else 4.0
        P = probZUT(z) if z >= 0 else 1 - probZUT(-1 * z)
        pen = (pt - P) / pt if (pt > P and pt > 0) else 0.0
        out.append((z, P, pen))
        total += pen
    return out, total * float(problem.parameters.get("PROBABILITYWEIGHTING", 1.0))


def _hand_problem(**params) -> ConservationProblem:
    """Five PUs (shuffled ids), four features.

    f1: p ≠ 0.5 cells, E < target (positive Z). f2: E > target (negative Z).
    f3: only p = 0 PUs selected → zero variance, ptarget1d > probZUT(4).
    f4: only in an unselected PU. ptarget1d differs between features.
    """
    pu = pd.DataFrame({
        "id": [40, 10, 30, 20, 50], "cost": [1.0, 2.0, 3.0, 4.0, 5.0], "status": 0,
        "prob": [0.1, 0.3, 0.0, 0.25, 0.6],
    })
    features = pd.DataFrame({
        "id": [1, 2, 3, 4], "name": ["f1", "f2", "f3", "f4"],
        "target": [12.0, 3.0, 4.0, 2.0], "spf": 1.0,
        "ptarget1d": [0.9, 0.6, 0.5, 0.75],
        "ptarget": [-1.0, -1.0, -1.0, -1.0],
    })
    pv = pd.DataFrame({
        "species": [1, 1, 1, 2, 2, 3, 4],
        "pu": [40, 10, 20, 40, 20, 30, 50],
        "amount": [5.0, 6.0, 3.0, 4.0, 2.0, 7.0, 9.0],
    })
    return ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=pv,
        parameters={"PROBMODE": 4, "PROBABILITYWEIGHTING": 3.0, **params},
    )


HAND_SEL = np.array([True, True, True, True, False])  # PU 50 unselected


class TestMarxan3Transcription:
    def test_score_matches_marxan3(self):
        problem = _hand_problem()
        ref, ref_total = marxan3_probability1d(problem, HAND_SEL)
        assert ref[0][0] > 0 and ref[1][0] < 0 and ref[2][0] == 4.0  # the cases are hit
        shortfalls, penalty = evaluate_solution_chance(problem, HAND_SEL, "1d")
        for fid, (_, P, _) in zip([1, 2, 3, 4], ref):
            pt = float(problem.features.set_index("id").loc[fid, "ptarget1d"])
            assert shortfalls[fid] == pytest.approx(max(0.0, pt - P), abs=1e-6)
        # penalty differences are P differences / ptarget (spec §1)
        assert penalty == pytest.approx(ref_total, abs=3.0 * 1e-6 / 0.5 * 4)

    def test_solution_fields_and_objective(self):
        problem = _hand_problem()
        sol = build_solution(problem, HAND_SEL, 0.0)
        assert sol.prob_shortfalls is None and sol.prob_penalty is None  # 2D off
        assert sol.prob1d_shortfalls is not None
        _, penalty = evaluate_solution_chance(problem, HAND_SEL, "1d")
        assert sol.prob1d_penalty == pytest.approx(penalty)
        terms = compute_objective_terms(
            problem, HAND_SEL, {int(i): k for k, i in enumerate(problem.planning_units.id)},
            0.0,
        )
        assert terms["probability"] == pytest.approx(penalty)

    def test_unselected_feature_is_zero_variance(self):
        problem = _hand_problem()
        sf, _ = evaluate_solution_chance(problem, HAND_SEL, "1d")
        assert sf[4] == pytest.approx(0.75 - probZUT(4.0), abs=1e-6)


# --------------------------------------------------------------- equivalences


def _random_pair(seed: int, n_pu: int = 30, n_feat: int = 4):
    """A mode-4 problem with a prob.dat source (shuffled, non-contiguous ids, a PU missing
    from prob.dat) and its mode-3 twin with q = 1 − p, ptarget explicitly mapped from
    ptarget1d (which differs from the mode-4 problem's own ptarget on purpose)."""
    rng = np.random.default_rng(seed)
    ids = rng.permutation(np.arange(1, n_pu + 1) * 11)
    pu = pd.DataFrame({"id": ids, "cost": rng.uniform(1, 5, n_pu), "status": 0})
    rows = [(f, int(i)) for f in range(1, n_feat + 1) for i in ids if rng.random() < 0.5]
    pv = pd.DataFrame(rows, columns=["species", "pu"])
    pv["amount"] = rng.uniform(0.5, 5, len(pv))
    p = rng.uniform(1e-6, 0.6, n_pu)
    missing = int(ids[3])  # absent from prob.dat: p = 0
    table = pd.DataFrame({"pu": ids, "probability": p})
    table = table[table.pu != missing].sample(frac=1.0, random_state=seed)
    p_of = dict(zip(table.pu, table.probability))
    tot = pv.amount.groupby(pv.species).sum().reindex(range(1, n_feat + 1)).to_numpy()
    feat = pd.DataFrame({
        "id": range(1, n_feat + 1), "name": [f"f{j}" for j in range(n_feat)],
        "target": tot * rng.uniform(0.2, 0.5, n_feat), "spf": 1.0,
        "ptarget1d": [0.9, 0.55, 0.7, -1.0][:n_feat],
        "ptarget": [0.1, 0.95, -1.0, 0.8][:n_feat],
    })
    mode4 = ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pv, probability=table,
        parameters={"PROBMODE": 4, "PROBABILITYWEIGHTING": 7.0},
    )
    pv3 = pv.assign(prob=[1.0 - p_of.get(int(u), 0.0) for u in pv.pu])
    feat3 = feat.assign(ptarget=feat["ptarget1d"]).drop(columns=["ptarget1d"])
    mode3 = ConservationProblem(
        planning_units=pu, features=feat3, pu_vs_features=pv3,
        parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 7.0},
    )
    return mode4, mode3


@pytest.mark.parametrize("seed", range(4))
def test_mode4_equals_mode3_with_presence_one_minus_p(seed):
    mode4, mode3 = _random_pair(seed)
    rng = np.random.default_rng(seed + 100)
    for _ in range(5):
        sel = rng.random(len(mode4.planning_units)) < 0.6
        sf4, pen4 = evaluate_solution_chance(mode4, sel, "1d")
        sf3, pen3 = evaluate_solution_chance(mode3, sel, "2d")
        assert pen4 == pytest.approx(pen3, rel=1e-12, abs=1e-14)
        for fid in sf3:
            assert sf4[fid] == pytest.approx(sf3[fid], rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("seed", range(3))
def test_mode5_equals_mode3_plus_mode4(seed):
    mode4, _ = _random_pair(seed)
    rng = np.random.default_rng(seed + 7)
    # A 2D distribution unrelated to the 1D one, and ptarget ≠ ptarget1d.
    pv = mode4.pu_vs_features.assign(prob=rng.uniform(0.2, 1.0, len(mode4.pu_vs_features)))
    base = mode4.copy_with(pu_vs_features=pv)
    p5 = base.copy_with(parameters={**base.parameters, "PROBMODE": 5})
    p4 = base.copy_with(parameters={**base.parameters, "PROBMODE": 4})
    p3 = base.copy_with(parameters={**base.parameters, "PROBMODE": 3})
    assert _active_chance_terms(p5) == {"1d", "2d"}
    for _ in range(4):
        sel = rng.random(len(base.planning_units)) < 0.5
        s5, s4, s3 = (build_solution(q, sel, 0.0) for q in (p5, p4, p3))
        assert s5.prob1d_penalty == pytest.approx(s4.prob1d_penalty, rel=1e-12)
        assert s5.prob_penalty == pytest.approx(s3.prob_penalty, rel=1e-12)
        assert s5.objective == pytest.approx(
            s4.objective + s3.objective - s4.cost - s4.penalty, rel=1e-12,
        )
        assert s5.prob1d_shortfalls == s4.prob1d_shortfalls
        assert s5.prob_shortfalls == s3.prob_shortfalls


def test_mode5_keeps_held_amounts_and_costs_raw():
    mode4, _ = _random_pair(1)
    p5 = mode4.copy_with(
        pu_vs_features=mode4.pu_vs_features.assign(prob=0.7),
        parameters={**mode4.parameters, "PROBMODE": 5},
    )
    p0 = mode4.copy_with(parameters={**mode4.parameters, "PROBMODE": 0})
    sel = np.ones(len(p5.planning_units), dtype=bool)
    idx = {int(i): k for k, i in enumerate(p5.planning_units.id)}
    assert compute_held(p5, sel, idx) == compute_held(p0, sel, idx)
    s5, s0 = build_solution(p5, sel, 0.0), build_solution(p0, sel, 0.0)
    assert s5.cost == s0.cost
    assert s5.objective == pytest.approx(s0.objective + s5.prob_penalty + s5.prob1d_penalty)
    from pymarxan.solvers.cache import ProblemCache
    c5, c0 = ProblemCache.from_problem(p5), ProblemCache.from_problem(p0)
    assert not c5.prob_active
    np.testing.assert_array_equal(c5.held_csr.data, c0.held_csr.data)


def test_z_uses_the_raw_target_not_misslevel():
    problem = _hand_problem()
    a = evaluate_solution_chance(problem, HAND_SEL, "1d")
    b = evaluate_solution_chance(
        problem.copy_with(parameters={**problem.parameters, "MISSLEVEL": 0.5}), HAND_SEL, "1d",
    )
    assert a == b


class TestZeroVariance:
    def _all_safe(self, **params) -> ConservationProblem:
        problem = _hand_problem(**params)
        return problem.copy_with(planning_units=problem.planning_units.assign(prob=0.0))

    def test_marxan_rule_by_default(self):
        sf, pen = evaluate_solution_chance(self._all_safe(), HAND_SEL, "1d")
        assert sf[1] == pytest.approx(0.9 - probZUT(4.0), abs=1e-6)
        assert pen > 0

    def test_nothing_selected(self):
        sf, _ = evaluate_solution_chance(_hand_problem(), np.zeros(5, dtype=bool), "1d")
        assert sf[2] == pytest.approx(0.6 - probZUT(4.0), abs=1e-6)

    def test_opt_out_counts_as_met(self):
        sf, pen = evaluate_solution_chance(self._all_safe(PROBZEROVAR=1), HAND_SEL, "1d")
        assert pen == 0.0 and set(sf.values()) == {0.0}

    def test_exact_rule(self):
        sf, _ = evaluate_solution_chance(self._all_safe(PROBZEROVAR=2), HAND_SEL, "1d")
        # f1 holds 14 ≥ 12 for certain; f4 (unselected) holds 0 < 2.
        assert sf[1] == 0.0
        assert sf[4] == 0.75


def test_tiny_loss_probability_has_variance():
    problem = _hand_problem()
    pu = problem.planning_units.assign(prob=1e-17)
    sf, _ = evaluate_solution_chance(problem.copy_with(planning_units=pu), HAND_SEL, "1d")
    # V = Σ a²·1e-17 > 0 (not Z = 4): E = 14 ≥ 12, so Z is hugely negative and P = 1.
    assert sf[1] == 0.0


class TestTermWithoutData:
    def test_explicit_4_without_a_1d_source_is_off(self):
        problem = _hand_problem()
        problem = problem.copy_with(planning_units=problem.planning_units.drop(columns="prob"))
        assert _chance_terms(problem) == {"1d"}
        assert _active_chance_terms(problem) == frozenset()
        sol = build_solution(problem, HAND_SEL, 0.0)
        assert sol.prob1d_shortfalls is None and sol.prob1d_penalty is None
        assert evaluate_solution_chance(problem, HAND_SEL, "1d") == (
            {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}, 0.0,
        )

    def test_missing_ptarget1d_column_is_disabled(self):
        problem = _hand_problem()
        problem = problem.copy_with(features=problem.features.drop(columns="ptarget1d"))
        sol = build_solution(problem, HAND_SEL, 0.0)
        assert sol.prob1d_penalty == 0.0
        assert set(sol.prob1d_shortfalls.values()) == {0.0}

    def test_non_finite_pu_prob_raises(self):
        problem = _hand_problem()
        pu = problem.planning_units.assign(prob=[0.1, np.nan, 0.0, 0.2, 0.3])
        with pytest.raises(ValueError, match=r"\[10\]"):
            build_solution(problem.copy_with(planning_units=pu), HAND_SEL, 0.0)


def test_requested_terms_per_mode():
    problem = _hand_problem()
    for mode, terms in [(0, set()), (1, set()), (2, set()), (3, {"2d"}), (4, {"1d"}),
                        (5, {"1d", "2d"})]:
        p = problem.copy_with(parameters={**problem.parameters, "PROBMODE": mode})
        assert _chance_terms(p) == terms


# ------------------------------------------------------------------- reports


class TestMvbest:
    def test_prob1d_gap_column_in_mode_4(self, tmp_path):
        problem = _hand_problem()
        sol = build_solution(problem, HAND_SEL, 0.0)
        out = tmp_path / "mvbest.csv"
        write_mvbest(problem, sol, out)
        df = pd.read_csv(out)
        assert "Prob1D_Gap" in df.columns and "Prob_Gap" not in df.columns
        assert df["Prob1D_Gap"].tolist() == [sol.prob1d_shortfalls[f] for f in [1, 2, 3, 4]]

    def test_both_columns_in_mode_5(self, tmp_path):
        problem = _hand_problem(PROBMODE=5)
        problem = problem.copy_with(
            pu_vs_features=problem.pu_vs_features.assign(prob=0.8),
            features=problem.features.assign(ptarget=0.7),
        )
        sol = build_solution(problem, HAND_SEL, 0.0)
        out = tmp_path / "mvbest.csv"
        write_mvbest(problem, sol, out)
        df = pd.read_csv(out)
        assert {"Prob_Gap", "Prob1D_Gap"} <= set(df.columns)


# ------------------------------------------------------------- MIP and decisions


@pytest.mark.parametrize("mode", [4, 5])
def test_mip_drop_reports_the_chance_terms(mode):
    problem = _hand_problem(PROBMODE=mode)
    problem = problem.copy_with(
        pu_vs_features=problem.pu_vs_features.assign(prob=0.8),
        features=problem.features.assign(ptarget=0.7),
    )
    (sol,) = MIPSolver().solve(problem, SolverConfig(num_solutions=1))
    assert sol.prob1d_penalty is not None and sol.prob1d_penalty > 0  # incl. the Z = 4 rule
    if mode == 5:
        assert sol.prob_penalty is not None
    assert sol.metadata.get("mip_chance_strategy") == "drop"


@pytest.mark.parametrize("strategy", ["piecewise", "socp"])
@pytest.mark.parametrize("mode", [4, 5])
def test_mip_other_strategies_raise(mode, strategy):
    problem = _hand_problem(PROBMODE=mode)
    with pytest.raises(NotImplementedError):
        MIPSolver(mip_chance_strategy=strategy).solve(problem, SolverConfig(num_solutions=1))


@pytest.mark.parametrize("mode", [4, 5])
def test_non_binary_decisions_are_rejected(mode):
    problem = _hand_problem(PROBMODE=mode)
    with pytest.raises(ValueError, match="binary"):
        build_solution(problem, np.full(5, 0.5), 0.0)
    with pytest.raises(ValueError, match="PROBMODE"):
        MIPSolver(decisions="proportion").solve(problem, SolverConfig(num_solutions=1))
