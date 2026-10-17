"""Stage 4 (spec §2-§4, §8): MIPSolver proportion / semicontinuous decisions."""
from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pulp
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers._backends import _available_backends
from pymarxan.solvers._decisions import (
    SNAP_TOL,
    ZERO_TOL,
    _extract_selected,
    _validate_decisions,
)
from pymarxan.solvers.base import Solution, SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver
from tests.pymarxan.fractional_fixtures import rejected_cases, with_conn
from tests.pymarxan.lp_canon import captured_models

ONE = SolverConfig(num_solutions=1)


def _solve(problem: ConservationProblem, config: SolverConfig = ONE, **kw: Any) -> list[Solution]:
    kw.setdefault("mip_backend", "cbc")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # PuLP 3.3.2
        return MIPSolver(**kw).solve(problem, config)


# Spec §8: the targets-met tests run under every installed LP backend (HiGHS in CI).
BACKENDS = [
    pytest.param(
        b, marks=pytest.mark.skipif(not _available_backends().get(b), reason=f"{b} missing"),
    )
    for b in ("cbc", "highs")
]


def _one(problem: ConservationProblem, **kw: Any) -> Solution:
    sols = _solve(problem, **kw)
    assert sols, kw
    return sols[0]


def _obj(problem: ConservationProblem, **kw: Any) -> float:
    return float(_one(problem, **kw).metadata["mip_objective_value"])


def random_problem(
    seed: int, *, scale: float = 1.0, connectivity: str | None = None, n_pu: int = 10,
) -> ConservationProblem:
    """Seeded random problem: chain boundary, targets at 20-50% of each feature's total."""
    rng = np.random.default_rng(seed)
    ids = list(range(1, n_pu + 1))
    pu = pd.DataFrame({
        "id": ids, "cost": rng.uniform(1.0, 20.0, n_pu).round(3), "status": [0] * n_pu,
    })
    rows = []
    for f in (1, 2, 3, 4):
        for p in ids:
            if rng.random() < 0.6:
                rows.append((f, p, float(rng.uniform(0.5, 10.0)) * scale))
    pvf = pd.DataFrame(rows, columns=["species", "pu", "amount"])
    totals = pvf.groupby("species")["amount"].sum()
    feat = pd.DataFrame({
        "id": [1, 2, 3, 4], "name": list("abcd"),
        "target": [float(totals.get(f, 0.0)) * rng.uniform(0.2, 0.5) for f in (1, 2, 3, 4)],
        "spf": [1.0] * 4,
    })
    bnd = pd.DataFrame({
        "id1": ids[:-1] + ids, "id2": ids[1:] + ids,
        "boundary": list(rng.uniform(0.5, 3.0, n_pu - 1)) + list(rng.uniform(0.0, 2.0, n_pu)),
    })
    params: dict[str, Any] = {"BLM": float(rng.uniform(0.1, 2.0))}
    conn = None
    if connectivity is not None:
        pairs = rng.choice(n_pu, size=(8, 2))
        pairs = np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)  # one row per pair
        conn = pd.DataFrame({
            "id1": pairs[:, 0] + 1, "id2": pairs[:, 1] + 1,
            "value": rng.uniform(0.5, 5.0, len(pairs)),
        })
        params["CONNECTIVITY_WEIGHT"] = 1.0
        params["ASYMMETRIC_CONNECTIVITY"] = int(connectivity == "asym")
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pvf, boundary=bnd,
        parameters=params, connectivity=conn,
    )


def identity_problem() -> ConservationProblem:
    """One feature per PU (identity rij), targets = amounts: the LP optimum is integral."""
    pu = pd.DataFrame({"id": [1, 2, 3, 4], "cost": [3.0, 1.0, 4.0, 1.5], "status": [0] * 4})
    feat = pd.DataFrame({
        "id": [1, 2, 3, 4], "name": list("abcd"), "target": [2.0, 5.0, 1.0, 7.0],
        "spf": [1.0] * 4,
    })
    pvf = pd.DataFrame({"species": [1, 2, 3, 4], "pu": [1, 2, 3, 4],
                        "amount": [2.0, 5.0, 1.0, 7.0]})
    return ConservationProblem(planning_units=pu, features=feat, pu_vs_features=pvf)


# --- validation ---------------------------------------------------------------------


def test_defaults_are_binary() -> None:
    s = MIPSolver()
    assert (s.decisions, s.upper, s.lower) == ("binary", 1.0, 0.0)
    s = MIPSolver(decisions="semicontinuous", upper=0.8, lower=0.2)
    assert (s.decisions, s.upper, s.lower) == ("semicontinuous", 0.8, 0.2)
    assert _validate_decisions("proportion", 1, 0) == ("proportion", 1.0, 0.0)


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"decisions": "x"}, "decisions must be one of 'binary', 'proportion', "
                             "'semicontinuous', got 'x'"),
        ({"decisions": "binary", "upper": 0.5}, "upper only applies"),
        ({"decisions": "binary", "lower": 0.5}, "lower only applies to "
                                                "decisions='semicontinuous'"),
        ({"decisions": "proportion", "lower": 0.1}, "lower only applies"),
        ({"decisions": "proportion", "upper": 1.5}, r"upper must be in \(0, 1\], got 1.5"),
        ({"decisions": "proportion", "upper": 0.0}, r"upper must be in \(0, 1\]"),
        ({"decisions": "proportion", "upper": True}, "upper must be a finite number"),
        ({"decisions": "proportion", "upper": float("nan")}, "upper must be a finite"),
        ({"decisions": "proportion", "upper": "1"}, "upper must be a finite"),
        ({"decisions": "semicontinuous", "lower": 0.0}, r"lower must be in \(0, upper\]"),
        ({"decisions": "semicontinuous", "upper": 0.5, "lower": 0.6}, "lower must be in"),
        ({"decisions": "semicontinuous", "lower": float("inf")}, "lower must be a finite"),
    ],
)
def test_validation(kw: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        MIPSolver(**kw)


# --- rejected combinations (spec §3.5) ----------------------------------------------


@pytest.mark.parametrize("decisions", ["proportion", "semicontinuous"])
@pytest.mark.parametrize(
    "cause",
    ["PROBMODE 3", "clumping", "separation", "space", "forbidden_selections",
     "negative connectivity"],
)
def test_rejected_combinations(
    tiny_problem: ConservationProblem, decisions: str, cause: str,
) -> None:
    problem, config = rejected_cases(tiny_problem)[cause]
    kw: dict[str, Any] = {"decisions": decisions}
    if decisions == "semicontinuous":
        kw["lower"] = 0.3
    solver = MIPSolver(mip_backend="cbc", **kw)
    with pytest.raises(ValueError, match=cause) as exc:
        solver.solve(problem, config)
    assert "decisions='binary'" in str(exc.value)
    with pytest.raises(ValueError, match=cause):
        solver.build_model(problem, config)


def test_rejection_precedes_strategy_errors(tiny_problem: ConservationProblem) -> None:
    """The gate sits above the NotImplementedError strategy checks."""
    p = tiny_problem.copy_with(parameters={**tiny_problem.parameters, "PROBMODE": 3})
    with pytest.raises(ValueError, match="PROBMODE 3"):
        MIPSolver(decisions="proportion", mip_chance_strategy="socp").build_model(p)


def test_binary_allows_the_rejected_combinations(tiny_problem: ConservationProblem) -> None:
    problem, config = rejected_cases(tiny_problem)["negative connectivity"]
    assert _solve(problem, config)
    problem, config = rejected_cases(tiny_problem)["forbidden_selections"]
    assert _solve(problem, config)


def test_negative_connectivity_ignored_without_weight(tiny_problem: ConservationProblem) -> None:
    p = with_conn(tiny_problem, [4.0, -3.0, 2.5, 6.0, 1.5], weight=0.0)
    assert _one(p, decisions="proportion").is_fractional


# --- model shape --------------------------------------------------------------------


def test_proportion_model_variables(tiny_problem: ConservationProblem) -> None:
    p = with_conn(tiny_problem)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        model = MIPSolver(decisions="proportion", upper=0.7).build_model(p)
    v = {var.name: var for var in model.variables()}
    assert v["x_1"].cat == "Continuous"
    assert (v["x_1"].lowBound, v["x_1"].upBound) == (0, 0.7)
    assert v["zc_1_2"].cat == "Continuous"
    assert (v["zc_1_2"].lowBound, v["zc_1_2"].upBound) == (0, 1)
    assert not any(n.startswith("s_") for n in v)


def test_semicontinuous_model_rows(tiny_problem: ConservationProblem) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        model = MIPSolver(decisions="semicontinuous", upper=0.9, lower=0.3).build_model(
            tiny_problem,
        )
        rows = dict(model.constraints)
    v = {var.name: var for var in model.variables()}
    assert v["s_1"].cat == "Integer" and (v["s_1"].lowBound, v["s_1"].upBound) == (0, 1)
    ub, lb = rows["sc_ub_1"], rows["sc_lb_1"]
    assert ub.sense == pulp.LpConstraintLE and lb.sense == pulp.LpConstraintGE
    assert {k.name: a for k, a in ub.items()} == {"x_1": 1, "s_1": -0.9}
    assert {k.name: a for k, a in lb.items()} == {"x_1": 1, "s_1": -0.3}


# --- solves -------------------------------------------------------------------------


def test_proportion_solution_is_float(tiny_problem: ConservationProblem) -> None:
    s = _one(tiny_problem, decisions="proportion")
    assert s.is_fractional
    assert s.selected.dtype == np.float64
    assert ((s.selected >= 0) & (s.selected <= 1)).all()
    assert _obj(tiny_problem, decisions="proportion") == pytest.approx(35.583, abs=1e-3)


def test_lp_bound_tiny(tiny_problem: ConservationProblem) -> None:
    lp = _obj(tiny_problem, decisions="proportion")
    assert lp <= _obj(tiny_problem) + 1e-6
    assert _obj(tiny_problem) == pytest.approx(43.0)  # the binary BLM-1 optimum


@pytest.mark.parametrize("seed", range(21))
def test_lp_bound_random(seed: int) -> None:
    connectivity = ("sym", "asym")[seed % 2]
    p = random_problem(seed, connectivity=connectivity)
    binary = _obj(p)
    assert _obj(p, decisions="proportion") <= binary + 1e-6 * max(1.0, abs(binary))


def test_integral_lp_fixture() -> None:
    p = identity_problem()
    s = _one(p, decisions="proportion")
    assert set(s.selected.tolist()) <= {0.0, 1.0}
    assert s.metadata["mip_objective_value"] == pytest.approx(_obj(p))


@pytest.mark.parametrize("lower", [0.3, 0.6, 0.9])
def test_semicontinuous_floor(tiny_problem: ConservationProblem, lower: float) -> None:
    s = _one(tiny_problem, decisions="semicontinuous", lower=lower)
    x = s.selected
    assert s.is_fractional
    assert ((x == 0.0) | ((x >= lower) & (x <= 1.0))).all(), x
    sc = float(s.metadata["mip_objective_value"])
    prop, binary = _obj(tiny_problem, decisions="proportion"), _obj(tiny_problem)
    assert prop - 1e-6 <= sc <= binary + 1e-6


def test_upper_cap_and_locks(tiny_problem: ConservationProblem) -> None:
    pu = tiny_problem.planning_units.copy()
    pu.loc[pu["id"] == 3, "status"] = 2
    pu.loc[pu["id"] == 6, "status"] = 3
    p = tiny_problem.copy_with(
        planning_units=pu, parameters={**tiny_problem.parameters, "MISSLEVEL": 0.8},
    )
    s = _one(p, decisions="proportion", upper=0.5)
    free = np.ones(6, dtype=bool)
    free[[2, 5]] = False
    assert (s.selected[free] <= 0.5).all()
    assert s.selected[2] == 1.0  # locked in: whole in every mode (review R3)
    assert s.selected[5] == 0.0
    sc = _one(p, decisions="semicontinuous", upper=0.5, lower=0.2)
    assert sc.selected[2] == 1.0 and sc.selected[5] == 0.0
    assert (sc.selected[free] <= 0.5).all()


@pytest.mark.parametrize("backend", BACKENDS)
def test_targets_met_at_lp_optimum_tiny(tiny_problem: ConservationProblem, backend: str) -> None:
    s = _one(tiny_problem, decisions="proportion", mip_backend=backend)
    assert s.all_targets_met
    assert s.shortfall == 0.0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("scale", [1e-6, 1e-3, 1.0, 1e4, 1e8])
@pytest.mark.parametrize("seed", range(20))
def test_targets_met_at_lp_optimum_random(seed: int, scale: float, backend: str) -> None:
    s = _one(random_problem(100 + seed, scale=scale), decisions="proportion", mip_backend=backend)
    assert s.all_targets_met, s.targets_met
    assert s.shortfall == 0.0


def test_connectivity_reward_survives(tiny_problem: ConservationProblem) -> None:
    """Fractional z > 0 occurs, and the objective beats the same model with z binary."""
    p = with_conn(tiny_problem, weight=2.0)
    with captured_models() as models:
        s = _one(p, decisions="proportion")
    z = {v.name: v.varValue for v in models[0].variables() if v.name.startswith("zc_")}
    assert any(0.0 < (val or 0.0) < 1.0 for val in z.values()), z
    lp = float(s.metadata["mip_objective_value"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        model = MIPSolver(decisions="proportion").build_model(p)
        for v in model.variables():
            if v.name.startswith("zc_"):
                v.cat = pulp.LpInteger  # bounds are already [0, 1]
        model.solve(pulp.PULP_CBC_CMD(msg=False))
    assert lp < float(pulp.value(model.objective)) - 1e-6


@pytest.mark.parametrize(
    "kw", [{"decisions": "proportion"}, {"decisions": "proportion", "upper": 0.8},
           {"decisions": "semicontinuous", "lower": 0.4}],
)
def test_objective_agreement_min_set(tiny_problem: ConservationProblem, kw: dict) -> None:
    for p in (tiny_problem, random_problem(7), random_problem(8, scale=1e4)):
        if kw.get("upper", 1.0) < 1.0:
            p = p.copy_with(parameters={**p.parameters, "MISSLEVEL": 0.5})
        s = _one(p, **kw)
        obj = float(s.metadata["mip_objective_value"])
        tol = 1e-6 * max(1.0, abs(obj)) + SNAP_TOL * float(p.planning_units["cost"].sum())
        assert abs(s.objective - obj) <= tol


@pytest.mark.parametrize("asymmetric", [False, True])
@pytest.mark.parametrize(
    "kw", [{}, {"decisions": "proportion"}, {"decisions": "semicontinuous", "lower": 0.4}],
)
def test_objective_agreement_connectivity(
    tiny_problem: ConservationProblem, kw: dict, asymmetric: bool,
) -> None:
    """The reported objective uses the MIP's connectivity term, so the two agree."""
    kind = "asym" if asymmetric else "sym"
    for p in (with_conn(tiny_problem, asymmetric=asymmetric),
              random_problem(11, connectivity=kind), random_problem(12, connectivity=kind)):
        s = _one(p, **kw)
        obj = float(s.metadata["mip_objective_value"])
        tol = 1e-6 * max(1.0, abs(obj)) + SNAP_TOL * float(p.planning_units["cost"].sum())
        assert abs(s.objective - obj) <= tol, (kind, kw)


def test_infeasible_returns_empty() -> None:
    """upper 0.5 cannot reach targets equal to the full amounts: [] as for binary."""
    p = identity_problem()
    assert _solve(p, decisions="proportion", upper=0.5) == []
    assert _solve(p, decisions="semicontinuous", upper=0.5, lower=0.2) == []


def test_binary_solve_unchanged(tiny_problem: ConservationProblem) -> None:
    s = _one(tiny_problem.copy_with(parameters={**tiny_problem.parameters, "BLM": 0.0}))
    assert s.selected.dtype == bool
    assert s.selected.tolist() == [False, True, False, True, False, True]
    assert s.cost == 35.0


# --- extraction ---------------------------------------------------------------------


def test_extract_binary() -> None:
    out = _extract_selected([0.0, 1.0, None, 0.9999, 0.4], [], "binary", 1.0, 0.0)
    assert out.dtype == bool
    assert out.tolist() == [False, True, False, True, False]


def test_extract_proportion_snapping() -> None:
    """Review R5: no snap up to upper; only clip and dust."""
    vals = [None, 5e-10, 1e-8, 0.4, 0.8 - 5e-7, 0.8, 0.8 + 1e-9, 0.7]
    out = _extract_selected(vals, [], "proportion", 0.8, 0.0)
    assert out.dtype == np.float64
    assert out.tolist() == [0.0, 0.0, 1e-8, 0.4, 0.8 - 5e-7, 0.8, 0.8, 0.7]
    assert ZERO_TOL == 1e-9 and SNAP_TOL == 1e-6


def test_extract_per_pu_upper() -> None:
    """Each PU is clipped to its own bound (1.0 for a locked-in unit, review R3)."""
    out = _extract_selected([1.0, 1.0, 0.5 + 1e-9], [], "proportion", [1.0, 0.5, 0.5], 0.0)
    assert out.tolist() == [1.0, 0.5, 0.5]


def test_extract_semicontinuous_snapping() -> None:
    vals = [0.3 - 5e-7, 0.3, 0.5, 1e-7, 1.0 - 1e-7, 0.2, 1.0]
    s = [1.0, 1.0, 1.0, 0.0, 1.0, 1e-9, None]  # None: no on/off variable (locked)
    out = _extract_selected(vals, s, "semicontinuous", 1.0, 0.3)
    assert out.tolist() == [0.3, 0.3, 0.5, 0.0, 1.0 - 1e-7, 0.0, 1.0]


@pytest.mark.parametrize("seed", range(5))
def test_extract_never_lowers_except_dust_s0_or_bound(seed: int) -> None:
    rng = np.random.default_rng(seed)
    upper, lower = 0.9, 0.25
    vals = rng.uniform(0.0, 1.0, 200)
    vals[:20] = rng.uniform(0.0, 2e-9, 20)
    s = np.ones(200)
    for decisions in ("proportion", "semicontinuous"):
        out = _extract_selected(list(vals), list(s) if decisions != "proportion" else [],
                                decisions, upper, lower if decisions != "proportion" else 0.0)
        for v, o in zip(vals, out, strict=True):
            if o < v:
                assert (v <= ZERO_TOL and o == 0.0) or (v > upper and o == upper), (v, o)
            assert 0.0 <= o <= upper
            assert not math.isnan(o)


def test_semicontinuous_locks(tiny_problem: ConservationProblem) -> None:
    """Review focus (R3): locked in is whole (1.0), locked out 0, free x in {0} ∪ [lower,
    upper], with lower < upper < 1."""
    pu = tiny_problem.planning_units.copy()
    pu.loc[pu["id"] == 3, "status"] = 2
    pu.loc[pu["id"] == 6, "status"] = 3
    p = tiny_problem.copy_with(
        planning_units=pu, parameters={**tiny_problem.parameters, "MISSLEVEL": 0.6},
    )
    s = _one(p, decisions="semicontinuous", upper=0.7, lower=0.3)
    assert s.selected[2] == 1.0
    assert s.selected[5] == 0.0
    x = np.delete(s.selected, [2, 5])
    assert ((x == 0.0) | ((x >= 0.3) & (x <= 0.7))).all(), x


def test_proportion_unwritten_pu_reads_zero(tiny_problem: ConservationProblem) -> None:
    """Review focus: a zero-cost PU in no pu_vs_features row never enters the LP (None)."""
    pu = pd.concat(
        [tiny_problem.planning_units,
         pd.DataFrame({"id": [7], "cost": [0.0], "status": [0]})],
        ignore_index=True,
    )
    p = tiny_problem.copy_with(planning_units=pu, boundary=None)
    for kw in ({"decisions": "proportion"}, {"decisions": "semicontinuous", "lower": 0.3}):
        s = _one(p, **kw)
        assert s.selected.dtype == np.float64
        assert s.selected[6] == 0.0
        assert s.all_targets_met


def test_snap_does_not_raise_genuine_zero_with_tiny_lower() -> None:
    out = _extract_selected([0.0, 0.5], [None, 1.0], "semicontinuous", 1.0, 1e-7)
    assert out.tolist() == [0.0, 0.5]


def test_locked_out_unit_stays_zero_with_tiny_lower() -> None:
    from pymarxan.io.readers import load_project

    problem = load_project("tests/data/simple")
    problem.planning_units.loc[problem.planning_units["id"] == 1, "status"] = 3
    sol = MIPSolver(
        decisions="semicontinuous", upper=1.0, lower=1e-7, mip_backend="cbc",
    ).solve(problem, ONE)[0]
    assert sol.selected[0] == 0.0
    assert sol.n_selected == int((sol.selected > 0).sum())
