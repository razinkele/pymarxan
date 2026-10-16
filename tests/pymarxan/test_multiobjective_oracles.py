"""Enumeration oracles and prioritizr parity for MultiObjectiveMIP (Stage 3 spec §7.2, §7.3).

Every expected value is derived by brute force over all 2^7 selections of a 7-PU toy, so a
formulation mistake shows up as an oracle mismatch rather than a silent pass.
"""
from __future__ import annotations

import itertools
import warnings
from collections.abc import Sequence
from typing import Any

import pandas as pd
import pytest

import pymarxan.solvers.mip_solver as ms
from pymarxan.models.problem import ConservationProblem
from pymarxan.multiobjective import (
    Hierarchical,
    MaxFeatures,
    MaxWtdSum,
    MinCost,
    MinShortfall,
    MultiObjectiveMIP,
    ObjectiveSpec,
    WeightedSum,
    nondominated,
    weights_matrix,
)
from pymarxan.solvers.base import Solution, SolverConfig
from tests.pymarxan.scripted_solver import Scripted, fail

COST = [4.0, 4.0, 1.0, 3.0, 5.0, 4.0, 2.0]
AMOUNTS = {1: [2.0, 1.0, 2.0, 1.0, 3.0, 0.0, 1.0], 2: [3.0, 0.0, 1.0, 0.0, 0.0, 3.0, 1.0]}
TARGETS = {1: 5.0, 2: 4.0}
SPF = {1: 1.0, 2: 2.0}
WTD = {1: 1.0, 2: 2.0}  # MaxWtdSum weights
ABS_TOL = 1e-9


def toy(status: Sequence[int] | None = None, **params: Any) -> ConservationProblem:
    ids = list(range(1, 8))
    pvf = [(f, pu, a) for f, row in AMOUNTS.items() for pu, a in zip(ids, row) if a > 0]
    return ConservationProblem(
        planning_units=pd.DataFrame({
            "id": ids, "cost": COST, "status": list(status or [0] * 7),
        }),
        features=pd.DataFrame({
            "id": [1, 2], "name": ["a", "b"], "target": [TARGETS[1], TARGETS[2]],
            "spf": [SPF[1], SPF[2]],
        }),
        pu_vs_features=pd.DataFrame(pvf, columns=["species", "pu", "amount"]),
        parameters=params,
    )


# --- the oracle ------------------------------------------------------------------


def value(spec: ObjectiveSpec, sel: tuple[int, ...], problem: ConservationProblem) -> float:
    held = {f: sum(a for a, s in zip(row, sel) if s) for f, row in AMOUNTS.items()}
    if isinstance(spec, MinCost):
        return sum(c for c, s in zip(COST, sel) if s)
    if isinstance(spec, MinShortfall):
        return sum(SPF[f] * max(0.0, TARGETS[f] - held[f]) / TARGETS[f] for f in held)
    if isinstance(spec, MaxWtdSum):
        weights = spec.weights or {f: 1.0 for f in held}
        return sum(weights[f] * held[f] for f in held)
    if isinstance(spec, MaxFeatures):
        return float(sum(held[f] >= TARGETS[f] for f in held))
    raise AssertionError(spec)


def points(
    problem: ConservationProblem, specs: Sequence[ObjectiveSpec],
    forbidden: Sequence[Sequence[int]] = (),
) -> list[tuple[tuple[int, ...], list[float]]]:
    """Every feasible selection with its objective values (locks, budget, no-good cuts)."""
    status = problem.planning_units["status"].tolist()
    budget = problem.parameters.get("COSTBUDGET")
    out = []
    for sel in itertools.product((0, 1), repeat=7):
        if any((st == 2 and not s) or (st == 3 and s) for st, s in zip(status, sel)):
            continue
        if budget is not None and sum(c for c, s in zip(COST, sel) if s) > budget:
            continue
        if any(list(sel) == list(f) for f in forbidden):
            continue
        out.append((sel, [value(spec, sel, problem) for spec in specs]))
    return out


def lexicographic(
    pts: list[tuple[tuple[int, ...], list[float]]], specs: Sequence[ObjectiveSpec],
    rel_tol: Sequence[float], abs_tol: float = ABS_TOL,
) -> tuple[list[tuple[tuple[int, ...], list[float]]], list[float]]:
    """Enumerated hierarchical result: the optimal set of the last stage, and each v."""
    vs = []
    for k, spec in enumerate(specs):
        vals = [p[1][k] for p in pts]
        v = min(vals) if spec.sense == "min" else max(vals)
        vs.append(v)
        if k == len(specs) - 1:
            return [p for p in pts if abs(p[1][k] - v) <= 1e-9], vs
        slack = rel_tol[k] * abs(v) + abs_tol
        pts = [p for p in pts if (
            p[1][k] <= v + slack if spec.sense == "min" else p[1][k] >= v - slack
        )]
    raise AssertionError


def run(specs: Sequence[ObjectiveSpec], approach: Any, problem: ConservationProblem,
        config: SolverConfig | None = None) -> list[Solution]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # PuLP 3.3.2
        return MultiObjectiveMIP(specs, approach, mip_backend="cbc").solve(problem, config)


def got(sol: Solution, specs: Sequence[ObjectiveSpec]) -> list[float]:
    return [sol.metadata["objective_values"][s.key] for s in specs]


def sel(sol: Solution) -> tuple[int, ...]:
    return tuple(int(v) for v in sol.selected)


# --- hierarchical, 2 objectives (§7.2) ----------------------------------------------


@pytest.mark.parametrize("budget", [None, 4.0])  # 4 binds: best shortfall 1.1
@pytest.mark.parametrize("order", ["cost_first", "shortfall_first"])
def test_hierarchical_rel_tol_0_is_the_lexicographic_optimum(
    order: str, budget: float | None,
) -> None:
    specs: list[ObjectiveSpec] = [MinCost(), MinShortfall()]
    if order == "shortfall_first":
        specs.reverse()
    p = toy() if budget is None else toy(COSTBUDGET=budget)
    best, _ = lexicographic(points(p, specs), specs, [0.0])
    sol = run(specs, Hierarchical(0.0), p)[0]
    assert got(sol, specs) == pytest.approx(best[0][1], abs=1e-9)
    assert sel(sol) in {b[0] for b in best}


@pytest.mark.parametrize("r", [0.25, 0.6])
def test_hierarchical_rel_tol_r_caps_stage_1(r: float) -> None:
    specs = [MinShortfall(), MinCost()]
    p = toy(COSTBUDGET=4.0)  # best shortfall 1.1 at cost 4; r = 0.6 allows cost 3
    best, (v1, _) = lexicographic(points(p, specs), specs, [r])
    sol = run(specs, Hierarchical(r), p)[0]
    f1, f2 = got(sol, specs)
    assert f2 == pytest.approx(best[0][1][1], abs=1e-9)  # stage 2 = enumerated min f2
    assert f1 <= v1 + r * abs(v1) + ABS_TOL + 1e-9  # the cap holds
    assert v1 > 0  # the fixture makes r matter


@pytest.mark.parametrize("r", [0.25, 1.0])
def test_hierarchical_rel_tol_r_caps_cost_first(r: float) -> None:
    """MinCost → MinShortfall with PU1 (cost 4) locked in, so v = 4 > 0 and r matters."""
    specs = [MinCost(), MinShortfall()]
    p = toy(status=[2, 0, 0, 0, 0, 0, 0])
    best, (v1, _) = lexicographic(points(p, specs), specs, [r])
    sol = run(specs, Hierarchical(r), p)[0]
    f1, f2 = got(sol, specs)
    assert f2 == pytest.approx(best[0][1][1], abs=1e-9)
    assert f1 <= v1 + r * abs(v1) + ABS_TOL + 1e-9
    # rel_tol 0 keeps cost 4 at shortfall 1.1; f1 itself can tie within the cap.
    assert v1 == 4.0 and f2 < 1.1


# --- 3 objectives: rel_tol indexed by solve order (§7.2) ----------------------------


def test_three_objectives_rel_tol_follows_stage_order() -> None:
    specs = [MaxWtdSum(weights=WTD), MinCost(), MinShortfall()]
    p = toy()
    pts = points(p, specs)
    r = 0.5
    results = {}
    for tols in ([0.0, r], [r, 0.0]):
        best, _ = lexicographic(pts, specs, tols)
        sol = run(specs, Hierarchical(tols), p)[0]
        assert got(sol, specs) == pytest.approx(best[0][1], abs=1e-9)
        results[tuple(tols)] = got(sol, specs)
    assert results[(0.0, r)] != results[(r, 0.0)]  # the fixture discriminates


# --- mixed senses (§7.2) ----------------------------------------------------------


@pytest.mark.parametrize("r", [0.0, 0.25])
def test_mixed_sense_hierarchy_max_then_min(r: float) -> None:
    specs = [MaxWtdSum(weights=WTD), MinCost()]
    p = toy(COSTBUDGET=10.0)
    best, (v1, _) = lexicographic(points(p, specs), specs, [r])
    sol = run(specs, Hierarchical(r), p)[0]
    f1, f2 = got(sol, specs)
    assert f2 == pytest.approx(best[0][1][1], abs=1e-9)
    assert f1 >= v1 - (r * abs(v1) + ABS_TOL) - 1e-9


@pytest.mark.parametrize("row", [[1, 1], [1, 3], [3, 1], [1, 0.2]])
def test_mixed_sense_weighted_sum(row: list[float]) -> None:
    specs = [MinCost(), MaxWtdSum(weights=WTD)]
    p = toy(COSTBUDGET=12.0)
    total = sum(row)
    score = [row[0] / total * v[0] - row[1] / total * v[1] for _, v in points(p, specs)]
    sol = run(specs, WeightedSum(row), p)[0]
    f = got(sol, specs)
    assert row[0] / total * f[0] - row[1] / total * f[1] == pytest.approx(min(score), abs=1e-9)


# --- locks and no-good cuts (§7.2) -----------------------------------------------


@pytest.mark.parametrize("r", [0.0, 0.5])
def test_hierarchical_respects_locks_and_no_good_cuts(r: float) -> None:
    specs = [MinShortfall(), MinCost()]
    # Budget 8 binds (best shortfall 0.9 > 0), so r = 0.5 lowers the cost from 8 to 7.
    p = toy(status=[0, 2, 0, 0, 3, 0, 0], COSTBUDGET=8.0)
    unforbidden = run(specs, Hierarchical(r), p)[0]
    forbidden = [list(sel(unforbidden))]
    config = SolverConfig(metadata={"forbidden_selections": forbidden})
    best, _ = lexicographic(points(p, specs, forbidden), specs, [r])
    sol = run(specs, Hierarchical(r), p, config)[0]
    assert got(sol, specs)[1] == pytest.approx(best[0][1][1], abs=1e-9)
    s = sel(sol)
    assert s[1] == 1 and s[4] == 0 and list(s) != forbidden[0]


# --- weighted-sum sweep reproduces the supported hull (§7.2) ------------------------


def _hull(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Lower-left convex hull vertices from (min cost, then min sf) to (min sf, then min cost)."""
    uniq = sorted(set(pts))
    lower: list[tuple[float, float]] = []
    for q in uniq:
        while len(lower) >= 2:
            (ax, ay), (bx, by) = lower[-2], lower[-1]
            if (bx - ax) * (q[1] - ay) - (by - ay) * (q[0] - ax) <= 1e-12:
                lower.pop()
            else:
                break
        lower.append(q)
    end = min(uniq, key=lambda q: (q[1], q[0]))
    return lower[: lower.index(end) + 1]


def _unit(v: tuple[float, float]) -> tuple[float, float]:
    n = (v[0] ** 2 + v[1] ** 2) ** 0.5
    return (v[0] / n, v[1] / n)


def _cone_rows(vertices: list[tuple[float, float]]) -> list[list[float]]:
    """One weight row strictly inside each vertex's normal cone."""
    normals = [
        _unit((a[1] - b[1], b[0] - a[0])) for a, b in zip(vertices, vertices[1:])
    ]  # edge a→b: cost rises, shortfall falls; both components > 0
    cones = [((1.0, 0.0), normals[0])]
    cones += [(normals[i - 1], normals[i]) for i in range(1, len(normals))]
    cones += [(normals[-1], (0.0, 1.0))]
    return [[(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] for a, b in cones]


SWEEP_SPECS = [MinCost(), MinShortfall()]


def _sweep_points() -> list[tuple[float, float]]:
    return [(v[0], v[1]) for _, v in points(toy(), SWEEP_SPECS)]


def test_weighted_sum_sweep_returns_each_hull_vertex() -> None:
    vertices = _hull(_sweep_points())
    assert len(vertices) >= 4  # several supported trade-offs
    lex_cost, _ = lexicographic(points(toy(), SWEEP_SPECS), SWEEP_SPECS, [0.0])
    lex_sf, _ = lexicographic(points(toy(), SWEEP_SPECS[::-1]), SWEEP_SPECS[::-1], [0.0])
    assert vertices[0] == tuple(lex_cost[0][1])
    assert vertices[-1] == tuple(lex_sf[0][1][::-1])
    sols = run(SWEEP_SPECS, WeightedSum(_cone_rows(vertices)), toy())
    for vertex, sol in zip(vertices, sols, strict=True):
        assert got(sol, SWEEP_SPECS) == pytest.approx(list(vertex), abs=1e-9)


def test_weighted_sum_grid_rows_reach_the_optimal_weighted_value() -> None:
    pts = _sweep_points()
    rows = weights_matrix(2, 5)
    for row, sol in zip(rows, run(SWEEP_SPECS, WeightedSum(rows), toy()), strict=True):
        w = [v / sum(row) for v in row]
        best = min(w[0] * c + w[1] * s for c, s in pts)
        f = got(sol, SWEEP_SPECS)
        assert w[0] * f[0] + w[1] * f[1] == pytest.approx(best, abs=1e-9)


def test_nondominated_sweep_equals_the_enumerated_pareto_points_found() -> None:
    pts = _sweep_points()
    pareto = {
        p for p in pts
        if not any(q[0] <= p[0] and q[1] <= p[1] and q != p for q in pts)
    }
    rows = weights_matrix(2, 5) + _cone_rows(_hull(pts))
    sols = run(SWEEP_SPECS, WeightedSum(rows), toy())
    found = {tuple(got(s, SWEEP_SPECS)) for s in sols}
    kept = [tuple(got(s, SWEEP_SPECS)) for s in nondominated(sols)]
    assert len(kept) == len(set(kept))  # identical vectors kept once
    assert set(kept) == {p for p in found if p in pareto}


# --- sweeps: rows are independent (§7.2, §9) ------------------------------------------


def test_hierarchical_sweep_rows_match_their_own_oracles() -> None:
    specs = [MinShortfall(), MinCost()]
    p = toy(COSTBUDGET=4.0)
    rows = [[0.6], [0.0]]  # the looser row first: its cap must not leak into row 2
    sols = run(specs, Hierarchical(rows), p)
    costs = []
    for row, sol in zip(rows, sols, strict=True):
        best, _ = lexicographic(points(p, specs), specs, row)
        assert got(sol, specs)[1] == pytest.approx(best[0][1][1], abs=1e-9)
        assert sol.metadata["rel_tol"] == row
        costs.append(got(sol, specs)[1])
    assert costs == [3.0, 4.0]


def test_failed_stage_leaves_no_cap_and_row_2_equals_a_fresh_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = [MinShortfall(), MinCost()]
    p = toy(COSTBUDGET=4.0)
    fresh = run(specs, Hierarchical(0.0), p)[0]
    solver = Scripted({2: fail})  # row 1, stage 2
    monkeypatch.setattr(ms, "_make_pulp_solver", lambda *a, **k: solver)
    with pytest.warns(UserWarning, match="stage 2"):
        sols = run(specs, Hierarchical([[0.0], [0.0]]), p)
    assert sels_equal(sols[1], fresh)
    assert "hier_cap_1" not in solver.models[2]  # row 2, stage 1


def sels_equal(a: Solution, b: Solution) -> bool:
    return sel(a) == sel(b) and a.metadata["objective_values"] == b.metadata[
        "objective_values"
    ]


# --- prioritizr parity (§7.3) ----------------------------------------------------------


def prioritizr_fixture() -> ConservationProblem:
    """prioritizr's 6-PU multi-objective test problem: cost 1 each, budget 2."""
    ft = {1: [5.0, 0.5, 0, 0, 0, 0], 2: [0, 0, 0, 0, 3.0, 2.0]}
    pvf = [(f, pu, a) for f, row in ft.items() for pu, a in enumerate(row, 1) if a > 0]
    return ConservationProblem(
        planning_units=pd.DataFrame({"id": range(1, 7), "cost": [1.0] * 6, "status": 0}),
        features=pd.DataFrame({"id": [1, 2], "name": ["ft1", "ft2"], "target": [0.0, 0.0]}),
        pu_vs_features=pd.DataFrame(pvf, columns=["species", "pu", "amount"]),
        parameters={"COSTBUDGET": 2.0},
    )


PARITY = [MaxWtdSum(weights={1: 1, 2: 0}, label="ft1"),
          MaxWtdSum(weights={1: 0, 2: 1}, label="ft2")]
PARITY_ROWS = [[1, 1], [1, 0], [0, 1], [1, 1], [1, 5]]
PARITY_EXPECTED = [
    ("100010", (5.0, 3.0)), ("110000", (5.5, 0.0)), ("000011", (0.0, 5.0)),
    ("100010", (5.0, 3.0)), ("000011", (0.0, 5.0)),
]


def test_prioritizr_parity_expectations_by_enumeration() -> None:
    ft = {1: [5.0, 0.5, 0, 0, 0, 0], 2: [0, 0, 0, 0, 3.0, 2.0]}
    for row, (x, vals) in zip(PARITY_ROWS, PARITY_EXPECTED, strict=True):
        scored = []
        for s in itertools.product((0, 1), repeat=6):
            if sum(s) > 2:
                continue
            v = tuple(sum(a for a, si in zip(ft[f], s) if si) for f in (1, 2))
            scored.append((row[0] * v[0] + row[1] * v[1], "".join(map(str, s)), v))
        best = max(sc[0] for sc in scored)
        winners = [sc for sc in scored if sc[0] == best]
        assert [(w[1], w[2]) for w in winners] == [(x, vals)]  # unique optimum


def test_prioritizr_parity_weighted_sum() -> None:
    sols = run(PARITY, WeightedSum(PARITY_ROWS), prioritizr_fixture())
    for sol, (x, vals) in zip(sols, PARITY_EXPECTED, strict=True):
        assert "".join(map(str, sel(sol)[:6])) == x
        assert got(sol, PARITY) == pytest.approx(list(vals))
