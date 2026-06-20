"""Two-stage stochastic reserve selection — Snyder–Haight–ReVelle (2005).

A scenario-based two-period model (Snyder, Haight & ReVelle 2005,
doi:10.1007/s10666-005-3799-1): choose sites to protect **now** (stage 1)
before an uncertain future is revealed, then add sites by **recourse**
(stage 2) once a scenario realises, each stage under its own budget. Maximise
the expected weighted feature coverage across scenarios — covering the value of
acting early against the flexibility of waiting.

Solved exactly on the shared CBC/HiGHS/Gurobi MIP backend:

    maximise   Σ_s π_s Σ_f w_f z_{f,s}
    s.t.       z_{f,s} ≤ Σ_{i∈contains[f]} (x_i + y_{i,s})        (coverage)
               Σ_i c_i x_i ≤ B1                                   (stage-1 budget)
               Σ_i c_i y_{i,s} ≤ B2          ∀ s                  (stage-2 budget)
               y_{i,s} = 0  if site i unavailable in scenario s
               x_i, y_{i,s} ∈ {0,1};  z_{f,s} ∈ [0, 1]

Stage-1 sites are protected before the scenario and assumed secure; stage-2
selections are limited to the sites available in that scenario (the recourse).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pulp

from pymarxan.solvers._backends import _make_pulp_solver


@dataclass
class TwoStageResult:
    """Outcome of a two-stage stochastic reserve solve."""

    stage1: set[int]
    stage2: dict[int, set[int]]          # scenario index → sites added by recourse
    expected_coverage: float
    optimal: bool = False
    metadata: dict = field(default_factory=dict)


def two_stage_reserve_mip(
    sites,
    features,
    contains: dict,
    cost: dict,
    scenarios: list[tuple[float, set]],
    budget1: float,
    budget2: float,
    weights: dict | None = None,
    *,
    mip_backend: str = "auto",
    time_limit: int = 120,
    gap: float = 0.0,
    verbose: bool = False,
) -> TwoStageResult:
    """Exact two-stage stochastic maximal-coverage reserve selection.

    Args:
        sites: Iterable of site ids.
        features: Iterable of feature ids.
        contains: ``{feature_id: set(site_ids containing it)}``.
        cost: ``{site_id: cost}``.
        scenarios: list of ``(probability, available_sites_set)`` — the sites
            selectable by stage-2 recourse under each scenario.
        budget1, budget2: stage-1 and (per-scenario) stage-2 cost caps.
        weights: ``{feature_id: weight}`` (default 1 each).

    Returns:
        :class:`TwoStageResult` with the stage-1 selection, per-scenario
        stage-2 recourse selections, and the expected weighted coverage.
    """
    sites = list(sites)
    features = list(features)
    weights = weights or {f: 1.0 for f in features}

    model = pulp.LpProblem("two_stage_reserve", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in sites}
    y: dict[tuple[int, int], pulp.LpVariable] = {}
    z: dict[tuple[int, int], pulp.LpVariable] = {}

    obj_terms = []
    for s, (prob, avail) in enumerate(scenarios):
        for i in sites:
            v = pulp.LpVariable(f"y_{i}_{s}", cat="Binary")
            y[(i, s)] = v
            if i not in avail:
                model += v == 0  # unavailable → no recourse here
        for f in features:
            zf = pulp.LpVariable(f"z_{f}_{s}", lowBound=0.0, upBound=1.0)
            z[(f, s)] = zf
            in_f = contains.get(f, set())
            model += zf <= pulp.lpSum(
                x[i] + y[(i, s)] for i in sites if i in in_f
            )
            obj_terms.append(prob * weights.get(f, 1.0) * zf)
        model += pulp.lpSum(cost[i] * y[(i, s)] for i in sites) <= budget2

    model += pulp.lpSum(obj_terms)
    model += pulp.lpSum(cost[i] * x[i] for i in sites) <= budget1

    solver = _make_pulp_solver(
        mip_backend, time_limit=time_limit, gap=gap, verbose=verbose
    )
    model.solve(solver)
    optimal = pulp.LpStatus[model.status] == "Optimal"

    def _val(var) -> bool:
        return var.value() is not None and var.value() > 0.5

    stage1 = {i for i in sites if _val(x[i])} if optimal else set()
    stage2: dict[int, set[int]] = {}
    if optimal:
        for s in range(len(scenarios)):
            stage2[s] = {i for i in sites if _val(y[(i, s)])}

    expected = float(pulp.value(model.objective)) if optimal else 0.0
    return TwoStageResult(
        stage1=stage1,
        stage2=stage2,
        expected_coverage=expected,
        optimal=optimal,
    )
