"""Project-prioritization optimizers: evaluator, exact MIP, greedy.

See ``model.py`` for the problem definition. The exact MIP maximises the
weighted expected persistence under a budget using the standard assignment
linearisation (Hanson et al. 2019): fund actions ``x_a``, complete projects
``y_p ≤ x_a``, assign each feature to its best completed project via a
continuous ``z_{f,p} ∈ [0,1]`` with ``Σ_p z_{f,p} ≤ 1`` and ``z_{f,p} ≤ y_p``.
"""
from __future__ import annotations

import pulp

from pymarxan.projects.model import ProjectProblem, ProjectSolution
from pymarxan.solvers._backends import _make_pulp_solver

_TOL = 1e-9


def evaluate_projects(
    problem: ProjectProblem, funded_actions: set[int]
) -> ProjectSolution:
    """Score a given set of funded actions (no optimisation).

    A project is completed iff all its required actions are funded; each
    feature takes the max persistence over the completed projects that benefit
    it (0 if none); benefit is the weighted sum. Used as the production scorer
    (e.g. for the brute-force exactness check) and to report any solve.
    """
    funded = {int(a) for a in funded_actions}
    cost = problem.action_cost()
    weight = problem.feature_weight()
    pa = problem.project_action_map()
    fpp = problem.feature_project_persistence()

    completed = {p for p, acts in pa.items() if acts <= funded}

    feature_persistence: dict[int, float] = {}
    for fid in weight:
        best = 0.0
        for pid, persistence in fpp.get(fid, {}).items():
            if pid in completed and persistence > best:
                best = persistence
        feature_persistence[fid] = best

    benefit = sum(weight[f] * feature_persistence[f] for f in weight)
    total_cost = sum(cost[a] for a in funded if a in cost)
    return ProjectSolution(
        funded_actions=set(funded),
        funded_projects=set(completed),
        cost=float(total_cost),
        benefit=float(benefit),
        feature_persistence=feature_persistence,
        optimal=False,
    )


def prioritize_projects_mip(
    problem: ProjectProblem,
    *,
    mip_backend: str = "auto",
    time_limit: int = 120,
    gap: float = 0.0,
    verbose: bool = False,
) -> ProjectSolution:
    """Exact project prioritization: maximise weighted expected persistence
    under the budget. Returns ``optimal=True`` when the solver proves it."""
    cost = problem.action_cost()
    weight = problem.feature_weight()
    pa = problem.project_action_map()
    fpp = problem.feature_project_persistence()

    model = pulp.LpProblem("project_prioritization", pulp.LpMaximize)
    x = {a: pulp.LpVariable(f"x_{a}", cat="Binary") for a in cost}
    y = {p: pulp.LpVariable(f"y_{p}", cat="Binary") for p in pa}
    # z_{f,p} assignment of feature f to (completed) project p; continuous.
    z: dict[tuple[int, int], pulp.LpVariable] = {}

    # objective + per-feature assignment constraints
    obj_terms = []
    for fid, projmap in fpp.items():
        w = weight.get(fid, 1.0)
        zf = []
        for pid, persistence in projmap.items():
            if pid not in y:
                continue
            v = pulp.LpVariable(f"z_{fid}_{pid}", lowBound=0.0, upBound=1.0)
            z[(fid, pid)] = v
            zf.append(v)
            model += v <= y[pid]
            obj_terms.append(w * persistence * v)
        if zf:
            model += pulp.lpSum(zf) <= 1  # assign to at most one project
    model += pulp.lpSum(obj_terms)

    # a project is completed only if all its actions are funded
    for pid, acts in pa.items():
        for a in acts:
            model += y[pid] <= x[a]

    if problem.budget is not None:
        model += pulp.lpSum(cost[a] * x[a] for a in cost) <= problem.budget

    solver = _make_pulp_solver(
        mip_backend, time_limit=time_limit, gap=gap, verbose=verbose
    )
    model.solve(solver)
    optimal = pulp.LpStatus[model.status] == "Optimal"

    funded_actions = {
        a for a, v in x.items() if v.value() is not None and v.value() > 0.5
    } if optimal else set()
    # Report via the production evaluator so cost/benefit/persistence are
    # computed one way (and a project counts as funded iff its actions are).
    sol = evaluate_projects(problem, funded_actions)
    sol.optimal = optimal
    return sol


def prioritize_projects_greedy(problem: ProjectProblem) -> ProjectSolution:
    """Greedy cost-effectiveness heuristic (Joseph et al. 2009): repeatedly
    fund the project whose remaining actions give the best benefit-gain per
    cost, until no positive-gain affordable project remains. Suboptimal under
    shared actions; returns ``optimal=False``."""
    cost = problem.action_cost()
    pa = problem.project_action_map()

    funded: set[int] = set()
    # baselines (no actions) are free and "already funded"
    cur = evaluate_projects(problem, funded)
    cur_cost = cur.cost

    while True:
        best_pid = None
        best_ratio = 0.0
        best_new_actions: set[int] = set()
        best_benefit = cur.benefit
        best_cost = cur_cost
        for pid, acts in pa.items():
            new_actions = acts - funded
            if not new_actions:
                continue  # already completed (or baseline)
            add_cost = sum(cost[a] for a in new_actions)
            if add_cost <= 0:
                continue
            cand_cost = cur_cost + add_cost
            if problem.budget is not None and cand_cost > problem.budget + _TOL:
                continue
            cand = evaluate_projects(problem, funded | new_actions)
            gain = cand.benefit - cur.benefit
            if gain <= _TOL:
                continue
            ratio = gain / add_cost
            if best_pid is None or ratio > best_ratio:
                best_pid = pid
                best_ratio = ratio
                best_new_actions = new_actions
                best_benefit = cand.benefit
                best_cost = cand_cost
        if best_pid is None:
            break
        funded |= best_new_actions
        cur = evaluate_projects(problem, funded)
        cur_cost = best_cost
        _ = best_benefit  # (kept for readability; cur recomputed authoritatively)

    sol = evaluate_projects(problem, funded)
    sol.optimal = False
    return sol
