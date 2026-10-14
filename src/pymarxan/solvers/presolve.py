"""Pre-solve data checks (inspired by prioritizr's ``presolve_check``; the checks differ).

prioritizr runs its check by default on the compiled model with absolute thresholds
(1e6 / 1e-6); here the data checks are opt-in, add ratio checks (P2-P4, P8), and the
absolute scan of the built model (P10) runs only through a solver's ``presolve_check=``
flag. prioritizr's single-feature and >= 50 % empty-PU checks are not ported.
No zone imports at module level (``pymarxan.zones`` imports ``pymarxan.solvers``).
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pulp

if TYPE_CHECKING:
    from pymarxan.models.problem import ConservationProblem
    from pymarxan.zones.model import ZonalProblem

RATIO_LIMIT = 1e6
ABS_HIGH = 1e6
ABS_LOW = 1e-6


class PresolveWarning(UserWarning):
    """``presolve_check=True`` found numerical or data problems before solving."""


def _validate_presolve_flag(value: object) -> bool | str:
    if value is True or value is False:
        return value
    if isinstance(value, str) and value == "strict":
        return value
    raise ValueError(f"presolve_check must be True, False or 'strict', got {value!r}")


def _emit_presolve(
    flag: bool | str,
    problem: ConservationProblem,
    model: pulp.LpProblem,
    solver_name: str,
    *,
    stacklevel: int,
) -> None:
    """Run the data checks plus the model scan and report them per ``flag``.

    ``stacklevel`` is counted from this function, so a solver method called directly by
    the user passes 3 (this helper <- solve()/build_model() <- the user's frame).
    """
    if not flag:
        return
    messages = presolve_check(problem) + scan_model(model)
    if not messages:
        return
    text = f"{solver_name}: presolve check: " + " | ".join(messages)
    if flag == "strict":
        raise ValueError(text)
    warnings.warn(text, PresolveWarning, stacklevel=stacklevel)


def _first(items: list[str], n: int = 5, total: int | None = None) -> str:
    """The first ``n`` items, then ``(+k more)`` against ``total`` (default ``len(items)``)."""
    total = len(items) if total is None else total
    more = f" (+{total - n} more)" if total > n else ""
    return "; ".join(items[:n]) + more


Label = Callable[[int], str]
"""Row index -> the entity it belongs to (``"PU 7"``, ``"species 2 PU 4"``); called only
for the rows a message reports, so labelling stays cheap on large tables."""


def _labeller(prefixes: tuple[str, ...], *columns: Any) -> Label:
    cols = [np.asarray(c) for c in columns]

    def label(i: int) -> str:
        return " ".join(f"{p} {_fmt_id(c[i])}" for p, c in zip(prefixes, cols, strict=True))

    return label


def _fmt_id(v: Any) -> str:
    f = float(v)
    return str(int(f)) if f.is_integer() else f"{f:g}"


def _nonfinite(code: str, named: Iterable[tuple[str, Any, Label]]) -> list[str]:
    out: list[str] = []
    for name, values, label in named:
        arr = np.asarray(values, dtype=float).ravel()
        (bad,) = np.nonzero(~np.isfinite(arr))
        if bad.size:
            items = [f"{label(int(i))} = {arr[i]:g}" for i in bad[:5]]
            out.append(f"{code}: {bad.size} non-finite {name} value(s): "
                       + _first(items, total=int(bad.size)))
    return out


def _ratio(code: str, label: str, values: Any) -> list[str]:
    arr = np.abs(np.asarray(values, dtype=float).ravel())
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size < 2:
        return []
    lo, hi = float(arr.min()), float(arr.max())
    if hi / lo > RATIO_LIMIT:
        return [f"{code}: {label} span {hi / lo:.1e}x (min {lo:.3g}, max {hi:.3g}): "
                "solvers may lose precision"]
    return []


def _blm_check(blm: float, boundary: Any, costs: Any) -> list[str]:
    bnd = np.abs(np.asarray(boundary, dtype=float))
    bnd = bnd[np.isfinite(bnd)]
    c = np.abs(np.asarray(costs, dtype=float))
    c = c[np.isfinite(c) & (c > 0)]
    if not blm > 0 or bnd.size == 0 or c.size == 0:
        return []
    ratio = blm * float(bnd.max()) / float(np.median(c))
    if ratio > RATIO_LIMIT or ratio < 1.0 / RATIO_LIMIT:
        return [f"P8: BLM x boundary is {ratio:.1e}x the median cost: "
                + ("boundary will dominate" if ratio > 1 else "boundary has no effect")]
    return []


def _budget_checks(budget: float | None, floor: float, cheapest: float | None,
                   ceiling: float) -> list[str]:
    if budget is None:
        return []
    out: list[str] = []
    if budget < floor - 1e-9:
        out.append(f"P9a: COSTBUDGET {budget:g} is below the cost that locked / must-assign "
                   f"units force ({floor:g}): infeasible")
    elif cheapest is not None and budget < cheapest - 1e-9:
        out.append(f"P9b: COSTBUDGET {budget:g} is below the cheapest selectable option "
                   f"({cheapest:g}): nothing can be selected")
    if budget >= ceiling and ceiling > 0:
        out.append(f"P11: COSTBUDGET {budget:g} reaches the total cost {ceiling:g}: "
                   "the budget does not constrain the solution")
    return out


def presolve_check(problem: ConservationProblem) -> list[str]:
    """Data problems likely to make a solve slow, imprecise or meaningless. Never raises.

    Returns ``[]`` when clean; every message starts with its code (``"P5: ..."``). When
    ``problem.validate()`` fails, returns one ``"P0: ..."`` message and skips the rest.

    Codes: P1 non-finite values / negative costs; P2-P4 cost, amount and boundary spans
    over 1e6x; P5 a target on a feature found in no planning unit; P6 a target above what
    reachable units hold; P7a / P7b every unit locked in / out; P8 BLM x boundary vs the
    median cost outside [1e-6, 1e6]; P9a COSTBUDGET below the forced cost; P9b below the
    cheapest selectable option; P11 COSTBUDGET at or above the total cost. P10 (model
    coefficients) needs a built model and is only reported through a solver's
    ``presolve_check=`` flag.
    """
    try:
        errors = problem.validate()
    except Exception as exc:  # validate() should not raise; presolve must not either
        return [f"P0: problem.validate() raised {exc!r}"]
    if errors:
        return [f"P0: problem fails validate(): {len(errors)} error(s); fix those first"]
    try:
        from pymarxan.zones.model import ZonalProblem

        if isinstance(problem, ZonalProblem):
            return _zone_checks(problem)
        return _core_checks(problem)
    except Exception as exc:  # a diagnostic must never crash the caller
        return [f"P0: presolve check failed: {exc!r}"]


def _budget(problem: ConservationProblem) -> float | None:
    budget = problem.parameters.get("COSTBUDGET")
    return None if budget is None else float(budget)


def _common(problem: ConservationProblem, costs: Any, cost_label: Label,
            named_extra: list[tuple[str, Any, Label]], totals: pd.Series,
            budget: float | None) -> list[str]:
    """Checks shared by core and zone problems (P1-P5, P8)."""
    params = problem.parameters
    pvf = problem.pu_vs_features
    amounts = pvf["amount"].to_numpy(dtype=float)
    feats = problem.features
    targets = feats["target"].to_numpy(dtype=float)
    b = problem.boundary
    bnd = b["boundary"].to_numpy(dtype=float) if b is not None else np.empty(0)
    id1: Any = b["id1"].to_numpy() if b is not None else np.empty(0)
    id2: Any = b["id2"].to_numpy() if b is not None else np.empty(0)
    blm = float(params.get("BLM", 0.0))
    named: list[tuple[str, Any, Label]] = [
        ("cost", costs, cost_label),
        ("amount", amounts, _labeller(("species", "PU"), pvf["species"], pvf["pu"])),
        ("target", targets, _labeller(("feature",), feats["id"])),
        ("boundary", bnd, lambda i: f"boundary {_fmt_id(id1[i])}-{_fmt_id(id2[i])}"),
        ("BLM", [blm], lambda i: "BLM"),
    ]
    if budget is not None:
        named.append(("COSTBUDGET", [budget], lambda i: "COSTBUDGET"))
    out = _nonfinite("P1", named + named_extra)
    cost_arr = np.asarray(costs, dtype=float)
    neg = [f"{cost_label(int(i))} has cost {cost_arr[i]:g}"
           for i in np.nonzero(cost_arr < 0)[0]]
    if neg:
        out.append(f"P1: {len(neg)} negative cost(s): {_first(neg)}")
    out += _ratio("P2", "costs", costs)
    out += _ratio("P3", "amounts", amounts)
    out += _ratio("P4", "boundary lengths", bnd)
    out += _blm_check(blm, bnd, costs)
    ghosts = [
        f"feature {int(f)} occurs in no planning unit but has target {t:g}"
        for f, t in zip(problem.features["id"].values, targets, strict=True)
        if t > 0 and not float(totals.get(int(f), 0.0)) > 0
    ]
    if ghosts:
        out.append("P5: " + _first(ghosts))
    return out


def _core_checks(problem: ConservationProblem) -> list[str]:
    pu = problem.planning_units
    costs = pu["cost"].to_numpy(dtype=float)
    status = (pu["status"].to_numpy(dtype=int) if "status" in pu.columns
              else np.zeros(len(pu), dtype=int))
    pvf = problem.pu_vs_features
    totals = pvf.groupby("species")["amount"].sum()
    budget = _budget(problem)
    out = _common(problem, costs, _labeller(("PU",), pu["id"]), [], totals, budget)
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    locked_out = {int(p) for p, s in zip(pu["id"].values, status, strict=True) if s == 3}
    avail = pvf[~pvf["pu"].isin(locked_out)].groupby("species")["amount"].sum()
    short = [
        f"feature {int(f)} target {t * misslevel:g} exceeds its total amount "
        f"{float(avail.get(int(f), 0.0)):g} outside locked-out units"
        for f, t in zip(problem.features["id"].values,
                        problem.features["target"].to_numpy(dtype=float), strict=True)
        if t > 0 and float(totals.get(int(f), 0.0)) > 0
        and float(avail.get(int(f), 0.0)) < t * misslevel - 1e-9
    ]
    if short:
        out.append("P6: " + _first(short))
    if len(status) and (status == 2).all():
        out.append("P7a: every planning unit is locked in: nothing to optimise")
    if len(status) and (status == 3).all():
        out.append("P7b: every planning unit is locked out: nothing can be selected")
    free = costs[(status == 0) | (status == 1)]
    out += _budget_checks(
        budget,
        floor=float(costs[status == 2].sum()),
        cheapest=float(free.min()) if free.size else None,
        ceiling=float(np.clip(costs[status != 3], 0, None).sum()),
    )
    return out


def _zone_checks(problem: ZonalProblem) -> list[str]:
    from pymarxan.zones.diagnostics import (
        _zone_cost_lookup,
        diagnose_zone_infeasibility,
        min_forced_cost,
    )

    zcf = problem.zone_costs
    zc = zcf["cost"].to_numpy(dtype=float)
    contrib = problem.zone_contributions
    extra: list[tuple[str, Any, Label]] = []
    if contrib is not None:
        extra.append(("zone contribution", contrib["contribution"].to_numpy(dtype=float),
                      _labeller(("feature", "zone"), contrib["feature"], contrib["zone"])))
    totals = problem.pu_vs_features.groupby("species")["amount"].sum()
    budget = _budget(problem)
    cost_label = _labeller(("PU", "zone"), zcf["pu"], zcf["zone"])
    out = _common(problem, zc, cost_label, extra, totals, budget)
    unreachable = diagnose_zone_infeasibility(problem)
    if unreachable:
        out.append("P6: " + _first(unreachable))
    rules = problem.pu_zone_rules()
    if rules and all(r.forced for r in rules):
        out.append("P7a: every planning unit is locked in: nothing to optimise")
    if rules and all(not r.allowed for r in rules):
        out.append("P7b: every planning unit is locked out: nothing can be selected")
    lookup = _zone_cost_lookup(problem)
    pids = [int(p) for p in problem.planning_units["id"].values]
    options = [[lookup.get((pid, z), 0.0) for z in r.allowed]
               for pid, r in zip(pids, rules, strict=True)]
    free_costs = [c for opts, r in zip(options, rules, strict=True)
                  if r.may_be_unassigned for c in opts]
    out += _budget_checks(
        budget,
        floor=min_forced_cost(problem),
        cheapest=min(free_costs) if free_costs else None,
        ceiling=float(sum(max(0.0, max(opts)) for opts in options if opts)),
    )
    return out


def scan_model(model: pulp.LpProblem) -> list[str]:
    """P10: absolute coefficient / rhs magnitudes outside [1e-6, 1e6] (prioritizr thresholds)."""
    worst: tuple[float, str] | None = None
    count = 0

    def visit(value: float, where: str) -> None:
        nonlocal worst, count
        a = abs(value)
        if a > ABS_HIGH or 0 < a < ABS_LOW:
            count += 1
            score = a if a > 1 else 1.0 / a
            if worst is None or score > worst[0]:
                worst = (score, f"{where} = {value:.3g}")

    if model.objective is not None:
        for v, a in model.objective.items():
            visit(float(a), f"objective coefficient of {v.name}")
    for name, c in model.constraints.items():
        visit(float(-c.constant), f"rhs of {name}")
        for v, a in c.items():
            visit(float(a), f"coefficient of {v.name} in {name}")
    if worst is None:
        return []
    if not math.isfinite(worst[0]):
        return [f"P10: {count} non-finite model value(s)"]
    return [f"P10: {count} model coefficient(s) outside [1e-6, 1e6]; worst: {worst[1]}"]
