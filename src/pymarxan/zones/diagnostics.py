"""Single-target infeasibility diagnostics for ZoneMIPSolver.

``_feature_groups`` (the MIP constraints' amount source) lives here; which zones a PU may
take is ``ZonalProblem.pu_zone_rules`` (``_pu_zone_rules`` is a thin alias kept for the MIP
and older imports).
"""
from __future__ import annotations

from pymarxan.zones.model import PUZoneRule, ZonalProblem
from pymarxan.zones.objective import _meets


class ZoneMIPNoSolutionWarning(UserWarning):
    """ZoneMIPSolver returned ``[]``; the message says why (diagnostic, compete, time limit)."""


def _feature_groups(problem: ZonalProblem) -> dict[int, list[tuple[int, float]]]:
    """feature id -> [(pu id, amount), ...] from pu_vs_features."""
    puvspr = problem.pu_vs_features
    groups: dict[int, list[tuple[int, float]]] = {}
    for pid, fid, amt in zip(
        puvspr["pu"].values, puvspr["species"].values, puvspr["amount"].values, strict=True,
    ):
        groups.setdefault(int(fid), []).append((int(pid), float(amt)))
    return groups


def _pu_zone_rules(problem: ZonalProblem) -> list[PUZoneRule]:
    """Alias of ``ZonalProblem.pu_zone_rules`` (kept for the MIP and older imports)."""
    return problem.pu_zone_rules()


def _best(rule: PUZoneRule, value_in: dict[int, float]) -> float:
    """A PU's largest possible contribution: the best allowed zone's value, or 0 when the
    PU may stay unassigned and that is better; a forced PU's value as-is."""
    values = [value_in.get(z, 0.0) for z in rule.allowed]
    if rule.may_be_unassigned:
        values.append(0.0)
    return max(values) if values else 0.0


def _fmt(x: float) -> str:
    return f"{x:.6g}"


def _zone_cost_lookup(problem: ZonalProblem) -> dict[tuple[int, int], float]:
    """(pu, zone) -> cost; a missing pair costs 0 (MarZone; prioritizr NA means forbidden)."""
    zc = problem.zone_costs
    return {
        (int(p), int(z)): float(c)
        for p, z, c in zip(zc["pu"].values, zc["zone"].values, zc["cost"].values, strict=True)
    }


def min_forced_cost(problem: ZonalProblem) -> float:
    """Least total zone cost any rule-respecting assignment must pay.

    Σ over must-assign PUs of the cheapest allowed zone cost, plus min(0, cheapest) for PUs
    that may stay unassigned (a negative cost lowers the floor). COSTBUDGET below this is
    infeasible for every budgeted objective.
    """
    lookup = _zone_cost_lookup(problem)
    total = 0.0
    for pid, rule in zip(
        (int(p) for p in problem.planning_units["id"].values),
        problem.pu_zone_rules(),
        strict=True,
    ):
        costs = [lookup.get((pid, z), 0.0) for z in rule.allowed]
        if not costs:
            continue
        cheapest = min(costs)
        total += min(0.0, cheapest) if rule.may_be_unassigned else cheapest
    return total


def diagnose_zone_infeasibility(problem: ZonalProblem) -> list[str]:
    """Targets no assignment can meet, each checked on its own (necessary conditions only).

    For PU ``i`` with per-zone value ``v_i(z)``, its best contribution is ``_best``;
    overall target ``f`` uses ``v_i(z) = amount[i, f] × contribution[z, f]`` and zone
    target ``(k, f)`` uses ``v_i(k) = amount[i, f] × w[k, f]`` (0 in other zones). A
    target ``t > 0`` is reported when ``Σ_i best_i < t × MISSLEVEL`` (1e-9 relative
    tolerance). Amounts come from ``_feature_groups`` — the MIP constraints' own source
    — so the bound is the MIP's. Zone targets are hard only in pymarxan's MIP (MarZone
    treats them as penalties). ``[]`` does not prove feasibility: targets may compete.
    """
    rule_of = dict(
        zip(
            (int(p) for p in problem.planning_units["id"].values),
            _pu_zone_rules(problem),
            strict=True,
        )
    )
    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
    contrib = problem.contribution_matrix()
    weight = problem.zone_target_weight_matrix()
    zidx = problem.zone_index()
    fidx = problem.feature_index()
    groups = _feature_groups(problem)

    def per_pu(fid: int) -> dict[int, float]:
        summed: dict[int, float] = {}
        for pid, amt in groups.get(fid, []):
            if pid in rule_of:
                summed[pid] = summed.get(pid, 0.0) + amt
        return summed

    messages: list[str] = []
    for fid_raw, target in zip(
        problem.features["id"].values, problem.features["target"].values, strict=True,
    ):
        fid, t = int(fid_raw), float(target)
        if not t > 0:
            continue
        col = fidx[fid]
        ub = 0.0
        for pid, amt in per_pu(fid).items():
            rule = rule_of[pid]
            ub += _best(rule, {z: amt * float(contrib[zidx[z], col]) for z in rule.allowed})
        if not _meets(ub, t * misslevel):
            messages.append(
                f"overall target for feature {fid} is {_fmt(t * misslevel)} "
                f"but at most {_fmt(ub)} is reachable"
            )

    for (zid, fid), t in problem.zone_target_lookup().items():
        w = float(weight[zidx[zid], fidx[fid]])
        ub = 0.0
        for pid, amt in per_pu(fid).items():
            rule = rule_of[pid]
            ub += _best(rule, {zid: amt * w} if zid in rule.allowed else {})
        if not _meets(ub, t * misslevel):
            messages.append(
                f"zone target (zone {zid}, feature {fid}) is {_fmt(t * misslevel)} "
                f"but at most {_fmt(ub)} is reachable"
            )
    return messages
