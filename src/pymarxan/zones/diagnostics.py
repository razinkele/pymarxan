"""Single-target infeasibility diagnostics for ZoneMIPSolver, and the rules it shares with it.

``_feature_groups`` (the MIP constraints' amount source) and ``_pu_zone_rules`` (which zones
a PU may take) live here so the MIP and the diagnostic read one definition each; #6 part 2
(lock into a named zone) extends ``_pu_zone_rules`` only.
"""
from __future__ import annotations

from typing import NamedTuple

from pymarxan.models.problem import STATUS_LOCKED_IN, STATUS_LOCKED_OUT
from pymarxan.zones.model import ZonalProblem
from pymarxan.zones.objective import _meets


class ZoneMIPNoSolutionWarning(UserWarning):
    """ZoneMIPSolver returned ``[]``; the message says why (diagnostic, compete, time limit)."""


class PUZoneRule(NamedTuple):
    """Zones a PU may take; ``forced`` means it must take the single allowed zone."""

    allowed: frozenset[int]
    forced: bool


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
    """Per PU in ``planning_units`` order: status 3 -> no zone; status 2 -> forced into
    ``min(zone_ids)``; otherwise free over every zone. No ``status`` column means free.

    This is pymarxan's status convention (ZoneMIPSolver / SA / heuristic / II), not a
    MarZone rule — MarZone locks PUs into named zones via its pulock file (#6 part 2).
    """
    every = frozenset(int(z) for z in problem.zone_ids)
    first = frozenset({min(every)}) if every else frozenset()
    pu = problem.planning_units
    if "status" not in pu.columns:
        return [PUZoneRule(every, False)] * len(pu)
    rules: list[PUZoneRule] = []
    for s in pu["status"].values.astype(int):
        if s == STATUS_LOCKED_OUT:
            rules.append(PUZoneRule(frozenset(), False))
        elif s == STATUS_LOCKED_IN:
            rules.append(PUZoneRule(first, True))
        else:
            rules.append(PUZoneRule(every, False))
    return rules


def _best(rule: PUZoneRule, value_in: dict[int, float]) -> float:
    """A PU's largest possible contribution: 0 if it can take no zone; its forced zone's
    value as-is (possibly negative); else max(0, best allowed) — it may stay unassigned."""
    if not rule.allowed:
        return 0.0
    if rule.forced:
        (zid,) = rule.allowed
        return value_in.get(zid, 0.0)
    return max(0.0, max((value_in.get(z, 0.0) for z in rule.allowed), default=0.0))


def _fmt(x: float) -> str:
    return f"{x:.6g}"


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
