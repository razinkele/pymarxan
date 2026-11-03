"""Read a zone solution back per habitat and action; re-score it at other success rates
(spec §3.7). Existing-protected and expected-restored amounts are reported side by side and
never added into one total (spec §3.4)."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pymarxan.recovery.builder import (
    ACTION_NAMES,
    ZONE_RESTORE_ACTIVE,
    ZONE_RESTORE_PASSIVE,
    RecoveryMapping,
)
from pymarxan.solvers.base import Solution
from pymarxan.zones.model import ZonalProblem
from pymarxan.zones.objective import _meets

RECOVERY_CAVEAT = (
    "Restored amounts are expected values: potential × probability that restoration "
    "succeeds (passive: × efficacy η), judged at horizon H — often only 1–2-year "
    "establishment survival. Targets are met on average, not with any stated probability: a "
    "target met by many low-probability units may well fall short in practice. Restored "
    "habitat usually stays below the quality of existing habitat for decades; {maturity}. "
    "Costs, η and the maturity factor are placeholders unless replaced with local evidence. "
    "This plan is a decision aid for stakeholder discussion, not a prescription."
)
NO_TIME_LAG = "no time lag is modelled"
DEFAULT_MULTIPLIERS = (0.5, 0.75, 1.0, 1.25)


def recovery_caveat(mapping: RecoveryMapping) -> str:
    """``RECOVERY_CAVEAT`` with the maturity clause filled in from the mapping."""
    hab = mapping.habitats
    g = hab["maturity"].to_numpy(dtype=float)
    if np.all(g == 1.0):
        clause = NO_TIME_LAG
    elif len(hab) == 1:
        clause = f"a maturity factor of {g[0]:g} is applied"
    else:
        parts = [f"{gi:g} ({name})" for gi, name in zip(g, hab["name"], strict=True)]
        clause = "a maturity factor of " + ", ".join(parts) + " is applied"
    return RECOVERY_CAVEAT.format(maturity=clause)


@dataclass(eq=False)
class RecoveryResult:
    """A recovery plan read back per habitat.

    ``habitats`` leads with shortfalls (columns and rows: largest relative shortfall
    first): habitat, name, restore_shortfall, protect_shortfall, restore_met, protect_met,
    existing_kept (existing amount in units under any action), existing_target,
    restored_active, restored_passive (expected units), restore_target.
    ``actions``: pu, zone, action. ``cost_by_action``: action -> total zone cost.
    """

    assignment: np.ndarray
    actions: pd.DataFrame
    habitats: pd.DataFrame
    cost_by_action: dict[str, float]
    total_cost: float
    targets_met: bool
    mode: str | None
    method: str | None
    solver: str | None
    boundary_dropped: bool
    caveat: str
    mapping: RecoveryMapping


def _achieved(
    problem: ZonalProblem, assignment: np.ndarray, feature: int, zone: int,
) -> float:
    """Σ amount × contribution of ``feature`` over the units assigned to ``zone``."""
    pv = problem.pu_vs_features
    rows = pv[pv["species"] == feature]
    zone_of = dict(zip(problem.planning_units["id"].astype(int), assignment, strict=True))
    contrib = problem.get_contribution(feature, zone)
    return float(sum(
        float(a) * contrib for u, a in zip(rows["pu"], rows["amount"], strict=True)
        if int(zone_of.get(int(u), 0)) == zone
    ))


def _sort_shortfall_first(df: pd.DataFrame) -> pd.DataFrame:
    """Rows by largest relative shortfall (restore or protect), then habitat id."""
    def rel(short: str, target: str) -> pd.Series:
        t = df[target].where(df[target] > 0)
        return (df[short] / t).fillna(0.0)

    key = pd.concat([rel("restore_shortfall", "restore_target"),
                     rel("protect_shortfall", "existing_target")], axis=1).max(axis=1)
    order = sorted(range(len(df)), key=lambda i: (-float(key.iloc[i]), int(df["habitat"].iloc[i])))
    return df.iloc[order].reset_index(drop=True)


def evaluate_recovery(
    solution: Solution, problem: ZonalProblem, mapping: RecoveryMapping,
) -> RecoveryResult:
    """Per-habitat amounts, targets and shortfalls, cost by action, solver provenance."""
    if solution.zone_assignment is None:
        raise ValueError("evaluate_recovery needs a zone solution (zone_assignment is None)")
    assignment = np.asarray(solution.zone_assignment, dtype=int)
    units = mapping.units["pu"].astype(int).tolist()
    actions = pd.DataFrame({
        "pu": units, "zone": assignment, "action": mapping.action_names(assignment),
    })
    zc = problem.zone_costs.set_index(["pu", "zone"])["cost"]
    cost_by_action: dict[str, float] = {}
    for z in sorted(k for k in mapping.zones if k != 0):
        cost_by_action[ACTION_NAMES[z]] = float(sum(
            float(zc.get((u, z), 0.0)) for u, a in zip(units, assignment, strict=True) if a == z
        ))
    rows = []
    for h in mapping.habitats.to_dict("records"):
        fe, fp = int(h["existing_feature"]), int(h["potential_feature"])
        kept = sum(_achieved(problem, assignment, fe, z) for z in mapping.zones if z != 0)
        active = _achieved(problem, assignment, fp, ZONE_RESTORE_ACTIVE)
        passive = (_achieved(problem, assignment, fp, ZONE_RESTORE_PASSIVE)
                   if mapping.passive else 0.0)
        e_t, r_t = float(h["existing_target"]), float(h["restore_target"])
        rows.append({
            "habitat": int(h["habitat"]), "name": h["name"],
            "restore_shortfall": max(0.0, r_t - (active + passive)),
            "protect_shortfall": max(0.0, e_t - kept),
            "restore_met": r_t <= 0 or _meets(active + passive, r_t),
            "protect_met": e_t <= 0 or _meets(kept, e_t),
            "existing_kept": kept, "existing_target": e_t,
            "restored_active": active, "restored_passive": passive,
            "restore_target": r_t,
        })
    habitats = _sort_shortfall_first(pd.DataFrame(rows))
    meta = solution.metadata or {}
    return RecoveryResult(
        assignment=assignment.copy(),
        actions=actions,
        habitats=habitats,
        cost_by_action=cost_by_action,
        total_cost=float(sum(cost_by_action.values())),
        targets_met=bool(habitats["restore_met"].all() and habitats["protect_met"].all()),
        mode=meta.get("recovery_mode"),
        method=meta.get("recovery_method"),
        solver=meta.get("solver"),
        boundary_dropped=bool(meta.get("boundary_dropped", False)),
        caveat=recovery_caveat(mapping),
        mapping=mapping,
    )


def success_sensitivity(
    result: RecoveryResult, multipliers: Sequence[float] = DEFAULT_MULTIPLIERS,
) -> pd.DataFrame:
    """Re-score the fixed plan with every unit's p_success × m (capped at 1) — no re-solve.

    One row per habitat: habitat, name, restore_target, ``restored_at_<m>`` (expected
    units, active + passive) per multiplier, and ``lowest_multiplier_met`` — the smallest
    tested multiplier at which the restore target is still met (NaN when none is).
    """
    mults = sorted(float(m) for m in multipliers)
    if not mults or any(m < 0 for m in mults):
        raise ValueError("multipliers must be a non-empty sequence of values >= 0")
    mapping = result.mapping
    zone_of = dict(zip(result.actions["pu"], result.actions["zone"], strict=True))
    p_of = dict(zip(mapping.units["pu"].astype(int), mapping.units["p_success"], strict=True))
    out = []
    for h in mapping.habitats.to_dict("records"):
        hid, g = int(h["habitat"]), float(h["maturity"])
        eta = float(h["efficacy"]) if mapping.passive else 0.0
        pot = mapping.potential[mapping.potential["habitat"] == hid]
        row: dict[str, object] = {
            "habitat": hid, "name": h["name"], "restore_target": float(h["restore_target"]),
        }
        lowest = float("nan")
        for m in mults:
            total = 0.0
            for u, amount in zip(pot["pu"].astype(int), pot["amount"], strict=True):
                z = int(zone_of.get(u, 0))
                credit = g if z == ZONE_RESTORE_ACTIVE else (
                    eta * g if z == ZONE_RESTORE_PASSIVE else 0.0
                )
                if credit:
                    total += float(amount) * min(1.0, float(p_of[u]) * m) * credit
            row[f"restored_at_{m:g}"] = total
            target = float(h["restore_target"])
            if np.isnan(lowest) and (target <= 0 or _meets(total, target)):
                lowest = m
        row["lowest_multiplier_met"] = lowest
        out.append(row)
    return pd.DataFrame(out)

