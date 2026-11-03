"""Persist recovery inputs next to the zone project they build (spec §3.8).

Sidecar files (in the project directory, beside ``input.dat``):

- ``potential.csv`` — ``pu, feature, amount``: raw incremental potential.
- ``recovery_units.csv`` — ``pu, status, p_success, restore_cost``: the status as given
  (the zone project's pu.dat holds the built status), p per unit and the active restore
  cost before per-habitat implementation costs.
- ``recovery.json`` — habitats and their feature ids, zone ids -> actions, passive η and
  cost factor, maturity, horizon, cost factors, targets, spf, provenance.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd

from pymarxan.models.problem import ConservationProblem
from pymarxan.recovery.builder import (
    RecoveryMapping,
    build_recovery_problem,
    existing_feature_id,
    potential_feature_id,
)
from pymarxan.recovery.inputs import PassiveAction, RecoveryInputs
from pymarxan.zones.model import ZonalProblem
from pymarxan.zones.readers import load_zone_project
from pymarxan.zones.writers import save_zone_project

FORMAT = "pymarxan-recovery"
FORMAT_VERSION = 1
POTENTIAL_FILE = "potential.csv"
UNITS_FILE = "recovery_units.csv"
SIDECAR_FILE = "recovery.json"


class RecoveryProject(NamedTuple):
    inputs: RecoveryInputs
    problem: ZonalProblem
    mapping: RecoveryMapping


def _int_keys(d: dict[str, Any] | None) -> dict[int, float] | None:
    return None if d is None else {int(k): float(v) for k, v in d.items()}


def _per_habitat_json(value: float | Any) -> float | dict[str, float]:
    if isinstance(value, int | float):
        return float(value)
    return {str(int(k)): float(v) for k, v in value.items()}


def write_recovery_inputs(inputs: RecoveryInputs, project_dir: str | Path) -> None:
    """Write the sidecar files for ``inputs`` (the base project is not written)."""
    errors = inputs.validate()
    if errors:
        raise ValueError("invalid recovery inputs: " + "; ".join(errors))
    d = Path(project_dir)
    d.mkdir(parents=True, exist_ok=True)
    pot = inputs.potential[["pu", "feature", "amount"]].sort_values(["feature", "pu"])
    pot.to_csv(d / POTENTIAL_FILE, index=False)
    units = inputs.unit_ids()
    base_rc = (
        [float(inputs.restore_cost[u]) for u in units] if inputs.restore_cost is not None
        else list(inputs.restore_cost_factor * inputs.unit_costs())
    )
    pd.DataFrame({
        "pu": units, "status": inputs.unit_statuses(), "p_success": inputs.p_by_unit(),
        "restore_cost": base_rc,
    }).to_csv(d / UNITS_FILE, index=False)
    names = inputs.habitat_names()
    passive = inputs.passive
    sidecar: dict[str, Any] = {
        "format": FORMAT,
        "version": FORMAT_VERSION,
        "habitats": [
            {"habitat": h, "name": names[h], "existing_feature": existing_feature_id(h),
             "potential_feature": potential_feature_id(h)}
            for h in inputs.habitat_ids()
        ],
        "zones": {"0": "no_action", "1": "protect", "2": "restore_active",
                  **({"3": "restore_passive"} if passive is not None else {})},
        "p_success_scalar": None if isinstance(inputs.p_success, pd.Series)
        else float(inputs.p_success),
        "restore_cost_from_factor": inputs.restore_cost is None,
        "restore_cost_factor": float(inputs.restore_cost_factor),
        "restore_cost_by_habitat": None if inputs.restore_cost_by_habitat is None
        else {str(k): float(v) for k, v in inputs.restore_cost_by_habitat.items()},
        "passive": None if passive is None else {
            "efficacy": float(passive.efficacy),
            "efficacy_by_habitat": None if passive.efficacy_by_habitat is None
            else {str(k): float(v) for k, v in passive.efficacy_by_habitat.items()},
            "cost_factor": float(passive.cost_factor),
        },
        "maturity": None if inputs.maturity is None
        else {str(k): float(v) for k, v in inputs.maturity.items()},
        "horizon_years": inputs.horizon_years,
        "protect_prop": _per_habitat_json(inputs.protect_prop),
        "restore_prop": _per_habitat_json(inputs.restore_prop),
        "restore_target": None if inputs.restore_target is None
        else {str(k): float(v) for k, v in inputs.restore_target.items()},
        "spf": inputs.spf,
        "provenance": inputs.provenance(),
    }
    (d / SIDECAR_FILE).write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n")


def _read_sidecar(d: Path) -> dict[str, Any]:
    sidecar: dict[str, Any] = json.loads((d / SIDECAR_FILE).read_text())
    if sidecar.get("format") != FORMAT or sidecar.get("version") != FORMAT_VERSION:
        raise ValueError(
            f"{d / SIDECAR_FILE} is not a {FORMAT} v{FORMAT_VERSION} file"
        )
    return sidecar


def read_recovery_inputs(project_dir: str | Path, base: ConservationProblem) -> RecoveryInputs:
    """``RecoveryInputs`` from the sidecar files plus ``base`` (habitats as features,
    existing amounts in ``pu_vs_features``)."""
    d = Path(project_dir)
    sidecar = _read_sidecar(d)
    potential = pd.read_csv(d / POTENTIAL_FILE).astype(
        {"pu": int, "feature": int, "amount": float}
    )
    units = pd.read_csv(d / UNITS_FILE).astype({"pu": int})
    p_scalar = sidecar["p_success_scalar"]
    p_success: float | pd.Series = (
        float(p_scalar) if p_scalar is not None
        else pd.Series(units["p_success"].astype(float).to_numpy(), index=units["pu"])
    )
    restore_cost = None if sidecar["restore_cost_from_factor"] else pd.Series(
        units["restore_cost"].astype(float).to_numpy(), index=units["pu"],
    )
    pa = sidecar["passive"]
    passive = None if pa is None else PassiveAction(
        efficacy=float(pa["efficacy"]),
        efficacy_by_habitat=_int_keys(pa["efficacy_by_habitat"]),
        cost_factor=float(pa["cost_factor"]),
    )

    def per_habitat(v: float | dict[str, float]) -> float | dict[int, float]:
        return float(v) if isinstance(v, int | float) else {int(k): float(x)
                                                             for k, x in v.items()}

    return RecoveryInputs(
        base=base,
        potential=potential,
        p_success=p_success,
        restore_cost=restore_cost,
        restore_cost_factor=float(sidecar["restore_cost_factor"]),
        restore_cost_by_habitat=_int_keys(sidecar["restore_cost_by_habitat"]),
        passive=passive,
        maturity=_int_keys(sidecar["maturity"]),
        horizon_years=sidecar["horizon_years"],
        protect_prop=per_habitat(sidecar["protect_prop"]),
        restore_prop=per_habitat(sidecar["restore_prop"]),
        restore_target=_int_keys(sidecar["restore_target"]),
        spf=sidecar["spf"],
    )


def save_recovery_project(
    inputs: RecoveryInputs, project_dir: str | Path,
) -> tuple[ZonalProblem, RecoveryMapping]:
    """Build the problem, save it as a zone project and write the sidecar beside it.
    Returns what ``build_recovery_problem`` returned."""
    problem, mapping = build_recovery_problem(inputs)
    save_zone_project(problem, project_dir)
    write_recovery_inputs(inputs, project_dir)
    return problem, mapping


def load_recovery_project(project_dir: str | Path) -> RecoveryProject:
    """Reload a saved recovery project: rebuild the base from the zone project and the
    sidecar (habitats, statuses as given, existing amounts from the ``*_existing``
    features), read the inputs and rebuild the problem and mapping from them."""
    d = Path(project_dir)
    sidecar = _read_sidecar(d)
    zonal = load_zone_project(d)
    units = pd.read_csv(d / UNITS_FILE).astype({"pu": int, "status": int})
    status_of = dict(zip(units["pu"], units["status"], strict=True))
    pu = zonal.planning_units[["id", "cost"]].copy()
    pu["status"] = [int(status_of[int(i)]) for i in pu["id"]]
    habitat_of = {int(h["existing_feature"]): int(h["habitat"]) for h in sidecar["habitats"]}
    pv = zonal.pu_vs_features
    existing = pv[pv["species"].astype(int).isin(habitat_of)].copy()
    existing["species"] = [habitat_of[int(f)] for f in existing["species"]]
    base = ConservationProblem(
        planning_units=pu,
        features=pd.DataFrame({
            "id": [int(h["habitat"]) for h in sidecar["habitats"]],
            "name": [str(h["name"]) for h in sidecar["habitats"]],
            "target": 0.0, "spf": 1.0,
        }),
        pu_vs_features=existing[["species", "pu", "amount"]].reset_index(drop=True),
        boundary=zonal.boundary,
        parameters=dict(zonal.parameters),
    )
    inputs = read_recovery_inputs(d, base)
    problem, mapping = build_recovery_problem(inputs)
    return RecoveryProject(inputs, problem, mapping)
