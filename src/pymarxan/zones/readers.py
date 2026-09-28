"""File readers for MarZone multi-zone projects."""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from pymarxan.io.readers import _read_dat, load_project
from pymarxan.zones.model import ZonalProblem


def read_zones(path: str | Path) -> pd.DataFrame:
    df = _read_dat(path)
    df["id"] = df["id"].astype(int)
    return df

def read_zone_costs(path: str | Path) -> pd.DataFrame:
    df = _read_dat(path)
    df["pu"] = df["pu"].astype(int)
    df["zone"] = df["zone"].astype(int)
    df["cost"] = df["cost"].astype(float)
    return df

def read_zone_contributions(path: str | Path) -> pd.DataFrame:
    df = _read_dat(path)
    df["feature"] = df["feature"].astype(int)
    df["zone"] = df["zone"].astype(int)
    df["contribution"] = df["contribution"].astype(float)
    return df

_ZONE_TARGET_TYPES_SUPPORTED = {0, 1}
_ZONE_TARGET_TYPES_KNOWN = {0, 1, 2, 3}


def read_zone_targets(path: str | Path) -> pd.DataFrame:
    """Read ``zonetarget.dat`` (``zone, feature, target[, targettype]``).

    ``targettype`` follows MarZone ``zones.hpp:96-160``: 0 = amount (default), 1 = proportion
    of the feature's total raw amount (resolved by :func:`resolve_zone_target_types`),
    2/3 = occurrence targets, which pymarxan does not support and rejects by row.
    """
    df = _read_dat(path)
    df["zone"] = df["zone"].astype(int)
    df["feature"] = df["feature"].astype(int)
    df["target"] = df["target"].astype(float)
    for pos, fid in enumerate(df["feature"].values, start=1):
        if int(fid) < 0:
            raise ValueError(
                f"{path}: row {pos} has feature {int(fid)}; MarZone's -1 'all species' "
                "wildcard (zones.hpp:127-138) is not supported"
            )
    if "targettype" in df.columns:
        df["targettype"] = df["targettype"].fillna(0).astype(int)
        for pos, (zid, fid, ttype) in enumerate(
            zip(df["zone"].values, df["feature"].values, df["targettype"].values, strict=True),
            start=1,
        ):
            if int(ttype) not in _ZONE_TARGET_TYPES_KNOWN:
                raise ValueError(
                    f"{path}: row {pos} (zone {int(zid)}, feature {int(fid)}) has "
                    f"targettype {int(ttype)}; known types are 0-3"
                )
            if int(ttype) not in _ZONE_TARGET_TYPES_SUPPORTED:
                raise ValueError(
                    f"{path}: row {pos} (zone {int(zid)}, feature {int(fid)}) has "
                    f"targettype {int(ttype)}: occurrence targets (MarZone types 2/3) are "
                    "not supported"
                )
    return df


def resolve_zone_target_types(
    zone_targets: pd.DataFrame,
    pu_vs_features: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve ``targettype == 1`` rows: target × the feature's total raw amount.

    MarZone ``zones.hpp:149-150``. Resolved rows are rewritten as type 0 so that a
    write → read cycle does not multiply again (same idempotence contract as
    ``io.readers._resolve_prop_targets``). Frames without the column pass through.
    The total is the feature's amount over every planning unit regardless of status
    (MarZone ``pu.TotalSpeciesAmount``, ``pu.hpp:142-151``; locked-out PUs included).
    """
    if "targettype" not in zone_targets.columns:
        return zone_targets
    df = zone_targets.copy()
    totals = pu_vs_features.groupby("species")["amount"].sum()
    is_prop = df["targettype"] == 1
    if is_prop.any():
        feature_totals = df.loc[is_prop, "feature"].map(totals).fillna(0.0)
        df.loc[is_prop, "target"] = df.loc[is_prop, "target"] * feature_totals
        df.loc[is_prop, "targettype"] = 0
    return df

def read_zone_boundary_costs(path: str | Path) -> pd.DataFrame:
    df = _read_dat(path)
    df["zone1"] = df["zone1"].astype(int)
    df["zone2"] = df["zone2"].astype(int)
    df["cost"] = df["cost"].astype(float)
    return df

def load_zone_project(project_dir: str | Path) -> ZonalProblem:
    project_dir = Path(project_dir)
    base = load_project(project_dir)

    input_dir = project_dir / base.parameters.get("INPUTDIR", "input")

    zones_name = base.parameters.get("ZONESNAME", "zones.dat")
    zones = read_zones(input_dir / zones_name)

    zonecost_name = base.parameters.get("ZONECOSTNAME", "zonecost.dat")
    zone_costs = read_zone_costs(input_dir / zonecost_name)

    zone_contributions = None
    zcontrib_name = base.parameters.get("ZONECONTRIBNAME", "zonecontrib.dat")
    zcontrib_path = input_dir / zcontrib_name
    if zcontrib_path.exists():
        zone_contributions = read_zone_contributions(zcontrib_path)

    zone_targets = None
    ztarget_name = base.parameters.get("ZONETARGETNAME", "zonetarget.dat")
    ztarget_path = input_dir / ztarget_name
    if ztarget_path.exists():
        zone_targets = resolve_zone_target_types(
            read_zone_targets(ztarget_path), base.pu_vs_features,
        )

    zone_boundary_costs = None
    zbc_name = base.parameters.get("ZONEBOUNDCOSTNAME", "zoneboundcost.dat")
    zbc_path = input_dir / zbc_name
    if zbc_path.exists():
        zone_boundary_costs = read_zone_boundary_costs(zbc_path)

    problem = ZonalProblem(
        planning_units=base.planning_units,
        features=base.features,
        pu_vs_features=base.pu_vs_features,
        boundary=base.boundary,
        parameters=base.parameters,
        zones=zones,
        zone_costs=zone_costs,
        zone_contributions=zone_contributions,
        zone_targets=zone_targets,
        zone_boundary_costs=zone_boundary_costs,
    )
    gaps = problem.contribution_gaps()
    total = len(problem.features) * len(problem.zone_ids)
    if gaps:
        warnings.warn(
            f"{project_dir}: zonecontrib.dat lists {total - len(gaps)} of {total} "
            f"(feature, zone) pairs; {len(gaps)} unlisted pair(s) default to 0.0 "
            f"(MarZone zones.hpp:619); first: feature {gaps[0][0]}, zone {gaps[0][1]}",
            stacklevel=2,
        )
    return problem
