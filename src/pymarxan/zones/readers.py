"""File readers for MarZone multi-zone projects."""
from __future__ import annotations

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

def read_zone_targets(path: str | Path) -> pd.DataFrame:
    df = _read_dat(path)
    df["zone"] = df["zone"].astype(int)
    df["feature"] = df["feature"].astype(int)
    df["target"] = df["target"].astype(float)
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
        zone_targets = read_zone_targets(ztarget_path)

    zone_boundary_costs = None
    zbc_name = base.parameters.get("ZONEBOUNDCOSTNAME", "zoneboundcost.dat")
    zbc_path = input_dir / zbc_name
    if zbc_path.exists():
        zone_boundary_costs = read_zone_boundary_costs(zbc_path)

    return ZonalProblem(
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
