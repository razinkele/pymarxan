"""File readers for MarZone multi-zone projects."""
from __future__ import annotations

import re
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

# MarZone util.hpp is_like_numerical_data: a line made only of these is data, not a header.
_NUMERIC_LINE_CHARS = frozenset(" ,;:^*\"/\t'\\\n\r.+-Ee0123456789")
_WS = re.compile(r"\s+")


def _read_pu_zone_pairs(path: str | Path) -> pd.DataFrame:
    """Two integer columns read by position (MarZone LoadPuLock / LoadPuZone)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError(f"{path}: file is empty (MarZone: 'cannot be read or is empty')")
    lines = text.splitlines()
    first = next(i for i, ln in enumerate(lines) if ln.strip())  # text is non-blank
    start = first if set(lines[first].strip()) <= _NUMERIC_LINE_CHARS else first + 1
    rows: list[tuple[int, int]] = []
    data = [ln for ln in lines[start:] if ln.strip()]
    comma = bool(data) and "," in data[0]
    for lineno, line in enumerate(lines[start:], start=start + 1):
        if not line.strip():
            continue
        raw = line.strip()
        cells = [c.strip() for c in raw.split(",")] if comma else _WS.split(raw)
        if len(cells) < 2:
            raise ValueError(f"{path}: line {lineno} needs two values (puid, zoneid)")
        try:
            a, b = float(cells[0]), float(cells[1])
        except ValueError as exc:
            raise ValueError(f"{path}: line {lineno} has a non-numeric id") from exc
        if not (a.is_integer() and b.is_integer()):
            raise ValueError(f"{path}: line {lineno} has a non-integral id")
        rows.append((int(a), int(b)))
    return pd.DataFrame(rows, columns=["pu", "zone"], dtype=int)


def read_pu_lock(path: str | Path) -> pd.DataFrame:
    """Read ``pulock.dat`` (``puid, zoneid``) into columns ``pu, zone``."""
    return _read_pu_zone_pairs(path)


def read_pu_zone(path: str | Path) -> pd.DataFrame:
    """Read ``puzone.dat`` (``puid, zoneid``, one row per allowed zone) into ``pu, zone``."""
    return _read_pu_zone_pairs(path)


def _named_input(
    project_dir: Path, input_dir: Path, parameters: dict, key: str,
) -> Path:
    """Path of the file named by ``key`` (MarZone loads pulock/puzone only when named)."""
    path = input_dir / str(parameters[key])
    if not path.exists():
        raise FileNotFoundError(f"{project_dir}: {key} names {path}, which does not exist")
    return path


def load_zone_project(project_dir: str | Path) -> ZonalProblem:
    """Load a MarZone project directory into a ``ZonalProblem``.

    ``zones.dat``, ``zonecost.dat``, ``zonecontrib.dat``, ``zonetarget.dat`` and
    ``zoneboundcost.dat`` fall back to those default names when their ``*NAME`` key is
    absent. ``pulock.dat`` / ``puzone.dat`` do not: as in MarZone they are read only when
    ``PULOCKNAME`` / ``PUZONENAME`` is set. A named file that does not exist raises
    ``FileNotFoundError``; a named empty file raises ``ValueError``.
    """
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

    pu_locks = (
        read_pu_lock(_named_input(project_dir, input_dir, base.parameters, "PULOCKNAME"))
        if "PULOCKNAME" in base.parameters else None
    )
    pu_zones = (
        read_pu_zone(_named_input(project_dir, input_dir, base.parameters, "PUZONENAME"))
        if "PUZONENAME" in base.parameters else None
    )

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
        pu_locks=pu_locks,
        pu_zones=pu_zones,
    )
    gaps = problem.contribution_gaps()
    total = len(problem.features) * len(problem.zone_ids)
    if gaps:
        warnings.warn(
            f"{project_dir}: {zcontrib_path.name} lists {total - len(gaps)} of {total} "
            f"(feature, zone) pairs; {len(gaps)} unlisted pair(s) default to 0.0 "
            f"(MarZone zones.hpp:619); first: feature {gaps[0][0]}, zone {gaps[0][1]}",
            stacklevel=2,
        )
    return problem
