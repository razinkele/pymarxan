"""Synthetic problem generators for performance benchmarks."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pymarxan.models.problem import ConservationProblem
from pymarxan.zones.model import ZonalProblem


def make_problem(
    n_pu: int,
    n_feat: int,
    density: float = 0.3,
    seed: int = 0,
) -> ConservationProblem:
    """Generate a synthetic ConservationProblem for benchmarking.

    Parameters
    ----------
    n_pu : int
        Number of planning units.
    n_feat : int
        Number of conservation features.
    density : float
        Proportion of (PU, feature) cells that have non-zero amounts.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    ConservationProblem
        Fully populated problem with grid-like boundary.
    """
    rng = np.random.default_rng(seed)

    # Planning units: id 1..n_pu, cost uniform(1, 100), status 0
    pu_ids = np.arange(1, n_pu + 1)
    pu_costs = rng.uniform(1.0, 100.0, size=n_pu)
    planning_units = pd.DataFrame({
        "id": pu_ids,
        "cost": pu_costs,
        "status": np.zeros(n_pu, dtype=int),
    })

    # Features: id 1..n_feat, target uniform(10, 50), spf 1.0
    feat_ids = np.arange(1, n_feat + 1)
    feat_targets = rng.uniform(10.0, 50.0, size=n_feat)
    features = pd.DataFrame({
        "id": feat_ids,
        "name": [f"feat_{i}" for i in feat_ids],
        "target": feat_targets,
        "spf": np.ones(n_feat),
    })

    # PU-feature matrix: random sparse with given density
    n_entries = int(n_pu * n_feat * density)
    pu_indices = rng.integers(1, n_pu + 1, size=n_entries)
    feat_indices = rng.integers(1, n_feat + 1, size=n_entries)
    amounts = rng.uniform(0.1, 10.0, size=n_entries)

    # Deduplicate by taking the last value for each (pu, feat) pair
    seen: dict[tuple[int, int], int] = {}
    unique_pu = []
    unique_feat = []
    unique_amount = []
    for k in range(n_entries):
        key = (int(pu_indices[k]), int(feat_indices[k]))
        if key in seen:
            # Replace previous entry
            idx = seen[key]
            unique_amount[idx] = float(amounts[k])
        else:
            seen[key] = len(unique_pu)
            unique_pu.append(key[0])
            unique_feat.append(key[1])
            unique_amount.append(float(amounts[k]))

    pu_vs_features = pd.DataFrame({
        "species": unique_feat,
        "pu": unique_pu,
        "amount": unique_amount,
    })

    # Boundary: grid-like (sqrt(n_pu) x sqrt(n_pu) grid)
    grid_side = int(math.isqrt(n_pu))
    boundary_rows = []
    for i in range(1, n_pu + 1):
        col = (i - 1) % grid_side
        # Right neighbor
        if col < grid_side - 1:
            right = i + 1
            if right <= n_pu:
                boundary_rows.append({"id1": i, "id2": right, "boundary": 1.0})
        # Bottom neighbor
        bottom = i + grid_side
        if bottom <= n_pu:
            boundary_rows.append({"id1": i, "id2": bottom, "boundary": 1.0})

    boundary = pd.DataFrame(boundary_rows) if boundary_rows else None

    return ConservationProblem(
        planning_units=planning_units,
        features=features,
        pu_vs_features=pu_vs_features,
        boundary=boundary,
        parameters={"BLM": 1.0},
    )


def make_zone_problem(
    n_pu: int,
    n_feat: int,
    n_zones: int,
    density: float = 0.3,
    seed: int = 0,
) -> ZonalProblem:
    """Generate a synthetic ZonalProblem for benchmarking.

    Parameters
    ----------
    n_pu : int
        Number of planning units.
    n_feat : int
        Number of conservation features.
    n_zones : int
        Number of zones.
    density : float
        Proportion of (PU, feature) cells that have non-zero amounts.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    ZonalProblem
        Fully populated zonal problem.
    """
    rng = np.random.default_rng(seed + 1000)

    # Use make_problem for base data
    base = make_problem(n_pu, n_feat, density, seed)

    # Zones: id 1..n_zones
    zone_ids = np.arange(1, n_zones + 1)
    zones = pd.DataFrame({
        "id": zone_ids,
        "name": [f"zone_{z}" for z in zone_ids],
    })

    # Zone costs: random for each (pu, zone) pair
    zc_pu = []
    zc_zone = []
    zc_cost = []
    for pu_id in range(1, n_pu + 1):
        for z_id in zone_ids:
            zc_pu.append(pu_id)
            zc_zone.append(z_id)
            zc_cost.append(float(rng.uniform(1.0, 100.0)))
    zone_costs = pd.DataFrame({
        "pu": zc_pu,
        "zone": zc_zone,
        "cost": zc_cost,
    })

    # Zone targets: random for each (zone, feature) pair
    zt_zone = []
    zt_feat = []
    zt_target = []
    for z_id in zone_ids:
        for f_id in range(1, n_feat + 1):
            zt_zone.append(z_id)
            zt_feat.append(f_id)
            zt_target.append(float(rng.uniform(5.0, 30.0)))
    zone_targets = pd.DataFrame({
        "zone": zt_zone,
        "feature": zt_feat,
        "target": zt_target,
    })

    # Zone boundary costs: random for each zone pair
    zbc_rows = []
    for z1 in zone_ids:
        for z2 in zone_ids:
            if z1 < z2:
                zbc_rows.append({
                    "zone1": int(z1),
                    "zone2": int(z2),
                    "cost": float(rng.uniform(0.1, 2.0)),
                })
    zone_boundary_costs = pd.DataFrame(zbc_rows) if zbc_rows else None

    return ZonalProblem(
        planning_units=base.planning_units,
        features=base.features,
        pu_vs_features=base.pu_vs_features,
        boundary=base.boundary,
        parameters={"BLM": 1.0, "NUMITNS": 10000, "NUMTEMP": 100},
        zones=zones,
        zone_costs=zone_costs,
        zone_targets=zone_targets,
        zone_boundary_costs=zone_boundary_costs,
    )
