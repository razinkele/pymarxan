# Phase 3: Multi-Zone & Connectivity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add MarZone (multi-zone) support and Marxan Connect (connectivity analysis) to pymarxan, enabling zone-based conservation planning and connectivity-aware reserve design.

**Architecture:** New subpackages `pymarxan.zones` (zone model, readers, objective, SA solver) and `pymarxan.connectivity` (connectivity matrix I/O, graph metrics, feature conversion). Both integrate with existing three-layer architecture. The `Solution` dataclass is extended with an optional `zone_assignment` field for zone problems. A new `ZonalProblem` dataclass holds zone-specific data alongside the base `ConservationProblem`. Connectivity metrics are computed as standalone functions and optionally injected as synthetic features into any problem.

**Tech Stack:** numpy, pandas, networkx (graph metrics), scipy.sparse (connectivity matrices), existing shiny/matplotlib stack

---

## Task 1: Zone Data Model

**Files:**
- Create: `src/pymarxan/zones/__init__.py`
- Create: `src/pymarxan/zones/model.py`
- Test: `tests/pymarxan/zones/__init__.py`
- Test: `tests/pymarxan/zones/test_model.py`

The zone data model defines `ZonalProblem` — a container that wraps a standard `ConservationProblem` with zone-specific DataFrames: zones definition, per-PU-per-zone costs, zone contributions (how much a feature "counts" in each zone), and zone-specific targets.

**Step 1: Write the failing tests**

`tests/pymarxan/zones/__init__.py`: empty

`tests/pymarxan/zones/test_model.py`:
```python
import pandas as pd

from pymarxan.zones.model import ZonalProblem


def _make_base_data():
    """Minimal base data: 4 PUs, 2 features."""
    planning_units = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "cost": [10.0, 15.0, 20.0, 12.0],
        "status": [0, 0, 0, 0],
    })
    features = pd.DataFrame({
        "id": [1, 2],
        "name": ["sp_a", "sp_b"],
        "target": [20.0, 15.0],
        "spf": [1.0, 1.0],
    })
    pu_vs_features = pd.DataFrame({
        "species": [1, 1, 1, 1, 2, 2, 2, 2],
        "pu": [1, 2, 3, 4, 1, 2, 3, 4],
        "amount": [10.0, 8.0, 6.0, 5.0, 5.0, 7.0, 9.0, 4.0],
    })
    return planning_units, features, pu_vs_features


def _make_zone_data():
    """Two zones: 'protected' and 'sustainable'."""
    zones = pd.DataFrame({
        "id": [1, 2],
        "name": ["protected", "sustainable"],
    })
    zone_costs = pd.DataFrame({
        "pu": [1, 1, 2, 2, 3, 3, 4, 4],
        "zone": [1, 2, 1, 2, 1, 2, 1, 2],
        "cost": [100.0, 50.0, 150.0, 80.0, 200.0, 100.0, 120.0, 60.0],
    })
    zone_contributions = pd.DataFrame({
        "feature": [1, 1, 2, 2],
        "zone": [1, 2, 1, 2],
        "contribution": [1.0, 0.5, 1.0, 0.3],
    })
    zone_targets = pd.DataFrame({
        "zone": [1, 1, 2, 2],
        "feature": [1, 2, 1, 2],
        "target": [10.0, 8.0, 5.0, 3.0],
    })
    zone_boundary_costs = pd.DataFrame({
        "zone1": [1, 1, 2],
        "zone2": [1, 2, 2],
        "cost": [0.0, 50.0, 0.0],
    })
    return zones, zone_costs, zone_contributions, zone_targets, zone_boundary_costs


class TestZonalProblem:
    def test_create(self):
        pu, feat, puvspr = _make_base_data()
        zones, zc, zcontrib, zt, zbc = _make_zone_data()
        zp = ZonalProblem(
            planning_units=pu, features=feat, pu_vs_features=puvspr,
            zones=zones, zone_costs=zc,
            zone_contributions=zcontrib, zone_targets=zt,
            zone_boundary_costs=zbc,
        )
        assert zp.n_zones == 2
        assert zp.n_planning_units == 4
        assert zp.n_features == 2

    def test_zone_ids(self):
        pu, feat, puvspr = _make_base_data()
        zones, zc, zcontrib, zt, zbc = _make_zone_data()
        zp = ZonalProblem(
            planning_units=pu, features=feat, pu_vs_features=puvspr,
            zones=zones, zone_costs=zc,
        )
        assert zp.zone_ids == {1, 2}

    def test_get_zone_cost(self):
        pu, feat, puvspr = _make_base_data()
        zones, zc, zcontrib, zt, zbc = _make_zone_data()
        zp = ZonalProblem(
            planning_units=pu, features=feat, pu_vs_features=puvspr,
            zones=zones, zone_costs=zc,
        )
        assert zp.get_zone_cost(1, 1) == 100.0
        assert zp.get_zone_cost(1, 2) == 50.0

    def test_get_contribution(self):
        pu, feat, puvspr = _make_base_data()
        zones, zc, zcontrib, zt, zbc = _make_zone_data()
        zp = ZonalProblem(
            planning_units=pu, features=feat, pu_vs_features=puvspr,
            zones=zones, zone_costs=zc,
            zone_contributions=zcontrib,
        )
        assert zp.get_contribution(1, 1) == 1.0
        assert zp.get_contribution(1, 2) == 0.5

    def test_default_contribution_is_one(self):
        pu, feat, puvspr = _make_base_data()
        zones, zc, _, _, _ = _make_zone_data()
        zp = ZonalProblem(
            planning_units=pu, features=feat, pu_vs_features=puvspr,
            zones=zones, zone_costs=zc,
        )
        assert zp.get_contribution(1, 1) == 1.0

    def test_validate_valid(self):
        pu, feat, puvspr = _make_base_data()
        zones, zc, zcontrib, zt, zbc = _make_zone_data()
        zp = ZonalProblem(
            planning_units=pu, features=feat, pu_vs_features=puvspr,
            zones=zones, zone_costs=zc,
            zone_contributions=zcontrib, zone_targets=zt,
            zone_boundary_costs=zbc,
        )
        errors = zp.validate()
        assert errors == []

    def test_validate_missing_zone_cost(self):
        pu, feat, puvspr = _make_base_data()
        zones, _, _, _, _ = _make_zone_data()
        # zone_costs missing PU 4
        zc_incomplete = pd.DataFrame({
            "pu": [1, 1, 2, 2, 3, 3],
            "zone": [1, 2, 1, 2, 1, 2],
            "cost": [100.0, 50.0, 150.0, 80.0, 200.0, 100.0],
        })
        zp = ZonalProblem(
            planning_units=pu, features=feat, pu_vs_features=puvspr,
            zones=zones, zone_costs=zc_incomplete,
        )
        errors = zp.validate()
        assert any("zone_costs" in e for e in errors)
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/pymarxan/zones/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pymarxan.zones'`

**Step 3: Implement zone data model**

`src/pymarxan/zones/__init__.py`: empty

`src/pymarxan/zones/model.py`:
```python
"""Zonal conservation problem data model for MarZone-style multi-zone planning."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from pymarxan.models.problem import ConservationProblem


@dataclass
class ZonalProblem(ConservationProblem):
    """Conservation problem with multiple management zones.

    Extends ConservationProblem with zone definitions, per-zone costs,
    contribution weights, zone-specific targets, and zone boundary costs.

    Parameters
    ----------
    zones : pd.DataFrame
        Zone definitions with columns ``id``, ``name``.
    zone_costs : pd.DataFrame
        Per-PU per-zone costs with columns ``pu``, ``zone``, ``cost``.
    zone_contributions : pd.DataFrame | None
        How much each feature counts in each zone: ``feature``, ``zone``,
        ``contribution`` (0-1). Defaults to 1.0 for all if None.
    zone_targets : pd.DataFrame | None
        Per-zone per-feature targets: ``zone``, ``feature``, ``target``.
    zone_boundary_costs : pd.DataFrame | None
        Zone adjacency costs: ``zone1``, ``zone2``, ``cost``.
    """

    zones: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    zone_costs: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    zone_contributions: pd.DataFrame | None = None
    zone_targets: pd.DataFrame | None = None
    zone_boundary_costs: pd.DataFrame | None = None

    @property
    def n_zones(self) -> int:
        return len(self.zones)

    @property
    def zone_ids(self) -> set:
        return set(self.zones["id"])

    def get_zone_cost(self, pu_id: int, zone_id: int) -> float:
        """Get the cost of placing a PU in a specific zone."""
        row = self.zone_costs[
            (self.zone_costs["pu"] == pu_id)
            & (self.zone_costs["zone"] == zone_id)
        ]
        if len(row) == 0:
            return 0.0
        return float(row.iloc[0]["cost"])

    def get_contribution(self, feature_id: int, zone_id: int) -> float:
        """Get the contribution weight of a feature in a zone.

        Returns 1.0 if no zone_contributions are defined.
        """
        if self.zone_contributions is None:
            return 1.0
        row = self.zone_contributions[
            (self.zone_contributions["feature"] == feature_id)
            & (self.zone_contributions["zone"] == zone_id)
        ]
        if len(row) == 0:
            return 1.0
        return float(row.iloc[0]["contribution"])

    def validate(self) -> list[str]:
        """Validate the zonal problem and return error messages."""
        errors = super().validate()

        # Check zones has required columns
        if not {"id", "name"}.issubset(set(self.zones.columns)):
            errors.append("zones missing columns: id, name")

        # Check zone_costs has required columns
        if not {"pu", "zone", "cost"}.issubset(set(self.zone_costs.columns)):
            errors.append("zone_costs missing columns: pu, zone, cost")
        else:
            # Every PU must have a cost for every zone
            pu_ids = set(self.planning_units["id"])
            z_ids = self.zone_ids
            for pid in pu_ids:
                for zid in z_ids:
                    match = self.zone_costs[
                        (self.zone_costs["pu"] == pid)
                        & (self.zone_costs["zone"] == zid)
                    ]
                    if len(match) == 0:
                        errors.append(
                            f"zone_costs missing entry for PU {pid}, zone {zid}"
                        )
                        break
                if errors and "zone_costs missing entry" in errors[-1]:
                    break

        # Optional: validate zone_contributions columns
        if self.zone_contributions is not None:
            req = {"feature", "zone", "contribution"}
            if not req.issubset(set(self.zone_contributions.columns)):
                errors.append(
                    f"zone_contributions missing columns: "
                    f"{sorted(req - set(self.zone_contributions.columns))}"
                )

        # Optional: validate zone_targets columns
        if self.zone_targets is not None:
            req = {"zone", "feature", "target"}
            if not req.issubset(set(self.zone_targets.columns)):
                errors.append(
                    f"zone_targets missing columns: "
                    f"{sorted(req - set(self.zone_targets.columns))}"
                )

        # Optional: validate zone_boundary_costs columns
        if self.zone_boundary_costs is not None:
            req = {"zone1", "zone2", "cost"}
            if not req.issubset(set(self.zone_boundary_costs.columns)):
                errors.append(
                    f"zone_boundary_costs missing columns: "
                    f"{sorted(req - set(self.zone_boundary_costs.columns))}"
                )

        return errors
```

**Step 4: Run tests**

Run: `pytest tests/pymarxan/zones/test_model.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add src/pymarxan/zones/ tests/pymarxan/zones/
git commit -m "feat: add ZonalProblem data model for multi-zone conservation planning"
```

---

## Task 2: Zone Test Fixture Data

**Files:**
- Create: `tests/data/zones/input.dat`
- Create: `tests/data/zones/input/pu.dat`
- Create: `tests/data/zones/input/spec.dat`
- Create: `tests/data/zones/input/puvspr.dat`
- Create: `tests/data/zones/input/bound.dat`
- Create: `tests/data/zones/input/zones.dat`
- Create: `tests/data/zones/input/zonecost.dat`
- Create: `tests/data/zones/input/zonecontrib.dat`
- Create: `tests/data/zones/input/zonetarget.dat`
- Create: `tests/data/zones/input/zoneboundcost.dat`

**Step 1: Create test fixtures**

A small zone problem: 4 PUs, 2 features, 2 zones ("protected", "sustainable"). Hand-verifiable.

`tests/data/zones/input.dat`:
```
INPUTDIR input
PUNAME pu.dat
SPECNAME spec.dat
PUVSPRNAME puvspr.dat
BOUNDNAME bound.dat
ZONESNAME zones.dat
ZONECOSTNAME zonecost.dat
ZONECONTRIBNAME zonecontrib.dat
ZONETARGETNAME zonetarget.dat
ZONEBOUNDCOSTNAME zoneboundcost.dat
BLM 1.0
NUMITNS 1000000
NUMREPS 10
```

`tests/data/zones/input/pu.dat`:
```
id,cost,status
1,10.0,0
2,15.0,0
3,20.0,0
4,12.0,0
```

`tests/data/zones/input/spec.dat`:
```
id,name,target,spf
1,species_a,20.0,1.0
2,species_b,15.0,1.0
```

`tests/data/zones/input/puvspr.dat`:
```
species,pu,amount
1,1,10.0
1,2,8.0
1,3,6.0
1,4,5.0
2,1,5.0
2,2,7.0
2,3,9.0
2,4,4.0
```

`tests/data/zones/input/bound.dat`:
```
id1,id2,boundary
1,1,2.0
2,2,1.0
3,3,1.0
4,4,2.0
1,2,1.0
2,3,1.0
3,4,1.0
```

`tests/data/zones/input/zones.dat`:
```
id,name
1,protected
2,sustainable
```

`tests/data/zones/input/zonecost.dat`:
```
pu,zone,cost
1,1,100.0
1,2,50.0
2,1,150.0
2,2,80.0
3,1,200.0
3,2,100.0
4,1,120.0
4,2,60.0
```

`tests/data/zones/input/zonecontrib.dat`:
```
feature,zone,contribution
1,1,1.0
1,2,0.5
2,1,1.0
2,2,0.3
```

`tests/data/zones/input/zonetarget.dat`:
```
zone,feature,target
1,1,10.0
1,2,8.0
2,1,5.0
2,2,3.0
```

`tests/data/zones/input/zoneboundcost.dat`:
```
zone1,zone2,cost
1,1,0.0
1,2,50.0
2,1,50.0
2,2,0.0
```

**Step 2: Verify files are readable**

Run: `python -c "import pandas as pd; print(pd.read_csv('tests/data/zones/input/zones.dat'))"`
Expected: DataFrame with 2 rows

**Step 3: Commit**

```bash
git add tests/data/zones/
git commit -m "test: add zone test fixture (4 PUs, 2 features, 2 zones)"
```

---

## Task 3: Zone File Readers

**Files:**
- Create: `src/pymarxan/zones/readers.py`
- Test: `tests/pymarxan/zones/test_readers.py`

**Step 1: Write the failing tests**

`tests/pymarxan/zones/test_readers.py`:
```python
from pathlib import Path

from pymarxan.zones.model import ZonalProblem
from pymarxan.zones.readers import (
    read_zones,
    read_zone_costs,
    read_zone_contributions,
    read_zone_targets,
    read_zone_boundary_costs,
    load_zone_project,
)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "zones"
INPUT_DIR = DATA_DIR / "input"


class TestReadZones:
    def test_reads_zones(self):
        df = read_zones(INPUT_DIR / "zones.dat")
        assert len(df) == 2
        assert set(df.columns) >= {"id", "name"}
        assert df["id"].dtype == int


class TestReadZoneCosts:
    def test_reads_costs(self):
        df = read_zone_costs(INPUT_DIR / "zonecost.dat")
        assert len(df) == 8
        assert set(df.columns) >= {"pu", "zone", "cost"}

    def test_types(self):
        df = read_zone_costs(INPUT_DIR / "zonecost.dat")
        assert df["pu"].dtype == int
        assert df["zone"].dtype == int
        assert df["cost"].dtype == float


class TestReadZoneContributions:
    def test_reads_contributions(self):
        df = read_zone_contributions(INPUT_DIR / "zonecontrib.dat")
        assert len(df) == 4
        assert set(df.columns) >= {"feature", "zone", "contribution"}

    def test_values_in_range(self):
        df = read_zone_contributions(INPUT_DIR / "zonecontrib.dat")
        assert (df["contribution"] >= 0).all()
        assert (df["contribution"] <= 1).all()


class TestReadZoneTargets:
    def test_reads_targets(self):
        df = read_zone_targets(INPUT_DIR / "zonetarget.dat")
        assert len(df) == 4
        assert set(df.columns) >= {"zone", "feature", "target"}


class TestReadZoneBoundaryCosts:
    def test_reads_boundary_costs(self):
        df = read_zone_boundary_costs(INPUT_DIR / "zoneboundcost.dat")
        assert len(df) == 4
        assert set(df.columns) >= {"zone1", "zone2", "cost"}


class TestLoadZoneProject:
    def test_loads_full_project(self):
        zp = load_zone_project(DATA_DIR)
        assert isinstance(zp, ZonalProblem)
        assert zp.n_planning_units == 4
        assert zp.n_features == 2
        assert zp.n_zones == 2

    def test_validates_clean(self):
        zp = load_zone_project(DATA_DIR)
        assert zp.validate() == []

    def test_zone_costs_loaded(self):
        zp = load_zone_project(DATA_DIR)
        assert zp.get_zone_cost(1, 1) == 100.0
        assert zp.get_zone_cost(1, 2) == 50.0

    def test_contributions_loaded(self):
        zp = load_zone_project(DATA_DIR)
        assert zp.get_contribution(1, 1) == 1.0
        assert zp.get_contribution(1, 2) == 0.5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pymarxan/zones/test_readers.py -v`
Expected: FAIL

**Step 3: Implement zone readers**

`src/pymarxan/zones/readers.py`:
```python
"""File readers for MarZone multi-zone projects."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pymarxan.io.readers import _read_dat, load_project
from pymarxan.zones.model import ZonalProblem


def read_zones(path: str | Path) -> pd.DataFrame:
    """Read a zones definition file (zones.dat)."""
    df = _read_dat(path)
    df["id"] = df["id"].astype(int)
    return df


def read_zone_costs(path: str | Path) -> pd.DataFrame:
    """Read per-PU per-zone costs (zonecost.dat)."""
    df = _read_dat(path)
    df["pu"] = df["pu"].astype(int)
    df["zone"] = df["zone"].astype(int)
    df["cost"] = df["cost"].astype(float)
    return df


def read_zone_contributions(path: str | Path) -> pd.DataFrame:
    """Read zone contribution weights (zonecontrib.dat)."""
    df = _read_dat(path)
    df["feature"] = df["feature"].astype(int)
    df["zone"] = df["zone"].astype(int)
    df["contribution"] = df["contribution"].astype(float)
    return df


def read_zone_targets(path: str | Path) -> pd.DataFrame:
    """Read per-zone per-feature targets (zonetarget.dat)."""
    df = _read_dat(path)
    df["zone"] = df["zone"].astype(int)
    df["feature"] = df["feature"].astype(int)
    df["target"] = df["target"].astype(float)
    return df


def read_zone_boundary_costs(path: str | Path) -> pd.DataFrame:
    """Read zone-to-zone boundary costs (zoneboundcost.dat)."""
    df = _read_dat(path)
    df["zone1"] = df["zone1"].astype(int)
    df["zone2"] = df["zone2"].astype(int)
    df["cost"] = df["cost"].astype(float)
    return df


def load_zone_project(project_dir: str | Path) -> ZonalProblem:
    """Load a MarZone project directory into a ZonalProblem.

    Reads standard Marxan files via load_project(), then loads zone-specific
    files if referenced in input.dat.
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
```

**Step 4: Run tests**

Run: `pytest tests/pymarxan/zones/test_readers.py -v`
Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add src/pymarxan/zones/ tests/pymarxan/zones/
git commit -m "feat: add zone file readers and load_zone_project"
```

---

## Task 4: Zone Objective Functions

**Files:**
- Create: `src/pymarxan/zones/objective.py`
- Test: `tests/pymarxan/zones/test_objective.py`

Zone assignment is an integer array where `zone_assignment[i]` is the zone ID for PU `i`. Zone 0 means "not selected" (available zone).

**Step 1: Write the failing tests**

`tests/pymarxan/zones/test_objective.py`:
```python
from pathlib import Path

import numpy as np

from pymarxan.zones.model import ZonalProblem
from pymarxan.zones.readers import load_zone_project
from pymarxan.zones.objective import (
    compute_zone_cost,
    compute_zone_boundary,
    check_zone_targets,
    compute_zone_objective,
)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "zones"


class TestComputeZoneCost:
    def setup_method(self):
        self.problem = load_zone_project(DATA_DIR)
        self.pu_ids = self.problem.planning_units["id"].tolist()

    def test_all_protected(self):
        # All PUs in zone 1 (protected): costs = 100+150+200+120 = 570
        assignment = np.array([1, 1, 1, 1])
        cost = compute_zone_cost(self.problem, assignment)
        assert cost == 570.0

    def test_all_sustainable(self):
        # All PUs in zone 2: costs = 50+80+100+60 = 290
        assignment = np.array([2, 2, 2, 2])
        cost = compute_zone_cost(self.problem, assignment)
        assert cost == 290.0

    def test_mixed(self):
        # PU1=protected(100), PU2=sustainable(80), PU3=protected(200), PU4=sustainable(60)
        assignment = np.array([1, 2, 1, 2])
        cost = compute_zone_cost(self.problem, assignment)
        assert cost == 440.0

    def test_unassigned_is_zero(self):
        # Zone 0 means not assigned — zero cost
        assignment = np.array([0, 0, 0, 0])
        cost = compute_zone_cost(self.problem, assignment)
        assert cost == 0.0


class TestComputeZoneBoundary:
    def setup_method(self):
        self.problem = load_zone_project(DATA_DIR)

    def test_same_zones_no_penalty(self):
        # All in same zone — zone boundary cost between same zones is 0
        assignment = np.array([1, 1, 1, 1])
        zbc = compute_zone_boundary(self.problem, assignment)
        assert zbc == 0.0

    def test_different_zones_penalty(self):
        # PU1=1, PU2=2 are adjacent — cost = 50.0
        # PU2=2, PU3=1 are adjacent — cost = 50.0
        # PU3=1, PU4=2 are adjacent — cost = 50.0
        assignment = np.array([1, 2, 1, 2])
        zbc = compute_zone_boundary(self.problem, assignment)
        assert zbc == 150.0


class TestCheckZoneTargets:
    def setup_method(self):
        self.problem = load_zone_project(DATA_DIR)

    def test_all_protected_targets(self):
        # All in zone 1 (contribution=1.0 for both features)
        # Feature 1: sum=10+8+6+5=29, zone 1 target=10 -> met
        # Feature 2: sum=5+7+9+4=25, zone 1 target=8 -> met
        assignment = np.array([1, 1, 1, 1])
        targets = check_zone_targets(self.problem, assignment)
        assert targets[(1, 1)] is True
        assert targets[(1, 2)] is True

    def test_none_assigned(self):
        assignment = np.array([0, 0, 0, 0])
        targets = check_zone_targets(self.problem, assignment)
        # All targets should be unmet
        assert not any(targets.values())


class TestComputeZoneObjective:
    def setup_method(self):
        self.problem = load_zone_project(DATA_DIR)

    def test_objective_components(self):
        assignment = np.array([1, 1, 1, 1])
        obj = compute_zone_objective(self.problem, assignment, blm=0.0)
        # Cost=570, zone boundary=0 (all same zone), blm*boundary=0
        assert obj >= 570.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pymarxan/zones/test_objective.py -v`
Expected: FAIL

**Step 3: Implement zone objective functions**

`src/pymarxan/zones/objective.py`:
```python
"""Objective function components for multi-zone conservation planning."""
from __future__ import annotations

import numpy as np

from pymarxan.zones.model import ZonalProblem


def compute_zone_cost(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
) -> float:
    """Compute total cost given zone assignments.

    Zone 0 = unassigned (no cost).
    """
    pu_ids = problem.planning_units["id"].tolist()
    total = 0.0
    for i, pid in enumerate(pu_ids):
        zid = int(zone_assignment[i])
        if zid == 0:
            continue
        total += problem.get_zone_cost(pid, zid)
    return total


def compute_zone_boundary(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
) -> float:
    """Compute zone boundary cost between adjacent PUs in different zones.

    For each pair of adjacent PUs (from boundary data, off-diagonal entries),
    if they are in different zones, add the zone boundary cost for that
    zone pair.
    """
    if problem.boundary is None or problem.zone_boundary_costs is None:
        return 0.0

    pu_ids = problem.planning_units["id"].tolist()
    pu_index = {pid: i for i, pid in enumerate(pu_ids)}

    # Build zone boundary cost lookup
    zbc_lookup: dict[tuple[int, int], float] = {}
    for _, row in problem.zone_boundary_costs.iterrows():
        z1 = int(row["zone1"])
        z2 = int(row["zone2"])
        cost = float(row["cost"])
        zbc_lookup[(z1, z2)] = cost

    total = 0.0
    for _, row in problem.boundary.iterrows():
        id1 = int(row["id1"])
        id2 = int(row["id2"])
        if id1 == id2:
            continue  # Skip diagonal entries

        idx1 = pu_index.get(id1)
        idx2 = pu_index.get(id2)
        if idx1 is None or idx2 is None:
            continue

        z1 = int(zone_assignment[idx1])
        z2 = int(zone_assignment[idx2])
        if z1 == 0 or z2 == 0:
            continue
        if z1 == z2:
            continue  # Same zone — no zone boundary penalty

        cost = zbc_lookup.get((z1, z2), 0.0)
        total += cost

    return total


def compute_standard_boundary(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
) -> float:
    """Compute standard (PU-level) boundary for selected PUs.

    Any PU with zone > 0 is considered "selected". Standard boundary logic:
    diagonal = external boundary when selected, off-diagonal = shared boundary
    when exactly one is selected.
    """
    if problem.boundary is None:
        return 0.0

    pu_ids = problem.planning_units["id"].tolist()
    pu_index = {pid: i for i, pid in enumerate(pu_ids)}
    selected = zone_assignment > 0

    total = 0.0
    for _, row in problem.boundary.iterrows():
        id1 = int(row["id1"])
        id2 = int(row["id2"])
        bval = float(row["boundary"])

        if id1 == id2:
            idx = pu_index.get(id1)
            if idx is not None and selected[idx]:
                total += bval
        else:
            idx1 = pu_index.get(id1)
            idx2 = pu_index.get(id2)
            if idx1 is not None and idx2 is not None:
                if selected[idx1] != selected[idx2]:
                    total += bval
    return total


def check_zone_targets(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
) -> dict[tuple[int, int], bool]:
    """Check which zone-specific targets are met.

    Returns dict of (zone_id, feature_id) -> bool.
    For each zone-target pair, sums the contributed amount
    (amount * contribution) of PUs assigned to that zone.
    """
    if problem.zone_targets is None:
        return {}

    pu_ids = problem.planning_units["id"].tolist()
    pu_index = {pid: i for i, pid in enumerate(pu_ids)}

    targets_met: dict[tuple[int, int], bool] = {}
    for _, trow in problem.zone_targets.iterrows():
        zid = int(trow["zone"])
        fid = int(trow["feature"])
        target = float(trow["target"])

        contribution = problem.get_contribution(fid, zid)
        feat_data = problem.pu_vs_features[
            problem.pu_vs_features["species"] == fid
        ]

        achieved = 0.0
        for _, r in feat_data.iterrows():
            pid = int(r["pu"])
            idx = pu_index.get(pid)
            if idx is not None and int(zone_assignment[idx]) == zid:
                achieved += float(r["amount"]) * contribution

        targets_met[(zid, fid)] = achieved >= target

    return targets_met


def compute_zone_penalty(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
) -> float:
    """Compute penalty for unmet zone targets (SPF * shortfall)."""
    if problem.zone_targets is None:
        return 0.0

    pu_ids = problem.planning_units["id"].tolist()
    pu_index = {pid: i for i, pid in enumerate(pu_ids)}

    # Build SPF lookup
    spf_lookup: dict[int, float] = {}
    for _, frow in problem.features.iterrows():
        spf_lookup[int(frow["id"])] = float(frow.get("spf", 1.0))

    total = 0.0
    for _, trow in problem.zone_targets.iterrows():
        zid = int(trow["zone"])
        fid = int(trow["feature"])
        target = float(trow["target"])

        contribution = problem.get_contribution(fid, zid)
        feat_data = problem.pu_vs_features[
            problem.pu_vs_features["species"] == fid
        ]

        achieved = 0.0
        for _, r in feat_data.iterrows():
            pid = int(r["pu"])
            idx = pu_index.get(pid)
            if idx is not None and int(zone_assignment[idx]) == zid:
                achieved += float(r["amount"]) * contribution

        shortfall = max(0.0, target - achieved)
        total += spf_lookup.get(fid, 1.0) * shortfall

    return total


def compute_zone_objective(
    problem: ZonalProblem,
    zone_assignment: np.ndarray,
    blm: float,
) -> float:
    """Compute the full MarZone objective.

    Objective = zone_cost + BLM * standard_boundary + zone_boundary + penalty
    """
    cost = compute_zone_cost(problem, zone_assignment)
    std_boundary = compute_standard_boundary(problem, zone_assignment)
    zone_boundary = compute_zone_boundary(problem, zone_assignment)
    penalty = compute_zone_penalty(problem, zone_assignment)
    return cost + blm * std_boundary + zone_boundary + penalty
```

**Step 4: Run tests**

Run: `pytest tests/pymarxan/zones/test_objective.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/pymarxan/zones/ tests/pymarxan/zones/
git commit -m "feat: add zone objective functions (cost, boundary, targets, penalty)"
```

---

## Task 5: Zone SA Solver

**Files:**
- Create: `src/pymarxan/zones/solver.py`
- Test: `tests/pymarxan/zones/test_solver.py`
- Modify: `src/pymarxan/solvers/base.py` — add `zone_assignment` to `Solution`

**Step 1: Extend Solution with zone_assignment**

Add to `src/pymarxan/solvers/base.py` `Solution` dataclass:
```python
zone_assignment: np.ndarray | None = None  # Zone ID per PU (None for standard problems)
```

**Step 2: Write the failing tests**

`tests/pymarxan/zones/test_solver.py`:
```python
from pathlib import Path

import numpy as np

from pymarxan.solvers.base import SolverConfig
from pymarxan.zones.readers import load_zone_project
from pymarxan.zones.solver import ZoneSASolver

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "zones"


class TestZoneSASolver:
    def setup_method(self):
        self.problem = load_zone_project(DATA_DIR)
        self.problem.parameters["NUMITNS"] = 5_000
        self.problem.parameters["NUMTEMP"] = 50
        self.solver = ZoneSASolver()

    def test_solver_name(self):
        assert self.solver.name() == "Zone SA (Python)"

    def test_supports_zones(self):
        assert self.solver.supports_zones()

    def test_solve_returns_solutions(self):
        config = SolverConfig(num_solutions=3, seed=42)
        solutions = self.solver.solve(self.problem, config)
        assert len(solutions) == 3

    def test_zone_assignment_present(self):
        config = SolverConfig(num_solutions=1, seed=42)
        sol = self.solver.solve(self.problem, config)[0]
        assert sol.zone_assignment is not None
        assert len(sol.zone_assignment) == 4

    def test_zone_values_valid(self):
        config = SolverConfig(num_solutions=1, seed=42)
        sol = self.solver.solve(self.problem, config)[0]
        valid_zones = {0, 1, 2}
        for z in sol.zone_assignment:
            assert int(z) in valid_zones

    def test_cost_nonnegative(self):
        config = SolverConfig(num_solutions=3, seed=42)
        for sol in self.solver.solve(self.problem, config):
            assert sol.cost >= 0
            assert sol.objective >= 0

    def test_seed_reproducibility(self):
        config = SolverConfig(num_solutions=1, seed=99)
        sol1 = self.solver.solve(self.problem, config)[0]
        sol2 = self.solver.solve(self.problem, config)[0]
        np.testing.assert_array_equal(sol1.zone_assignment, sol2.zone_assignment)

    def test_finds_feasible_on_simple_problem(self):
        import copy
        problem = copy.deepcopy(self.problem)
        problem.parameters["BLM"] = 0.0
        problem.features["spf"] = 100.0
        problem.parameters["NUMITNS"] = 10_000
        problem.parameters["NUMTEMP"] = 100
        config = SolverConfig(num_solutions=5, seed=42)
        solutions = self.solver.solve(problem, config)
        # Check if at least one solution has some PUs assigned
        assigned = sum(1 for s in solutions if s.zone_assignment.any())
        assert assigned > 0
```

**Step 3: Implement Zone SA solver**

`src/pymarxan/zones/solver.py`:
```python
"""Simulated annealing solver for multi-zone conservation planning."""
from __future__ import annotations

import math

import numpy as np

from pymarxan.solvers.base import Solution, Solver, SolverConfig
from pymarxan.zones.model import ZonalProblem
from pymarxan.zones.objective import (
    compute_zone_cost,
    compute_zone_objective,
    check_zone_targets,
    compute_standard_boundary,
    compute_zone_boundary,
)


class ZoneSASolver(Solver):
    """Simulated annealing solver for zonal conservation problems."""

    def __init__(
        self,
        num_iterations: int = 1_000_000,
        num_temp_steps: int = 10_000,
    ):
        self._num_iterations = num_iterations
        self._num_temp_steps = num_temp_steps

    def name(self) -> str:
        return "Zone SA (Python)"

    def supports_zones(self) -> bool:
        return True

    def solve(
        self, problem: ZonalProblem, config: SolverConfig | None = None
    ) -> list[Solution]:
        if config is None:
            config = SolverConfig()

        blm = float(problem.parameters.get("BLM", 0.0))
        num_iterations = int(
            problem.parameters.get("NUMITNS", self._num_iterations)
        )
        num_temp_steps = int(
            problem.parameters.get("NUMTEMP", self._num_temp_steps)
        )

        pu_ids = problem.planning_units["id"].tolist()
        n_pu = len(pu_ids)
        zone_ids_list = sorted(problem.zone_ids)
        # Options: 0 (unassigned) + each zone
        zone_options = [0] + zone_ids_list

        # Identify locked PUs
        locked: dict[int, int] = {}
        if "status" in problem.planning_units.columns:
            for _, row in problem.planning_units.iterrows():
                s = int(row["status"])
                idx = pu_ids.index(int(row["id"]))
                if s == 2:
                    locked[idx] = zone_ids_list[0]  # Lock into first zone
                elif s == 3:
                    locked[idx] = 0  # Lock out

        swappable = [i for i in range(n_pu) if i not in locked]

        solutions = []
        for run_idx in range(config.num_solutions):
            if config.seed is not None:
                rng = np.random.default_rng(config.seed + run_idx)
            else:
                rng = np.random.default_rng()

            # Initialize: random zone assignment
            assignment = np.zeros(n_pu, dtype=int)
            for idx, zid in locked.items():
                assignment[idx] = zid
            for idx in swappable:
                assignment[idx] = zone_options[
                    rng.integers(len(zone_options))
                ]

            current_obj = compute_zone_objective(problem, assignment, blm)

            # Estimate initial temperature
            deltas = []
            for _ in range(min(1000, num_iterations // 10)):
                idx = swappable[rng.integers(len(swappable))]
                old_zone = assignment[idx]
                new_zone = zone_options[rng.integers(len(zone_options))]
                if new_zone == old_zone:
                    continue
                assignment[idx] = new_zone
                new_obj = compute_zone_objective(problem, assignment, blm)
                delta = new_obj - current_obj
                if delta > 0:
                    deltas.append(delta)
                assignment[idx] = old_zone  # revert

            if deltas:
                avg_delta = sum(deltas) / len(deltas)
                initial_temp = -avg_delta / math.log(0.5)
            else:
                initial_temp = 1.0

            iters_per_step = max(1, num_iterations // num_temp_steps)
            if initial_temp > 0:
                alpha = (0.001 / initial_temp) ** (
                    1.0 / max(1, num_temp_steps)
                )
            else:
                alpha = 0.99

            temp = initial_temp
            best_assignment = assignment.copy()
            best_obj = current_obj
            step_count = 0

            for _ in range(num_iterations):
                idx = swappable[rng.integers(len(swappable))]
                old_zone = assignment[idx]
                new_zone = zone_options[rng.integers(len(zone_options))]
                if new_zone == old_zone:
                    continue

                assignment[idx] = new_zone
                new_obj = compute_zone_objective(problem, assignment, blm)
                delta = new_obj - current_obj

                if delta <= 0:
                    current_obj = new_obj
                elif temp > 0 and rng.random() < math.exp(-delta / temp):
                    current_obj = new_obj
                else:
                    assignment[idx] = old_zone

                if current_obj < best_obj:
                    best_assignment = assignment.copy()
                    best_obj = current_obj

                step_count += 1
                if step_count >= iters_per_step:
                    temp *= alpha
                    step_count = 0

            # Build solution
            selected = best_assignment > 0
            cost = compute_zone_cost(problem, best_assignment)
            std_boundary = compute_standard_boundary(
                problem, best_assignment
            )
            zone_boundary = compute_zone_boundary(
                problem, best_assignment
            )
            zone_targets = check_zone_targets(problem, best_assignment)

            sol = Solution(
                selected=selected,
                cost=cost,
                boundary=std_boundary,
                objective=best_obj,
                targets_met={},
                zone_assignment=best_assignment.copy(),
                metadata={
                    "solver": self.name(),
                    "run": run_idx + 1,
                    "zone_boundary_cost": round(zone_boundary, 4),
                    "zone_targets_met": {
                        f"z{z}_f{f}": v
                        for (z, f), v in zone_targets.items()
                    },
                },
            )
            solutions.append(sol)

        return solutions
```

**Step 4: Run tests**

Run: `pytest tests/pymarxan/zones/ -v`
Expected: All zone tests PASS

Run: `pytest tests/ -v` (ensure existing tests still pass with Solution change)
Expected: ALL tests pass

**Step 5: Commit**

```bash
git add src/pymarxan/ tests/pymarxan/zones/
git commit -m "feat: add Zone SA solver for multi-zone conservation planning"
```

---

## Task 6: Connectivity Metrics

**Files:**
- Create: `src/pymarxan/connectivity/__init__.py`
- Create: `src/pymarxan/connectivity/metrics.py`
- Test: `tests/pymarxan/connectivity/__init__.py`
- Test: `tests/pymarxan/connectivity/test_metrics.py`

**Step 1: Write the failing tests**

`tests/pymarxan/connectivity/__init__.py`: empty

`tests/pymarxan/connectivity/test_metrics.py`:
```python
import numpy as np

from pymarxan.connectivity.metrics import (
    compute_in_degree,
    compute_out_degree,
    compute_betweenness_centrality,
    compute_eigenvector_centrality,
)


def _make_matrix():
    """4-node connectivity matrix (asymmetric)."""
    return np.array([
        [0.0, 0.5, 0.0, 0.0],
        [0.1, 0.0, 0.8, 0.0],
        [0.0, 0.2, 0.0, 0.6],
        [0.0, 0.0, 0.3, 0.0],
    ])


class TestInDegree:
    def test_shape(self):
        m = _make_matrix()
        result = compute_in_degree(m)
        assert len(result) == 4

    def test_values(self):
        m = _make_matrix()
        result = compute_in_degree(m)
        # Column sums (excluding diagonal)
        # Node 0: receives from node 1 (0.1) = 0.1
        # Node 1: receives from node 0 (0.5) + node 2 (0.2) = 0.7
        # Node 2: receives from node 1 (0.8) + node 3 (0.3) = 1.1
        # Node 3: receives from node 2 (0.6) = 0.6
        np.testing.assert_array_almost_equal(result, [0.1, 0.7, 1.1, 0.6])


class TestOutDegree:
    def test_values(self):
        m = _make_matrix()
        result = compute_out_degree(m)
        # Row sums (excluding diagonal)
        # Node 0: sends to node 1 (0.5) = 0.5
        # Node 1: sends to node 0 (0.1) + node 2 (0.8) = 0.9
        # Node 2: sends to node 1 (0.2) + node 3 (0.6) = 0.8
        # Node 3: sends to node 2 (0.3) = 0.3
        np.testing.assert_array_almost_equal(result, [0.5, 0.9, 0.8, 0.3])


class TestBetweennessCentrality:
    def test_shape(self):
        m = _make_matrix()
        result = compute_betweenness_centrality(m)
        assert len(result) == 4

    def test_values_in_range(self):
        m = _make_matrix()
        result = compute_betweenness_centrality(m)
        assert all(0.0 <= v <= 1.0 for v in result)

    def test_middle_nodes_higher(self):
        """Middle nodes in a chain should have higher betweenness."""
        m = _make_matrix()
        result = compute_betweenness_centrality(m)
        # Nodes 1 and 2 are in the middle of the chain
        assert result[1] > result[0] or result[2] > result[3]


class TestEigenvectorCentrality:
    def test_shape(self):
        m = _make_matrix()
        result = compute_eigenvector_centrality(m)
        assert len(result) == 4

    def test_values_nonnegative(self):
        m = _make_matrix()
        result = compute_eigenvector_centrality(m)
        assert all(v >= 0 for v in result)

    def test_disconnected_zero(self):
        """A completely disconnected node should have low centrality."""
        m = np.zeros((4, 4))
        m[0, 1] = 1.0
        m[1, 0] = 1.0
        # Nodes 2, 3 are disconnected
        result = compute_eigenvector_centrality(m)
        assert result[2] == 0.0
        assert result[3] == 0.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pymarxan/connectivity/test_metrics.py -v`
Expected: FAIL

**Step 3: Implement connectivity metrics**

`src/pymarxan/connectivity/__init__.py`: empty

`src/pymarxan/connectivity/metrics.py`:
```python
"""Connectivity metrics for conservation planning.

Computes graph-theoretic metrics from a connectivity (adjacency) matrix.
Compatible with Marxan Connect workflow.
"""
from __future__ import annotations

import numpy as np


def compute_in_degree(matrix: np.ndarray) -> np.ndarray:
    """Compute in-degree (incoming flow) for each node.

    Sum of each column (excluding diagonal).
    """
    m = matrix.copy()
    np.fill_diagonal(m, 0)
    return m.sum(axis=0)


def compute_out_degree(matrix: np.ndarray) -> np.ndarray:
    """Compute out-degree (outgoing flow) for each node.

    Sum of each row (excluding diagonal).
    """
    m = matrix.copy()
    np.fill_diagonal(m, 0)
    return m.sum(axis=1)


def compute_betweenness_centrality(matrix: np.ndarray) -> np.ndarray:
    """Compute betweenness centrality using networkx.

    Returns normalized betweenness centrality [0, 1] for each node.
    """
    import networkx as nx

    n = matrix.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j and matrix[i, j] > 0:
                G.add_edge(i, j, weight=matrix[i, j])

    bc = nx.betweenness_centrality(G, weight="weight", normalized=True)
    return np.array([bc.get(i, 0.0) for i in range(n)])


def compute_eigenvector_centrality(matrix: np.ndarray) -> np.ndarray:
    """Compute eigenvector centrality using networkx.

    Falls back to zeros if the graph has no edges or convergence fails.
    """
    import networkx as nx

    n = matrix.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j and matrix[i, j] > 0:
                G.add_edge(i, j, weight=matrix[i, j])

    if G.number_of_edges() == 0:
        return np.zeros(n)

    try:
        ec = nx.eigenvector_centrality_numpy(G, weight="weight")
        return np.array([ec.get(i, 0.0) for i in range(n)])
    except (nx.NetworkXError, np.linalg.LinAlgError):
        return np.zeros(n)
```

**Step 4: Run tests**

Run: `pytest tests/pymarxan/connectivity/test_metrics.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/pymarxan/connectivity/ tests/pymarxan/connectivity/
git commit -m "feat: add connectivity metrics (in/out-degree, betweenness, eigenvector)"
```

---

## Task 7: Connectivity I/O and Feature Conversion

**Files:**
- Create: `src/pymarxan/connectivity/io.py`
- Create: `src/pymarxan/connectivity/features.py`
- Test: `tests/pymarxan/connectivity/test_io.py`
- Test: `tests/pymarxan/connectivity/test_features.py`

**Step 1: Write the failing tests**

`tests/pymarxan/connectivity/test_io.py`:
```python
from pathlib import Path

import numpy as np
import pandas as pd

from pymarxan.connectivity.io import (
    read_connectivity_edgelist,
    read_connectivity_matrix,
    connectivity_to_matrix,
)


class TestReadEdgelist:
    def test_reads_edgelist(self, tmp_path):
        p = tmp_path / "edges.csv"
        p.write_text("id1,id2,value\n1,2,0.5\n2,3,0.8\n3,1,0.3\n")
        pu_ids = [1, 2, 3]
        m = read_connectivity_edgelist(p, pu_ids)
        assert m.shape == (3, 3)
        assert m[0, 1] == 0.5
        assert m[1, 2] == 0.8
        assert m[2, 0] == 0.3

    def test_missing_edges_are_zero(self, tmp_path):
        p = tmp_path / "edges.csv"
        p.write_text("id1,id2,value\n1,2,0.5\n")
        pu_ids = [1, 2, 3]
        m = read_connectivity_edgelist(p, pu_ids)
        assert m[0, 2] == 0.0
        assert m[2, 1] == 0.0


class TestReadMatrix:
    def test_reads_csv_matrix(self, tmp_path):
        p = tmp_path / "matrix.csv"
        p.write_text(",1,2,3\n1,0,0.5,0\n2,0.1,0,0.8\n3,0,0.3,0\n")
        m = read_connectivity_matrix(p)
        assert m.shape == (3, 3)
        assert m[0, 1] == 0.5
        assert m[1, 2] == 0.8


class TestConnectivityToMatrix:
    def test_edgelist_df(self):
        df = pd.DataFrame({
            "id1": [1, 2], "id2": [2, 3], "value": [0.5, 0.8],
        })
        m = connectivity_to_matrix(df, pu_ids=[1, 2, 3])
        assert m.shape == (3, 3)
        assert m[0, 1] == 0.5
```

`tests/pymarxan/connectivity/test_features.py`:
```python
from pathlib import Path

import numpy as np
import pandas as pd

from pymarxan.io.readers import load_project
from pymarxan.connectivity.features import (
    metric_to_feature,
    add_connectivity_features,
)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


class TestMetricToFeature:
    def test_creates_puvspr_rows(self):
        pu_ids = [1, 2, 3, 4, 5, 6]
        metric_values = np.array([0.8, 0.2, 0.6, 0.9, 0.1, 0.5])
        feature_id = 99
        rows = metric_to_feature(pu_ids, metric_values, feature_id)
        assert isinstance(rows, pd.DataFrame)
        assert len(rows) == 6
        assert set(rows.columns) == {"species", "pu", "amount"}
        assert rows["species"].iloc[0] == 99

    def test_threshold_filters(self):
        pu_ids = [1, 2, 3]
        metric_values = np.array([0.8, 0.2, 0.6])
        rows = metric_to_feature(pu_ids, metric_values, 99, threshold=0.5)
        # Only PU 1 (0.8) and PU 3 (0.6) pass threshold
        assert len(rows) == 2


class TestAddConnectivityFeatures:
    def test_adds_feature(self):
        problem = load_project(DATA_DIR)
        pu_ids = problem.planning_units["id"].tolist()
        metric = np.array([0.5, 0.3, 0.7, 0.2, 0.8, 0.4])
        new_problem = add_connectivity_features(
            problem,
            metrics={"connectivity": metric},
            targets={"connectivity": 2.0},
        )
        # Should have one more feature
        assert new_problem.n_features == problem.n_features + 1
        # puvspr should have more rows
        assert len(new_problem.pu_vs_features) > len(problem.pu_vs_features)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/pymarxan/connectivity/test_io.py tests/pymarxan/connectivity/test_features.py -v`
Expected: FAIL

**Step 3: Implement connectivity I/O and feature conversion**

`src/pymarxan/connectivity/io.py`:
```python
"""I/O for connectivity matrices (edge lists and full matrices)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_connectivity_edgelist(
    path: str | Path,
    pu_ids: list[int],
) -> np.ndarray:
    """Read an edge list CSV and convert to NxN matrix.

    Expected columns: id1, id2, value.
    """
    df = pd.read_csv(path)
    n = len(pu_ids)
    id_to_idx = {pid: i for i, pid in enumerate(pu_ids)}
    matrix = np.zeros((n, n))
    for _, row in df.iterrows():
        i = id_to_idx.get(int(row["id1"]))
        j = id_to_idx.get(int(row["id2"]))
        if i is not None and j is not None:
            matrix[i, j] = float(row["value"])
    return matrix


def read_connectivity_matrix(path: str | Path) -> np.ndarray:
    """Read a full connectivity matrix from CSV (first column/row = IDs)."""
    df = pd.read_csv(path, index_col=0)
    return df.values.astype(float)


def connectivity_to_matrix(
    edgelist: pd.DataFrame,
    pu_ids: list[int],
) -> np.ndarray:
    """Convert an edge list DataFrame to NxN matrix."""
    n = len(pu_ids)
    id_to_idx = {pid: i for i, pid in enumerate(pu_ids)}
    matrix = np.zeros((n, n))
    for _, row in edgelist.iterrows():
        i = id_to_idx.get(int(row["id1"]))
        j = id_to_idx.get(int(row["id2"]))
        if i is not None and j is not None:
            matrix[i, j] = float(row["value"])
    return matrix
```

`src/pymarxan/connectivity/features.py`:
```python
"""Convert connectivity metrics into Marxan features."""
from __future__ import annotations

import pandas as pd
import numpy as np

from pymarxan.models.problem import ConservationProblem


def metric_to_feature(
    pu_ids: list[int],
    metric_values: np.ndarray,
    feature_id: int,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Convert a per-PU metric array into puvspr rows.

    Parameters
    ----------
    pu_ids : list[int]
        Planning unit IDs aligned with metric_values.
    metric_values : np.ndarray
        Metric value per PU.
    feature_id : int
        Feature ID to assign to the connectivity metric.
    threshold : float | None
        If set, only include PUs with metric >= threshold.

    Returns
    -------
    pd.DataFrame
        Rows with columns species, pu, amount.
    """
    rows = []
    for i, pid in enumerate(pu_ids):
        val = float(metric_values[i])
        if threshold is not None and val < threshold:
            continue
        rows.append({"species": feature_id, "pu": pid, "amount": val})
    return pd.DataFrame(rows)


def add_connectivity_features(
    problem: ConservationProblem,
    metrics: dict[str, np.ndarray],
    targets: dict[str, float],
    start_feature_id: int | None = None,
    threshold: float | None = None,
) -> ConservationProblem:
    """Add connectivity metrics as synthetic features to a problem.

    Creates a new ConservationProblem with additional features and
    puvspr rows for each connectivity metric.

    Parameters
    ----------
    problem : ConservationProblem
        The base problem.
    metrics : dict[str, np.ndarray]
        Mapping from metric name to per-PU values.
    targets : dict[str, float]
        Target value for each metric feature.
    start_feature_id : int | None
        First feature ID for connectivity features. Defaults to
        max(existing feature IDs) + 100.
    threshold : float | None
        Only include PUs above this threshold.

    Returns
    -------
    ConservationProblem
        New problem with connectivity features added.
    """
    pu_ids = problem.planning_units["id"].tolist()
    existing_max = int(problem.features["id"].max())
    if start_feature_id is None:
        start_feature_id = existing_max + 100

    new_features = problem.features.copy()
    new_puvspr = problem.pu_vs_features.copy()

    fid = start_feature_id
    for name, values in metrics.items():
        target = targets.get(name, 0.0)
        feat_row = pd.DataFrame({
            "id": [fid],
            "name": [f"conn_{name}"],
            "target": [target],
            "spf": [1.0],
        })
        new_features = pd.concat(
            [new_features, feat_row], ignore_index=True,
        )
        puvspr_rows = metric_to_feature(
            pu_ids, values, fid, threshold=threshold,
        )
        new_puvspr = pd.concat(
            [new_puvspr, puvspr_rows], ignore_index=True,
        )
        fid += 1

    return ConservationProblem(
        planning_units=problem.planning_units,
        features=new_features,
        pu_vs_features=new_puvspr,
        boundary=problem.boundary,
        parameters=problem.parameters,
    )
```

**Step 4: Run tests**

Run: `pytest tests/pymarxan/connectivity/ -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/pymarxan/connectivity/ tests/pymarxan/connectivity/
git commit -m "feat: add connectivity I/O and metric-to-feature conversion"
```

---

## Task 8: Zone Shiny Module

**Files:**
- Create: `src/pymarxan_shiny/modules/zones/__init__.py`
- Create: `src/pymarxan_shiny/modules/zones/zone_config.py`

**Step 1: Create zone configuration Shiny module**

`src/pymarxan_shiny/modules/zones/__init__.py`: empty

`src/pymarxan_shiny/modules/zones/zone_config.py`:
```python
"""Zone configuration Shiny module for multi-zone problems."""
from __future__ import annotations

from shiny import module, reactive, render, ui


@module.ui
def zone_config_ui():
    return ui.card(
        ui.card_header("Zone Configuration"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_action_button(
                    "load_zones", "Load Zone Project",
                    class_="btn-primary w-100",
                ),
                ui.hr(),
                ui.output_text_verbatim("zone_summary"),
                width=300,
            ),
            ui.div(
                ui.output_text_verbatim("zone_details"),
                ui.output_text_verbatim("zone_cost_summary"),
            ),
        ),
    )


@module.server
def zone_config_server(
    input, output, session,
    zone_problem: reactive.Value,
):
    @render.text
    def zone_summary():
        zp = zone_problem()
        if zp is None:
            return "No zone project loaded."
        return (
            f"Zones: {zp.n_zones}\n"
            f"Planning Units: {zp.n_planning_units}\n"
            f"Features: {zp.n_features}"
        )

    @render.text
    def zone_details():
        zp = zone_problem()
        if zp is None:
            return ""
        lines = ["Zone Definitions:"]
        for _, row in zp.zones.iterrows():
            lines.append(f"  Zone {int(row['id'])}: {row['name']}")
        return "\n".join(lines)

    @render.text
    def zone_cost_summary():
        zp = zone_problem()
        if zp is None:
            return ""
        lines = ["Zone Costs (avg per PU):"]
        for _, zrow in zp.zones.iterrows():
            zid = int(zrow["id"])
            zname = zrow["name"]
            costs = zp.zone_costs[zp.zone_costs["zone"] == zid]["cost"]
            avg = costs.mean() if len(costs) > 0 else 0.0
            lines.append(f"  {zname}: {avg:.2f}")
        return "\n".join(lines)
```

**Step 2: Commit**

```bash
git add src/pymarxan_shiny/modules/zones/
git commit -m "feat: add zone configuration Shiny module"
```

---

## Task 9: Update App with Zone and Connectivity Tabs

**Files:**
- Modify: `src/pymarxan_app/app.py`
- Modify: `src/pymarxan_shiny/modules/solver_config/solver_picker.py`

**Step 1: Update solver picker**

Add `"zone_sa": "Zone SA (Python)"` to solver choices. Add conditional panel:
```python
ui.panel_conditional(
    "input.solver_type === 'zone_sa'",
    ui.p("Uses Zone SA solver for multi-zone problems. "
         "Requires zone data to be loaded."),
),
```

Add `"zone_sa"` case in server info text.

**Step 2: Update app.py**

Add imports for zone and connectivity modules:
```python
from pymarxan.zones.readers import load_zone_project
from pymarxan.zones.solver import ZoneSASolver
from pymarxan_shiny.modules.zones.zone_config import zone_config_ui, zone_config_server
```

Add a "Zones" tab to the navbar. Add Zone SA to the `_run_solver` function. Wire the zone_config module.

**Step 3: Run all tests**

Run: `pytest tests/ -v`
Expected: ALL tests pass

**Step 4: Commit**

```bash
git add src/pymarxan_app/ src/pymarxan_shiny/
git commit -m "feat: integrate zones and connectivity into assembled app"
```

---

## Task 10: Integration Tests for Phase 3

**Files:**
- Create: `tests/test_integration_phase3.py`

**Step 1: Write integration tests**

`tests/test_integration_phase3.py`:
```python
"""Integration tests for Phase 3 (zones and connectivity)."""
from pathlib import Path

import numpy as np

from pymarxan.io.readers import load_project
from pymarxan.solvers.base import SolverConfig
from pymarxan.zones.readers import load_zone_project
from pymarxan.zones.solver import ZoneSASolver
from pymarxan.zones.objective import (
    compute_zone_cost,
    compute_zone_objective,
    check_zone_targets,
)
from pymarxan.connectivity.metrics import (
    compute_in_degree,
    compute_betweenness_centrality,
)
from pymarxan.connectivity.features import add_connectivity_features
from pymarxan.solvers.mip_solver import MIPSolver

ZONE_DIR = Path(__file__).parent / "data" / "zones"
SIMPLE_DIR = Path(__file__).parent / "data" / "simple"


class TestZoneIntegration:
    def test_load_and_solve_zones(self):
        problem = load_zone_project(ZONE_DIR)
        problem.parameters["NUMITNS"] = 5_000
        problem.parameters["NUMTEMP"] = 50
        solver = ZoneSASolver()
        config = SolverConfig(num_solutions=3, seed=42)
        solutions = solver.solve(problem, config)
        assert len(solutions) == 3
        for sol in solutions:
            assert sol.zone_assignment is not None
            assert len(sol.zone_assignment) == 4

    def test_zone_objective_components(self):
        problem = load_zone_project(ZONE_DIR)
        assignment = np.array([1, 2, 1, 2])
        cost = compute_zone_cost(problem, assignment)
        assert cost > 0
        obj = compute_zone_objective(problem, assignment, blm=1.0)
        assert obj >= cost


class TestConnectivityIntegration:
    def test_metrics_to_features_with_solver(self):
        problem = load_project(SIMPLE_DIR)
        pu_ids = problem.planning_units["id"].tolist()
        n = len(pu_ids)

        # Create a simple connectivity matrix
        matrix = np.zeros((n, n))
        for i in range(n - 1):
            matrix[i, i + 1] = 0.5
            matrix[i + 1, i] = 0.3

        in_deg = compute_in_degree(matrix)
        bc = compute_betweenness_centrality(matrix)

        enhanced = add_connectivity_features(
            problem,
            metrics={"in_degree": in_deg, "betweenness": bc},
            targets={"in_degree": 0.5, "betweenness": 0.1},
        )

        assert enhanced.n_features == problem.n_features + 2

        solver = MIPSolver()
        config = SolverConfig(num_solutions=1)
        solutions = solver.solve(enhanced, config)
        assert len(solutions) == 1
        assert solutions[0].all_targets_met
```

**Step 2: Run all tests**

Run: `pytest tests/ -v`
Expected: ALL tests pass

**Step 3: Commit**

```bash
git add tests/test_integration_phase3.py
git commit -m "test: add Phase 3 integration tests for zones and connectivity"
```

---

## Task 11: Lint and Final Cleanup

**Step 1: Run ruff**

Run: `ruff check src/ tests/ --fix`
Fix any remaining issues.

**Step 2: Run mypy**

Run: `mypy src/pymarxan/ --ignore-missing-imports`
Fix any type errors.

**Step 3: Full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: ALL tests pass

**Step 4: Commit if needed**

```bash
git add -A
git commit -m "chore: lint and type-check cleanup for Phase 3"
```

---

## Summary of Phase 3 Deliverables

| Component | Files | Tests |
|---|---|---|
| Zone data model | `pymarxan.zones.model` | 8 |
| Zone test fixtures | `tests/data/zones/` | — |
| Zone file readers | `pymarxan.zones.readers` | 10 |
| Zone objective functions | `pymarxan.zones.objective` | ~8 |
| Zone SA solver | `pymarxan.zones.solver` | 8 |
| Connectivity metrics | `pymarxan.connectivity.metrics` | ~9 |
| Connectivity I/O + features | `pymarxan.connectivity.io`, `.features` | ~7 |
| Zone Shiny module | `pymarxan_shiny.modules.zones.zone_config` | Manual |
| App updates (Zones tab) | `pymarxan_app.app` | Manual |
| Integration tests | `tests/test_integration_phase3.py` | ~4 |
| **Total new** | **~20 files** | **~54 new tests** |
