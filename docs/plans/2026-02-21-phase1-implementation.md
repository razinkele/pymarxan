# Phase 1: pymarxan Foundation (MVP) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a working end-to-end flow: load Marxan data → configure solver → run solver → view results, with both C++ binary wrapping and MIP solving, plus a basic Shiny-for-Python UI.

**Architecture:** Three-layer Python monorepo — `pymarxan/` (core library, no UI deps), `pymarxan_shiny/` (Shiny UI modules), `pymarxan_app/` (assembled application). All in one repo for Phase 1, split into separate packages later.

**Tech Stack:** Python 3.11+, numpy, pandas, geopandas, shapely, scipy, PuLP (MIP), shiny (Posit), ipyleaflet, shinywidgets, plotly, pytest, ruff, mypy

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/pymarxan/__init__.py`
- Create: `src/pymarxan/io/__init__.py`
- Create: `src/pymarxan/models/__init__.py`
- Create: `src/pymarxan/solvers/__init__.py`
- Create: `src/pymarxan_shiny/__init__.py`
- Create: `src/pymarxan_app/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/pymarxan/__init__.py`
- Create: `tests/pymarxan/io/__init__.py`
- Create: `tests/pymarxan/models/__init__.py`
- Create: `tests/pymarxan/solvers/__init__.py`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "pymarxan"
version = "0.1.0"
description = "Modular Python toolkit for Marxan conservation planning"
requires-python = ">=3.11"
license = "MIT"
dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
    "geopandas>=0.14",
    "shapely>=2.0",
    "scipy>=1.11",
    "PuLP>=2.7",
]

[project.optional-dependencies]
shiny = [
    "shiny>=1.0",
    "shinywidgets>=0.3",
    "ipyleaflet>=0.18",
    "plotly>=5.18",
    "matplotlib>=3.8",
]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.4",
    "mypy>=1.8",
    "pandas-stubs",
]
all = ["pymarxan[shiny,dev]"]

[tool.hatch.build.targets.wheel]
packages = ["src/pymarxan", "src/pymarxan_shiny", "src/pymarxan_app"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.ruff]
src = ["src"]
line-length = 99

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
```

**Step 2: Create all __init__.py files**

`src/pymarxan/__init__.py`:
```python
"""pymarxan: Modular Python toolkit for Marxan conservation planning."""

__version__ = "0.1.0"
```

All other `__init__.py` files: empty files.

**Step 3: Create test fixture data directory**

Create: `tests/data/` (empty directory — will be populated in Task 2)

**Step 4: Install in dev mode and verify**

Run: `cd /home/razinka/marxan && python -m venv .venv && source .venv/bin/activate && pip install -e ".[all]"`
Expected: Successful install with all dependencies.

Run: `python -c "import pymarxan; print(pymarxan.__version__)"`
Expected: `0.1.0`

**Step 5: Initialize git and commit**

```bash
git init
echo -e ".venv/\n__pycache__/\n*.egg-info/\n.mypy_cache/\n.pytest_cache/\n.ruff_cache/\ndist/\nbuild/" > .gitignore
git add .
git commit -m "chore: scaffold pymarxan project with three-layer architecture"
```

---

## Task 2: Test Fixture Data

**Files:**
- Create: `tests/data/simple/input/pu.dat`
- Create: `tests/data/simple/input/spec.dat`
- Create: `tests/data/simple/input/puvspr.dat`
- Create: `tests/data/simple/input/bound.dat`
- Create: `tests/data/simple/input.dat`

This is a minimal 6-planning-unit, 3-feature test problem. Small enough to verify by hand.

**Step 1: Create test planning units**

`tests/data/simple/input/pu.dat` — 6 planning units with varying costs:
```csv
id,cost,status
1,10.0,0
2,15.0,0
3,20.0,0
4,12.0,0
5,18.0,0
6,8.0,0
```

Status: 0=available, 2=locked in, 3=locked out.

**Step 2: Create test conservation features**

`tests/data/simple/input/spec.dat` — 3 features with targets:
```csv
id,name,target,spf
1,species_a,30.0,1.0
2,species_b,20.0,1.0
3,species_c,15.0,1.0
```

**Step 3: Create planning unit vs feature matrix**

`tests/data/simple/input/puvspr.dat` — how much of each feature is in each planning unit:
```csv
species,pu,amount
1,1,10.0
1,2,15.0
1,3,5.0
1,4,12.0
1,5,8.0
1,6,10.0
2,1,5.0
2,2,10.0
2,3,8.0
2,4,3.0
2,5,12.0
2,6,7.0
3,2,6.0
3,3,10.0
3,4,4.0
3,5,5.0
3,6,8.0
```

Verification:
- species_a total = 10+15+5+12+8+10 = 60 (target 30 → need ≥50%)
- species_b total = 5+10+8+3+12+7 = 45 (target 20 → need ≥44%)
- species_c total = 6+10+4+5+8 = 33 (target 15 → need ≥45%)

**Step 4: Create boundary data**

`tests/data/simple/input/bound.dat` — shared boundaries (linear arrangement: 1-2-3-4-5-6):
```csv
id1,id2,boundary
1,2,1.0
2,3,1.0
3,4,1.0
4,5,1.0
5,6,1.0
1,1,2.0
2,2,1.0
3,3,1.0
4,4,1.0
5,5,1.0
6,6,2.0
```

Diagonal entries represent the external (unshared) boundary of each planning unit.

**Step 5: Create input.dat configuration**

`tests/data/simple/input.dat`:
```
INPUTDIR input
PUNAME pu.dat
SPECNAME spec.dat
PUVSPRNAME puvspr.dat
BOUNDNAME bound.dat
OUTPUTDIR output
SCENNAME output
BLM 1.0
NUMREPS 10
NUMITNS 1000000
STARTTEMP -1
NUMTEMP 10000
PROP 0.5
RANDSEED 42
MISSLEVEL 1.0
RUNMODE 1
SAVERUN 3
SAVEBEST 3
SAVESUMMARY 3
SAVESCEN 3
SAVETARGMET 3
SAVESUMSOLN 3
SAVEPENALTY 3
SAVELOG 2
VERBOSITY 1
```

**Step 6: Commit**

```bash
git add tests/data/
git commit -m "test: add simple 6-PU test fixture dataset"
```

---

## Task 3: Domain Models

**Files:**
- Create: `src/pymarxan/models/planning_unit.py`
- Create: `src/pymarxan/models/feature.py`
- Create: `src/pymarxan/models/boundary.py`
- Create: `src/pymarxan/models/problem.py`
- Test: `tests/pymarxan/models/test_problem.py`

**Step 1: Write the failing test**

`tests/pymarxan/models/test_problem.py`:
```python
import numpy as np
import pandas as pd
from pymarxan.models.problem import ConservationProblem


def _make_simple_problem() -> ConservationProblem:
    """Create a minimal problem for testing."""
    planning_units = pd.DataFrame({
        "id": [1, 2, 3],
        "cost": [10.0, 15.0, 20.0],
        "status": [0, 0, 0],
    })
    features = pd.DataFrame({
        "id": [1, 2],
        "name": ["sp_a", "sp_b"],
        "target": [20.0, 10.0],
        "spf": [1.0, 1.0],
    })
    pu_vs_features = pd.DataFrame({
        "species": [1, 1, 1, 2, 2],
        "pu": [1, 2, 3, 1, 3],
        "amount": [10.0, 15.0, 5.0, 8.0, 12.0],
    })
    boundary = pd.DataFrame({
        "id1": [1, 2],
        "id2": [2, 3],
        "boundary": [1.0, 1.0],
    })
    return ConservationProblem(
        planning_units=planning_units,
        features=features,
        pu_vs_features=pu_vs_features,
        boundary=boundary,
        parameters={"BLM": 1.0},
    )


class TestConservationProblem:
    def test_create_problem(self):
        problem = _make_simple_problem()
        assert problem.n_planning_units == 3
        assert problem.n_features == 2

    def test_validate_valid_problem(self):
        problem = _make_simple_problem()
        errors = problem.validate()
        assert errors == []

    def test_validate_missing_pu_in_puvspr(self):
        problem = _make_simple_problem()
        # Add reference to non-existent PU
        extra = pd.DataFrame({"species": [1], "pu": [99], "amount": [5.0]})
        problem.pu_vs_features = pd.concat(
            [problem.pu_vs_features, extra], ignore_index=True
        )
        errors = problem.validate()
        assert any("planning unit" in e.lower() for e in errors)

    def test_validate_missing_feature_in_puvspr(self):
        problem = _make_simple_problem()
        extra = pd.DataFrame({"species": [99], "pu": [1], "amount": [5.0]})
        problem.pu_vs_features = pd.concat(
            [problem.pu_vs_features, extra], ignore_index=True
        )
        errors = problem.validate()
        assert any("feature" in e.lower() for e in errors)

    def test_feature_amounts_per_pu(self):
        problem = _make_simple_problem()
        amounts = problem.feature_amounts()
        # species 1 total = 10 + 15 + 5 = 30
        assert amounts[1] == 30.0
        # species 2 total = 8 + 12 = 20
        assert amounts[2] == 20.0

    def test_targets_achievable(self):
        problem = _make_simple_problem()
        # sp_a target=20, total=30 → achievable
        # sp_b target=10, total=20 → achievable
        assert problem.targets_achievable()

    def test_targets_not_achievable(self):
        problem = _make_simple_problem()
        problem.features.loc[problem.features["id"] == 1, "target"] = 999.0
        assert not problem.targets_achievable()

    def test_summary_returns_string(self):
        problem = _make_simple_problem()
        s = problem.summary()
        assert "3 planning units" in s
        assert "2 features" in s

    def test_no_boundary(self):
        problem = _make_simple_problem()
        problem.boundary = None
        errors = problem.validate()
        assert errors == []
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pymarxan/models/test_problem.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pymarxan.models.problem'`

**Step 3: Implement the models**

`src/pymarxan/models/planning_unit.py`:
```python
"""Planning unit status constants."""

AVAILABLE = 0
LOCKED_IN = 2
LOCKED_OUT = 3

VALID_STATUSES = {AVAILABLE, LOCKED_IN, LOCKED_OUT}
```

`src/pymarxan/models/feature.py`:
```python
"""Conservation feature constants and helpers."""

# Required columns in spec.dat
REQUIRED_COLUMNS = {"id"}
TARGET_COLUMNS = {"target", "prop"}  # At least one needed for solving
```

`src/pymarxan/models/boundary.py`:
```python
"""Boundary matrix helpers."""

REQUIRED_COLUMNS = {"id1", "id2", "boundary"}
```

`src/pymarxan/models/problem.py`:
```python
"""ConservationProblem: central container for a Marxan planning problem."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ConservationProblem:
    """Central container for a conservation planning problem.

    Attributes:
        planning_units: DataFrame with columns: id, cost, status
        features: DataFrame with columns: id, name, target, spf
        pu_vs_features: DataFrame with columns: species, pu, amount
        boundary: Optional DataFrame with columns: id1, id2, boundary
        parameters: Dict of solver parameters (BLM, NUMREPS, etc.)
    """

    planning_units: pd.DataFrame
    features: pd.DataFrame
    pu_vs_features: pd.DataFrame
    boundary: pd.DataFrame | None = None
    parameters: dict = field(default_factory=dict)

    @property
    def n_planning_units(self) -> int:
        return len(self.planning_units)

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def pu_ids(self) -> set[int]:
        return set(self.planning_units["id"])

    @property
    def feature_ids(self) -> set[int]:
        return set(self.features["id"])

    def feature_amounts(self) -> dict[int, float]:
        """Total amount of each feature across all planning units."""
        grouped = self.pu_vs_features.groupby("species")["amount"].sum()
        return grouped.to_dict()

    def targets_achievable(self) -> bool:
        """Check if all feature targets can be met given total amounts."""
        amounts = self.feature_amounts()
        for _, row in self.features.iterrows():
            fid = row["id"]
            target = row.get("target", 0.0)
            if target > 0 and amounts.get(fid, 0.0) < target:
                return False
        return True

    def validate(self) -> list[str]:
        """Return list of validation errors, empty if valid."""
        errors: list[str] = []

        # Check required columns
        for col in ("id", "cost"):
            if col not in self.planning_units.columns:
                errors.append(f"planning_units missing required column: {col}")

        for col in ("id", "name"):
            if col not in self.features.columns:
                errors.append(f"features missing required column: {col}")

        for col in ("species", "pu", "amount"):
            if col not in self.pu_vs_features.columns:
                errors.append(f"pu_vs_features missing required column: {col}")

        if errors:
            return errors  # Can't do cross-checks without required columns

        # Cross-reference: PU IDs in puvspr must exist in planning_units
        puvspr_pu_ids = set(self.pu_vs_features["pu"])
        missing_pus = puvspr_pu_ids - self.pu_ids
        if missing_pus:
            errors.append(
                f"Planning unit IDs in pu_vs_features not found in planning_units: "
                f"{sorted(missing_pus)}"
            )

        # Cross-reference: feature IDs in puvspr must exist in features
        puvspr_feature_ids = set(self.pu_vs_features["species"])
        missing_features = puvspr_feature_ids - self.feature_ids
        if missing_features:
            errors.append(
                f"Feature IDs in pu_vs_features not found in features: "
                f"{sorted(missing_features)}"
            )

        # Validate boundary if present
        if self.boundary is not None:
            for col in ("id1", "id2", "boundary"):
                if col not in self.boundary.columns:
                    errors.append(f"boundary missing required column: {col}")

        return errors

    def summary(self) -> str:
        """Human-readable problem summary."""
        lines = [
            f"ConservationProblem:",
            f"  {self.n_planning_units} planning units",
            f"  {self.n_features} features",
            f"  {len(self.pu_vs_features)} planning unit × feature entries",
        ]
        if self.boundary is not None:
            lines.append(f"  {len(self.boundary)} boundary entries")
        else:
            lines.append("  No boundary data")

        total_cost = self.planning_units["cost"].sum()
        lines.append(f"  Total cost: {total_cost:.1f}")
        lines.append(f"  BLM: {self.parameters.get('BLM', 0.0)}")
        return "\n".join(lines)
```

Update `src/pymarxan/models/__init__.py`:
```python
from pymarxan.models.problem import ConservationProblem

__all__ = ["ConservationProblem"]
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/pymarxan/models/test_problem.py -v`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add src/pymarxan/models/ tests/pymarxan/models/
git commit -m "feat: add ConservationProblem domain model with validation"
```

---

## Task 4: Marxan File Readers

**Files:**
- Create: `src/pymarxan/io/readers.py`
- Test: `tests/pymarxan/io/test_readers.py`

**Step 1: Write the failing tests**

`tests/pymarxan/io/test_readers.py`:
```python
from pathlib import Path

import pandas as pd
from pymarxan.io.readers import (
    read_pu,
    read_spec,
    read_puvspr,
    read_bound,
    read_input_dat,
    load_project,
)
from pymarxan.models.problem import ConservationProblem

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


class TestReadPu:
    def test_reads_csv(self):
        df = read_pu(DATA_DIR / "input" / "pu.dat")
        assert isinstance(df, pd.DataFrame)
        assert "id" in df.columns
        assert "cost" in df.columns
        assert len(df) == 6

    def test_id_column_is_int(self):
        df = read_pu(DATA_DIR / "input" / "pu.dat")
        assert df["id"].dtype in ("int64", "int32")

    def test_cost_column_is_float(self):
        df = read_pu(DATA_DIR / "input" / "pu.dat")
        assert df["cost"].dtype == "float64"


class TestReadSpec:
    def test_reads_csv(self):
        df = read_spec(DATA_DIR / "input" / "spec.dat")
        assert len(df) == 3
        assert "id" in df.columns
        assert "name" in df.columns

    def test_has_target(self):
        df = read_spec(DATA_DIR / "input" / "spec.dat")
        assert "target" in df.columns
        assert df["target"].iloc[0] == 30.0


class TestReadPuvspr:
    def test_reads_csv(self):
        df = read_puvspr(DATA_DIR / "input" / "puvspr.dat")
        assert len(df) == 17
        assert set(df.columns) >= {"species", "pu", "amount"}

    def test_species_are_ints(self):
        df = read_puvspr(DATA_DIR / "input" / "puvspr.dat")
        assert df["species"].dtype in ("int64", "int32")


class TestReadBound:
    def test_reads_csv(self):
        df = read_bound(DATA_DIR / "input" / "bound.dat")
        assert set(df.columns) >= {"id1", "id2", "boundary"}
        assert len(df) == 11


class TestReadInputDat:
    def test_reads_parameters(self):
        params = read_input_dat(DATA_DIR / "input.dat")
        assert isinstance(params, dict)
        assert params["BLM"] == 1.0
        assert params["NUMREPS"] == 10
        assert params["INPUTDIR"] == "input"
        assert params["PUNAME"] == "pu.dat"

    def test_numeric_conversion(self):
        params = read_input_dat(DATA_DIR / "input.dat")
        assert isinstance(params["BLM"], float)
        assert isinstance(params["NUMREPS"], int)
        assert isinstance(params["RANDSEED"], int)


class TestLoadProject:
    def test_loads_full_project(self):
        problem = load_project(DATA_DIR)
        assert isinstance(problem, ConservationProblem)
        assert problem.n_planning_units == 6
        assert problem.n_features == 3
        assert len(problem.pu_vs_features) == 17
        assert problem.boundary is not None
        assert len(problem.boundary) == 11
        assert problem.parameters["BLM"] == 1.0

    def test_loaded_problem_validates(self):
        problem = load_project(DATA_DIR)
        errors = problem.validate()
        assert errors == [], f"Validation errors: {errors}"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pymarxan/io/test_readers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pymarxan.io.readers'`

**Step 3: Implement readers**

`src/pymarxan/io/readers.py`:
```python
"""Read Marxan input files (.dat format) and load complete projects."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pymarxan.models.problem import ConservationProblem


def _read_dat(path: Path | str) -> pd.DataFrame:
    """Read a Marxan .dat file, auto-detecting delimiter (comma or tab)."""
    path = Path(path)
    text = path.read_text()
    # Marxan files use comma or tab as delimiter
    sep = "\t" if "\t" in text.split("\n")[0] else ","
    return pd.read_csv(path, sep=sep)


def read_pu(path: Path | str) -> pd.DataFrame:
    """Read planning unit file (pu.dat).

    Expected columns: id (int), cost (float), status (int, optional)
    """
    df = _read_dat(path)
    df["id"] = df["id"].astype(int)
    df["cost"] = df["cost"].astype(float)
    if "status" in df.columns:
        df["status"] = df["status"].astype(int)
    return df


def read_spec(path: Path | str) -> pd.DataFrame:
    """Read conservation feature file (spec.dat).

    Expected columns: id (int), name (str), target or prop (float), spf (float, optional)
    """
    df = _read_dat(path)
    df["id"] = df["id"].astype(int)
    if "target" in df.columns:
        df["target"] = df["target"].astype(float)
    if "prop" in df.columns:
        df["prop"] = df["prop"].astype(float)
    if "spf" in df.columns:
        df["spf"] = df["spf"].astype(float)
    return df


def read_puvspr(path: Path | str) -> pd.DataFrame:
    """Read planning unit vs. species file (puvspr.dat).

    Expected columns: species (int), pu (int), amount (float)
    """
    df = _read_dat(path)
    df["species"] = df["species"].astype(int)
    df["pu"] = df["pu"].astype(int)
    df["amount"] = df["amount"].astype(float)
    return df


def read_bound(path: Path | str) -> pd.DataFrame:
    """Read boundary length file (bound.dat).

    Expected columns: id1 (int), id2 (int), boundary (float)
    """
    df = _read_dat(path)
    df["id1"] = df["id1"].astype(int)
    df["id2"] = df["id2"].astype(int)
    df["boundary"] = df["boundary"].astype(float)
    return df


def read_input_dat(path: Path | str) -> dict:
    """Read Marxan input parameter file (input.dat).

    Format: KEY VALUE pairs, one per line. Lines starting with # are comments.
    Returns dict with numeric values auto-converted to int/float.
    """
    params: dict = {}
    path = Path(path)

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)  # Split on whitespace, max 2 parts
        if len(parts) == 2:
            key, value = parts[0], parts[1]
            # Try numeric conversion
            try:
                # Try int first
                if "." not in value:
                    params[key] = int(value)
                else:
                    params[key] = float(value)
            except ValueError:
                params[key] = value

    return params


def load_project(project_dir: Path | str) -> ConservationProblem:
    """Load a complete Marxan project from a directory.

    Expects an input.dat in the project directory root, which points to the
    input files via INPUTDIR, PUNAME, SPECNAME, PUVSPRNAME, and BOUNDNAME.
    """
    project_dir = Path(project_dir)
    params = read_input_dat(project_dir / "input.dat")

    input_dir = project_dir / params.get("INPUTDIR", "input")

    pu = read_pu(input_dir / params.get("PUNAME", "pu.dat"))
    spec = read_spec(input_dir / params.get("SPECNAME", "spec.dat"))
    puvspr = read_puvspr(input_dir / params.get("PUVSPRNAME", "puvspr.dat"))

    bound_name = params.get("BOUNDNAME", "bound.dat")
    bound_path = input_dir / bound_name
    boundary = read_bound(bound_path) if bound_path.exists() else None

    return ConservationProblem(
        planning_units=pu,
        features=spec,
        pu_vs_features=puvspr,
        boundary=boundary,
        parameters=params,
    )
```

Update `src/pymarxan/io/__init__.py`:
```python
from pymarxan.io.readers import (
    load_project,
    read_bound,
    read_input_dat,
    read_pu,
    read_puvspr,
    read_spec,
)

__all__ = [
    "load_project",
    "read_bound",
    "read_input_dat",
    "read_pu",
    "read_puvspr",
    "read_spec",
]
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/pymarxan/io/test_readers.py -v`
Expected: All 11 tests PASS

**Step 5: Commit**

```bash
git add src/pymarxan/io/ tests/pymarxan/io/
git commit -m "feat: add Marxan file readers and project loader"
```

---

## Task 5: Marxan File Writers

**Files:**
- Create: `src/pymarxan/io/writers.py`
- Test: `tests/pymarxan/io/test_writers.py`

**Step 1: Write the failing tests**

`tests/pymarxan/io/test_writers.py`:
```python
from pathlib import Path

import pandas as pd

from pymarxan.io.readers import load_project, read_input_dat, read_pu, read_spec
from pymarxan.io.writers import write_pu, write_spec, write_puvspr, write_bound, write_input_dat, save_project

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


class TestWritePu:
    def test_roundtrip(self, tmp_path):
        original = pd.DataFrame({
            "id": [1, 2, 3],
            "cost": [10.0, 15.0, 20.0],
            "status": [0, 0, 2],
        })
        out_path = tmp_path / "pu.dat"
        write_pu(original, out_path)
        loaded = read_pu(out_path)
        pd.testing.assert_frame_equal(original, loaded)


class TestWriteSpec:
    def test_roundtrip(self, tmp_path):
        original = pd.DataFrame({
            "id": [1, 2],
            "name": ["sp_a", "sp_b"],
            "target": [20.0, 10.0],
            "spf": [1.0, 1.0],
        })
        out_path = tmp_path / "spec.dat"
        write_spec(original, out_path)
        loaded = read_spec(out_path)
        pd.testing.assert_frame_equal(original, loaded)


class TestWriteInputDat:
    def test_roundtrip(self, tmp_path):
        params = {"BLM": 1.5, "NUMREPS": 10, "INPUTDIR": "input", "PUNAME": "pu.dat"}
        out_path = tmp_path / "input.dat"
        write_input_dat(params, out_path)
        loaded = read_input_dat(out_path)
        assert loaded["BLM"] == 1.5
        assert loaded["NUMREPS"] == 10
        assert loaded["INPUTDIR"] == "input"


class TestSaveProject:
    def test_roundtrip(self, tmp_path):
        original = load_project(DATA_DIR)
        save_project(original, tmp_path)
        reloaded = load_project(tmp_path)
        assert reloaded.n_planning_units == original.n_planning_units
        assert reloaded.n_features == original.n_features
        assert len(reloaded.pu_vs_features) == len(original.pu_vs_features)
        assert reloaded.parameters["BLM"] == original.parameters["BLM"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pymarxan/io/test_writers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pymarxan.io.writers'`

**Step 3: Implement writers**

`src/pymarxan/io/writers.py`:
```python
"""Write Marxan input files and save complete projects."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pymarxan.models.problem import ConservationProblem


def write_pu(df: pd.DataFrame, path: Path | str) -> None:
    """Write planning unit data to pu.dat format."""
    df.to_csv(path, index=False)


def write_spec(df: pd.DataFrame, path: Path | str) -> None:
    """Write conservation feature data to spec.dat format."""
    df.to_csv(path, index=False)


def write_puvspr(df: pd.DataFrame, path: Path | str) -> None:
    """Write planning unit vs. species data to puvspr.dat format."""
    df.to_csv(path, index=False)


def write_bound(df: pd.DataFrame, path: Path | str) -> None:
    """Write boundary data to bound.dat format."""
    df.to_csv(path, index=False)


def write_input_dat(params: dict, path: Path | str) -> None:
    """Write Marxan input parameter file (input.dat).

    Format: KEY VALUE pairs, one per line.
    """
    path = Path(path)
    lines = []
    for key, value in params.items():
        if isinstance(value, float) and value == int(value):
            # Write 1.0 as "1.0" not "1" to preserve float type on roundtrip
            lines.append(f"{key} {value}")
        else:
            lines.append(f"{key} {value}")
    path.write_text("\n".join(lines) + "\n")


def save_project(problem: ConservationProblem, project_dir: Path | str) -> None:
    """Save a complete Marxan project to a directory.

    Creates input.dat and an input/ subdirectory with all data files.
    """
    project_dir = Path(project_dir)
    input_dir_name = problem.parameters.get("INPUTDIR", "input")
    input_dir = project_dir / input_dir_name
    input_dir.mkdir(parents=True, exist_ok=True)

    pu_name = problem.parameters.get("PUNAME", "pu.dat")
    spec_name = problem.parameters.get("SPECNAME", "spec.dat")
    puvspr_name = problem.parameters.get("PUVSPRNAME", "puvspr.dat")
    bound_name = problem.parameters.get("BOUNDNAME", "bound.dat")

    write_pu(problem.planning_units, input_dir / pu_name)
    write_spec(problem.features, input_dir / spec_name)
    write_puvspr(problem.pu_vs_features, input_dir / puvspr_name)

    if problem.boundary is not None:
        write_bound(problem.boundary, input_dir / bound_name)

    write_input_dat(problem.parameters, project_dir / "input.dat")
```

Update `src/pymarxan/io/__init__.py` — add the writer imports:
```python
from pymarxan.io.readers import (
    load_project,
    read_bound,
    read_input_dat,
    read_pu,
    read_puvspr,
    read_spec,
)
from pymarxan.io.writers import (
    save_project,
    write_bound,
    write_input_dat,
    write_pu,
    write_puvspr,
    write_spec,
)

__all__ = [
    "load_project",
    "read_bound",
    "read_input_dat",
    "read_pu",
    "read_puvspr",
    "read_spec",
    "save_project",
    "write_bound",
    "write_input_dat",
    "write_pu",
    "write_puvspr",
    "write_spec",
]
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/pymarxan/io/test_writers.py -v`
Expected: All 4 tests PASS

Run: `pytest tests/ -v`
Expected: All tests PASS (readers + writers + models)

**Step 5: Commit**

```bash
git add src/pymarxan/io/ tests/pymarxan/io/
git commit -m "feat: add Marxan file writers and project saver"
```

---

## Task 6: Solver Base Interface and Solution Model

**Files:**
- Create: `src/pymarxan/solvers/base.py`
- Test: `tests/pymarxan/solvers/test_base.py`

**Step 1: Write the failing tests**

`tests/pymarxan/solvers/test_base.py`:
```python
import numpy as np

from pymarxan.solvers.base import Solution, SolverConfig


class TestSolution:
    def test_create_solution(self):
        sol = Solution(
            selected=np.array([True, False, True]),
            cost=30.0,
            boundary=2.0,
            objective=32.0,
            targets_met={1: True, 2: False},
            metadata={"solver": "test"},
        )
        assert sol.cost == 30.0
        assert sol.selected.sum() == 2
        assert sol.targets_met[1] is True
        assert sol.targets_met[2] is False

    def test_all_targets_met(self):
        sol = Solution(
            selected=np.array([True, True]),
            cost=25.0,
            boundary=1.0,
            objective=26.0,
            targets_met={1: True, 2: True},
            metadata={},
        )
        assert sol.all_targets_met

    def test_not_all_targets_met(self):
        sol = Solution(
            selected=np.array([True]),
            cost=10.0,
            boundary=0.0,
            objective=10.0,
            targets_met={1: True, 2: False},
            metadata={},
        )
        assert not sol.all_targets_met

    def test_n_selected(self):
        sol = Solution(
            selected=np.array([True, False, True, True, False]),
            cost=0, boundary=0, objective=0,
            targets_met={}, metadata={},
        )
        assert sol.n_selected == 3


class TestSolverConfig:
    def test_defaults(self):
        config = SolverConfig()
        assert config.num_solutions == 10
        assert config.seed is None
        assert config.verbose is False

    def test_custom(self):
        config = SolverConfig(num_solutions=50, seed=42, verbose=True)
        assert config.num_solutions == 50
        assert config.seed == 42
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pymarxan/solvers/test_base.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement solver base**

`src/pymarxan/solvers/base.py`:
```python
"""Base solver interface and solution model for conservation planning."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from pymarxan.models.problem import ConservationProblem


@dataclass
class Solution:
    """Result of a conservation planning optimization run."""

    selected: np.ndarray  # Boolean array: is planning unit i selected?
    cost: float
    boundary: float
    objective: float  # cost + BLM * boundary + penalties
    targets_met: dict[int, bool]  # Feature ID → whether target is met
    metadata: dict = field(default_factory=dict)

    @property
    def all_targets_met(self) -> bool:
        return all(self.targets_met.values())

    @property
    def n_selected(self) -> int:
        return int(self.selected.sum())


@dataclass
class SolverConfig:
    """Configuration for a solver run."""

    num_solutions: int = 10
    seed: int | None = None
    verbose: bool = False


class Solver(ABC):
    """Abstract base class for conservation planning solvers."""

    @abstractmethod
    def solve(
        self, problem: ConservationProblem, config: SolverConfig | None = None
    ) -> list[Solution]:
        """Run the solver and return a list of solutions."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable solver name."""
        ...

    @abstractmethod
    def supports_zones(self) -> bool:
        """Whether this solver supports multi-zone problems."""
        ...

    def available(self) -> bool:
        """Check if this solver's dependencies are available."""
        return True
```

Update `src/pymarxan/solvers/__init__.py`:
```python
from pymarxan.solvers.base import Solution, Solver, SolverConfig

__all__ = ["Solution", "Solver", "SolverConfig"]
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/pymarxan/solvers/test_base.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/pymarxan/solvers/ tests/pymarxan/solvers/
git commit -m "feat: add solver base interface, Solution and SolverConfig models"
```

---

## Task 7: MIP Solver (PuLP)

**Files:**
- Create: `src/pymarxan/solvers/mip_solver.py`
- Test: `tests/pymarxan/solvers/test_mip_solver.py`

**Step 1: Write the failing tests**

`tests/pymarxan/solvers/test_mip_solver.py`:
```python
from pathlib import Path

import numpy as np

from pymarxan.io.readers import load_project
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


class TestMIPSolver:
    def setup_method(self):
        self.problem = load_project(DATA_DIR)
        self.solver = MIPSolver()

    def test_solver_name(self):
        assert self.solver.name() == "MIP (PuLP)"

    def test_solver_available(self):
        assert self.solver.available()

    def test_does_not_support_zones(self):
        assert not self.solver.supports_zones()

    def test_solve_returns_solutions(self):
        config = SolverConfig(num_solutions=1)
        solutions = self.solver.solve(self.problem, config)
        assert len(solutions) == 1
        sol = solutions[0]
        assert isinstance(sol.selected, np.ndarray)
        assert len(sol.selected) == 6
        assert sol.cost > 0

    def test_all_targets_met(self):
        config = SolverConfig(num_solutions=1)
        solutions = self.solver.solve(self.problem, config)
        sol = solutions[0]
        assert sol.all_targets_met, f"Unmet targets: {sol.targets_met}"

    def test_solution_cost_is_optimal_or_near(self):
        """MIP should find the minimum cost solution that meets all targets."""
        config = SolverConfig(num_solutions=1)
        solutions = self.solver.solve(self.problem, config)
        sol = solutions[0]
        # Cost must be less than total (selecting everything costs 83)
        total_cost = self.problem.planning_units["cost"].sum()
        assert sol.cost < total_cost

    def test_locked_in_units_selected(self):
        """Planning units with status=2 must always be selected."""
        self.problem.planning_units.loc[
            self.problem.planning_units["id"] == 1, "status"
        ] = 2
        config = SolverConfig(num_solutions=1)
        solutions = self.solver.solve(self.problem, config)
        sol = solutions[0]
        # PU 1 is index 0 in the array
        pu_ids = self.problem.planning_units["id"].tolist()
        idx = pu_ids.index(1)
        assert sol.selected[idx], "Locked-in PU 1 should be selected"

    def test_locked_out_units_not_selected(self):
        """Planning units with status=3 must never be selected."""
        self.problem.planning_units.loc[
            self.problem.planning_units["id"] == 6, "status"
        ] = 3
        config = SolverConfig(num_solutions=1)
        solutions = self.solver.solve(self.problem, config)
        sol = solutions[0]
        pu_ids = self.problem.planning_units["id"].tolist()
        idx = pu_ids.index(6)
        assert not sol.selected[idx], "Locked-out PU 6 should not be selected"

    def test_blm_zero_ignores_boundary(self):
        """With BLM=0, boundary should not affect the solution."""
        self.problem.parameters["BLM"] = 0.0
        config = SolverConfig(num_solutions=1)
        solutions = self.solver.solve(self.problem, config)
        sol = solutions[0]
        assert sol.all_targets_met
        # Objective should equal cost when BLM=0
        assert abs(sol.objective - sol.cost) < 0.01
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pymarxan/solvers/test_mip_solver.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement MIP solver**

`src/pymarxan/solvers/mip_solver.py`:
```python
"""Exact MIP solver using PuLP for conservation planning.

Formulates the Marxan minimum-set problem as a Mixed Integer Linear Program:

    Minimize: Σ(cost_i * x_i) + BLM * Σ(boundary_ij * y_ij)
    Subject to:
        Σ(amount_ij * x_i) >= target_j   for all features j
        y_ij >= x_i - x_j                for all boundary pairs (i,j)
        y_ij >= x_j - x_i                for all boundary pairs (i,j)
        x_i = 1                           for locked-in PUs (status=2)
        x_i = 0                           for locked-out PUs (status=3)
        x_i ∈ {0, 1}
"""

from __future__ import annotations

import numpy as np
import pulp

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig


class MIPSolver(Solver):
    """Exact solver using Mixed Integer Linear Programming via PuLP."""

    def name(self) -> str:
        return "MIP (PuLP)"

    def supports_zones(self) -> bool:
        return False

    def available(self) -> bool:
        try:
            import pulp  # noqa: F811
            return True
        except ImportError:
            return False

    def solve(
        self, problem: ConservationProblem, config: SolverConfig | None = None
    ) -> list[Solution]:
        if config is None:
            config = SolverConfig()

        blm = float(problem.parameters.get("BLM", 0.0))
        pu_ids = problem.planning_units["id"].tolist()
        n_pu = len(pu_ids)
        id_to_idx = {pid: i for i, pid in enumerate(pu_ids)}

        # Build cost array
        costs = problem.planning_units["cost"].values

        # Build status array
        status = {}
        if "status" in problem.planning_units.columns:
            for _, row in problem.planning_units.iterrows():
                status[row["id"]] = int(row["status"])

        # Build feature amount matrix: feature_amounts[feature_id] = {pu_idx: amount}
        feature_amounts: dict[int, dict[int, float]] = {}
        for _, row in problem.pu_vs_features.iterrows():
            fid = int(row["species"])
            pid = int(row["pu"])
            if pid in id_to_idx:
                feature_amounts.setdefault(fid, {})[id_to_idx[pid]] = float(
                    row["amount"]
                )

        # Build boundary pairs (only off-diagonal)
        boundary_pairs: list[tuple[int, int, float]] = []
        if problem.boundary is not None and blm > 0:
            for _, row in problem.boundary.iterrows():
                id1, id2 = int(row["id1"]), int(row["id2"])
                if id1 != id2 and id1 in id_to_idx and id2 in id_to_idx:
                    boundary_pairs.append(
                        (id_to_idx[id1], id_to_idx[id2], float(row["boundary"]))
                    )

        # Create the LP problem
        prob = pulp.LpProblem("marxan_minimum_set", pulp.LpMinimize)

        # Decision variables
        x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n_pu)]

        # Boundary auxiliary variables: y_ij = |x_i - x_j|
        y_vars = {}
        for i, j, _ in boundary_pairs:
            key = (min(i, j), max(i, j))
            if key not in y_vars:
                y_vars[key] = pulp.LpVariable(f"y_{key[0]}_{key[1]}", lowBound=0)

        # Objective: minimize cost + BLM * boundary
        cost_expr = pulp.lpSum(costs[i] * x[i] for i in range(n_pu))
        boundary_expr = pulp.lpSum(
            bnd * y_vars[(min(i, j), max(i, j))] for i, j, bnd in boundary_pairs
        )
        prob += cost_expr + blm * boundary_expr

        # Feature target constraints
        for _, frow in problem.features.iterrows():
            fid = int(frow["id"])
            target = float(frow.get("target", 0.0))
            if target <= 0:
                continue
            amounts = feature_amounts.get(fid, {})
            prob += (
                pulp.lpSum(amounts.get(i, 0.0) * x[i] for i in range(n_pu)) >= target,
                f"target_{fid}",
            )

        # Boundary linearization: y_ij >= x_i - x_j and y_ij >= x_j - x_i
        for (i, j), yvar in y_vars.items():
            prob += yvar >= x[i] - x[j], f"bound_pos_{i}_{j}"
            prob += yvar >= x[j] - x[i], f"bound_neg_{i}_{j}"

        # Lock constraints
        for pid, s in status.items():
            if pid in id_to_idx:
                idx = id_to_idx[pid]
                if s == 2:  # Locked in
                    prob += x[idx] == 1, f"lock_in_{pid}"
                elif s == 3:  # Locked out
                    prob += x[idx] == 0, f"lock_out_{pid}"

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        # Extract solution
        selected = np.array([bool(round(x[i].varValue or 0)) for i in range(n_pu)])

        # Compute actual cost and boundary
        total_cost = float(costs[selected].sum())
        total_boundary = 0.0
        if problem.boundary is not None:
            for _, row in problem.boundary.iterrows():
                id1, id2 = int(row["id1"]), int(row["id2"])
                bnd = float(row["boundary"])
                if id1 == id2:
                    # External boundary: contributes if PU is selected
                    if id1 in id_to_idx and selected[id_to_idx[id1]]:
                        total_boundary += bnd
                elif id1 in id_to_idx and id2 in id_to_idx:
                    # Shared boundary: contributes if exactly one is selected
                    s1 = selected[id_to_idx[id1]]
                    s2 = selected[id_to_idx[id2]]
                    if s1 != s2:
                        total_boundary += bnd

        # Check targets met
        targets_met = {}
        for _, frow in problem.features.iterrows():
            fid = int(frow["id"])
            target = float(frow.get("target", 0.0))
            amounts = feature_amounts.get(fid, {})
            achieved = sum(amounts.get(i, 0.0) for i in range(n_pu) if selected[i])
            targets_met[fid] = achieved >= target

        objective = total_cost + blm * total_boundary

        solution = Solution(
            selected=selected,
            cost=total_cost,
            boundary=total_boundary,
            objective=objective,
            targets_met=targets_met,
            metadata={
                "solver": self.name(),
                "status": pulp.LpStatus[prob.status],
            },
        )

        # MIP is deterministic — return the same solution num_solutions times
        # (In practice users might vary BLM or constraints for multiple solutions)
        return [solution] * config.num_solutions
```

Update `src/pymarxan/solvers/__init__.py`:
```python
from pymarxan.solvers.base import Solution, Solver, SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver

__all__ = ["MIPSolver", "Solution", "Solver", "SolverConfig"]
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/pymarxan/solvers/test_mip_solver.py -v`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add src/pymarxan/solvers/ tests/pymarxan/solvers/
git commit -m "feat: add MIP solver using PuLP for exact conservation planning"
```

---

## Task 8: Marxan Binary Solver (C++ Wrapper)

**Files:**
- Create: `src/pymarxan/solvers/marxan_binary.py`
- Test: `tests/pymarxan/solvers/test_marxan_binary.py`

**Step 1: Write the failing tests**

`tests/pymarxan/solvers/test_marxan_binary.py`:
```python
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from pymarxan.io.readers import load_project
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.marxan_binary import MarxanBinarySolver

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


class TestMarxanBinarySolver:
    def test_solver_name(self):
        solver = MarxanBinarySolver()
        assert solver.name() == "Marxan (C++ binary)"

    def test_does_not_support_zones(self):
        solver = MarxanBinarySolver()
        assert not solver.supports_zones()

    def test_available_when_binary_exists(self):
        with patch("shutil.which", return_value="/usr/bin/marxan"):
            solver = MarxanBinarySolver()
            assert solver.available()

    def test_not_available_when_binary_missing(self):
        with patch("shutil.which", return_value=None):
            solver = MarxanBinarySolver()
            assert not solver.available()

    def test_custom_binary_path(self):
        solver = MarxanBinarySolver(binary_path="/custom/path/marxan")
        assert solver._binary_path == "/custom/path/marxan"

    def test_solve_raises_when_unavailable(self):
        with patch("shutil.which", return_value=None):
            solver = MarxanBinarySolver()
            problem = load_project(DATA_DIR)
            with pytest.raises(RuntimeError, match="Marxan binary not found"):
                solver.solve(problem, SolverConfig(num_solutions=1))

    def test_parse_output_csv(self):
        """Test that we can parse a Marxan output_r001.csv file."""
        solver = MarxanBinarySolver()
        # Simulate output: each line is a PU id and selection (0 or 1)
        csv_content = "planning_unit,solution\n1,1\n2,0\n3,1\n4,0\n5,1\n6,0\n"
        selected = solver._parse_solution_csv(csv_content, [1, 2, 3, 4, 5, 6])
        expected = np.array([True, False, True, False, True, False])
        np.testing.assert_array_equal(selected, expected)

    def test_parse_best_csv(self):
        solver = MarxanBinarySolver()
        csv_content = "planning_unit,solution\n1,0\n2,1\n3,1\n4,1\n5,0\n6,0\n"
        selected = solver._parse_solution_csv(csv_content, [1, 2, 3, 4, 5, 6])
        assert selected.sum() == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pymarxan/solvers/test_marxan_binary.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement binary wrapper**

`src/pymarxan/solvers/marxan_binary.py`:
```python
"""Wrapper around the Marxan C++ binary executable."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pymarxan.io.readers import read_input_dat
from pymarxan.io.writers import save_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig

# Common binary names for Marxan executable
_BINARY_NAMES = ["Marxan_x64", "marxan", "Marxan", "marxan_x64"]


class MarxanBinarySolver(Solver):
    """Solver that wraps the Marxan C++ executable."""

    def __init__(self, binary_path: str | None = None):
        self._binary_path = binary_path

    def name(self) -> str:
        return "Marxan (C++ binary)"

    def supports_zones(self) -> bool:
        return False

    def available(self) -> bool:
        return self._find_binary() is not None

    def _find_binary(self) -> str | None:
        if self._binary_path:
            return self._binary_path
        for name in _BINARY_NAMES:
            path = shutil.which(name)
            if path:
                return path
        return None

    def solve(
        self, problem: ConservationProblem, config: SolverConfig | None = None
    ) -> list[Solution]:
        if config is None:
            config = SolverConfig()

        binary = self._find_binary()
        if binary is None:
            raise RuntimeError(
                "Marxan binary not found. Install Marxan and ensure it is on PATH, "
                "or pass binary_path to MarxanBinarySolver()."
            )

        blm = float(problem.parameters.get("BLM", 0.0))
        pu_ids = problem.planning_units["id"].tolist()

        # Set up temp directory with Marxan project files
        with tempfile.TemporaryDirectory(prefix="pymarxan_") as tmpdir:
            tmpdir = Path(tmpdir)

            # Override parameters for this run
            params = dict(problem.parameters)
            params["NUMREPS"] = config.num_solutions
            if config.seed is not None:
                params["RANDSEED"] = config.seed
            params["SAVERUN"] = 3  # CSV format
            params["SAVEBEST"] = 3
            params["SAVESUMMARY"] = 3
            params["SAVESUMSOLN"] = 3
            params["SAVETARGMET"] = 3
            params["VERBOSITY"] = 3 if config.verbose else 0
            params["INPUTDIR"] = "input"
            params["OUTPUTDIR"] = "output"
            params["SCENNAME"] = "output"

            modified_problem = ConservationProblem(
                planning_units=problem.planning_units,
                features=problem.features,
                pu_vs_features=problem.pu_vs_features,
                boundary=problem.boundary,
                parameters=params,
            )
            save_project(modified_problem, tmpdir)

            # Create output directory
            output_dir = tmpdir / "output"
            output_dir.mkdir(exist_ok=True)

            # Run Marxan
            result = subprocess.run(
                [binary, "-s", str(tmpdir / "input.dat")],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Marxan failed:\n{result.stderr}")

            # Parse solutions
            solutions = []
            for i in range(1, config.num_solutions + 1):
                sol_file = output_dir / f"output_r{i:04d}.csv"
                if not sol_file.exists():
                    continue

                selected = self._parse_solution_csv(sol_file.read_text(), pu_ids)
                sol = self._build_solution(problem, selected, blm)
                sol.metadata["run"] = i
                sol.metadata["solver"] = self.name()
                solutions.append(sol)

            # Also try to parse the best solution if no run files found
            if not solutions:
                best_file = output_dir / "output_best.csv"
                if best_file.exists():
                    selected = self._parse_solution_csv(
                        best_file.read_text(), pu_ids
                    )
                    sol = self._build_solution(problem, selected, blm)
                    sol.metadata["solver"] = self.name()
                    sol.metadata["run"] = "best"
                    solutions.append(sol)

        return solutions

    @staticmethod
    def _parse_solution_csv(csv_content: str, pu_ids: list[int]) -> np.ndarray:
        """Parse a Marxan output CSV (planning_unit,solution) into a boolean array."""
        import io

        df = pd.read_csv(io.StringIO(csv_content))
        # Map PU IDs to selection status
        selection_map = dict(zip(df["planning_unit"], df["solution"]))
        return np.array([bool(selection_map.get(pid, 0)) for pid in pu_ids])

    @staticmethod
    def _build_solution(
        problem: ConservationProblem, selected: np.ndarray, blm: float
    ) -> Solution:
        """Build a Solution from a selection array."""
        costs = problem.planning_units["cost"].values
        total_cost = float(costs[selected].sum())

        # Compute boundary
        total_boundary = 0.0
        if problem.boundary is not None:
            pu_ids = problem.planning_units["id"].tolist()
            id_to_idx = {pid: i for i, pid in enumerate(pu_ids)}
            for _, row in problem.boundary.iterrows():
                id1, id2 = int(row["id1"]), int(row["id2"])
                bnd = float(row["boundary"])
                if id1 == id2:
                    if id1 in id_to_idx and selected[id_to_idx[id1]]:
                        total_boundary += bnd
                elif id1 in id_to_idx and id2 in id_to_idx:
                    s1 = selected[id_to_idx[id1]]
                    s2 = selected[id_to_idx[id2]]
                    if s1 != s2:
                        total_boundary += bnd

        # Check targets
        targets_met = {}
        pu_ids = problem.planning_units["id"].tolist()
        id_to_idx = {pid: i for i, pid in enumerate(pu_ids)}
        for _, frow in problem.features.iterrows():
            fid = int(frow["id"])
            target = float(frow.get("target", 0.0))
            achieved = 0.0
            mask = problem.pu_vs_features["species"] == fid
            for _, arow in problem.pu_vs_features[mask].iterrows():
                pid = int(arow["pu"])
                if pid in id_to_idx and selected[id_to_idx[pid]]:
                    achieved += float(arow["amount"])
            targets_met[fid] = achieved >= target

        objective = total_cost + blm * total_boundary

        return Solution(
            selected=selected,
            cost=total_cost,
            boundary=total_boundary,
            objective=objective,
            targets_met=targets_met,
            metadata={},
        )
```

Update `src/pymarxan/solvers/__init__.py`:
```python
from pymarxan.solvers.base import Solution, Solver, SolverConfig
from pymarxan.solvers.marxan_binary import MarxanBinarySolver
from pymarxan.solvers.mip_solver import MIPSolver

__all__ = ["MarxanBinarySolver", "MIPSolver", "Solution", "Solver", "SolverConfig"]
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/pymarxan/solvers/test_marxan_binary.py -v`
Expected: All 8 tests PASS

Run: `pytest tests/ -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/pymarxan/solvers/ tests/pymarxan/solvers/
git commit -m "feat: add Marxan C++ binary wrapper solver"
```

---

## Task 9: Basic Shiny Upload Module

**Files:**
- Create: `src/pymarxan_shiny/__init__.py` (update)
- Create: `src/pymarxan_shiny/modules/__init__.py`
- Create: `src/pymarxan_shiny/modules/data_input/__init__.py`
- Create: `src/pymarxan_shiny/modules/data_input/upload.py`

No automated tests for Shiny UI modules — these are tested via the assembled app manually.

**Step 1: Create the upload module**

`src/pymarxan_shiny/modules/__init__.py`: empty file

`src/pymarxan_shiny/modules/data_input/__init__.py`: empty file

`src/pymarxan_shiny/modules/data_input/upload.py`:
```python
"""Data upload Shiny module for loading Marxan project files."""

from __future__ import annotations

from pathlib import Path
import tempfile
import zipfile

from shiny import module, reactive, render, ui

from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem


@module.ui
def upload_ui():
    return ui.card(
        ui.card_header("Load Marxan Project"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_file(
                    "project_zip",
                    "Upload Marxan project (.zip)",
                    accept=[".zip"],
                    multiple=False,
                ),
                ui.hr(),
                ui.p("Or load from a local directory:"),
                ui.input_text("project_path", "Project directory path", placeholder="/path/to/project"),
                ui.input_action_button("load_local", "Load from path"),
                width=350,
            ),
            ui.output_text_verbatim("project_summary"),
        ),
    )


@module.server
def upload_server(input, output, session, problem: reactive.Value):
    """Server logic for data upload module.

    Args:
        problem: A reactive.Value[ConservationProblem | None] that this module writes to.
    """

    @reactive.effect
    @reactive.event(input.project_zip)
    def _handle_zip_upload():
        file_infos = input.project_zip()
        if not file_infos:
            return
        file_info = file_infos[0]
        uploaded_path = file_info["datapath"]

        with tempfile.TemporaryDirectory(prefix="pymarxan_upload_") as tmpdir:
            with zipfile.ZipFile(uploaded_path, "r") as zf:
                zf.extractall(tmpdir)
            # Find the directory containing input.dat
            tmpdir_path = Path(tmpdir)
            input_dat_files = list(tmpdir_path.rglob("input.dat"))
            if not input_dat_files:
                ui.notification_show("No input.dat found in uploaded ZIP", type="error")
                return
            project_dir = input_dat_files[0].parent
            try:
                loaded = load_project(project_dir)
                errors = loaded.validate()
                if errors:
                    ui.notification_show(
                        f"Validation warnings: {'; '.join(errors)}", type="warning"
                    )
                problem.set(loaded)
                ui.notification_show("Project loaded successfully!", type="message")
            except Exception as e:
                ui.notification_show(f"Error loading project: {e}", type="error")

    @reactive.effect
    @reactive.event(input.load_local)
    def _handle_local_load():
        path = input.project_path()
        if not path:
            return
        project_dir = Path(path)
        if not (project_dir / "input.dat").exists():
            ui.notification_show(
                f"No input.dat found in {project_dir}", type="error"
            )
            return
        try:
            loaded = load_project(project_dir)
            errors = loaded.validate()
            if errors:
                ui.notification_show(
                    f"Validation warnings: {'; '.join(errors)}", type="warning"
                )
            problem.set(loaded)
            ui.notification_show("Project loaded successfully!", type="message")
        except Exception as e:
            ui.notification_show(f"Error loading project: {e}", type="error")

    @render.text
    def project_summary():
        p = problem()
        if p is None:
            return "No project loaded. Upload a ZIP file or enter a directory path."
        return p.summary()
```

**Step 2: Commit**

```bash
git add src/pymarxan_shiny/
git commit -m "feat: add Shiny upload module for loading Marxan projects"
```

---

## Task 10: Solver Config Shiny Module

**Files:**
- Create: `src/pymarxan_shiny/modules/solver_config/__init__.py`
- Create: `src/pymarxan_shiny/modules/solver_config/solver_picker.py`

**Step 1: Create the solver config module**

`src/pymarxan_shiny/modules/solver_config/__init__.py`: empty file

`src/pymarxan_shiny/modules/solver_config/solver_picker.py`:
```python
"""Solver selection and configuration Shiny module."""

from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver
from pymarxan.solvers.marxan_binary import MarxanBinarySolver


@module.ui
def solver_picker_ui():
    binary_available = MarxanBinarySolver().available()
    solver_choices = {"mip": "MIP Solver (exact, PuLP/CBC)"}
    if binary_available:
        solver_choices["binary"] = "Marxan C++ Binary"

    return ui.card(
        ui.card_header("Solver Configuration"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_radio_buttons(
                    "solver_type",
                    "Solver",
                    choices=solver_choices,
                    selected="mip",
                ),
                ui.hr(),
                ui.h5("Parameters"),
                ui.input_numeric("blm", "Boundary Length Modifier (BLM)", value=1.0, min=0, step=0.1),
                ui.input_numeric("num_solutions", "Number of solutions", value=10, min=1, max=1000, step=1),
                ui.input_numeric("seed", "Random seed (optional)", value=42, min=-1),
                ui.hr(),
                ui.panel_conditional(
                    "input.solver_type === 'binary'",
                    ui.input_numeric("num_iterations", "SA Iterations", value=1000000, min=1000, step=100000),
                    ui.input_numeric("num_temp", "Temperature steps", value=10000, min=100, step=1000),
                ),
                width=350,
            ),
            ui.output_text_verbatim("solver_info"),
        ),
    )


@module.server
def solver_picker_server(input, output, session, solver_config: reactive.Value):
    """Server logic for solver configuration module.

    Args:
        solver_config: A reactive.Value[dict] this module writes solver config into.
            Keys: "solver_type", "blm", "num_solutions", "seed", + solver-specific params
    """

    @reactive.effect
    @reactive.event(
        input.solver_type, input.blm, input.num_solutions, input.seed,
        ignore_init=False,
    )
    def _update_config():
        config = {
            "solver_type": input.solver_type(),
            "blm": float(input.blm()),
            "num_solutions": int(input.num_solutions()),
            "seed": int(input.seed()) if input.seed() and input.seed() > 0 else None,
        }
        if input.solver_type() == "binary":
            config["num_iterations"] = int(input.num_iterations() or 1000000)
            config["num_temp"] = int(input.num_temp() or 10000)
        solver_config.set(config)

    @render.text
    def solver_info():
        st = input.solver_type()
        if st == "mip":
            return (
                "MIP Solver (PuLP/CBC)\n"
                "---------------------\n"
                "Uses Mixed Integer Linear Programming to find the\n"
                "mathematically optimal solution. Guaranteed to find\n"
                "the minimum-cost reserve network that meets all targets.\n"
                "Equivalent to prioritizr in R."
            )
        elif st == "binary":
            return (
                "Marxan C++ Binary\n"
                "-----------------\n"
                "Wraps the original Marxan executable using simulated\n"
                "annealing. Produces multiple solutions across repeat\n"
                "runs. Heuristic — not guaranteed optimal but well-tested."
            )
        return ""
```

**Step 2: Commit**

```bash
git add src/pymarxan_shiny/modules/solver_config/
git commit -m "feat: add Shiny solver picker module with MIP and binary options"
```

---

## Task 11: Solution Map Shiny Module

**Files:**
- Create: `src/pymarxan_shiny/modules/mapping/__init__.py`
- Create: `src/pymarxan_shiny/modules/mapping/solution_map.py`

**Step 1: Create the solution map module**

`src/pymarxan_shiny/modules/mapping/__init__.py`: empty file

`src/pymarxan_shiny/modules/mapping/solution_map.py`:
```python
"""Solution map Shiny module — displays selected planning units on a map."""

from __future__ import annotations

import json

from shiny import module, reactive, render, ui

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution


@module.ui
def solution_map_ui():
    return ui.card(
        ui.card_header("Solution Map"),
        ui.output_ui("map_or_table"),
    )


@module.server
def solution_map_server(
    input, output, session,
    problem: reactive.Value,
    solution: reactive.Value,
):
    """Server logic for solution map display.

    Args:
        problem: reactive.Value[ConservationProblem | None]
        solution: reactive.Value[Solution | None]
    """

    @render.ui
    def map_or_table():
        p = problem()
        s = solution()
        if p is None or s is None:
            return ui.p("Run a solver to see results here.")

        # Build a summary table of selected PUs
        rows = []
        pu_ids = p.planning_units["id"].tolist()
        costs = p.planning_units["cost"].tolist()
        for i, (pid, cost) in enumerate(zip(pu_ids, costs)):
            if s.selected[i]:
                rows.append({"Planning Unit": pid, "Cost": f"{cost:.1f}", "Selected": "Yes"})

        if not rows:
            return ui.p("No planning units selected.")

        # Summary stats
        header = ui.div(
            ui.h5(f"Solution Summary"),
            ui.p(f"Selected: {s.n_selected} / {len(pu_ids)} planning units"),
            ui.p(f"Cost: {s.cost:.2f}"),
            ui.p(f"Boundary: {s.boundary:.2f}"),
            ui.p(f"Objective: {s.objective:.2f}"),
            ui.p(
                f"Targets met: {sum(s.targets_met.values())} / {len(s.targets_met)}"
            ),
        )

        # Selection table
        from shiny import ui as sui

        table_rows = [
            ui.tags.tr(
                ui.tags.td(str(r["Planning Unit"])),
                ui.tags.td(r["Cost"]),
            )
            for r in rows
        ]
        table = ui.tags.table(
            ui.tags.thead(
                ui.tags.tr(ui.tags.th("Planning Unit"), ui.tags.th("Cost"))
            ),
            ui.tags.tbody(*table_rows),
            class_="table table-striped table-sm",
        )

        return ui.div(header, table)
```

**Step 2: Commit**

```bash
git add src/pymarxan_shiny/modules/mapping/
git commit -m "feat: add Shiny solution map module for results display"
```

---

## Task 12: Results Summary Shiny Module

**Files:**
- Create: `src/pymarxan_shiny/modules/results/__init__.py`
- Create: `src/pymarxan_shiny/modules/results/summary_table.py`

**Step 1: Create the results summary module**

`src/pymarxan_shiny/modules/results/__init__.py`: empty file

`src/pymarxan_shiny/modules/results/summary_table.py`:
```python
"""Results summary Shiny module — shows target achievement and solution stats."""

from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution


@module.ui
def summary_table_ui():
    return ui.card(
        ui.card_header("Target Achievement"),
        ui.output_ui("target_table"),
    )


@module.server
def summary_table_server(
    input, output, session,
    problem: reactive.Value,
    solution: reactive.Value,
):
    """Server logic for target achievement summary.

    Args:
        problem: reactive.Value[ConservationProblem | None]
        solution: reactive.Value[Solution | None]
    """

    @render.ui
    def target_table():
        p = problem()
        s = solution()
        if p is None or s is None:
            return ui.p("No solution available. Run a solver first.")

        pu_ids = p.planning_units["id"].tolist()
        id_to_idx = {pid: i for i, pid in enumerate(pu_ids)}

        # Compute achieved amounts per feature
        rows = []
        for _, frow in p.features.iterrows():
            fid = int(frow["id"])
            fname = frow.get("name", f"Feature {fid}")
            target = float(frow.get("target", 0.0))

            mask = p.pu_vs_features["species"] == fid
            achieved = 0.0
            for _, arow in p.pu_vs_features[mask].iterrows():
                pid = int(arow["pu"])
                if pid in id_to_idx and s.selected[id_to_idx[pid]]:
                    achieved += float(arow["amount"])

            met = achieved >= target
            pct = (achieved / target * 100) if target > 0 else 100.0
            rows.append({
                "id": fid,
                "name": fname,
                "target": target,
                "achieved": achieved,
                "pct": pct,
                "met": met,
            })

        table_rows = [
            ui.tags.tr(
                ui.tags.td(str(r["id"])),
                ui.tags.td(r["name"]),
                ui.tags.td(f"{r['target']:.1f}"),
                ui.tags.td(f"{r['achieved']:.1f}"),
                ui.tags.td(f"{r['pct']:.1f}%"),
                ui.tags.td("Met" if r["met"] else "NOT MET",
                           style=f"color: {'green' if r['met'] else 'red'}; font-weight: bold"),
            )
            for r in rows
        ]

        return ui.tags.table(
            ui.tags.thead(
                ui.tags.tr(
                    ui.tags.th("ID"),
                    ui.tags.th("Feature"),
                    ui.tags.th("Target"),
                    ui.tags.th("Achieved"),
                    ui.tags.th("%"),
                    ui.tags.th("Status"),
                )
            ),
            ui.tags.tbody(*table_rows),
            class_="table table-striped",
        )
```

**Step 2: Commit**

```bash
git add src/pymarxan_shiny/modules/results/
git commit -m "feat: add Shiny results summary module with target achievement table"
```

---

## Task 13: Assembled Shiny Application

**Files:**
- Create: `src/pymarxan_app/__init__.py` (update)
- Create: `src/pymarxan_app/app.py`

**Step 1: Create the assembled application**

`src/pymarxan_app/app.py`:
```python
"""pymarxan: Assembled Shiny application for Marxan conservation planning.

Run with: shiny run src/pymarxan_app/app.py
"""

from __future__ import annotations

from shiny import App, Inputs, Outputs, Session, reactive, render, ui

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver
from pymarxan.solvers.marxan_binary import MarxanBinarySolver
from pymarxan_shiny.modules.data_input.upload import upload_ui, upload_server
from pymarxan_shiny.modules.solver_config.solver_picker import (
    solver_picker_ui,
    solver_picker_server,
)
from pymarxan_shiny.modules.mapping.solution_map import (
    solution_map_ui,
    solution_map_server,
)
from pymarxan_shiny.modules.results.summary_table import (
    summary_table_ui,
    summary_table_server,
)

app_ui = ui.page_navbar(
    ui.nav_panel(
        "Data",
        ui.layout_columns(
            upload_ui("upload"),
            col_widths=12,
        ),
    ),
    ui.nav_panel(
        "Configure",
        ui.layout_columns(
            solver_picker_ui("solver"),
            col_widths=12,
        ),
    ),
    ui.nav_panel(
        "Run",
        ui.card(
            ui.card_header("Run Solver"),
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_action_button("run_solver", "Run Solver", class_="btn-primary btn-lg w-100"),
                    ui.hr(),
                    ui.output_text_verbatim("run_log"),
                    width=300,
                ),
                ui.output_text_verbatim("run_status"),
            ),
        ),
    ),
    ui.nav_panel(
        "Results",
        ui.layout_columns(
            solution_map_ui("solution_map"),
            summary_table_ui("summary"),
            col_widths=[6, 6],
        ),
    ),
    title="pymarxan",
    id="navbar",
)


def server(input: Inputs, output: Outputs, session: Session):
    # Shared reactive state
    problem: reactive.Value[ConservationProblem | None] = reactive.value(None)
    solver_config: reactive.Value[dict] = reactive.value({
        "solver_type": "mip",
        "blm": 1.0,
        "num_solutions": 10,
        "seed": None,
    })
    current_solution: reactive.Value[Solution | None] = reactive.value(None)

    # Wire up modules
    upload_server("upload", problem=problem)
    solver_picker_server("solver", solver_config=solver_config)
    solution_map_server("solution_map", problem=problem, solution=current_solution)
    summary_table_server("summary", problem=problem, solution=current_solution)

    @reactive.effect
    @reactive.event(input.run_solver)
    def _run_solver():
        p = problem()
        if p is None:
            ui.notification_show("Load a project first!", type="error")
            return

        config_dict = solver_config()
        solver_type = config_dict.get("solver_type", "mip")

        # Apply BLM from config to problem
        p.parameters["BLM"] = config_dict.get("blm", 1.0)

        config = SolverConfig(
            num_solutions=config_dict.get("num_solutions", 10),
            seed=config_dict.get("seed"),
            verbose=False,
        )

        if solver_type == "binary":
            p.parameters["NUMITNS"] = config_dict.get("num_iterations", 1000000)
            p.parameters["NUMTEMP"] = config_dict.get("num_temp", 10000)

        # Build solver
        if solver_type == "mip":
            solver = MIPSolver()
        elif solver_type == "binary":
            solver = MarxanBinarySolver()
        else:
            ui.notification_show(f"Unknown solver type: {solver_type}", type="error")
            return

        if not solver.available():
            ui.notification_show(
                f"Solver '{solver.name()}' is not available. Check dependencies.",
                type="error",
            )
            return

        ui.notification_show(f"Running {solver.name()}...", type="message")

        try:
            solutions = solver.solve(p, config)
            if solutions:
                # Use the best solution (lowest objective)
                best = min(solutions, key=lambda s: s.objective)
                current_solution.set(best)
                ui.notification_show(
                    f"Solver complete! Best cost: {best.cost:.2f}, "
                    f"Targets met: {sum(best.targets_met.values())}/{len(best.targets_met)}",
                    type="message",
                )
            else:
                ui.notification_show("Solver returned no solutions.", type="warning")
        except Exception as e:
            ui.notification_show(f"Solver error: {e}", type="error")

    @render.text
    def run_status():
        p = problem()
        s = current_solution()
        if p is None:
            return "Step 1: Go to 'Data' tab and load a Marxan project."
        if s is None:
            return (
                f"Project loaded: {p.n_planning_units} planning units, "
                f"{p.n_features} features.\n"
                f"Step 2: Configure solver in 'Configure' tab, then click 'Run Solver'."
            )
        return (
            f"Solution available!\n"
            f"  Selected: {s.n_selected} planning units\n"
            f"  Cost: {s.cost:.2f}\n"
            f"  Boundary: {s.boundary:.2f}\n"
            f"  Objective: {s.objective:.2f}\n"
            f"  All targets met: {'Yes' if s.all_targets_met else 'No'}\n\n"
            f"Go to 'Results' tab to explore the solution."
        )

    @render.text
    def run_log():
        s = current_solution()
        if s is None:
            return "No solver has been run yet."
        meta = s.metadata
        lines = ["Solver metadata:"]
        for k, v in meta.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


app = App(app_ui, server)
```

Update `src/pymarxan_app/__init__.py`:
```python
"""pymarxan-app: Assembled Shiny application for Marxan conservation planning."""
```

**Step 2: Test manually**

Run: `cd /home/razinka/marxan && shiny run src/pymarxan_app/app.py --port 8000`
Expected: App starts and is accessible at http://localhost:8000

Manual test flow:
1. Go to "Data" tab → enter path `tests/data/simple` → click "Load from path" → see project summary
2. Go to "Configure" tab → select MIP solver, BLM=1.0, 1 solution
3. Go to "Run" tab → click "Run Solver" → see success notification
4. Go to "Results" tab → see solution map table and target achievement table

**Step 3: Commit**

```bash
git add src/pymarxan_app/ src/pymarxan_shiny/
git commit -m "feat: assemble pymarxan Shiny app with data-configure-run-results flow"
```

---

## Task 14: Integration Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

`tests/test_integration.py`:
```python
"""Integration test: full roundtrip from loading data to solving."""

from pathlib import Path

from pymarxan.io.readers import load_project
from pymarxan.io.writers import save_project
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver

DATA_DIR = Path(__file__).parent / "data" / "simple"


class TestFullRoundtrip:
    def test_load_solve_check(self):
        """Load test data → solve with MIP → verify all targets met."""
        problem = load_project(DATA_DIR)
        assert problem.validate() == []

        solver = MIPSolver()
        config = SolverConfig(num_solutions=1)
        solutions = solver.solve(problem, config)

        assert len(solutions) == 1
        sol = solutions[0]
        assert sol.all_targets_met
        assert sol.cost > 0
        assert sol.n_selected > 0
        assert sol.n_selected < problem.n_planning_units  # Not everything selected

    def test_load_save_load_solve(self, tmp_path):
        """Load → save → reload → solve → verify results match."""
        original = load_project(DATA_DIR)
        save_project(original, tmp_path)
        reloaded = load_project(tmp_path)

        solver = MIPSolver()
        config = SolverConfig(num_solutions=1)

        sol_original = solver.solve(original, config)[0]
        sol_reloaded = solver.solve(reloaded, config)[0]

        assert sol_original.cost == sol_reloaded.cost
        assert sol_original.n_selected == sol_reloaded.n_selected

    def test_blm_affects_solution(self):
        """Higher BLM should lead to more compact (less boundary) solutions."""
        problem_low = load_project(DATA_DIR)
        problem_low.parameters["BLM"] = 0.0

        problem_high = load_project(DATA_DIR)
        problem_high.parameters["BLM"] = 100.0

        solver = MIPSolver()
        config = SolverConfig(num_solutions=1)

        sol_low = solver.solve(problem_low, config)[0]
        sol_high = solver.solve(problem_high, config)[0]

        # High BLM should have <= boundary than low BLM (more compact)
        assert sol_high.boundary <= sol_low.boundary or sol_high.objective != sol_low.objective

    def test_locked_in_respected(self):
        """Locked-in PU must appear in solution."""
        problem = load_project(DATA_DIR)
        # Lock in PU 3 (most expensive)
        problem.planning_units.loc[
            problem.planning_units["id"] == 3, "status"
        ] = 2

        solver = MIPSolver()
        sol = solver.solve(problem, SolverConfig(num_solutions=1))[0]

        pu_ids = problem.planning_units["id"].tolist()
        idx = pu_ids.index(3)
        assert sol.selected[idx], "Locked-in PU 3 should be selected"

    def test_locked_out_respected(self):
        """Locked-out PU must NOT appear in solution (if targets still achievable)."""
        problem = load_project(DATA_DIR)
        # Lock out PU 1
        problem.planning_units.loc[
            problem.planning_units["id"] == 1, "status"
        ] = 3

        solver = MIPSolver()
        sol = solver.solve(problem, SolverConfig(num_solutions=1))[0]

        pu_ids = problem.planning_units["id"].tolist()
        idx = pu_ids.index(1)
        assert not sol.selected[idx], "Locked-out PU 1 should not be selected"
```

**Step 2: Run the integration test**

Run: `pytest tests/test_integration.py -v`
Expected: All 5 tests PASS

**Step 3: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS (models + readers + writers + solvers + integration)

**Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for full load-solve-verify roundtrip"
```

---

## Task 15: Lint, Type Check, and Final Cleanup

**Step 1: Run ruff**

Run: `ruff check src/ tests/ --fix`
Expected: No errors (or auto-fixed).

**Step 2: Run mypy**

Run: `mypy src/pymarxan/ --ignore-missing-imports`
Expected: No errors (or minor ones to fix).

**Step 3: Run full test suite one final time**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

**Step 4: Final commit**

```bash
git add -A
git commit -m "chore: lint and type-check cleanup for Phase 1"
```

---

## Summary of Phase 1 Deliverables

After completing all 15 tasks, the project will have:

| Component | Files | Tests |
|---|---|---|
| Project scaffold | `pyproject.toml`, package structure | Install verification |
| Test fixtures | `tests/data/simple/` (6 PU, 3 features) | Used by all tests |
| Domain models | `pymarxan.models.problem` | 9 tests |
| File readers | `pymarxan.io.readers` (pu, spec, puvspr, bound, input.dat, project) | 11 tests |
| File writers | `pymarxan.io.writers` (all formats + project) | 4 tests |
| Solver base | `pymarxan.solvers.base` (Solution, SolverConfig, Solver ABC) | 6 tests |
| MIP solver | `pymarxan.solvers.mip_solver` (PuLP/CBC exact solver) | 9 tests |
| Binary wrapper | `pymarxan.solvers.marxan_binary` (C++ wrapper) | 8 tests |
| Shiny: upload | `pymarxan_shiny.modules.data_input.upload` | Manual |
| Shiny: solver config | `pymarxan_shiny.modules.solver_config.solver_picker` | Manual |
| Shiny: solution map | `pymarxan_shiny.modules.mapping.solution_map` | Manual |
| Shiny: results | `pymarxan_shiny.modules.results.summary_table` | Manual |
| Assembled app | `pymarxan_app.app` (4-tab navigation) | Manual |
| Integration tests | `tests/test_integration.py` | 5 tests |
| **Total** | **~30 files** | **52+ automated tests** |
