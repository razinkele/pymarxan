# Marxan Ecosystem Review & Modular Shiny-for-Python Design

**Date:** 2026-02-21
**Status:** Design Document
**Author:** Generated with Claude Code

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Review of Existing Marxan Implementations](#2-review-of-existing-marxan-implementations)
3. [Gap Analysis](#3-gap-analysis)
4. [Proposed Architecture: `pymarxan`](#4-proposed-architecture-pymarxan)
5. [Module Specifications](#5-module-specifications)
6. [Technology Stack](#6-technology-stack)
7. [Implementation Roadmap](#7-implementation-roadmap)
8. [Design Decisions & Trade-offs](#8-design-decisions--trade-offs)

---

## 1. Executive Summary

Marxan is the world's most widely used conservation planning software, supporting over 1,300 organizations in 100+ countries. The current ecosystem is fragmented across C++ binaries, R packages, JavaScript/TypeScript web platforms, and Python utilities — with no comprehensive Python-native implementation that covers the full workflow.

This document reviews all major Marxan implementations, identifies gaps, and proposes **`pymarxan`**: a modular, three-layer Python system built on Shiny for Python that serves conservation practitioners, researchers, and developers alike.

The proposed system provides:
- A **pure Python core library** (`pymarxan-core`) for headless use
- **Shiny module components** (`pymarxan-shiny`) for UI building blocks
- An **assembled application** (`pymarxan-app`) for end-to-end conservation planning
- Support for classic **simulated annealing**, the existing **C++ binary**, and modern **exact MIP solvers**

---

## 2. Review of Existing Marxan Implementations

### 2.1 Core Solvers

#### Marxan v4 (C++17)
- **Repository:** https://github.com/Marxan-source-code/marxan
- **Language:** C++17 (98.4%), Python (1.4%)
- **License:** MIT
- **Status:** Active — v4.0.6 (March 2021)
- **Features:**
  - Simulated annealing (thermal and quantum-inspired variants)
  - Hill climbing and iterative improvement
  - Heuristic methods
  - OpenMP parallelization for multi-core utilization
  - Cross-platform binaries (Windows, Linux, macOS including M1)
- **Architecture:** Single monolithic C++ executable reading `.dat` input files, producing `.csv` output files. Build system uses g++ with OpenMP. Microsoft Azure Quantum team contributed quantum-inspired optimization enhancements including Substochastic Monte Carlo (SSMC).
- **Input format:** `input.dat` (parameters), `pu.dat` (planning units), `spec.dat` (conservation features), `puvspr.dat` (planning unit vs. species), `bound.dat` (boundary lengths)

#### Marxan with Zones (MarZone) v4
- **Repository:** https://github.com/Marxan-source-code/marzone
- **Language:** C++
- **License:** MIT
- **Status:** Active — v4.0.6
- **Features:**
  - Multi-zone spatial optimization
  - Multiple zones with independent costs, contributions, and spatial relationships
  - Supports land/sea use zoning beyond binary reserve/non-reserve
  - Version 4 includes major code refactoring into classes
- **Use case:** Regional planning across marine, freshwater, and terrestrial areas where multiple competing objectives must be balanced (conservation, sustainable use, extraction, etc.)

#### prioritizr (R)
- **Repository:** https://github.com/prioritizr/prioritizr
- **Language:** R
- **License:** GPL-3
- **Status:** Very active
- **Features:**
  - Mixed Integer Linear Programming (MILP) for exact optimal solutions
  - Supports Gurobi, SYMPHONY, CBC, lpsymphony, HiGHS solvers
  - Can read Marxan input files directly via `marxan_problem()`
  - Mathematically guaranteed optimal solutions (vs. SA heuristics)
  - Multi-zone support
  - Finds cheaper solutions faster than Marxan SA
- **Significance:** Represents the state-of-the-art in conservation planning solvers. Its MILP approach consistently outperforms simulated annealing in solution quality and speed. Any modern reimplementation should support similar exact solver backends.

### 2.2 Desktop & Web User Interfaces

#### marxan.io (R/Shiny)
- **Repository:** https://github.com/mattwatts/marxan.io
- **Language:** R (74.6%), JavaScript (24.5%)
- **License:** AGPL-3.0
- **Status:** Legacy (minimal maintenance)
- **Features:**
  - Graphical UI running on Nectar research cloud
  - Upload custom Marxan datasets
  - Edit conservation targets, SPF, and BLM parameters
  - Parameter testing and comparative analysis
  - Output visualization (maps, figures, tables)
  - Dataset publishing for public access
- **Architecture:** Single R Shiny application with multiple versioned iterations (marxan5 through marxan24), user authentication module, and data upload functionality.
- **Limitations:** Monolithic structure, R-only, difficult to extend or decompose.

#### marxanui (R Package)
- **Repository:** https://github.com/mattwatts/marxanui
- **Language:** R (80.3%), JavaScript (19.4%)
- **License:** Not specified
- **Status:** Stale — v0.1 (June 2016, 4 commits)
- **Features:**
  - Desktop Marxan UI for Windows, Mac, Linux
  - Import datasets, run Marxan, edit parameters
  - Parameter testing, output visualization
  - Derived from marxan.io web apps
- **Limitations:** Unmaintained since 2016, limited to R ecosystem.

#### marxanr (R Package)
- **Repository:** https://github.com/remi-daigle/marxanr
- **Language:** R
- **Status:** Active
- **Features:**
  - Prepare Marxan input files programmatically in R
  - Download Marxan binary
  - Run Marxan from R
  - Write parameter files
- **Significance:** Clean R wrapper, but no UI. Demonstrates the pattern of wrapping the C++ binary from a higher-level language.

### 2.3 Web Platforms

#### Marxan Web
- **Repository:** https://github.com/marxanweb/general (hub), plus marxan-server and marxan-client
- **Language:** Python (63.5%), JavaScript, HTML, CSS
- **License:** EUPL-1.2
- **Status:** Maintained
- **Architecture:**
  - **Server:** PostGIS database + Tornado web server + Marxan binary + custom Python backend
  - **Client:** React web application + MapboxGL mapping + Vector Tile services
  - Communication via REST services and WebSockets
  - Server registry system for discovering Marxan Server instances
- **Features:**
  - Full web-based conservation planning
  - Multi-user with access control
  - Spatial data management via PostGIS
  - Interactive mapping via MapboxGL
  - Windows installer and hosted deployment options
- **Limitations:** Tied to Tornado/React stack, not easily decomposable into reusable Python modules.

#### Marxan Cloud / MaPP (Marxan Planning Platform)
- **Repository:** https://github.com/Vizzuality/marxan-cloud
- **Language:** TypeScript/Node.js
- **License:** MIT
- **Status:** Active (Vizzuality + The Nature Conservancy + Microsoft)
- **Architecture:**
  - Monorepo with 4+ microservices
  - **Frontend:** React web application (port 3000)
  - **Core API:** RESTful backend for platform operations
  - **Geoprocessing Service:** Spatial computation microservice
  - **Webshot Service:** HTML to PDF/PNG report generation
  - PostgreSQL v14 + PostGIS v3, Redis v6
  - Docker Compose (local) / Kubernetes (production)
  - Azure cloud infrastructure, Terraform IaC
  - GitHub Actions CI/CD
- **Features:**
  - World's first cloud-hosted Marxan platform
  - Scenario modeling and comparison
  - Multi-user collaboration with organization management
  - GADM administrative boundaries, WDPA protected areas integration
  - Free and open-source
- **Significance:** The most sophisticated Marxan deployment, but requires significant infrastructure (16GB RAM, 40GB disk minimum). Aimed at institutional users, not individual researchers.
- **Limitations:** Heavy infrastructure requirements, TypeScript/Node stack (not Python), not usable as a library.

### 2.4 Connectivity Tools

#### Marxan Connect (GUI)
- **Repository:** https://github.com/remi-daigle/MarxanConnect
- **Language:** Python (GUI application)
- **License:** MIT
- **Status:** Active
- **Features:**
  - Graphical UI for connectivity-informed conservation planning
  - Calculates connectivity metrics as conservation features
  - Prepares connectivity as spatial dependencies
  - Supports demographic and landscape connectivity
  - Exports `.MarCon` project files (JSON format)
- **Architecture:** Python desktop GUI that wraps the `marxanconpy` library.

#### marxanconpy (Python Library)
- **Repository:** https://github.com/remi-daigle/marxanconpy
- **PyPI:** https://pypi.org/project/marxanconpy/
- **Language:** Python
- **License:** MIT
- **Status:** Active — v0.1.2
- **Features:**
  - Command-line interface to all Marxan Connect functionality
  - Connectivity metrics calculation
  - Spatial analysis and post-hoc evaluation
  - Project management (create, load, save `.MarCon` files)
  - Data format conversion (Probability, Migration, Flow matrices)
- **Significance:** The closest existing Python library to what we propose, but focused only on connectivity — not the full Marxan workflow.

### 2.5 GIS Plugins

#### CLUZ (QGIS Plugin)
- **Website:** https://cluz-systematic-conservation-planning.github.io/
- **Language:** Python
- **Status:** Active (CLUZ3)
- **Features:**
  - QGIS plug-in for designing conservation area networks
  - On-screen interactive planning
  - Links to Marxan and Marxan with Zones executables
  - Conservation land-use zoning

#### QMarxan Toolbox (QGIS Plugin)
- **Website:** https://www.aproposinfosystems.com/en/solutions/qgis-plugins/qmarxan-toolbox/
- **Language:** Python
- **Status:** Active
- **Features:**
  - 9 tools in 4 groups for QGIS processing framework
  - Create planning grids
  - Measure conservation features
  - Generate all Marxan input files
  - Import results for visualization
  - Edit input.dat without external tools
  - BLM calibration and SPF adjustment

---

## 3. Gap Analysis

### What Exists
| Capability | R Ecosystem | Python Ecosystem | Web Platforms |
|---|---|---|---|
| Core SA solver | marxanr (wraps binary) | None (binary only) | Marxan Web, MaPP |
| Exact MIP solver | prioritizr | None | None |
| Multi-zone support | prioritizr, marxanui | None | MaPP |
| Connectivity analysis | None | marxanconpy | None |
| Interactive UI | marxan.io, marxanui | Marxan Connect (desktop) | Marxan Web, MaPP |
| Calibration tools | marxan.io | None | MaPP |
| Headless library API | prioritizr, marxanr | marxanconpy (connectivity only) | MaPP API |
| Spatial visualization | marxan.io (Leaflet) | CLUZ, QMarxan (QGIS) | MaPP (Mapbox) |

### What's Missing: The Python Gap

1. **No Python-native conservation problem solver** — All solving is done via the C++ binary or R packages
2. **No Python Shiny UI for Marxan** — The most modern Python UI framework (Shiny for Python) is unused
3. **No unified Python library** covering data prep + solving + calibration + analysis
4. **No Python MIP solver integration** for conservation planning (prioritizr's advantage is R-only)
5. **Connectivity is isolated** — marxanconpy exists but isn't integrated into a full workflow
6. **No modular, composable system** — All existing UIs are monolithic

### Opportunity

Python has surpassed R as the most-used language in data science and spatial analysis. Libraries like `geopandas`, `shapely`, `PuLP`, `OR-Tools`, and `scipy` provide all the building blocks for a comprehensive conservation planning toolkit. Shiny for Python (by Posit) brings the same reactive UI paradigm that made marxan.io successful to the Python ecosystem.

---

## 4. Proposed Architecture: `pymarxan`

### 4.1 Three-Layer Design

```
┌─────────────────────────────────────────────────────────┐
│                    pymarxan-app                          │
│         Assembled Shiny Application (Pages)             │
│   Home │ Data │ Configure │ Calibrate │ Run │ Results   │
├─────────────────────────────────────────────────────────┤
│                   pymarxan-shiny                        │
│            Reusable Shiny UI Modules                    │
│  data_input │ solver_config │ calibration │ mapping │   │
│  connectivity │ results │ run_control                   │
├─────────────────────────────────────────────────────────┤
│                   pymarxan-core                         │
│             Pure Python Library (No UI)                 │
│  io │ models │ solvers │ connectivity │ calibration │   │
│  analysis │ utils                                       │
└─────────────────────────────────────────────────────────┘
```

**Design principle:** Each layer only depends on the layers below it. `pymarxan-core` has zero UI dependencies and can be used standalone by developers and in scripts.

### 4.2 Layer 1: `pymarxan-core` — Pure Python Library

#### 4.2.1 `io` — Data Input/Output

```
pymarxan/io/
├── readers.py         # Read pu.dat, spec.dat, puvspr.dat, bound.dat, input.dat
├── writers.py         # Write Marxan-format .dat files
├── spatial.py         # GeoPackage/Shapefile ↔ PlanningUnit conversion (geopandas)
└── project.py         # Load/save complete Marxan project directories
```

**Responsibilities:**
- Parse all standard Marxan input file formats (`.dat` files with comma/tab delimiters)
- Read/write spatial data formats (Shapefile, GeoPackage, GeoJSON) via `geopandas`
- Convert between spatial geometries and planning unit data frames
- Manage complete project directories (input folder, output folder, input.dat configuration)
- Validate file schemas and report clear error messages

#### 4.2.2 `models` — Domain Models

```
pymarxan/models/
├── planning_unit.py   # PlanningUnit, PlanningUnitSet
├── feature.py         # ConservationFeature, Target
├── boundary.py        # BoundaryMatrix (sparse), connectivity weights
├── zone.py            # Zone, ZoneContribution, ZoneCost (for MarZone)
└── problem.py         # ConservationProblem — top-level container
```

**Key classes:**
- `ConservationProblem`: The central object that ties planning units, features, boundaries, targets, and solver configuration together. Analogous to `prioritizr::problem()`.
- `PlanningUnit`: Represents a spatial unit with id, cost, status (locked in/out/available), and optional geometry.
- `ConservationFeature`: A biodiversity feature with target amount, SPF, and name.
- `BoundaryMatrix`: Sparse representation of shared boundaries between planning units.
- `Zone`: Definition of a management zone with associated costs and feature contributions (for multi-zone problems).

#### 4.2.3 `solvers` — Optimization Engines

```
pymarxan/solvers/
├── base.py                    # Abstract Solver interface
├── marxan_binary.py           # Wraps C++ Marxan/MarZone executable
├── simulated_annealing.py     # Native Python SA (NumPy-accelerated)
├── mip_solver.py              # Exact solver via PuLP / OR-Tools / Gurobi
├── heuristic.py               # Greedy and iterative heuristics
└── zones.py                   # Multi-zone solver (MarZone equivalent)
```

**Abstract interface (`base.py`):**
```python
class Solver(ABC):
    @abstractmethod
    def solve(self, problem: ConservationProblem) -> Solution: ...

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def supports_zones(self) -> bool: ...
```

**Solver implementations:**

| Solver | Method | Optimality | Speed | Dependencies |
|---|---|---|---|---|
| `MarxanBinarySolver` | Wraps C++ executable | Heuristic | Fast (C++) | Marxan binary on PATH |
| `SimulatedAnnealingSolver` | Native Python SA | Heuristic | Medium (NumPy) | numpy, scipy |
| `MIPSolver` | Integer Linear Programming | Exact optimal | Variable | PuLP + CBC/HiGHS (free) or Gurobi (commercial) |
| `HeuristicSolver` | Greedy/iterative | Heuristic | Very fast | numpy |
| `ZoneSolver` | Multi-zone optimization | Heuristic/Exact | Variable | Depends on backend |

The `MIPSolver` formulates the Marxan minimum-set problem as:

```
Minimize: Σ(cost_i × x_i) + BLM × Σ(boundary_ij × |x_i - x_j|)
Subject to: Σ(amount_ij × x_i) ≥ target_j  for all features j
            x_i ∈ {0, 1}
```

This is equivalent to what prioritizr does in R, brought to the Python ecosystem.

#### 4.2.4 `connectivity` — Marxan Connect Equivalent

```
pymarxan/connectivity/
├── metrics.py         # Connectivity metrics (betweenness, eigenvector, etc.)
├── landscape.py       # Landscape connectivity (resistance surfaces, corridors)
├── demographic.py     # Demographic connectivity (migration, dispersal)
└── spatial_deps.py    # Generate spatial dependency matrices for solvers
```

**Responsibilities:**
- Calculate graph-theoretic connectivity metrics using `networkx`
- Generate connectivity matrices from movement/dispersal data
- Convert connectivity to Marxan-compatible boundary/spatial dependency formats
- Support both landscape (structural) and demographic (functional) connectivity
- Interoperate with `marxanconpy` `.MarCon` project files

#### 4.2.5 `calibration` — Parameter Calibration

```
pymarxan/calibration/
├── blm.py             # Boundary Length Modifier calibration
├── spf.py             # Species Penalty Factor calibration
├── sensitivity.py     # Target sensitivity analysis
└── parameter_sweep.py # General n-dimensional parameter exploration
```

**BLM calibration** runs the solver at multiple BLM values and plots cost vs. boundary length to find the "elbow" where increasing BLM yields diminishing fragmentation reduction. This is a critical workflow from marxan.io that must be preserved.

**SPF calibration** adjusts species penalty factors to ensure all conservation targets are met. Iterative process: run → check unmet targets → increase SPF for unmet features → re-run.

#### 4.2.6 `analysis` — Post-Hoc Analysis

```
pymarxan/analysis/
├── selection_freq.py    # Selection frequency across multiple runs
├── irreplaceability.py  # Irreplaceability indices
├── gap_analysis.py      # Current protection gap analysis
└── comparison.py        # Scenario comparison metrics
```

### 4.3 Layer 2: `pymarxan-shiny` — Reusable UI Modules

Each module is a self-contained Shiny module (with `_ui()` and `_server()` functions) that can be used independently or composed into larger applications.

#### Module Inventory

| Module | Purpose | Core Dependency |
|---|---|---|
| `data_input.upload` | File upload widget (CSV, Shapefile, GeoPackage) | `pymarxan.io` |
| `data_input.spatial_grid` | Interactive planning grid editor on map | `pymarxan.models`, ipyleaflet |
| `data_input.feature_table` | Editable conservation feature table with targets | `pymarxan.models` |
| `solver_config.solver_picker` | Solver selection (SA / MIP / Binary / Heuristic) | `pymarxan.solvers` |
| `solver_config.sa_params` | SA parameter sliders (iterations, cooling schedule) | `pymarxan.solvers.simulated_annealing` |
| `solver_config.mip_params` | MIP parameter controls (gap, time limit, solver backend) | `pymarxan.solvers.mip_solver` |
| `solver_config.zones_config` | Multi-zone definition and configuration | `pymarxan.models.zone` |
| `calibration.blm_explorer` | Interactive BLM calibration with live cost-vs-boundary plot | `pymarxan.calibration.blm` |
| `calibration.spf_explorer` | SPF calibration with target achievement indicators | `pymarxan.calibration.spf` |
| `calibration.sensitivity` | Target sensitivity analysis dashboard | `pymarxan.calibration.sensitivity` |
| `connectivity.matrix_input` | Connectivity matrix upload and editor | `pymarxan.connectivity` |
| `connectivity.metrics_viz` | Connectivity metrics visualization (bar charts, network) | `pymarxan.connectivity.metrics` |
| `connectivity.network_view` | Interactive network graph visualization | `networkx`, plotly |
| `mapping.solution_map` | Planning unit selection map with legend | ipyleaflet |
| `mapping.frequency_map` | Selection frequency heatmap across runs | ipyleaflet |
| `mapping.comparison_map` | Side-by-side scenario comparison maps | ipyleaflet |
| `results.summary_table` | Solution summary statistics table | `pymarxan.analysis` |
| `results.convergence` | SA convergence plot (objective value over iterations) | plotly |
| `results.target_met` | Target achievement dashboard (met/unmet per feature) | `pymarxan.analysis` |
| `results.export` | Export results (CSV, GeoPackage, PDF report) | `pymarxan.io.writers` |
| `run_control.run_panel` | Start/stop/monitor solver execution | `pymarxan.solvers` |
| `run_control.progress` | Real-time progress bar and log stream | Shiny reactive |

### 4.4 Layer 3: `pymarxan-app` — Assembled Application

The application composes Shiny modules into a multi-page navigation:

```
pymarxan-app/
├── app.py                # Entry point: shiny.App(app_ui, server)
├── pages/
│   ├── home.py           # Project manager: create, load, save projects
│   ├── data.py           # Data preparation: upload spatial data, define features
│   ├── configure.py      # Solver selection, parameter configuration, zones
│   ├── calibrate.py      # BLM, SPF, sensitivity calibration tools
│   ├── connectivity.py   # Connectivity matrix import, metrics, visualization
│   ├── run.py            # Execute solver, progress monitoring, logs
│   ├── results.py        # Maps, tables, charts, target achievement
│   └── compare.py        # Multi-scenario comparison
└── state.py              # Shared application state (reactive values)
```

**Navigation flow:** Home → Data → Configure → (Calibrate) → (Connectivity) → Run → Results → (Compare)

The parenthesized steps are optional — users can go directly from Configure to Run for simple analyses.

---

## 5. Module Specifications

### 5.1 Solver Interface Contract

All solvers implement this interface, allowing seamless backend swapping:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class Solution:
    """Result of a conservation planning optimization."""
    selected: np.ndarray          # Boolean array: is planning unit selected?
    cost: float                   # Total cost of selected units
    boundary: float               # Total boundary length of selection
    objective: float              # Objective function value (cost + BLM * boundary)
    targets_met: dict[int, bool]  # Feature ID → whether target is met
    metadata: dict                # Solver-specific metadata (iterations, time, etc.)

@dataclass
class SolverConfig:
    """Configuration for a solver run."""
    num_solutions: int = 100      # Number of repeat runs
    seed: int | None = None       # Random seed for reproducibility
    verbose: bool = False
    # Subclasses add solver-specific parameters

class Solver(ABC):
    @abstractmethod
    def solve(self, problem: ConservationProblem, config: SolverConfig) -> list[Solution]:
        """Run the solver and return a list of solutions."""
        ...

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def supports_zones(self) -> bool: ...

    def available(self) -> bool:
        """Check if this solver's dependencies are available."""
        return True
```

### 5.2 Conservation Problem Model

```python
@dataclass
class ConservationProblem:
    """Central container for a conservation planning problem."""
    planning_units: gpd.GeoDataFrame  # id, cost, status, geometry
    features: pd.DataFrame             # id, name, target, spf
    pu_vs_features: pd.DataFrame       # pu, species, amount (sparse)
    boundary: pd.DataFrame | None      # id1, id2, boundary (optional)
    zones: list[Zone] | None           # Zone definitions (optional, for MarZone)
    connectivity: pd.DataFrame | None  # Connectivity matrix (optional)
    parameters: dict                   # BLM, num_reps, and other config

    def validate(self) -> list[str]:
        """Return list of validation errors, empty if valid."""
        ...

    def summary(self) -> str:
        """Human-readable problem summary."""
        ...

    @classmethod
    def from_marxan_files(cls, input_dir: str) -> "ConservationProblem":
        """Load from a standard Marxan file directory."""
        ...

    def to_marxan_files(self, output_dir: str) -> None:
        """Export to standard Marxan file format."""
        ...
```

### 5.3 Shiny Module Pattern

Each Shiny module follows the standard Shiny-for-Python module pattern:

```python
# Example: pymarxan_shiny/modules/calibration/blm_explorer.py

from shiny import module, ui, render, reactive

@module.ui
def blm_explorer_ui():
    return ui.card(
        ui.card_header("BLM Calibration"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_slider("blm_min", "Min BLM", 0, 100, 0),
                ui.input_slider("blm_max", "Max BLM", 0, 100, 50),
                ui.input_numeric("blm_steps", "Steps", 10),
                ui.input_action_button("run_calibration", "Run Calibration"),
            ),
            ui.output_plot("blm_plot"),
        ),
    )

@module.server
def blm_explorer_server(input, output, session, problem: reactive.Value):
    # Server logic using pymarxan.calibration.blm
    ...
```

---

## 6. Technology Stack

### 6.1 Core Library Dependencies

| Package | Purpose | Why |
|---|---|---|
| `numpy` | Numerical arrays | Foundation for all matrix operations |
| `pandas` | DataFrames | Tabular data handling (feature tables, solutions) |
| `geopandas` | Spatial DataFrames | Planning unit geometries, spatial I/O |
| `shapely` | Geometry operations | Boundary length calculations, spatial queries |
| `scipy` | Sparse matrices, optimization | Efficient boundary matrix storage, SA helpers |
| `PuLP` | MIP modeling | Free, solver-agnostic LP/MILP formulation |
| `networkx` | Graph analysis | Connectivity metrics calculation |
| `pydantic` | Data validation | Input validation and configuration schemas |

### 6.2 Optional Solver Backends

| Package | Purpose | License |
|---|---|---|
| CBC (via PuLP) | Free MIP solver | EPL-2.0 (open source) |
| HiGHS (via PuLP) | Free MIP solver | MIT |
| OR-Tools | Google's optimization suite | Apache-2.0 |
| Gurobi | Commercial MIP solver | Commercial (free academic) |

### 6.3 UI Dependencies

| Package | Purpose | Why |
|---|---|---|
| `shiny` (Posit) | Web framework | Reactive UI, Python-native, familiar to R/Shiny users |
| `ipyleaflet` | Interactive maps | Best Leaflet binding for Python, works with Shiny |
| `shinywidgets` | Widget integration | Bridge ipyleaflet/plotly into Shiny |
| `plotly` | Interactive charts | Convergence plots, calibration curves, comparison charts |
| `matplotlib` | Static plots | Publication-quality figures for export |

### 6.4 Development & Testing

| Tool | Purpose |
|---|---|
| `pytest` | Unit and integration testing |
| `pytest-cov` | Coverage reporting |
| `mypy` | Static type checking |
| `ruff` | Linting and formatting |
| `hatch` / `uv` | Build system and dependency management |

---

## 7. Implementation Roadmap

### Phase 1: Foundation (MVP)
**Goal:** Working end-to-end flow with core Marxan functionality

1. `pymarxan.io.readers` / `writers` — Read and write all Marxan `.dat` files
2. `pymarxan.models.problem` — `ConservationProblem` data model
3. `pymarxan.solvers.marxan_binary` — Wrap C++ Marxan executable
4. `pymarxan.solvers.mip_solver` — PuLP-based exact solver (minimum-set formulation)
5. `pymarxan-shiny` basic modules: upload, solver picker, solution map, summary table
6. `pymarxan-app` minimal: Data → Configure → Run → Results pages

**Validates:** Full roundtrip from data upload to solution visualization.

### Phase 2: Native Solvers & Calibration
**Goal:** Remove C++ dependency for basic use, add calibration

7. `pymarxan.solvers.simulated_annealing` — Native Python SA
8. `pymarxan.calibration.blm` / `spf` — BLM and SPF calibration
9. Calibration Shiny modules with interactive plots
10. Selection frequency analysis and irreplaceability
11. Results export (CSV, GeoPackage, PDF)

**Validates:** Users can run full analyses without the C++ binary.

### Phase 3: Multi-Zone & Connectivity
**Goal:** MarZone and Marxan Connect equivalents

12. `pymarxan.models.zone` — Zone data model
13. `pymarxan.solvers.zones` — Multi-zone solver
14. `pymarxan.connectivity` — Full connectivity module
15. Zone configuration and connectivity Shiny modules
16. Scenario comparison page

**Validates:** Feature parity with the broader Marxan ecosystem.

### Phase 4: Advanced Features
**Goal:** Exceed current ecosystem capabilities

17. Parameter sweep and sensitivity analysis
18. Batch processing and parallel execution
19. Cloud deployment configuration (Docker, Shinylive)
20. Plugin system for custom solvers and data sources
21. Integration with global datasets (WDPA, GADM)

---

## 8. Design Decisions & Trade-offs

### 8.1 Why Shiny for Python (not Streamlit, Dash, or Panel)?

| Framework | Pros | Cons | Verdict |
|---|---|---|---|
| **Shiny for Python** | Reactive paradigm ideal for parameter exploration; familiar to conservation community (R/Shiny history); module system for composability; Posit backing | Newer, smaller ecosystem than Streamlit | **Selected** — reactivity + module system + conservation community familiarity |
| Streamlit | Popular, easy to start | No module system, re-runs entire script on each interaction, hard to build complex multi-page apps | Not suitable for complex interactive workflows |
| Dash (Plotly) | Powerful, mature | Callback-based (not reactive), verbose, no module system | Too verbose for this many components |
| Panel (HoloViz) | Good for dashboards | Smaller community, less intuitive module composition | Less ecosystem support |

### 8.2 Why Hybrid Solver Strategy?

- **C++ binary wrapper:** Needed for backward compatibility with existing Marxan workflows. Users who already have Marxan installed can continue using it.
- **Native Python SA:** Removes the C++ dependency for environments where installing binaries is difficult (cloud notebooks, Docker, WebAssembly). NumPy-accelerated SA is ~10-50x slower than C++ but acceptable for moderate problem sizes.
- **MIP solver:** Provides mathematically optimal solutions that SA cannot guarantee. Fills the role that prioritizr plays in the R ecosystem. PuLP + CBC is free and open source.

### 8.3 Why Three Packages?

- **Developers** install only `pymarxan-core` to use in scripts, pipelines, and custom apps.
- **Researchers** install `pymarxan-core` + `pymarxan-shiny` to embed specific modules (e.g., just the BLM calibration widget) into their own Shiny apps.
- **Practitioners** install `pymarxan-app` (which depends on both) for the complete experience.

This avoids forcing UI dependencies on headless users while still providing a complete application.

### 8.4 Interoperability with Existing Ecosystem

- **Marxan files:** Full read/write support for all `.dat` formats ensures existing Marxan projects can be loaded directly.
- **marxanconpy:** Import/export `.MarCon` project files for connectivity data.
- **prioritizr:** While we don't call R packages directly, the MIP formulation is equivalent, so solutions are comparable.
- **QGIS:** GeoPackage export allows results to be opened in QGIS for further analysis with CLUZ or QMarxan Toolbox.

---

## References

1. Ball, I.R., Possingham, H.P., and Watts, M. (2009). *Marxan and relatives: Software for spatial conservation prioritisation.* Chapter 14 in Spatial conservation prioritisation: Quantitative methods and computational tools.
2. Watts, M.E., et al. (2009). *Marxan with Zones: Software for optimal conservation based land- and sea-use zoning.* Environmental Modelling & Software, 24(12), 1513-1521.
3. Hanson, J.O., et al. (2024). *Systematic conservation prioritization with the prioritizr R package.* Conservation Biology.
4. Daigle, R.M., et al. (2020). *Operationalizing ecological connectivity in spatial conservation planning with Marxan Connect.* Methods in Ecology and Evolution, 11(4), 570-579.

### Source Repositories

- Marxan v4: https://github.com/Marxan-source-code/marxan
- MarZone v4: https://github.com/Marxan-source-code/marzone
- marxan.io: https://github.com/mattwatts/marxan.io
- marxanui: https://github.com/mattwatts/marxanui
- Marxan Web: https://github.com/marxanweb/general
- Marxan Cloud: https://github.com/Vizzuality/marxan-cloud
- Marxan Connect: https://github.com/remi-daigle/MarxanConnect
- marxanconpy: https://github.com/remi-daigle/marxanconpy
- marxanr: https://github.com/remi-daigle/marxanr
- prioritizr: https://github.com/prioritizr/prioritizr
- CLUZ: https://cluz-systematic-conservation-planning.github.io/
- Shiny for Python: https://shiny.posit.co/py/
- Marxan Solutions: https://marxansolutions.org/
- Marxan Planning Platform: https://marxanplanning.org/
