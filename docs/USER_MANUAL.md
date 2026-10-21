# pyMarxan User Manual

**Version 0.1.0**

A comprehensive guide to using the pyMarxan Shiny application for systematic conservation planning.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Application Overview](#application-overview)
4. [Data Input](#data-input)
   - [Upload Project](#upload-project)
   - [Import GIS File](#import-gis-file)
   - [Generate Grid](#generate-grid)
5. [Spatial Data](#spatial-data)
   - [GADM Boundaries](#gadm-boundaries)
   - [Protected Areas (WDPA)](#protected-areas-wdpa)
   - [Cost Surface](#cost-surface)
6. [Features](#features)
7. [Connectivity](#connectivity)
   - [Matrix Input](#matrix-input)
   - [Connectivity Metrics](#connectivity-metrics)
8. [Configure](#configure)
   - [Solver Selection](#solver-selection)
   - [Solver Parameters](#solver-parameters)
   - [Probability (PROBMODE)](#probability-probmode)
   - [Partial Selection](#partial-selection)
   - [Zone Configuration](#zone-configuration)
9. [Calibrate](#calibrate)
   - [BLM Calibration](#blm-calibration)
   - [Target Sensitivity](#target-sensitivity)
   - [SPF Calibration](#spf-calibration)
   - [Parameter Sweep](#parameter-sweep)
10. [Run Solver](#run-solver)
11. [Maps](#maps)
    - [Planning Unit Map](#planning-unit-map)
    - [Solution Map](#solution-map)
    - [Selection Frequency](#selection-frequency)
    - [Solution Comparison](#solution-comparison)
    - [Connectivity Network](#connectivity-network)
12. [Results](#results)
    - [Summary Table](#summary-table)
    - [Target Achievement](#target-achievement)
    - [SA Convergence](#sa-convergence)
    - [Scenario Comparison](#scenario-comparison)
    - [Export](#export)
13. [Marxan Concepts Glossary](#marxan-concepts-glossary)
14. [Troubleshooting](#troubleshooting)

---

## Introduction

pyMarxan is a Python-based web application for **systematic conservation planning** using the Marxan framework. It provides an interactive interface for:

- Loading and creating Marxan conservation problems
- Configuring and running multiple solver algorithms
- Calibrating key parameters (BLM, SPF)
- Visualising results on interactive maps
- Comparing scenarios and exporting results

### What is Marxan?

Marxan is the world's most widely used conservation planning software. It solves the **minimum set problem**: find the smallest-cost set of planning units that meets all conservation feature targets while optionally promoting spatial compactness (via the Boundary Length Modifier).

The **objective function** that Marxan minimises is:

```
Objective = Σ(cost × selected) + BLM × Σ(boundary × selected) + Σ(SPF × shortfall)
```

Where:
- **Cost**: the cost of including each planning unit
- **BLM**: Boundary Length Modifier — controls compactness
- **SPF**: Species Penalty Factor — penalises unmet targets
- **Shortfall**: how much a feature falls short of its target

---

## Getting Started

### Prerequisites

- Python 3.11+
- Required packages: `shiny`, `pandas`, `numpy`, `pulp`
- Optional: `geopandas`, `shapely`, `ipyleaflet`, `shinywidgets`, `plotly`, `matplotlib`

### Installation

```bash
pip install -e ".[all]"
```

### Running the Application

```bash
cd src/pymarxan_shiny
shiny run app.py
```

The application opens in your browser at `http://localhost:8000`.

### Quick Start Workflow

1. **Data Input** → Upload a Marxan project ZIP or browse to a project directory
2. **Features** → Review and edit conservation targets and SPF values
3. **Configure** → Select a solver and set parameters (BLM, number of solutions)
4. **Run** → Click "Run Solver" and wait for completion
5. **Results** → Inspect target achievement, maps, and export CSVs

---

## Application Overview

The application is organised into tabs along the top navigation bar:

| Tab | Purpose |
|-----|---------|
| **Data Input** | Load or create a Marxan conservation problem |
| **Spatial** | Fetch boundaries, overlay protected areas, apply cost surfaces |
| **Features** | Edit conservation feature targets and penalty factors |
| **Connectivity** | Upload and analyse connectivity matrices |
| **Configure** | Select solver algorithm and set parameters |
| **Calibrate** | Tune BLM, SPF, and other parameters |
| **Run** | Execute the solver with progress monitoring |
| **Maps** | Interactive maps of planning units, solutions, and networks |
| **Results** | Summary tables, convergence plots, scenario comparison, export |

---

## Data Input

### Upload Project

Load an existing Marxan project from a ZIP archive or a server directory.

**A valid Marxan project contains:**
- `input.dat` — configuration file specifying input file paths and parameters
- `pu.dat` — planning unit data (ID, cost, status)
- `spec.dat` — conservation feature specifications (ID, name, target, SPF)
- `puvspr.dat` — planning unit vs. species matrix (which features occur in which PUs)
- `bound.dat` (optional) — boundary length data between adjacent PUs

**Upload a ZIP:**
Upload a `.zip` file containing the complete project directory. The application will extract it, locate `input.dat`, and load the project.

**Browse server directory:**
Use the directory browser to navigate to a Marxan project folder on the server. The browser shows a badge when it detects a valid project directory (contains `input.dat`).

### Import GIS File

Import planning units directly from a geospatial file instead of loading a pre-built Marxan project.

**Supported formats:** Shapefile (`.shp`), GeoJSON (`.geojson`), GeoPackage (`.gpkg`)

**Column mapping:**
- **ID Column** — unique identifier for each planning unit (corresponds to Marxan's `PUID`)
- **Cost Column** — numeric cost of including each PU (e.g. area, land price, opportunity cost)
- **Status Column** (optional) — Marxan status codes:
  - `0` = available (can be selected or not)
  - `1` = initial include (starts in the solution but can be removed)
  - `2` = locked in (must be selected)
  - `3` = locked out (cannot be selected)

After import, adjacency boundaries are automatically computed from polygon geometry.

### Generate Grid

Create a regular grid of planning units over a geographic bounding box.

**Parameters:**
- **Min X / Max X (longitude)**: Western and eastern bounds in decimal degrees
- **Min Y / Max Y (latitude)**: Southern and northern bounds in decimal degrees
- **Cell Size**: Width/height of each cell in coordinate units (degrees for geographic CRS)
- **Grid Type**: Square or hexagonal cells. Hexagonal grids reduce edge effects and provide more uniform neighbour distances.
- **Clip to GADM**: If checked, cells outside the GADM boundary are removed

---

## Spatial Data

### GADM Boundaries

Fetch administrative boundaries from the **Global Administrative Areas (GADM)** database.

- **Country**: Select the target country by ISO3 code
- **Admin Level**:
  - ADM0 = entire country border
  - ADM1 = state/province boundaries
  - ADM2 = district/county boundaries
- **Region Name Filter**: Optionally filter to a specific region name (e.g. "California")

Fetched boundaries can be used to clip the planning unit grid.

### Protected Areas (WDPA)

Overlay existing protected areas from the **World Database on Protected Areas** onto your planning units.

- **API Token**: Optional Protected Planet API token (a public endpoint is used if blank)
- **Overlap Threshold**: Minimum fraction (0.1–1.0) of a PU's area that must overlap a protected area. Default 0.5 = at least 50% overlap required.
- **Set Status**: Choose the Marxan status to assign:
  - **Locked In (2)**: PU must be in the reserve (reflects existing protection)
  - **Initial Include (1)**: PU starts in the solution but can be removed by the solver

### Cost Surface

Upload a custom cost layer to replace default planning unit costs.

- **Cost File**: Vector layer (Shapefile, GeoJSON, GeoPackage) with a numeric cost attribute
- **Cost Column**: Column containing the cost values
- **Aggregation Method**: How to handle PUs overlapping multiple cost polygons:
  - **Area-Weighted Mean** (recommended): Weighted average by intersection area
  - **Sum**: Total cost from all overlapping polygons
  - **Maximum**: Highest cost from any overlapping polygon

---

## Features

The Features tab displays an editable table of all conservation features in the project.

**Columns:**
- **ID**: Feature identifier
- **Name**: Feature name (species, habitat type, etc.)
- **Target**: Minimum amount of this feature that must be captured in the reserve. This is specified in the same units as the `puvspr.dat` amounts (e.g. area in km², proportion, count).
- **SPF** (Species Penalty Factor): Controls how heavily Marxan penalises solutions that fail to meet this feature's target. Higher SPF = stronger incentive to meet the target. Default is 1.0.

**How to edit:** Click any cell in the target or SPF column to change its value. Click **Save Changes** to apply edits to the active conservation problem. Changes are immediate for the next solver run.

**Tips:**
- Set SPF > 1 for high-priority features that must be met
- Use the SPF Calibration tool to automatically tune SPF values
- Target = 0 means the feature has no conservation requirement

---

## Connectivity

### Matrix Input

Upload a connectivity matrix describing spatial relationships between planning units. Marxan uses connectivity in the boundary term of the objective function.

**Two formats supported:**

1. **Edge List**: A CSV/TSV file with three columns:
   - `id1` — first planning unit ID
   - `id2` — second planning unit ID
   - `value` — connection strength/weight

2. **Full Matrix**: A square N×N CSV where entry (i, j) represents the connectivity between PU i and PU j

After upload, the matrix shape, non-zero count, and density are displayed.

**How connectivity enters the objective** (scaled by `CONNECTIVITY_WEIGHT`; a weight of 0 or
below switches it off):

- **Symmetric** (default): a reward of `v` for each connected pair with both units selected.
  A pair listed twice, as `(i, j)` and `(j, i)` (as in a full matrix), counts once: the first
  row wins.
- **Asymmetric** (`ASYMMETRIC_CONNECTIVITY = 1`): a penalty of `v` for each row whose source
  `id1` is selected without its sink `id2`. Repeated `(id1, id2)` rows add up.

These follow prioritizr's `add_connectivity_penalties()` and
`add_asym_connectivity_penalties()` (each symmetric pair counted once, as prioritizr does by
using one triangle of the matrix). Edge cases differ: for a pair given twice with different
values prioritizr keeps the `id1 > id2` row and sums repeated rows, where pymarxan keeps the
first row; prioritizr treats an asymmetric self-row `(i, i)` as a reward for selecting `i`,
where pymarxan, like Marxan's `ASYMMETRICCONNECTIVITY`, ignores it; and rows naming unknown
planning units are skipped here but are an error in prioritizr. For clean input (no
conflicting reversed rows, no repeated symmetric rows, no self-rows, no unknown ids) the terms
are the same.

pymarxan's MIP solver, simulated annealing and iterative improvement optimise them. Every other
solver — the greedy heuristics (HEURTYPE 0–7), the Zonation solver and the external Marxan
executable (which has no `CONNECTIVITY_WEIGHT`) — only reports them in the objective.

This term is separate from Marxan's own way of using connectivity, which Marxan Connect also
follows (Beger et al. 2010; Daigle et al. 2020): connection strengths written into the boundary
file and weighted by `BLM`, penalising a unit selected without the units it is connected to.
That boundary-based connectivity has always been optimised by simulated annealing. pymarxan
does not read Marxan's `ASYMMETRICCONNECTIVITY` option; its asymmetric term has the same form
(source selected without sink, self-rows ignored, repeated rows added) but applies to the
connectivity table and `CONNECTIVITY_WEIGHT`.

### Connectivity Metrics

Compute graph-theoretic metrics from the loaded connectivity matrix:

| Metric | Description |
|--------|-------------|
| **In-Degree** | Number of incoming connections (sum of column) |
| **Out-Degree** | Number of outgoing connections (sum of row) |
| **Betweenness Centrality** | How often a node lies on shortest paths between other nodes — identifies corridor bottlenecks |
| **Eigenvector Centrality** | How connected a node is to other well-connected nodes — identifies hubs |

These metrics help identify critical planning units for maintaining ecological connectivity.

---

## Configure

### Solver Selection

pyMarxan supports ten solver algorithms:

| Solver | Type | Zones | Description |
|--------|------|-------|-------------|
| **MIP (PuLP/CBC)** | Exact | ✗ | Mixed Integer Linear Programming. Guaranteed optimal solution. Best for small–medium problems. |
| **Zone MIP** | Exact | ✓ | Multi-zone MIP with zone costs, contributions, and targets. |
| **Simulated Annealing (Python)** | Heuristic | ✗ | Native Python SA with 4 cooling schedules (adaptive, geometric, linear, logarithmic). |
| **Zone SA** | Heuristic | ✓ | Simulated annealing for multi-zone problems. Each PU is assigned to a zone. |
| **Greedy Heuristic** | Heuristic | ✗ | Selects PUs one-by-one based on a scoring strategy. Very fast baseline. |
| **Zone Heuristic** | Heuristic | ✓ | Greedy zone assignment minimizing zone objective. |
| **Iterative Improvement** | Heuristic | ✗ | Refines an existing solution by trying removals, additions, or swaps. |
| **Zone II** | Heuristic | ✓ | Zone-aware iterative improvement with all 4 ITIMPTYPE modes. |
| **Marxan C++ Binary** | Heuristic | ✗ | Wraps the original Marxan executable. Requires the binary to be installed. |
| **Pipeline** | Hybrid | ✓ | Chains heuristic, SA, and iterative improvement in RUNMODE sequences (auto-selects zone solvers). |

### Solver Parameters

**Common parameters (all solvers):**

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| **BLM** (Boundary Length Modifier) | Controls cost-vs-compactness tradeoff. BLM=0 ignores boundaries; higher BLM favours clumped reserves. | Use BLM calibration to find the elbow |
| **Number of Solutions** | Independent solver runs. Multiple solutions reveal selection frequency. | 10–100 for SA; 1 for MIP |
| **Random Seed** | For reproducibility. Leave blank or -1 for random. | — |

**SA parameters (SA, Binary, Zone SA):**

| Parameter | Marxan Name | Description |
|-----------|------------|-------------|
| **SA Iterations** | `NUMITNS` | Total iterations per run. More = better exploration. Typical: 1,000,000 |
| **Temperature Steps** | `NUMTEMP` | Number of cooling steps. Controls cooling schedule granularity. Typical: 10,000 |

**MIP parameters:**

| Parameter | Description |
|-----------|-------------|
| **Time Limit** | Maximum seconds before returning best-so-far solution |
| **Optimality Gap** | Acceptable gap from proven optimum (0.0 = exact; 0.01 = within 1%) |
| **Verbose** | Print solver progress to console |
| **Decisions** | `binary` (default), `proportion` or `semicontinuous` — see [Partial Selection](#partial-selection) |
| **Upper bound** | Largest share of a unit that can be selected (non-binary decisions) |
| **Lower bound** | Smallest share of a unit that is selected at all (semicontinuous) |

**Greedy parameters:**

| HEURTYPE | Strategy |
|----------|----------|
| 0 | Richness — most features |
| 1 | Greedy — cheapest PU |
| 2 | Max Rarity — most irreplaceable first (default) |
| 3 | Best Rarity/Cost ratio |
| 4 | Average Rarity |
| 5 | Sum Rarity |
| 6 | Product Irreplaceability |
| 7 | Summation Irreplaceability |

**Iterative Improvement parameters:**

| ITIMPTYPE | Strategy |
|-----------|----------|
| 0 | No improvement |
| 1 | Removal pass — try removing each selected PU |
| 2 | Two-step — remove then add pass |
| 3 | Swap — try swapping selected/unselected pairs |

**Pipeline parameters (RUNMODE):**

| Mode | Sequence |
|------|----------|
| 0 | SA only (classic Marxan default) |
| 1 | Heuristic only |
| 2 | SA + iterative improvement |
| 3 | Heuristic + iterative improvement |
| 4 | Heuristic + SA (pick best) |
| 5 | Heuristic + SA + improvement (highest quality) |
| 6 | Iterative improvement only |

### Probability (PROBMODE)

A probability table (`prob.dat`: columns `pu`, `probability`) gives, for each planning unit,
the probability that the unit (and the features in it) is **lost** over the planning horizon;
persistence is 1 − probability. It is attached automatically when the project folder has one.
`PROBMODE` chooses how it is used; with a table and no `PROBMODE`, mode 1 applies.

| PROBMODE | Meaning | Origin |
|----------|---------|--------|
| 0 | Probabilities ignored | — |
| 1 (default with a table) | Risk premium `PROBABILITYWEIGHTING × Σ probability × cost` over selected units, added to the objective (not to the reported cost; off when the weighting is 0 or below) — the expected cost of protecting units that are then lost; it does not depend on the features a unit holds, and a zero-cost unit carries no premium | pymarxan extension |
| 2 | Expected amounts: every feature amount counts as `amount × (1 − probability)`, and targets, shortfalls, penalties, representation and gap analysis are judged on them | pymarxan extension |
| 3 | Z-score probability penalty (a soft chance target, not a hard constraint) from a per-cell probability of **presence** (a `prob` column in `puvspr.dat`) and a per-feature `ptarget` in spec.dat (Marxan's column is `ptarget2d`) | Based on Marxan's Probability 2D, except that a feature with zero variance is treated as met (Marxan 3 treats it as failing: Z = 4) |

Modes 1 and 2 have the same meaning in every single-zone solver: the MIP, simulated annealing and
iterative improvement optimise them, the greedy heuristics use them in their scores (mode 1 as an
effective cost, mode 2 as scaled amounts), and every reported value (`Solution.objective`, targets
met, the summary and target-met tables) uses them. A unit missing from the table has probability 0;
a unit listed twice uses its last row. `validate()` reports duplicates, values outside [0, 1] and
non-finite values (a blank cell reads as NaN) in every mode. Wherever the table is read — mode 1
with `PROBABILITYWEIGHTING` above 0, and mode 2 — every single-zone solver and the evaluator raise
`ValueError` on a non-finite probability; modes 0 and 3, and mode 1 with a weighting of 0 or below,
ignore the table. The zone solvers ignore the probability table.

Marxan's own **Probability 1D** (a `prob` column in `pu.dat`, a Z-score on the expected
amount `Σ amount × (1 − p)` with variance `Σ amount² × p × (1 − p)` against `ptarget1d`) is
**not yet implemented**; mode 2 uses the same expected amount but as a deterministic target,
without the variance or the chance constraint. Meeting a target in expectation is not the same
as meeting it reliably: when the expected amount just equals the target, the chance of actually
retaining the target amount is only about 50% (higher if the reserve holds more, lower with few
high-risk units). Mode 2 also ignores how the risk is spread — one unit at p = 0.5 and ten units
at p = 0.05 with the same expected amount count the same. Use it as a risk-discounted
representation target; for a stated confidence level use a chance formulation (Watts et al.
2021). Until Probability 1D is implemented, it can be reproduced with PROBMODE 3 by writing
`prob = 1 − p` (the unit's persistence) on every puvspr.dat row of that unit and setting
`ptarget` (same mean and variance; zero-variance features differ as noted above).

Under PROBMODE 2 some paths still use raw amounts: clumping occupancy (`target2`), separation
occurrence and space-target weights (simulated annealing and iterative improvement warn when
PROBMODE 2 meets active clumping or separation — once per SA solve, once per solution
iterative improvement improves: they optimise those features on raw amounts while reports use
expected amounts); the MIP presolve reachability check; irreplaceability and Ferrier importance; and
relative (`prop`) target resolution, which is measured against raw feature totals — so a
relative target under mode 2 asks for a share of the raw total out of expected amounts, and is
infeasible when `prop` exceeds `Σ_i a_ij(1 − p_i) / Σ_i a_ij` (one minus the amount-weighted
mean loss probability).

Reference: Watts, M., Klein, C. J., Tulloch, V., et al. (2021). Software for prioritizing conservation actions based on probabilistic information.
*Conservation Biology*, 35(4), 1299–1308. https://doi.org/10.1111/cobi.13681

### Partial Selection

By default the MIP selects whole planning units (`decisions="binary"`). Two other decision
types let the plan take **part of a unit** — protecting or restoring a share of its area,
or committing a share of management effort, e.g. restoration shares or land-use fractions.
`x` is that share of the unit's area (or effort), between 0 and 1. If you mean protection
that is only partly *effective* across a whole unit, use zones with a zone contribution
instead (`ZoneMIPSolver`).

| Decisions | Each unit | Model |
|-----------|-----------|-------|
| `binary` | selected or not | integer program |
| `proportion` | any share `0 ≤ x ≤ upper` | linear program (fast) unless the objective counts targets met |
| `semicontinuous` | `x = 0` or `lower ≤ x ≤ upper` | integer program (one on/off variable per free unit; locked units have none) |

```python
from pymarxan.io.readers import load_project
from pymarxan.solvers import MIPSolver, binarize_solution

problem = load_project("tests/data/simple")

sol = MIPSolver(decisions="proportion", upper=0.8).solve(problem)[0]
print(sol.selected)         # float64 shares in [0, 0.8]
print(sol.all_targets_met)  # True: the shares meet every target
print(sol.selected_sum)     # Σ x; sol.n_selected counts units with x > 0

plan = binarize_solution(problem, sol, threshold=0.5)
print(plan.all_targets_met)  # False: rounding the shares is not a plan
```

`binarize_solution` converts shares to a 0/1 plan by threshold; it does not optimise. The
result can miss targets the shares met (here it misses all three), the exact binary target
rule is re-applied (so a target met within the fractional tolerance can flip to unmet), and a
threshold above `upper` selects nothing. For a feasible whole-unit plan, solve with
`decisions="binary"`, or with `decisions="semicontinuous"` for "nothing, or at least `lower`
of a unit".

**prioritizr mapping.** `add_proportion_decisions()` is `decisions="proportion"`.
prioritizr's `add_semicontinuous_decisions(u)` has no lower bound — it is
`decisions="proportion", upper=u`. `decisions="semicontinuous"` is a true floor.

**How results are computed.** Every number is linear in the shares: cost `Σ c·x` and held
amount `Σ r·x`. Held amounts assume each feature is spread evenly within a unit, the same
assumption prioritizr makes. Boundary is `Σ b_ii·x_i + Σ b_ij·|x_i − x_j|`, the optimiser's own
term; with Marxan's boundary convention it equals prioritizr's default (non-knapsack) boundary
formulation, whose evaluators instead report `x_i·x_j`. For fractional plans the boundary is a
linear compactness penalty, not a measured perimeter. Connectivity is the optimiser's own
term: `−v·min(x_i, x_j)` per pair (symmetric) or `v·max(0, x_i − x_j)` per row (asymmetric), so
the reported objective equals the MIP objective with connectivity too.

A target counts as met when the held amount is within `1e-7·|target|` of it (relative only;
a target of 0 is always met), because a linear-program optimum leaves active targets a solver
tolerance short; binary plans are judged exactly. This is looser than prioritizr's
`eval_target_coverage_summary` (shortfall below `1e-10`, absolute), which can report LP-optimal
targets as unmet where pymarxan reports them met.

For a set of fractional solutions, "selection frequency" (`compute_selection_frequency`,
`export_frequency_spatial`) is the mean share and the summed-solution `Number`
(`write_ssoln`) the sum of shares; a set mixing binary and fractional solutions aggregates
shares (a binary selection counts as 1).

**Differences and limits.**
- Locked-in units (status 2) are fixed whole (`x = 1.0`) in every mode; `upper` and `lower`
  bound only the free units. prioritizr makes the model infeasible when `upper < 1`.
- Non-binary decisions cannot be combined with PROBMODE 3, clumping (`target2`), separation,
  space targets, no-good cuts or negative connectivity values; the solver raises `ValueError`.
- Solver values are clipped to each unit's bounds and values at or below `1e-9` set to 0;
  semicontinuous values within `1e-6` below `lower` are raised to `lower`; a semicontinuous unit whose on/off
  variable is 0 is set to 0. Nothing else is
  rounded. A genuine optimum below `1e-9` (one unit holding a billion times a target) would be
  lost.
- Tools that only make sense for whole units raise `FractionalSolutionError`: iterative
  improvement and RUNMODE chains through it, portfolio cuts, replacement cost, solution
  diversity, scenario overlap, post-hoc clusters, rank importance, phylogenetic diversity and
  zone writers; the solution comparison map shows an explanation instead. Convert first
  with `binarize_solution`.
- The raw-array APIs of `pymarxan.constraints` and `pymarxan.objectives` (`evaluate`,
  `penalty`) are binary-only: they treat any non-zero share as selected.
- Zone solvers stay binary.

### Zone Configuration

For multi-zone conservation planning, zones are loaded from the project files. Typical zones include:
- **No-take zone**: Full protection, no resource extraction
- **Buffer zone**: Limited activities allowed
- **Sustainable use zone**: Managed resource extraction permitted

The zone configuration panel displays zone definitions and average costs per zone.

---

## Calibrate

### BLM Calibration

The **Boundary Length Modifier (BLM)** controls the tradeoff between reserve cost and spatial compactness. The classic calibration approach:

1. Set a range of BLM values to test (e.g. 0–50)
2. Run the solver at each value
3. Plot **cost vs. boundary length**
4. Choose the **"elbow" point** — where increasing BLM yields diminishing compactness gains

**Parameters:**
- **Min BLM / Max BLM**: Range to explore
- **Steps**: Number of BLM values to test (more = smoother curve, slower)

**Interpreting results:**
- Left side of the curve: low BLM, low cost, fragmented reserves
- Right side: high BLM, higher cost, compact reserves
- The elbow represents the best cost-efficiency for compactness

### Target Sensitivity

Test how sensitive the reserve design is to changes in conservation targets.

**How it works:** Each feature's target is multiplied by values in a range (e.g. 0.8–1.2 = 80–120% of original target). The solver runs once per multiplier value per feature.

**Interpreting results:**
- Features with steep objective curves are **driving the solution** — small target changes cause large cost changes
- Flat curves indicate the feature's target is easily met regardless
- This reveals which features are most critical and where target uncertainty matters most

### SPF Calibration

Automatically tune SPF (Species Penalty Factor) values to meet all conservation targets.

**How it works:**
1. Run the solver with current SPF values
2. For features not meeting their targets, multiply their SPF by the multiplier
3. Repeat until all targets are met or the iteration limit is reached

**Parameters:**
- **Max Iterations**: Maximum calibration rounds (stops early if all targets met)
- **SPF Multiplier**: Factor to increase unmet features' SPF each round (e.g. 2.0 = double each iteration)

### Parameter Sweep

Run a systematic sweep over any Marxan parameter to understand its effect on solution quality.

**Sweep-able parameters:**
- **BLM**: Boundary Length Modifier
- **NUMITNS**: SA iteration count
- **NUMTEMP**: Temperature steps

Results are displayed in a table with the parameter value, cost, boundary, and objective for each step.

---

## Run Solver

Click **Run Solver** to execute the configured solver on the loaded conservation problem.

**Workflow:**
1. Ensure a project is loaded (Data Input tab)
2. Configure solver and parameters (Configure tab)
3. Click "Run Solver"
4. Monitor real-time progress bar and status messages
5. When complete, results are automatically available in Results and Maps tabs

The solver runs in a background thread so the interface remains responsive. The progress bar shows completion percentage.

**Status messages guide you through the workflow:**
- "Step 1: Go to 'Data' tab and load a Marxan project" — if no project is loaded
- "Step 2: Configure solver, then click 'Run Solver'" — if a project is loaded but not yet run
- Solution summary — after a successful run

---

## Maps

All maps require `ipyleaflet` and `shinywidgets` to be installed. Without these packages, a text summary is shown instead.

### Planning Unit Map

Interactive map of all planning units colored by:
- **Cost**: Yellow (low) to red (high) gradient
- **Status**: Gray (available), green (locked-in), red (locked-out)

### Solution Map

Shows the solver result: green PUs are **selected** for the reserve; gray PUs are not.

Displays summary statistics: number selected, total cost, boundary length, objective value, and targets met.

### Selection Frequency

Heatmap of how often each PU is selected across all solver runs. White (0%) to dark blue (100%).

**Higher frequency = higher irreplaceability** — these PUs are consistently needed to meet conservation targets regardless of the specific solution.

Requires multiple solutions (set "Number of solutions" > 1 in the Configure tab).

### Solution Comparison

Compare two solutions side-by-side to identify shared and unique planning units:

| Color | Meaning |
|-------|---------|
| Green | Selected in **both** solutions |
| Blue | Selected in **Solution A only** |
| Orange | Selected in **Solution B only** |
| Gray | Not selected in **either** |

### Connectivity Network

Overlay the connectivity graph on planning units:
- **Node colour**: In-degree or out-degree metric (yellow to purple gradient)
- **Edges**: Lines between connected PUs (filtered by minimum weight threshold)
- Up to 5,000 edges are displayed to prevent browser slowdown

---

## Results

### Summary Table

HTML table showing target achievement for each conservation feature:
- Feature ID, name, target amount, achieved amount, percentage, and status (✓ Met / ✗ NOT MET)
- Summary line: "X of Y targets met"

### Target Achievement

Sortable data table with the same information in a filterable format.

### SA Convergence

Plot the simulated annealing convergence curve:
- **Blue line**: Current objective value at each iteration
- **Green line**: Best objective found so far
- **Red dotted line** (optional): Temperature schedule on a log-scale secondary axis

**What to look for:**
- Both lines should converge and flatten — if still declining, increase NUMITNS
- Temperature should decrease smoothly — erratic temperature suggests NUMTEMP is too low

### Scenario Comparison

Save multiple solutions as named scenarios and compare them:

1. Run the solver with different settings
2. Enter a descriptive name (e.g. "BLM=10, SA, 100 runs")
3. Click "Save Current Solution"
4. Repeat with different configurations
5. The comparison table shows all saved scenarios with their metrics

### Export

Download results as CSV files:

- **Solution CSV**: One row per planning unit with columns: PU ID, selected (1/0), cost
- **Target Summary CSV**: Feature ID, name, target, achieved amount, met status

---

## Marxan Concepts Glossary

| Term | Definition |
|------|-----------|
| **Planning Unit (PU)** | A discrete spatial unit (grid cell or polygon) that can be selected or not in the reserve design |
| **Feature** | A conservation target entity — typically a species, habitat type, or ecosystem service |
| **Target** | The minimum amount of a feature that must be represented in the reserve network |
| **Cost** | The cost of including a PU in the reserve (land price, opportunity cost, area, etc.) |
| **BLM** (Boundary Length Modifier) | Weight on the boundary penalty term. Higher BLM = more spatially compact reserves |
| **SPF** (Species Penalty Factor) | Weight on the penalty for not meeting a feature's target. Higher SPF = stronger incentive to meet that target |
| **Objective Function** | `Σ(cost) + BLM × Σ(boundary) + Σ(SPF × shortfall)` — what Marxan minimises |
| **Status** | PU status code: 0=available, 1=initial include, 2=locked in, 3=locked out |
| **NUMITNS** | Number of simulated annealing iterations per run |
| **NUMTEMP** | Number of temperature decrease steps in the SA cooling schedule |
| **HEURTYPE** | Scoring mode for the greedy heuristic (0–7) |
| **ITIMPTYPE** | Improvement strategy for iterative improvement (0–3) |
| **RUNMODE** | Pipeline combination of heuristic, SA, and improvement algorithms (0–6) |
| **Selection Frequency** | How often a PU appears in the solution across multiple runs (0–100%) |
| **Irreplaceability** | How critical a PU is for meeting targets — PUs selected in nearly all solutions are highly irreplaceable |
| **Connectivity** | Spatial connections between PUs — encourages selecting linked PUs to maintain ecological corridors |
| **Probability** | Per-PU probability that the unit (and the features in it) is **lost** over the planning horizon (prob.dat). Persistence is 1 − probability. PROBMODE 3 instead reads a per-(PU, feature) probability of **presence** from puvspr.dat |
| **PROBMODE** | Probability mode (pymarxan extensions over `prob.dat`): 1 = risk premium in the objective (default with a probability table), 2 = expected (persistence-adjusted) amounts; 3 = Z-score probability penalty based on Marxan's Probability 2D. Marxan's Probability 1D is not yet implemented — see [Probability (PROBMODE)](#probability-probmode) |
| **Zone** | A management category (e.g. no-take, buffer, sustainable use) in multi-zone planning |
| **Zone Contribution** | How much a PU in a given zone contributes to a feature's target (0–1 multiplier) |
| **Zone Boundary Cost** | Cost per unit of shared boundary between adjacent PUs in a given pair of zones, including the same zone twice (`zone1 = zone2`); negative values reward that adjacency. Scaled by BLM, as in Marxan with Zones, so BLM = 0 switches zone boundary costs off. Unassigned PUs (zone 0) pay nothing |
| **Constraint** | An additional rule beyond targets — contiguity, minimum neighbors, budget caps, etc. |
| **Objective** | The MIP objective: `min_set` (default), `max_features`, `max_weighted_features`, `min_shortfall`, `min_largest_shortfall`, `min_penalties` (core only); the Shiny selector does not yet drive it |
| **puvspr.dat** | Planning Unit vs. Species file — maps features to PUs with amounts |
| **bound.dat** | Boundary definition file — pairwise boundary lengths between adjacent PUs |
| **prob.dat** | Probability file — per-PU probability of loss (pymarxan's `PROBNAME`; not a Marxan file) |

---

## Troubleshooting

### Common Issues

**"No input.dat found"**
- Ensure the ZIP or directory contains a valid `input.dat` file
- Check that file paths inside `input.dat` are relative and correct

**"Solver not available"**
- MIP solver requires `pulp` package: `pip install pulp`
- Marxan binary requires the compiled `Marxan` executable on your PATH
- Python SA solver is always available

**"Load a project first"**
- Go to the Data Input tab and load a project before running the solver

**No maps shown**
- Install `ipyleaflet` and `shinywidgets`: `pip install ipyleaflet shinywidgets`
- Maps require planning units with geometry (generated grid or imported GIS file)

**Solver runs slowly**
- Reduce NUMITNS or NUMTEMP for faster (but potentially lower quality) SA runs
- Use the MIP solver for exact solutions on small problems (< 10,000 PUs)
- Reduce the number of solutions for faster batch runs

**All targets not met**
- Increase SPF values for unmet features (or use SPF Calibration)
- Check that targets are achievable given available planning units
- Increase NUMITNS for better SA exploration

**BLM calibration — no elbow visible**
- Expand the BLM range (try 0–500 or higher)
- Increase the number of steps for a smoother curve
- If the curve is always flat, the boundary data may be missing or all zeros

---

*For more information on the Marxan framework, see:*
- Ball, I.R., Possingham, H.P., & Watts, M. (2009). *Marxan and relatives: Software for spatial conservation prioritisation.* In Spatial conservation prioritisation: Quantitative methods and computational tools.
- [marxan.org](https://marxan.org) — Official Marxan documentation and resources
