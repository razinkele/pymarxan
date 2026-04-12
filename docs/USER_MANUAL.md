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

- Python 3.10+
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
| **Probability** | Per-PU persistence probability — accounts for habitat loss risk in reserve design |
| **PROBMODE** | Probability mode: 1 = risk premium on cost, 2 = persistence-adjusted feature amounts |
| **Zone** | A management category (e.g. no-take, buffer, sustainable use) in multi-zone planning |
| **Zone Contribution** | How much a PU in a given zone contributes to a feature's target (0–1 multiplier) |
| **Zone Boundary Cost** | Penalty for adjacent PUs being in different zone types |
| **Constraint** | An additional rule beyond targets — contiguity, minimum neighbors, budget caps, etc. |
| **Objective** | The objective function type: MinSet (default), MaxCoverage, MaxUtility, MinShortfall |
| **puvspr.dat** | Planning Unit vs. Species file — maps features to PUs with amounts |
| **bound.dat** | Boundary definition file — pairwise boundary lengths between adjacent PUs |
| **prob.dat** | Probability file — per-PU habitat persistence probabilities |

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
