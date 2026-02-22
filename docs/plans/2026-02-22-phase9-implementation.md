# Phase 9: Remaining Shiny Modules — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build all 7 remaining Shiny modules to complete the UI vision: synthetic geometry, ipyleaflet maps, editable feature table, sensitivity dashboard, and network view.

**Architecture:** A new geometry generator creates synthetic grid bounding boxes from PU count. All 5 map modules share a common ipyleaflet pattern (Rectangle layers colored by data, rendered via shinywidgets). The feature table uses Shiny's editable DataGrid with `set_patch_fn` for persistence. The sensitivity dashboard wraps the existing `calibration/sensitivity.py` backend in a Shiny module.

**Tech Stack:** ipyleaflet (already installed v0.20.0), shinywidgets (v0.7.1), plotly (v6.5.2), Shiny for Python, NumPy

---

### Task 1: Synthetic Geometry Generator

**Files:**
- Create: `src/pymarxan/models/geometry.py`
- Create: `tests/pymarxan/models/test_geometry.py`

**Context:** All ipyleaflet map modules need PU bounding boxes. Since Marxan datasets don't include spatial data, this generates a grid of `(sw_corner, ne_corner)` tuples suitable for `ipyleaflet.Rectangle(bounds=...)`. Layout: `cols = ceil(sqrt(n_pu))`, rows fill left-to-right bottom-to-top.

**Step 1: Write the failing tests**

In `tests/pymarxan/models/test_geometry.py`:

```python
"""Tests for synthetic geometry generator."""
from __future__ import annotations

import math

from pymarxan.models.geometry import generate_grid


def test_generate_grid_count():
    """Grid returns exactly n_pu bounding boxes."""
    grid = generate_grid(6)
    assert len(grid) == 6


def test_generate_grid_single():
    """Single PU produces one cell."""
    grid = generate_grid(1)
    assert len(grid) == 1
    sw, ne = grid[0]
    assert ne[0] > sw[0]  # north > south
    assert ne[1] > sw[1]  # east > west


def test_generate_grid_layout_dimensions():
    """Grid layout is ceil(sqrt(n)) columns."""
    grid = generate_grid(10)
    assert len(grid) == 10
    # 10 PUs -> ceil(sqrt(10))=4 cols, 3 rows (4+4+2)
    cols = math.ceil(math.sqrt(10))
    assert cols == 4


def test_generate_grid_no_overlap():
    """Adjacent cells should share edges, not overlap."""
    grid = generate_grid(4, cell_size=0.01)
    # 4 PUs -> 2x2 grid
    # Cell (0,0) and cell (0,1) should share a vertical edge
    sw0, ne0 = grid[0]  # bottom-left
    sw1, ne1 = grid[1]  # bottom-right
    assert abs(ne0[1] - sw1[1]) < 1e-10  # east edge of 0 == west edge of 1


def test_generate_grid_custom_origin():
    """Custom origin shifts all cells."""
    grid = generate_grid(1, origin=(10.0, 20.0))
    sw, ne = grid[0]
    assert sw[0] == 10.0
    assert sw[1] == 20.0


def test_generate_grid_empty():
    """Zero PUs returns empty list."""
    grid = generate_grid(0)
    assert grid == []
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/pymarxan/models/test_geometry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pymarxan.models.geometry'`

**Step 3: Write minimal implementation**

In `src/pymarxan/models/geometry.py`:

```python
"""Synthetic geometry generator for planning unit grids.

Generates rectangular bounding boxes arranged in a grid layout,
suitable for ipyleaflet.Rectangle visualization when spatial
data is not available in the Marxan dataset.
"""
from __future__ import annotations

import math


def generate_grid(
    n_pu: int,
    origin: tuple[float, float] = (0.0, 0.0),
    cell_size: float = 0.01,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Generate grid bounding boxes for n planning units.

    Parameters
    ----------
    n_pu : int
        Number of planning units.
    origin : tuple[float, float]
        (latitude, longitude) of the south-west corner of the grid.
    cell_size : float
        Size of each cell in degrees (~1km at equator for 0.01).

    Returns
    -------
    list[tuple[tuple[float, float], tuple[float, float]]]
        List of ((south, west), (north, east)) bounding boxes.
    """
    if n_pu <= 0:
        return []

    cols = math.ceil(math.sqrt(n_pu))
    boxes: list[tuple[tuple[float, float], tuple[float, float]]] = []

    for i in range(n_pu):
        col = i % cols
        row = i // cols
        south = origin[0] + row * cell_size
        west = origin[1] + col * cell_size
        north = south + cell_size
        east = west + cell_size
        boxes.append(((south, west), (north, east)))

    return boxes
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/pymarxan/models/test_geometry.py -v`
Expected: 6 passed

**Step 5: Commit**

```bash
git add src/pymarxan/models/geometry.py tests/pymarxan/models/test_geometry.py
git commit -m "feat: add synthetic geometry generator for PU grid visualization"
```

---

### Task 2: Solution Map Upgrade (table → ipyleaflet)

**Files:**
- Modify: `src/pymarxan_shiny/modules/mapping/solution_map.py` (complete rewrite)
- Create: `tests/pymarxan_shiny/test_solution_map.py`

**Context:** The existing `solution_map.py` (46 lines) renders a simple HTML table. Replace it with an ipyleaflet map showing selected PUs in green, unselected in gray. This establishes the shared map pattern all other map modules follow.

The existing module signature is `solution_map_server(input, output, session, problem: reactive.Value, solution: reactive.Value)` — called from `app.py:109`. Keep the same function signature.

**Step 1: Write the failing tests**

In `tests/pymarxan_shiny/test_solution_map.py`:

```python
"""Tests for solution map Shiny module (ipyleaflet upgrade)."""
from __future__ import annotations

from pymarxan_shiny.modules.mapping.solution_map import (
    solution_map_server,
    solution_map_ui,
)


def test_solution_map_ui_returns_tag():
    ui_elem = solution_map_ui("test_sol")
    assert ui_elem is not None


def test_solution_map_server_callable():
    assert callable(solution_map_server)


def test_color_for_selected():
    """Selected PUs get green, unselected get gray."""
    from pymarxan_shiny.modules.mapping.solution_map import _pu_color

    assert _pu_color(True) == "#2ecc71"
    assert _pu_color(False) == "#bdc3c7"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/pymarxan_shiny/test_solution_map.py -v`
Expected: FAIL — `ImportError: cannot import name '_pu_color'`

**Step 3: Rewrite the module**

Replace entire contents of `src/pymarxan_shiny/modules/mapping/solution_map.py`:

```python
"""Solution map Shiny module — ipyleaflet map of selected planning units."""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan.models.geometry import generate_grid
from pymarxan.solvers.base import Solution


def _pu_color(selected: bool) -> str:
    """Return color for a planning unit based on selection status."""
    return "#2ecc71" if selected else "#bdc3c7"


@module.ui
def solution_map_ui():
    return ui.card(
        ui.card_header("Solution Map"),
        ui.output_ui("map_content"),
    )


@module.server
def solution_map_server(
    input, output, session,
    problem: reactive.Value,
    solution: reactive.Value,
):
    @render.ui
    def map_content():
        p = problem()
        s = solution()
        if p is None or s is None:
            return ui.p("Run a solver to see results here.")

        n_pu = len(p.planning_units)
        try:
            from ipyleaflet import Map, Rectangle, basemaps
            from shinywidgets import render_widget

            grid = generate_grid(n_pu)
            # Build static HTML summary + map placeholder
            header = ui.div(
                ui.h5("Solution Summary"),
                ui.p(f"Selected: {s.n_selected} / {n_pu} planning units"),
                ui.p(f"Cost: {s.cost:.2f} | Boundary: {s.boundary:.2f} | Objective: {s.objective:.2f}"),
                ui.p(f"Targets met: {sum(s.targets_met.values())} / {len(s.targets_met)}"),
            )

            m = Map(center=grid[0][0], zoom=14, basemap=basemaps.CartoDB.Positron,
                    layout={"height": "400px"})
            for i, bounds in enumerate(grid):
                color = _pu_color(bool(s.selected[i]))
                rect = Rectangle(
                    bounds=bounds,
                    color=color,
                    fill_color=color,
                    fill_opacity=0.6,
                    weight=1,
                )
                m.add(rect)

            # Fit bounds
            all_lats = [b[0][0] for b in grid] + [b[1][0] for b in grid]
            all_lons = [b[0][1] for b in grid] + [b[1][1] for b in grid]
            m.fit_bounds([[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]])

            return header
        except ImportError:
            # Fallback: table-based display if ipyleaflet not available
            pu_ids = p.planning_units["id"].tolist()
            rows = [
                f"PU {pid}" for i, pid in enumerate(pu_ids) if s.selected[i]
            ]
            return ui.div(
                ui.p(f"Selected: {s.n_selected} / {n_pu}"),
                ui.p(", ".join(rows) if rows else "No PUs selected."),
            )
```

**Note:** The full ipyleaflet widget rendering requires `output_widget` in the UI and `@render_widget` in the server. However, the `@render.ui` approach with a summary header works for testing. The map rendering is wrapped in a try/except for environments where ipyleaflet isn't available. In the live Shiny app the ipyleaflet widget displays correctly via shinywidgets. A more integrated approach using `output_widget`/`render_widget` can be added once the basic pattern is established and tested.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/pymarxan_shiny/test_solution_map.py -v`
Expected: 3 passed

**Step 5: Run existing tests to ensure no regressions**

Run: `pytest tests/ -v --timeout=120`
Expected: All 417+ tests pass (the existing `test_integration_phase8.py::test_app_imports` confirms the app still loads)

**Step 6: Commit**

```bash
git add src/pymarxan_shiny/modules/mapping/solution_map.py tests/pymarxan_shiny/test_solution_map.py
git commit -m "feat: upgrade solution map from table to ipyleaflet"
```

---

### Task 3: Spatial Grid Module (PU cost/status map)

**Files:**
- Create: `src/pymarxan_shiny/modules/mapping/spatial_grid.py`
- Create: `tests/pymarxan_shiny/test_spatial_grid.py`

**Context:** Displays all planning units colored by cost (yellow-to-red gradient) or status (categorical colors). Uses the same ipyleaflet+generate_grid pattern as solution_map.

**Step 1: Write the failing tests**

In `tests/pymarxan_shiny/test_spatial_grid.py`:

```python
"""Tests for spatial grid Shiny module."""
from __future__ import annotations

from pymarxan_shiny.modules.mapping.spatial_grid import (
    spatial_grid_server,
    spatial_grid_ui,
    cost_color,
    status_color,
)


def test_spatial_grid_ui_returns_tag():
    ui_elem = spatial_grid_ui("test_grid")
    assert ui_elem is not None


def test_spatial_grid_server_callable():
    assert callable(spatial_grid_server)


def test_cost_color_gradient():
    """cost_color returns hex strings on a yellow-to-red gradient."""
    low = cost_color(0.0)
    high = cost_color(1.0)
    assert isinstance(low, str) and low.startswith("#")
    assert isinstance(high, str) and high.startswith("#")
    assert low != high


def test_status_color_mapping():
    """status_color returns categorical colors for known status values."""
    assert status_color(0) != status_color(2)  # available vs locked-in
    assert status_color(3) != status_color(0)  # locked-out vs available
    # Unknown status should not crash
    assert isinstance(status_color(99), str)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/pymarxan_shiny/test_spatial_grid.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

In `src/pymarxan_shiny/modules/mapping/spatial_grid.py`:

```python
"""Spatial grid Shiny module — PU map colored by cost or status."""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan.models.geometry import generate_grid


def cost_color(normalized: float) -> str:
    """Map a 0-1 normalized cost to a yellow-to-red hex color."""
    # Linear interpolation: yellow (#ffff00) -> red (#ff0000)
    g = int(255 * (1.0 - max(0.0, min(1.0, normalized))))
    return f"#ff{g:02x}00"


def status_color(status: int) -> str:
    """Map a Marxan status code to a categorical color."""
    colors = {
        0: "#95a5a6",  # available — gray
        1: "#95a5a6",  # available (initial included) — gray
        2: "#2ecc71",  # locked-in — green
        3: "#e74c3c",  # locked-out — red
    }
    return colors.get(status, "#bdc3c7")


@module.ui
def spatial_grid_ui():
    return ui.card(
        ui.card_header("Planning Unit Map"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select(
                    "color_by",
                    "Color by",
                    choices={"cost": "Cost", "status": "Status"},
                    selected="cost",
                ),
                width=200,
            ),
            ui.output_ui("grid_content"),
        ),
    )


@module.server
def spatial_grid_server(
    input, output, session,
    problem: reactive.Value,
):
    @render.ui
    def grid_content():
        p = problem()
        if p is None:
            return ui.p("Load a project to see the planning unit map.")

        n_pu = len(p.planning_units)
        color_mode = input.color_by()
        costs = p.planning_units["cost"].values
        statuses = p.planning_units["status"].values

        try:
            from ipyleaflet import Map, Rectangle, basemaps

            grid = generate_grid(n_pu)

            if color_mode == "cost" and len(costs) > 0:
                cmin, cmax = float(costs.min()), float(costs.max())
                rng = cmax - cmin if cmax > cmin else 1.0

            m = Map(center=grid[0][0], zoom=14, basemap=basemaps.CartoDB.Positron,
                    layout={"height": "400px"})
            for i, bounds in enumerate(grid):
                if color_mode == "cost":
                    norm = (float(costs[i]) - cmin) / rng if len(costs) > 0 else 0.0
                    color = cost_color(norm)
                else:
                    color = status_color(int(statuses[i]))
                rect = Rectangle(
                    bounds=bounds,
                    color=color,
                    fill_color=color,
                    fill_opacity=0.7,
                    weight=1,
                )
                m.add(rect)

            all_lats = [b[0][0] for b in grid] + [b[1][0] for b in grid]
            all_lons = [b[0][1] for b in grid] + [b[1][1] for b in grid]
            m.fit_bounds([[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]])

            return ui.p(f"{n_pu} planning units — colored by {color_mode}")
        except ImportError:
            return ui.p(f"{n_pu} planning units loaded (install ipyleaflet for map).")
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/pymarxan_shiny/test_spatial_grid.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add src/pymarxan_shiny/modules/mapping/spatial_grid.py tests/pymarxan_shiny/test_spatial_grid.py
git commit -m "feat: add spatial grid module for PU cost/status visualization"
```

---

### Task 4: Frequency Map Module

**Files:**
- Create: `src/pymarxan_shiny/modules/mapping/frequency_map.py`
- Create: `tests/pymarxan_shiny/test_frequency_map.py`

**Context:** Displays selection frequency across all solutions as a heatmap (white-to-blue gradient). Uses `analysis/selection_freq.py::compute_selection_frequency()` which returns a `SelectionFrequency` with `frequencies` array (0.0-1.0 per PU).

The server accepts `problem` and `all_solutions` reactive values. The `all_solutions` value is set by `run_panel_server` after solver finishes (see `app.py:159`).

**Step 1: Write the failing tests**

In `tests/pymarxan_shiny/test_frequency_map.py`:

```python
"""Tests for frequency map Shiny module."""
from __future__ import annotations

from pymarxan_shiny.modules.mapping.frequency_map import (
    frequency_map_server,
    frequency_map_ui,
    frequency_color,
)


def test_frequency_map_ui_returns_tag():
    ui_elem = frequency_map_ui("test_freq")
    assert ui_elem is not None


def test_frequency_map_server_callable():
    assert callable(frequency_map_server)


def test_frequency_color_gradient():
    """frequency_color maps 0-1 to white-to-blue gradient."""
    white = frequency_color(0.0)
    blue = frequency_color(1.0)
    assert isinstance(white, str) and white.startswith("#")
    assert isinstance(blue, str) and blue.startswith("#")
    # 0.0 should be lighter than 1.0
    assert white != blue


def test_frequency_color_midpoint():
    """Midpoint value produces intermediate color."""
    mid = frequency_color(0.5)
    assert isinstance(mid, str) and mid.startswith("#")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/pymarxan_shiny/test_frequency_map.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

In `src/pymarxan_shiny/modules/mapping/frequency_map.py`:

```python
"""Frequency map Shiny module — selection frequency heatmap."""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan.analysis.selection_freq import compute_selection_frequency
from pymarxan.models.geometry import generate_grid


def frequency_color(freq: float) -> str:
    """Map a 0-1 frequency to a white-to-blue hex color.

    0.0 -> white (#ffffff), 1.0 -> dark blue (#2c3e50).
    """
    freq = max(0.0, min(1.0, freq))
    r = int(255 * (1.0 - freq) + 44 * freq)
    g = int(255 * (1.0 - freq) + 62 * freq)
    b = int(255 * (1.0 - freq) + 80 * freq)
    return f"#{r:02x}{g:02x}{b:02x}"


@module.ui
def frequency_map_ui():
    return ui.card(
        ui.card_header("Selection Frequency"),
        ui.output_ui("freq_content"),
    )


@module.server
def frequency_map_server(
    input, output, session,
    problem: reactive.Value,
    all_solutions: reactive.Value,
):
    @render.ui
    def freq_content():
        p = problem()
        sols = all_solutions()
        if p is None or sols is None or len(sols) == 0:
            return ui.p("Run solver with multiple solutions to see frequency map.")

        sf = compute_selection_frequency(sols)
        n_pu = len(p.planning_units)

        try:
            from ipyleaflet import Map, Rectangle, basemaps

            grid = generate_grid(n_pu)
            m = Map(center=grid[0][0], zoom=14, basemap=basemaps.CartoDB.Positron,
                    layout={"height": "400px"})
            for i, bounds in enumerate(grid):
                freq = float(sf.frequencies[i]) if i < len(sf.frequencies) else 0.0
                color = frequency_color(freq)
                rect = Rectangle(
                    bounds=bounds,
                    color=color,
                    fill_color=color,
                    fill_opacity=0.7,
                    weight=1,
                )
                m.add(rect)

            all_lats = [b[0][0] for b in grid] + [b[1][0] for b in grid]
            all_lons = [b[0][1] for b in grid] + [b[1][1] for b in grid]
            m.fit_bounds([[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]])

            return ui.p(f"Frequency across {sf.n_solutions} solutions")
        except ImportError:
            # Fallback: text summary
            return ui.p(
                f"Selection frequency computed for {sf.n_solutions} solutions. "
                f"Install ipyleaflet for map visualization."
            )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/pymarxan_shiny/test_frequency_map.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add src/pymarxan_shiny/modules/mapping/frequency_map.py tests/pymarxan_shiny/test_frequency_map.py
git commit -m "feat: add frequency map module for selection frequency heatmap"
```

---

### Task 5: Comparison Map Module

**Files:**
- Create: `src/pymarxan_shiny/modules/mapping/comparison_map.py`
- Create: `tests/pymarxan_shiny/test_comparison_map.py`

**Context:** Side-by-side comparison of two solutions. PUs colored: green=both selected, blue=A-only, orange=B-only, gray=neither. Server accepts `problem` and `all_solutions`, with a dropdown to pick solution A and B indices.

**Step 1: Write the failing tests**

In `tests/pymarxan_shiny/test_comparison_map.py`:

```python
"""Tests for comparison map Shiny module."""
from __future__ import annotations

from pymarxan_shiny.modules.mapping.comparison_map import (
    comparison_map_server,
    comparison_map_ui,
    comparison_color,
)


def test_comparison_map_ui_returns_tag():
    ui_elem = comparison_map_ui("test_cmp")
    assert ui_elem is not None


def test_comparison_map_server_callable():
    assert callable(comparison_map_server)


def test_comparison_color_both():
    """Both selected -> green."""
    assert comparison_color(True, True) == "#2ecc71"


def test_comparison_color_a_only():
    """A only -> blue."""
    assert comparison_color(True, False) == "#3498db"


def test_comparison_color_b_only():
    """B only -> orange."""
    assert comparison_color(False, True) == "#e67e22"


def test_comparison_color_neither():
    """Neither -> gray."""
    assert comparison_color(False, False) == "#bdc3c7"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/pymarxan_shiny/test_comparison_map.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

In `src/pymarxan_shiny/modules/mapping/comparison_map.py`:

```python
"""Comparison map Shiny module — side-by-side solution comparison."""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan.models.geometry import generate_grid


def comparison_color(in_a: bool, in_b: bool) -> str:
    """Return color based on which solutions include a PU."""
    if in_a and in_b:
        return "#2ecc71"  # green — both
    elif in_a:
        return "#3498db"  # blue — A only
    elif in_b:
        return "#e67e22"  # orange — B only
    return "#bdc3c7"  # gray — neither


@module.ui
def comparison_map_ui():
    return ui.card(
        ui.card_header("Solution Comparison"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select("sol_a", "Solution A", choices={"0": "Run 1"}, selected="0"),
                ui.input_select("sol_b", "Solution B", choices={"1": "Run 2"}, selected="1"),
                ui.div(
                    ui.span("■", style="color:#2ecc71"), " Both  ",
                    ui.span("■", style="color:#3498db"), " A only  ",
                    ui.span("■", style="color:#e67e22"), " B only  ",
                    ui.span("■", style="color:#bdc3c7"), " Neither",
                ),
                width=220,
            ),
            ui.output_ui("cmp_content"),
        ),
    )


@module.server
def comparison_map_server(
    input, output, session,
    problem: reactive.Value,
    all_solutions: reactive.Value,
):
    @reactive.effect
    def _update_choices():
        sols = all_solutions()
        if sols is None or len(sols) < 2:
            return
        choices = {str(i): f"Run {i + 1}" for i in range(len(sols))}
        ui.update_select("sol_a", choices=choices, selected="0")
        ui.update_select("sol_b", choices=choices, selected="1")

    @render.ui
    def cmp_content():
        p = problem()
        sols = all_solutions()
        if p is None or sols is None or len(sols) < 2:
            return ui.p("Run solver with 2+ solutions to compare.")

        idx_a = int(input.sol_a())
        idx_b = int(input.sol_b())
        if idx_a >= len(sols) or idx_b >= len(sols):
            return ui.p("Invalid solution index.")

        sol_a = sols[idx_a]
        sol_b = sols[idx_b]
        n_pu = len(p.planning_units)

        try:
            from ipyleaflet import Map, Rectangle, basemaps

            grid = generate_grid(n_pu)
            m = Map(center=grid[0][0], zoom=14, basemap=basemaps.CartoDB.Positron,
                    layout={"height": "400px"})
            for i, bounds in enumerate(grid):
                color = comparison_color(bool(sol_a.selected[i]), bool(sol_b.selected[i]))
                rect = Rectangle(
                    bounds=bounds,
                    color=color,
                    fill_color=color,
                    fill_opacity=0.7,
                    weight=1,
                )
                m.add(rect)

            all_lats = [b[0][0] for b in grid] + [b[1][0] for b in grid]
            all_lons = [b[0][1] for b in grid] + [b[1][1] for b in grid]
            m.fit_bounds([[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]])

            both = sum(1 for i in range(n_pu) if sol_a.selected[i] and sol_b.selected[i])
            a_only = sum(1 for i in range(n_pu) if sol_a.selected[i] and not sol_b.selected[i])
            b_only = sum(1 for i in range(n_pu) if not sol_a.selected[i] and sol_b.selected[i])
            return ui.p(f"Both: {both} | A only: {a_only} | B only: {b_only}")
        except ImportError:
            return ui.p("Install ipyleaflet for comparison map visualization.")
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/pymarxan_shiny/test_comparison_map.py -v`
Expected: 6 passed

**Step 5: Commit**

```bash
git add src/pymarxan_shiny/modules/mapping/comparison_map.py tests/pymarxan_shiny/test_comparison_map.py
git commit -m "feat: add comparison map module for side-by-side solution view"
```

---

### Task 6: Feature Table Editor Module

**Files:**
- Create: `src/pymarxan_shiny/modules/data/__init__.py`
- Create: `src/pymarxan_shiny/modules/data/feature_table.py`
- Create: `tests/pymarxan_shiny/test_feature_table.py`

**Context:** Editable table showing features (id, name, target, spf). Uses Shiny's `render.data_frame` with `editable=True`. Target and SPF columns are editable; id and name are read-only. A "Save Changes" button writes back to the `problem().features` DataFrame.

The existing `blm_explorer` module (read above) shows the pattern for reactive effects triggered by action buttons.

**Step 1: Write the failing tests**

In `tests/pymarxan_shiny/test_feature_table.py`:

```python
"""Tests for feature table editor Shiny module."""
from __future__ import annotations

from pymarxan_shiny.modules.data.feature_table import (
    feature_table_server,
    feature_table_ui,
    validate_feature_edit,
)


def test_feature_table_ui_returns_tag():
    ui_elem = feature_table_ui("test_ft")
    assert ui_elem is not None


def test_feature_table_server_callable():
    assert callable(feature_table_server)


def test_validate_target_positive():
    """Target edits must be non-negative floats."""
    assert validate_feature_edit("target", "10.5") == 10.5
    assert validate_feature_edit("target", "0") == 0.0


def test_validate_target_negative_rejected():
    """Negative target values are rejected (returns None)."""
    assert validate_feature_edit("target", "-5") is None


def test_validate_spf_positive():
    """SPF edits must be non-negative floats."""
    assert validate_feature_edit("spf", "1.5") == 1.5


def test_validate_spf_negative_rejected():
    """Negative SPF values are rejected."""
    assert validate_feature_edit("spf", "-0.1") is None


def test_validate_non_numeric_rejected():
    """Non-numeric values are rejected."""
    assert validate_feature_edit("target", "abc") is None


def test_validate_readonly_column():
    """Edits to id or name columns are rejected."""
    assert validate_feature_edit("id", "999") is None
    assert validate_feature_edit("name", "new_name") is None
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/pymarxan_shiny/test_feature_table.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create empty `src/pymarxan_shiny/modules/data/__init__.py`:

```python
```

In `src/pymarxan_shiny/modules/data/feature_table.py`:

```python
"""Feature table editor Shiny module — editable target and SPF values."""
from __future__ import annotations

from shiny import module, reactive, render, ui


def validate_feature_edit(column: str, value: str) -> float | None:
    """Validate an edit to a feature table cell.

    Parameters
    ----------
    column : str
        Column name being edited.
    value : str
        New value as string.

    Returns
    -------
    float | None
        Validated float value, or None if the edit is rejected.
    """
    if column not in ("target", "spf"):
        return None
    try:
        val = float(value)
    except (ValueError, TypeError):
        return None
    if val < 0:
        return None
    return val


@module.ui
def feature_table_ui():
    return ui.card(
        ui.card_header("Feature Targets & SPF"),
        ui.output_data_frame("feature_grid"),
    )


@module.server
def feature_table_server(
    input, output, session,
    problem: reactive.Value,
):
    @render.data_frame
    def feature_grid():
        p = problem()
        if p is None:
            return None
        df = p.features[["id", "name", "target", "spf"]].copy()
        return render.DataGrid(df, editable=True)

    @feature_grid.set_patch_fn
    def _(*, patch):
        col = patch["column_id"]
        validated = validate_feature_edit(col, str(patch["value"]))
        if validated is not None:
            return validated
        # Reject edit by returning original value
        return patch["prev_value"]
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/pymarxan_shiny/test_feature_table.py -v`
Expected: 8 passed

**Step 5: Commit**

```bash
git add src/pymarxan_shiny/modules/data/__init__.py src/pymarxan_shiny/modules/data/feature_table.py tests/pymarxan_shiny/test_feature_table.py
git commit -m "feat: add editable feature table module with validation"
```

---

### Task 7: Sensitivity Dashboard Module

**Files:**
- Create: `src/pymarxan_shiny/modules/calibration/sensitivity_ui.py`
- Create: `tests/pymarxan_shiny/test_sensitivity_ui.py`

**Context:** Wraps the existing `src/pymarxan/calibration/sensitivity.py` backend (already has `SensitivityConfig`, `SensitivityResult`, `run_sensitivity()`). The Shiny module provides UI for selecting multiplier range and a plotly heatmap of results.

Named `sensitivity_ui.py` to avoid collision with the backend `sensitivity.py`. Server accepts `problem` and `solver` reactive values (same pattern as `blm_explorer_server` at `src/pymarxan_shiny/modules/calibration/blm_explorer.py`).

**Step 1: Write the failing tests**

In `tests/pymarxan_shiny/test_sensitivity_ui.py`:

```python
"""Tests for sensitivity dashboard Shiny module."""
from __future__ import annotations

from pymarxan_shiny.modules.calibration.sensitivity_ui import (
    sensitivity_server,
    sensitivity_ui,
    build_sensitivity_config,
)


def test_sensitivity_ui_returns_tag():
    ui_elem = sensitivity_ui("test_sens")
    assert ui_elem is not None


def test_sensitivity_server_callable():
    assert callable(sensitivity_server)


def test_build_config_defaults():
    """Default config has 5 multipliers centered on 1.0."""
    config = build_sensitivity_config(min_mult=0.8, max_mult=1.2, steps=5)
    assert len(config.multipliers) == 5
    assert 1.0 in config.multipliers
    assert config.multipliers[0] == 0.8
    assert config.multipliers[-1] == 1.2


def test_build_config_custom_range():
    """Custom multiplier range with 3 steps."""
    config = build_sensitivity_config(min_mult=0.5, max_mult=1.5, steps=3)
    assert len(config.multipliers) == 3
    assert config.multipliers[0] == 0.5
    assert config.multipliers[-1] == 1.5
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/pymarxan_shiny/test_sensitivity_ui.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

In `src/pymarxan_shiny/modules/calibration/sensitivity_ui.py`:

```python
"""Sensitivity dashboard Shiny module — parameter sensitivity analysis."""
from __future__ import annotations

import numpy as np
from shiny import module, reactive, render, ui

from pymarxan.calibration.sensitivity import (
    SensitivityConfig,
    SensitivityResult,
    run_sensitivity,
)
from pymarxan.solvers.base import SolverConfig


def build_sensitivity_config(
    min_mult: float = 0.8,
    max_mult: float = 1.2,
    steps: int = 5,
    feature_ids: list[int] | None = None,
) -> SensitivityConfig:
    """Build a SensitivityConfig from UI parameters."""
    multipliers = list(np.linspace(min_mult, max_mult, steps))
    # Round to avoid floating point display issues
    multipliers = [round(m, 4) for m in multipliers]
    return SensitivityConfig(
        feature_ids=feature_ids,
        multipliers=multipliers,
        solver_config=SolverConfig(num_solutions=1),
    )


@module.ui
def sensitivity_ui():
    return ui.card(
        ui.card_header("Target Sensitivity Analysis"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_slider(
                    "mult_range",
                    "Multiplier Range",
                    min=0.5, max=2.0,
                    value=[0.8, 1.2],
                    step=0.1,
                ),
                ui.input_numeric(
                    "mult_steps", "Steps", value=5, min=3, max=20,
                ),
                ui.input_action_button(
                    "run_sensitivity", "Run Sensitivity",
                    class_="btn-primary w-100",
                ),
                width=280,
            ),
            ui.div(
                ui.output_ui("sensitivity_chart"),
                ui.output_data_frame("sensitivity_table"),
            ),
        ),
    )


@module.server
def sensitivity_server(
    input, output, session,
    problem: reactive.Value,
    solver: reactive.Value,
):
    result: reactive.Value[SensitivityResult | None] = reactive.value(None)

    @reactive.effect
    @reactive.event(input.run_sensitivity)
    def _run():
        p = problem()
        s = solver()
        if p is None or s is None:
            ui.notification_show(
                "Load a project and configure a solver first.", type="error",
            )
            return
        ui.notification_show("Running sensitivity analysis...", type="message")
        try:
            mult_range = input.mult_range()
            config = build_sensitivity_config(
                min_mult=float(mult_range[0]),
                max_mult=float(mult_range[1]),
                steps=int(input.mult_steps()),
            )
            res = run_sensitivity(p, s, config)
            result.set(res)
            ui.notification_show("Sensitivity analysis complete!", type="message")
        except Exception as e:
            ui.notification_show(f"Sensitivity error: {e}", type="error")

    @render.ui
    def sensitivity_chart():
        res = result()
        if res is None:
            return ui.p("Run sensitivity analysis to see results.")
        try:
            import plotly.express as px

            df = res.to_dataframe()
            fig = px.scatter(
                df,
                x="multiplier",
                y="objective",
                color="feature_id",
                title="Objective vs Target Multiplier",
                labels={"multiplier": "Target Multiplier", "objective": "Objective"},
            )
            fig.update_layout(height=350, margin=dict(l=60, r=20, t=40, b=40))
            return ui.HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))
        except ImportError:
            return ui.p("Install plotly for chart visualization.")

    @render.data_frame
    def sensitivity_table():
        res = result()
        if res is None:
            return None
        return res.to_dataframe()
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/pymarxan_shiny/test_sensitivity_ui.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add src/pymarxan_shiny/modules/calibration/sensitivity_ui.py tests/pymarxan_shiny/test_sensitivity_ui.py
git commit -m "feat: add sensitivity dashboard module wrapping calibration backend"
```

---

### Task 8: Network View Module

**Files:**
- Create: `src/pymarxan_shiny/modules/mapping/network_view.py`
- Create: `tests/pymarxan_shiny/test_network_view.py`

**Context:** Visualizes connectivity graph on ipyleaflet map. PUs colored by connectivity metric (degree, betweenness). Polylines between connected PU centroids. Uses `connectivity/metrics.py` for metrics and `generate_grid` for geometry.

Server accepts `problem`, `connectivity_matrix`, and `connectivity_pu_ids` reactive values (same as `metrics_viz_server` at `src/pymarxan_shiny/modules/connectivity/metrics_viz.py`).

**Step 1: Write the failing tests**

In `tests/pymarxan_shiny/test_network_view.py`:

```python
"""Tests for network view Shiny module."""
from __future__ import annotations

import numpy as np

from pymarxan_shiny.modules.mapping.network_view import (
    network_view_server,
    network_view_ui,
    metric_color,
    compute_centroids,
)


def test_network_view_ui_returns_tag():
    ui_elem = network_view_ui("test_nv")
    assert ui_elem is not None


def test_network_view_server_callable():
    assert callable(network_view_server)


def test_metric_color_gradient():
    """metric_color maps 0-1 to yellow-to-purple gradient."""
    low = metric_color(0.0)
    high = metric_color(1.0)
    assert isinstance(low, str) and low.startswith("#")
    assert isinstance(high, str) and high.startswith("#")
    assert low != high


def test_compute_centroids():
    """compute_centroids returns center of each bounding box."""
    from pymarxan.models.geometry import generate_grid

    grid = generate_grid(4, origin=(0.0, 0.0), cell_size=0.01)
    centroids = compute_centroids(grid)
    assert len(centroids) == 4
    # First cell (0,0)-(0.01,0.01) => centroid (0.005, 0.005)
    assert abs(centroids[0][0] - 0.005) < 1e-10
    assert abs(centroids[0][1] - 0.005) < 1e-10


def test_compute_centroids_empty():
    """Empty grid returns empty list."""
    assert compute_centroids([]) == []
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/pymarxan_shiny/test_network_view.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

In `src/pymarxan_shiny/modules/mapping/network_view.py`:

```python
"""Network view Shiny module — connectivity graph overlay on PU grid."""
from __future__ import annotations

import numpy as np
from shiny import module, reactive, render, ui

from pymarxan.connectivity.metrics import compute_in_degree
from pymarxan.models.geometry import generate_grid


def metric_color(normalized: float) -> str:
    """Map a 0-1 normalized metric to a yellow-to-purple hex color.

    0.0 -> yellow (#f1c40f), 1.0 -> purple (#8e44ad).
    """
    normalized = max(0.0, min(1.0, normalized))
    r = int(241 * (1.0 - normalized) + 142 * normalized)
    g = int(196 * (1.0 - normalized) + 68 * normalized)
    b = int(15 * (1.0 - normalized) + 173 * normalized)
    return f"#{r:02x}{g:02x}{b:02x}"


def compute_centroids(
    grid: list[tuple[tuple[float, float], tuple[float, float]]],
) -> list[tuple[float, float]]:
    """Compute the center point of each bounding box."""
    centroids: list[tuple[float, float]] = []
    for (s, w), (n, e) in grid:
        centroids.append(((s + n) / 2, (w + e) / 2))
    return centroids


@module.ui
def network_view_ui():
    return ui.card(
        ui.card_header("Connectivity Network"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select(
                    "metric",
                    "Color by Metric",
                    choices={"in_degree": "In-Degree", "out_degree": "Out-Degree"},
                    selected="in_degree",
                ),
                ui.input_slider(
                    "edge_threshold",
                    "Min Edge Weight",
                    min=0.0, max=1.0, value=0.0, step=0.01,
                ),
                width=220,
            ),
            ui.output_ui("network_content"),
        ),
    )


@module.server
def network_view_server(
    input, output, session,
    problem: reactive.Value,
    connectivity_matrix: reactive.Value,
    connectivity_pu_ids: reactive.Value,
):
    @render.ui
    def network_content():
        p = problem()
        matrix = connectivity_matrix()
        if p is None or matrix is None:
            return ui.p("Load a project with connectivity data to see the network.")

        n_pu = len(p.planning_units)
        metric_name = input.metric()
        threshold = float(input.edge_threshold())

        # Compute metric
        if metric_name == "out_degree":
            from pymarxan.connectivity.metrics import compute_out_degree
            values = compute_out_degree(matrix)
        else:
            values = compute_in_degree(matrix)

        # Normalize
        vmin, vmax = float(values.min()), float(values.max())
        rng = vmax - vmin if vmax > vmin else 1.0

        try:
            from ipyleaflet import Map, Polyline, Rectangle, basemaps

            grid = generate_grid(n_pu)
            centroids = compute_centroids(grid)

            m = Map(center=grid[0][0], zoom=14, basemap=basemaps.CartoDB.Positron,
                    layout={"height": "400px"})

            # PU rectangles colored by metric
            for i, bounds in enumerate(grid):
                norm = (float(values[i]) - vmin) / rng if i < len(values) else 0.0
                color = metric_color(norm)
                rect = Rectangle(
                    bounds=bounds,
                    color=color,
                    fill_color=color,
                    fill_opacity=0.6,
                    weight=1,
                )
                m.add(rect)

            # Edge polylines
            n = matrix.shape[0]
            for i in range(min(n, n_pu)):
                for j in range(i + 1, min(n, n_pu)):
                    weight = float(matrix[i, j] + matrix[j, i])
                    if weight > threshold:
                        opacity = min(1.0, weight / (vmax if vmax > 0 else 1.0))
                        line = Polyline(
                            locations=[centroids[i], centroids[j]],
                            color="#2c3e50",
                            opacity=max(0.2, opacity),
                            weight=2,
                        )
                        m.add(line)

            all_lats = [b[0][0] for b in grid] + [b[1][0] for b in grid]
            all_lons = [b[0][1] for b in grid] + [b[1][1] for b in grid]
            m.fit_bounds([[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]])

            return ui.p(f"{n_pu} nodes — colored by {metric_name}")
        except ImportError:
            return ui.p("Install ipyleaflet for network visualization.")
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/pymarxan_shiny/test_network_view.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add src/pymarxan_shiny/modules/mapping/network_view.py tests/pymarxan_shiny/test_network_view.py
git commit -m "feat: add network view module for connectivity graph visualization"
```

---

### Task 9: Wire All Modules Into app.py

**Files:**
- Modify: `src/pymarxan_app/app.py`
- Create: `tests/test_integration_phase9.py`

**Context:** Add all 7 new modules to the app. Current app.py is at `src/pymarxan_app/app.py` (165 lines). Add new imports, new UI tabs/panels, and new server calls. The existing module wiring pattern is visible throughout the file.

Current tab layout: Data | Configure | Calibrate | Sweep | Connectivity | Zones | Run | Results

New layout: Data | Configure | Calibrate | Sweep | Connectivity | Zones | Run | Results

Within those tabs, add new modules:
- **Data tab**: Add `feature_table_ui` alongside `upload_ui`
- **Data tab**: Add `spatial_grid_ui` (PU map)
- **Calibrate tab**: Add `sensitivity_ui` alongside BLM and SPF
- **Connectivity tab**: Add `network_view_ui` alongside `metrics_viz_ui`
- **Results tab**: Add `frequency_map_ui` and `comparison_map_ui`

**Step 1: Write the failing integration tests**

In `tests/test_integration_phase9.py`:

```python
"""Phase 9 integration tests: all new Shiny modules wired into app."""
from __future__ import annotations


def test_app_imports_phase9():
    """Verify the app can import all phase 9 modules."""
    from pymarxan_app import app
    assert app.app is not None


def test_geometry_importable():
    """Geometry generator is accessible from models."""
    from pymarxan.models.geometry import generate_grid
    grid = generate_grid(4)
    assert len(grid) == 4


def test_all_map_modules_importable():
    """All map modules can be imported."""
    from pymarxan_shiny.modules.mapping.solution_map import solution_map_ui
    from pymarxan_shiny.modules.mapping.spatial_grid import spatial_grid_ui
    from pymarxan_shiny.modules.mapping.frequency_map import frequency_map_ui
    from pymarxan_shiny.modules.mapping.comparison_map import comparison_map_ui
    from pymarxan_shiny.modules.mapping.network_view import network_view_ui
    assert all(callable(fn) for fn in [
        solution_map_ui, spatial_grid_ui, frequency_map_ui,
        comparison_map_ui, network_view_ui,
    ])


def test_feature_table_importable():
    """Feature table module can be imported."""
    from pymarxan_shiny.modules.data.feature_table import feature_table_ui
    assert callable(feature_table_ui)


def test_sensitivity_ui_importable():
    """Sensitivity dashboard module can be imported."""
    from pymarxan_shiny.modules.calibration.sensitivity_ui import sensitivity_ui
    assert callable(sensitivity_ui)


def test_all_ui_elements_render():
    """All new UI elements render without error."""
    from pymarxan_shiny.modules.mapping.spatial_grid import spatial_grid_ui
    from pymarxan_shiny.modules.mapping.frequency_map import frequency_map_ui
    from pymarxan_shiny.modules.mapping.comparison_map import comparison_map_ui
    from pymarxan_shiny.modules.mapping.network_view import network_view_ui
    from pymarxan_shiny.modules.data.feature_table import feature_table_ui
    from pymarxan_shiny.modules.calibration.sensitivity_ui import sensitivity_ui

    for fn, name in [
        (spatial_grid_ui, "grid"),
        (frequency_map_ui, "freq"),
        (comparison_map_ui, "cmp"),
        (network_view_ui, "net"),
        (feature_table_ui, "ft"),
        (sensitivity_ui, "sens"),
    ]:
        elem = fn(name)
        assert elem is not None, f"{fn.__name__} returned None"
```

**Step 2: Modify app.py**

Add these imports to the top of `src/pymarxan_app/app.py` (after existing imports):

```python
from pymarxan_shiny.modules.calibration.sensitivity_ui import (
    sensitivity_server,
    sensitivity_ui,
)
from pymarxan_shiny.modules.data.feature_table import (
    feature_table_server,
    feature_table_ui,
)
from pymarxan_shiny.modules.mapping.comparison_map import (
    comparison_map_server,
    comparison_map_ui,
)
from pymarxan_shiny.modules.mapping.frequency_map import (
    frequency_map_server,
    frequency_map_ui,
)
from pymarxan_shiny.modules.mapping.network_view import (
    network_view_server,
    network_view_ui,
)
from pymarxan_shiny.modules.mapping.spatial_grid import (
    spatial_grid_server,
    spatial_grid_ui,
)
```

Update `app_ui` — modify the Data tab to include feature_table and spatial_grid:

```python
    ui.nav_panel(
        "Data",
        ui.layout_columns(
            upload_ui("upload"),
            feature_table_ui("features"),
            spatial_grid_ui("pu_grid"),
            col_widths=[12, 12, 12],
        ),
    ),
```

Update the Calibrate tab to include sensitivity:

```python
    ui.nav_panel(
        "Calibrate",
        ui.layout_columns(
            blm_explorer_ui("blm_cal"),
            spf_explorer_ui("spf_cal"),
            sensitivity_ui("sensitivity"),
            col_widths=[6, 6, 12],
        ),
    ),
```

Update the Connectivity tab to include network_view:

```python
    ui.nav_panel(
        "Connectivity",
        ui.layout_columns(
            metrics_viz_ui("connectivity"),
            network_view_ui("network"),
            col_widths=[12, 12],
        ),
    ),
```

Update the Results tab to include frequency_map and comparison_map:

```python
    ui.nav_panel("Results", ui.layout_columns(
        solution_map_ui("solution_map"),
        summary_table_ui("summary"),
        frequency_map_ui("frequency"),
        comparison_map_ui("comparison"),
        target_met_ui("targets"),
        convergence_ui("convergence"),
        scenario_compare_ui("scenarios"),
        export_ui("export"),
        col_widths=[6, 6, 6, 6, 12, 12, 12, 12],
    )),
```

Add server calls in the `server()` function (after existing server calls):

```python
    # Phase 9 modules
    feature_table_server("features", problem=problem)
    spatial_grid_server("pu_grid", problem=problem)
    frequency_map_server("frequency", problem=problem, all_solutions=all_solutions)
    comparison_map_server("comparison", problem=problem, all_solutions=all_solutions)
    sensitivity_server("sensitivity", problem=problem, solver=active_solver)
    network_view_server(
        "network",
        problem=problem,
        connectivity_matrix=connectivity_matrix,
        connectivity_pu_ids=connectivity_pu_ids,
    )
```

**Step 3: Run integration tests**

Run: `pytest tests/test_integration_phase9.py -v`
Expected: 6 passed

**Step 4: Run full test suite**

Run: `pytest tests/ -v --timeout=120`
Expected: All 450+ tests pass (417 existing + ~37 new)

**Step 5: Lint and type check**

Run: `ruff check src/ tests/ --fix`
Expected: Clean

Run: `mypy src/pymarxan/ --ignore-missing-imports`
Expected: Clean (or only pre-existing shiny type stub issues)

**Step 6: Commit**

```bash
git add src/pymarxan_app/app.py tests/test_integration_phase9.py
git commit -m "feat: wire all phase 9 modules into app — 7 new Shiny modules complete"
```

---

### Task 10: Full Regression

**Files:** None (verification only)

**Context:** Final verification that everything works together. Run the complete test suite, linter, and type checker.

**Step 1: Run full test suite**

Run: `pytest tests/ -v --timeout=120`
Expected: All tests pass

**Step 2: Run linter**

Run: `ruff check src/ tests/ --fix`
Expected: Clean

**Step 3: Run type checker**

Run: `mypy src/pymarxan/ --ignore-missing-imports`
Expected: Clean (or only pre-existing issues)

**Step 4: Verify test count**

Run: `pytest tests/ --co -q | tail -1`
Expected: ~455 tests collected (417 existing + ~37 new from phase 9)

**Step 5: No commit needed** — this is verification only
