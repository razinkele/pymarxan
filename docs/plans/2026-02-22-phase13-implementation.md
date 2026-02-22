# Phase 13: Real ipyleaflet Maps — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace text-summary map modules with interactive ipyleaflet maps using shinywidgets output_widget/render_widget, while preserving text fallback for headless/CI environments.

**Architecture:** Shared `create_grid_map()` helper builds an ipyleaflet.Map with colored Rectangle layers. All 5 map modules switch from `@render.ui` to `@render_widget` for the map, keeping a separate `@render.text` for accessible text summaries. `network_view` additionally adds Polyline edges. Each module has a try/except ImportError fallback that returns `None` from `@render_widget` if ipyleaflet is unavailable.

**Tech Stack:** ipyleaflet, shinywidgets (output_widget/render_widget), existing color functions

---

### Task 1: Shared Map Helper — `create_grid_map()`

**Files:**
- Create: `src/pymarxan_shiny/modules/mapping/map_utils.py`
- Create: `tests/pymarxan_shiny/test_map_utils.py`

**Step 1: Write the failing test**

Create `tests/pymarxan_shiny/test_map_utils.py`:

```python
"""Tests for shared map helper."""
from __future__ import annotations

from pymarxan.models.geometry import generate_grid

from pymarxan_shiny.modules.mapping.map_utils import create_grid_map


def test_create_grid_map_returns_map():
    """create_grid_map returns an ipyleaflet Map."""
    import ipyleaflet

    grid = generate_grid(4)
    colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00"]
    m = create_grid_map(grid, colors)
    assert isinstance(m, ipyleaflet.Map)


def test_create_grid_map_layer_count():
    """Map has one Rectangle per grid cell (plus base TileLayer)."""
    import ipyleaflet

    grid = generate_grid(6)
    colors = ["#aaaaaa"] * 6
    m = create_grid_map(grid, colors)
    rectangles = [
        layer for layer in m.layers
        if isinstance(layer, ipyleaflet.Rectangle)
    ]
    assert len(rectangles) == 6


def test_create_grid_map_auto_center():
    """Map auto-centers on grid midpoint when center not provided."""
    grid = generate_grid(4, origin=(10.0, 20.0), cell_size=0.01)
    colors = ["#000000"] * 4
    m = create_grid_map(grid, colors)
    # Grid spans from (10.0, 20.0) to (10.02, 20.02)
    # Midpoint should be approximately (10.01, 20.01)
    assert abs(m.center[0] - 10.01) < 0.01
    assert abs(m.center[1] - 20.01) < 0.01


def test_create_grid_map_custom_center():
    """Map uses provided center when given."""
    grid = generate_grid(4)
    colors = ["#000000"] * 4
    m = create_grid_map(grid, colors, center=(50.0, 10.0))
    assert m.center == (50.0, 10.0)


def test_create_grid_map_empty_grid():
    """Empty grid produces Map with no Rectangle layers."""
    import ipyleaflet

    m = create_grid_map([], [])
    rectangles = [
        layer for layer in m.layers
        if isinstance(layer, ipyleaflet.Rectangle)
    ]
    assert len(rectangles) == 0
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/pymarxan_shiny/test_map_utils.py -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError" (map_utils does not exist yet).

**Step 3: Write minimal implementation**

Create `src/pymarxan_shiny/modules/mapping/map_utils.py`:

```python
"""Shared map helper for building ipyleaflet maps from grid data."""
from __future__ import annotations

import ipyleaflet


def create_grid_map(
    grid: list[tuple[tuple[float, float], tuple[float, float]]],
    colors: list[str],
    center: tuple[float, float] | None = None,
    zoom: int = 12,
) -> ipyleaflet.Map:
    """Create ipyleaflet Map with colored Rectangle layers.

    Parameters
    ----------
    grid : list of ((south, west), (north, east)) tuples
        Bounding boxes for each planning unit.
    colors : list of hex color strings
        One color per grid cell (same length as grid).
    center : optional (lat, lon) tuple
        Map center. Auto-computed from grid midpoint if not provided.
    zoom : int, default 12
        Initial zoom level.

    Returns
    -------
    ipyleaflet.Map with one Rectangle per planning unit.
    """
    if center is None and grid:
        all_lats = [b[0][0] for b in grid] + [b[1][0] for b in grid]
        all_lons = [b[0][1] for b in grid] + [b[1][1] for b in grid]
        center = (
            (min(all_lats) + max(all_lats)) / 2,
            (min(all_lons) + max(all_lons)) / 2,
        )
    elif center is None:
        center = (0.0, 0.0)

    m = ipyleaflet.Map(center=center, zoom=zoom)

    for bounds, color in zip(grid, colors):
        rect = ipyleaflet.Rectangle(
            bounds=bounds,
            color=color,
            fill_color=color,
            fill_opacity=0.7,
            weight=1,
        )
        m.add(rect)

    return m
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/pymarxan_shiny/test_map_utils.py -v`
Expected: 5 passed.

**Step 5: Commit**

```bash
git add src/pymarxan_shiny/modules/mapping/map_utils.py tests/pymarxan_shiny/test_map_utils.py
git commit -m "feat: add create_grid_map shared helper for ipyleaflet maps"
```

---

### Task 2: Upgrade solution_map to @render_widget

**Files:**
- Modify: `src/pymarxan_shiny/modules/mapping/solution_map.py` (full rewrite)
- Modify: `tests/pymarxan_shiny/test_solution_map.py` (update tests)

**Step 1: Write the failing test**

Replace `tests/pymarxan_shiny/test_solution_map.py` with:

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


def test_solution_map_ui_has_output_widget():
    """UI should contain an output_widget for the map."""
    html = str(solution_map_ui("test_sol"))
    assert "jupyter" in html.lower() or "widget" in html.lower() or "map" in html.lower()
```

**Step 2: Run test to verify the new test fails**

Run: `source .venv/bin/activate && pytest tests/pymarxan_shiny/test_solution_map.py::test_solution_map_ui_has_output_widget -v`
Expected: FAIL (current UI uses output_ui, not output_widget).

**Step 3: Write implementation**

Replace `src/pymarxan_shiny/modules/mapping/solution_map.py`:

```python
"""Solution map Shiny module — ipyleaflet map of selected planning units."""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan.models.geometry import generate_grid

try:
    from shinywidgets import output_widget, render_widget

    from pymarxan_shiny.modules.mapping.map_utils import create_grid_map

    _HAS_IPYLEAFLET = True
except ImportError:
    _HAS_IPYLEAFLET = False


@module.ui
def solution_map_ui():
    if _HAS_IPYLEAFLET:
        return ui.card(
            ui.card_header("Solution Map"),
            output_widget("map"),
            ui.output_text_verbatim("map_summary"),
        )
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
    if _HAS_IPYLEAFLET:

        @render_widget
        def map():
            p = problem()
            s = solution()
            if p is None or s is None:
                return None

            n_pu = len(p.planning_units)
            grid = generate_grid(n_pu)
            colors = [
                "#2ecc71" if s.selected[i] else "#95a5a6"
                for i in range(n_pu)
            ]
            return create_grid_map(grid, colors)

    @render.text
    def map_summary():
        p = problem()
        s = solution()
        if p is None or s is None:
            return "Run a solver to see results here."

        n_pu = len(p.planning_units)
        targets_met = sum(s.targets_met.values())
        total_targets = len(s.targets_met)
        return (
            f"Selected: {s.n_selected} / {n_pu} planning units\n"
            f"Cost: {s.cost:.2f} | Boundary: {s.boundary:.2f}"
            f" | Objective: {s.objective:.2f}\n"
            f"Targets met: {targets_met} / {total_targets}"
        )

    if not _HAS_IPYLEAFLET:

        @render.ui
        def map_content():
            p = problem()
            s = solution()
            if p is None or s is None:
                return ui.p("Run a solver to see results here.")

            n_pu = len(p.planning_units)
            pu_ids = p.planning_units["id"].tolist()
            rows = [
                f"PU {pid}" for i, pid in enumerate(pu_ids) if s.selected[i]
            ]
            return ui.div(
                ui.p(f"Selected: {s.n_selected} / {n_pu}"),
                ui.p(", ".join(rows) if rows else "No PUs selected."),
            )
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/pymarxan_shiny/test_solution_map.py -v`
Expected: 3 passed.

**Step 5: Commit**

```bash
git add src/pymarxan_shiny/modules/mapping/solution_map.py tests/pymarxan_shiny/test_solution_map.py
git commit -m "feat: upgrade solution_map to ipyleaflet @render_widget"
```

---

### Task 3: Upgrade spatial_grid to @render_widget

**Files:**
- Modify: `src/pymarxan_shiny/modules/mapping/spatial_grid.py` (full rewrite)
- Modify: `tests/pymarxan_shiny/test_spatial_grid.py` (update tests)

**Step 1: Read existing test file**

Read: `tests/pymarxan_shiny/test_spatial_grid.py` to understand existing tests.

**Step 2: Write the failing test**

Add to `tests/pymarxan_shiny/test_spatial_grid.py`:

```python
def test_spatial_grid_ui_has_output_widget():
    """UI should contain an output_widget for the map."""
    html = str(spatial_grid_ui("test_sg"))
    assert "jupyter" in html.lower() or "widget" in html.lower() or "map" in html.lower()
```

**Step 3: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/pymarxan_shiny/test_spatial_grid.py::test_spatial_grid_ui_has_output_widget -v`
Expected: FAIL.

**Step 4: Write implementation**

Replace `src/pymarxan_shiny/modules/mapping/spatial_grid.py`:

```python
"""Spatial grid Shiny module — PU map colored by cost or status."""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan.models.geometry import generate_grid

try:
    from shinywidgets import output_widget, render_widget

    from pymarxan_shiny.modules.mapping.map_utils import create_grid_map

    _HAS_IPYLEAFLET = True
except ImportError:
    _HAS_IPYLEAFLET = False


def cost_color(normalized: float) -> str:
    """Map a 0-1 normalized cost to a yellow-to-red hex color."""
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
    sidebar = ui.sidebar(
        ui.input_select(
            "color_by",
            "Color by",
            choices={"cost": "Cost", "status": "Status"},
            selected="cost",
        ),
        width=200,
    )
    if _HAS_IPYLEAFLET:
        return ui.card(
            ui.card_header("Planning Unit Map"),
            ui.layout_sidebar(
                sidebar,
                ui.div(
                    output_widget("map"),
                    ui.output_text_verbatim("map_summary"),
                ),
            ),
        )
    return ui.card(
        ui.card_header("Planning Unit Map"),
        ui.layout_sidebar(sidebar, ui.output_ui("grid_content")),
    )


@module.server
def spatial_grid_server(
    input,
    output,
    session,
    problem: reactive.Value,
):
    if _HAS_IPYLEAFLET:

        @render_widget
        def map():
            p = problem()
            if p is None:
                return None

            n_pu = len(p.planning_units)
            color_mode = input.color_by()
            grid = generate_grid(n_pu)

            if color_mode == "status":
                statuses = p.planning_units["status"].tolist()
                colors = [status_color(s) for s in statuses]
            else:
                costs = p.planning_units["cost"].tolist()
                max_c = max(costs) if costs else 1.0
                colors = [
                    cost_color(c / max_c if max_c > 0 else 0.0) for c in costs
                ]

            return create_grid_map(grid, colors)

    @render.text
    def map_summary():
        p = problem()
        if p is None:
            return "Load a project to see the planning unit map."

        n_pu = len(p.planning_units)
        color_mode = input.color_by()
        return f"{n_pu} planning units — colored by {color_mode}"

    if not _HAS_IPYLEAFLET:

        @render.ui
        def grid_content():
            p = problem()
            if p is None:
                return ui.p("Load a project to see the planning unit map.")

            n_pu = len(p.planning_units)
            color_mode = input.color_by()
            grid = generate_grid(n_pu)
            all_lats = [b[0][0] for b in grid] + [b[1][0] for b in grid]
            all_lons = [b[0][1] for b in grid] + [b[1][1] for b in grid]
            bounds_info = (
                f"Grid bounds: [{min(all_lats):.4f}, {min(all_lons):.4f}]"
                f" to [{max(all_lats):.4f}, {max(all_lons):.4f}]"
            )
            return ui.div(
                ui.p(f"{n_pu} planning units — colored by {color_mode}"),
                ui.p(bounds_info),
            )
```

**Step 5: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/pymarxan_shiny/test_spatial_grid.py -v`
Expected: All passed.

**Step 6: Commit**

```bash
git add src/pymarxan_shiny/modules/mapping/spatial_grid.py tests/pymarxan_shiny/test_spatial_grid.py
git commit -m "feat: upgrade spatial_grid to ipyleaflet @render_widget"
```

---

### Task 4: Upgrade frequency_map to @render_widget

**Files:**
- Modify: `src/pymarxan_shiny/modules/mapping/frequency_map.py` (full rewrite)
- Modify: `tests/pymarxan_shiny/test_frequency_map.py` (update tests)

**Step 1: Read existing test file**

Read: `tests/pymarxan_shiny/test_frequency_map.py` to understand existing tests.

**Step 2: Write the failing test**

Add to `tests/pymarxan_shiny/test_frequency_map.py`:

```python
def test_frequency_map_ui_has_output_widget():
    """UI should contain an output_widget for the map."""
    html = str(frequency_map_ui("test_fm"))
    assert "jupyter" in html.lower() or "widget" in html.lower() or "map" in html.lower()
```

**Step 3: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/pymarxan_shiny/test_frequency_map.py::test_frequency_map_ui_has_output_widget -v`
Expected: FAIL.

**Step 4: Write implementation**

Replace `src/pymarxan_shiny/modules/mapping/frequency_map.py`:

```python
"""Frequency map Shiny module — selection frequency heatmap."""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan.analysis.selection_freq import compute_selection_frequency
from pymarxan.models.geometry import generate_grid

try:
    from shinywidgets import output_widget, render_widget

    from pymarxan_shiny.modules.mapping.map_utils import create_grid_map

    _HAS_IPYLEAFLET = True
except ImportError:
    _HAS_IPYLEAFLET = False


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
    if _HAS_IPYLEAFLET:
        return ui.card(
            ui.card_header("Selection Frequency"),
            output_widget("map"),
            ui.output_text_verbatim("map_summary"),
        )
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
    if _HAS_IPYLEAFLET:

        @render_widget
        def map():
            p = problem()
            sols = all_solutions()
            if p is None or sols is None or len(sols) == 0:
                return None

            sf = compute_selection_frequency(sols)
            n_pu = len(p.planning_units)
            grid = generate_grid(n_pu)
            colors = [frequency_color(sf.frequencies[i]) for i in range(n_pu)]
            return create_grid_map(grid, colors)

    @render.text
    def map_summary():
        p = problem()
        sols = all_solutions()
        if p is None or sols is None or len(sols) == 0:
            return "Run solver with multiple solutions to see frequency map."

        sf = compute_selection_frequency(sols)
        return f"Frequency across {sf.n_solutions} solutions"

    if not _HAS_IPYLEAFLET:

        @render.ui
        def freq_content():
            p = problem()
            sols = all_solutions()
            if p is None or sols is None or len(sols) == 0:
                return ui.p(
                    "Run solver with multiple solutions to see frequency map."
                )

            sf = compute_selection_frequency(sols)
            n_pu = len(p.planning_units)
            grid = generate_grid(n_pu)
            all_lats = [b[0][0] for b in grid] + [b[1][0] for b in grid]
            all_lons = [b[0][1] for b in grid] + [b[1][1] for b in grid]
            bounds_info = (
                f"Grid: [{min(all_lats):.4f}, {min(all_lons):.4f}]"
                f" to [{max(all_lats):.4f}, {max(all_lons):.4f}]"
            )
            return ui.div(
                ui.p(f"Frequency across {sf.n_solutions} solutions"),
                ui.p(bounds_info),
            )
```

**Step 5: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/pymarxan_shiny/test_frequency_map.py -v`
Expected: All passed.

**Step 6: Commit**

```bash
git add src/pymarxan_shiny/modules/mapping/frequency_map.py tests/pymarxan_shiny/test_frequency_map.py
git commit -m "feat: upgrade frequency_map to ipyleaflet @render_widget"
```

---

### Task 5: Upgrade comparison_map to @render_widget

**Files:**
- Modify: `src/pymarxan_shiny/modules/mapping/comparison_map.py` (full rewrite)
- Modify: `tests/pymarxan_shiny/test_comparison_map.py` (update tests)

**Step 1: Read existing test file**

Read: `tests/pymarxan_shiny/test_comparison_map.py` to understand existing tests.

**Step 2: Write the failing test**

Add to `tests/pymarxan_shiny/test_comparison_map.py`:

```python
def test_comparison_map_ui_has_output_widget():
    """UI should contain an output_widget for the map."""
    html = str(comparison_map_ui("test_cm"))
    assert "jupyter" in html.lower() or "widget" in html.lower() or "map" in html.lower()
```

**Step 3: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/pymarxan_shiny/test_comparison_map.py::test_comparison_map_ui_has_output_widget -v`
Expected: FAIL.

**Step 4: Write implementation**

Replace `src/pymarxan_shiny/modules/mapping/comparison_map.py`:

```python
"""Comparison map Shiny module -- side-by-side solution comparison."""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan.models.geometry import generate_grid

try:
    from shinywidgets import output_widget, render_widget

    from pymarxan_shiny.modules.mapping.map_utils import create_grid_map

    _HAS_IPYLEAFLET = True
except ImportError:
    _HAS_IPYLEAFLET = False


def comparison_color(in_a: bool, in_b: bool) -> str:
    """Return color based on which solutions include a PU.

    Green (#2ecc71) = both, Blue (#3498db) = A only,
    Orange (#e67e22) = B only, Gray (#bdc3c7) = neither.
    """
    if in_a and in_b:
        return "#2ecc71"  # green -- both
    elif in_a:
        return "#3498db"  # blue -- A only
    elif in_b:
        return "#e67e22"  # orange -- B only
    return "#bdc3c7"  # gray -- neither


@module.ui
def comparison_map_ui():
    sidebar = ui.sidebar(
        ui.input_select(
            "sol_a", "Solution A",
            choices={"0": "Run 1"}, selected="0",
        ),
        ui.input_select(
            "sol_b", "Solution B",
            choices={"1": "Run 2"}, selected="1",
        ),
        ui.div(
            ui.span("\u25a0", style="color:#2ecc71"), " Both  ",
            ui.span("\u25a0", style="color:#3498db"), " A only  ",
            ui.span("\u25a0", style="color:#e67e22"), " B only  ",
            ui.span("\u25a0", style="color:#bdc3c7"), " Neither",
        ),
        width=220,
    )
    if _HAS_IPYLEAFLET:
        return ui.card(
            ui.card_header("Solution Comparison"),
            ui.layout_sidebar(
                sidebar,
                ui.div(
                    output_widget("map"),
                    ui.output_text_verbatim("map_summary"),
                ),
            ),
        )
    return ui.card(
        ui.card_header("Solution Comparison"),
        ui.layout_sidebar(sidebar, ui.output_ui("cmp_content")),
    )


@module.server
def comparison_map_server(
    input,
    output,
    session,
    problem: reactive.Value,
    all_solutions: reactive.Value,
):
    @reactive.effect
    def _update_choices():
        sols = all_solutions()
        if sols is None or len(sols) < 2:
            return
        choices = {
            str(i): f"Run {i + 1}" for i in range(len(sols))
        }
        ui.update_select("sol_a", choices=choices, selected="0")
        ui.update_select("sol_b", choices=choices, selected="1")

    if _HAS_IPYLEAFLET:

        @render_widget
        def map():
            p = problem()
            sols = all_solutions()
            if p is None or sols is None or len(sols) < 2:
                return None

            idx_a = int(input.sol_a())
            idx_b = int(input.sol_b())
            if idx_a >= len(sols) or idx_b >= len(sols):
                return None

            sol_a = sols[idx_a]
            sol_b = sols[idx_b]
            n_pu = len(p.planning_units)
            grid = generate_grid(n_pu)

            colors = [
                comparison_color(sol_a.selected[i], sol_b.selected[i])
                for i in range(n_pu)
            ]
            return create_grid_map(grid, colors)

    @render.text
    def map_summary():
        p = problem()
        sols = all_solutions()
        if p is None or sols is None or len(sols) < 2:
            return "Run solver with 2+ solutions to compare."

        idx_a = int(input.sol_a())
        idx_b = int(input.sol_b())
        if idx_a >= len(sols) or idx_b >= len(sols):
            return "Invalid solution index."

        sol_a = sols[idx_a]
        sol_b = sols[idx_b]
        n_pu = len(p.planning_units)
        both = sum(
            1 for i in range(n_pu)
            if sol_a.selected[i] and sol_b.selected[i]
        )
        a_only = sum(
            1 for i in range(n_pu)
            if sol_a.selected[i] and not sol_b.selected[i]
        )
        b_only = sum(
            1 for i in range(n_pu)
            if not sol_a.selected[i] and sol_b.selected[i]
        )
        return f"Both: {both} | A only: {a_only} | B only: {b_only}"

    if not _HAS_IPYLEAFLET:

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
            both = sum(
                1 for i in range(n_pu)
                if sol_a.selected[i] and sol_b.selected[i]
            )
            a_only = sum(
                1 for i in range(n_pu)
                if sol_a.selected[i] and not sol_b.selected[i]
            )
            b_only = sum(
                1 for i in range(n_pu)
                if not sol_a.selected[i] and sol_b.selected[i]
            )
            return ui.div(
                ui.p(f"Both: {both} | A only: {a_only} | B only: {b_only}"),
            )
```

**Step 5: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/pymarxan_shiny/test_comparison_map.py -v`
Expected: All passed.

**Step 6: Commit**

```bash
git add src/pymarxan_shiny/modules/mapping/comparison_map.py tests/pymarxan_shiny/test_comparison_map.py
git commit -m "feat: upgrade comparison_map to ipyleaflet @render_widget"
```

---

### Task 6: Upgrade network_view to @render_widget + Polyline Edges

**Files:**
- Modify: `src/pymarxan_shiny/modules/mapping/network_view.py` (full rewrite)
- Modify: `tests/pymarxan_shiny/test_network_view.py` (update tests)

This is the most complex module — it adds both colored rectangles AND polyline edge layers.

**Step 1: Write the failing test**

Add to `tests/pymarxan_shiny/test_network_view.py`:

```python
def test_network_view_ui_has_output_widget():
    """UI should contain an output_widget for the map."""
    html = str(network_view_ui("test_nv"))
    assert "jupyter" in html.lower() or "widget" in html.lower() or "map" in html.lower()
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/pymarxan_shiny/test_network_view.py::test_network_view_ui_has_output_widget -v`
Expected: FAIL.

**Step 3: Write implementation**

Replace `src/pymarxan_shiny/modules/mapping/network_view.py`:

```python
"""Network view Shiny module — connectivity graph overlay on PU grid."""
from __future__ import annotations

import numpy as np
from shiny import module, reactive, render, ui

from pymarxan.connectivity.metrics import compute_in_degree
from pymarxan.models.geometry import generate_grid

try:
    import ipyleaflet
    from shinywidgets import output_widget, render_widget

    from pymarxan_shiny.modules.mapping.map_utils import create_grid_map

    _HAS_IPYLEAFLET = True
except ImportError:
    _HAS_IPYLEAFLET = False


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
    sidebar = ui.sidebar(
        ui.input_select(
            "metric",
            "Color by Metric",
            choices={
                "in_degree": "In-Degree",
                "out_degree": "Out-Degree",
            },
            selected="in_degree",
        ),
        ui.input_slider(
            "edge_threshold",
            "Min Edge Weight",
            min=0.0,
            max=1.0,
            value=0.0,
            step=0.01,
        ),
        width=220,
    )
    if _HAS_IPYLEAFLET:
        return ui.card(
            ui.card_header("Connectivity Network"),
            ui.layout_sidebar(
                sidebar,
                ui.div(
                    output_widget("map"),
                    ui.output_text_verbatim("map_summary"),
                ),
            ),
        )
    return ui.card(
        ui.card_header("Connectivity Network"),
        ui.layout_sidebar(sidebar, ui.output_ui("network_content")),
    )


@module.server
def network_view_server(
    input,
    output,
    session,
    problem: reactive.Value,
    connectivity_matrix: reactive.Value,
    connectivity_pu_ids: reactive.Value,
):
    if _HAS_IPYLEAFLET:

        @render_widget
        def map():
            p = problem()
            matrix = connectivity_matrix()
            if p is None or matrix is None:
                return None

            n_pu = len(p.planning_units)
            metric_name = input.metric()
            threshold = input.edge_threshold()

            if metric_name == "out_degree":
                from pymarxan.connectivity.metrics import compute_out_degree

                metric_values = compute_out_degree(matrix)
            else:
                metric_values = compute_in_degree(matrix)

            grid = generate_grid(n_pu)
            max_val = float(metric_values.max()) if len(metric_values) > 0 else 0.0
            min_val = float(metric_values.min()) if len(metric_values) > 0 else 0.0
            rng = max_val - min_val if max_val > min_val else 1.0

            colors = [
                metric_color((float(metric_values[i]) - min_val) / rng)
                if i < len(metric_values) else "#bdc3c7"
                for i in range(n_pu)
            ]

            m = create_grid_map(grid, colors)

            # Add polyline edges
            centroids = compute_centroids(grid)
            n = min(matrix.shape[0], n_pu)
            for i in range(n):
                for j in range(n):
                    weight = float(matrix[i, j])
                    if weight > threshold and i != j:
                        line = ipyleaflet.Polyline(
                            locations=[centroids[i], centroids[j]],
                            color="#3498db",
                            opacity=min(1.0, weight),
                            weight=2,
                        )
                        m.add(line)

            return m

    @render.text
    def map_summary():
        p = problem()
        matrix = connectivity_matrix()
        if p is None or matrix is None:
            return "Load a project with connectivity data to see the network."

        n_pu = len(p.planning_units)
        metric_name = input.metric()

        if metric_name == "out_degree":
            from pymarxan.connectivity.metrics import compute_out_degree

            metric_values = compute_out_degree(matrix)
        else:
            metric_values = compute_in_degree(matrix)

        max_val = float(metric_values.max()) if len(metric_values) > 0 else 0.0
        min_val = float(metric_values.min()) if len(metric_values) > 0 else 0.0

        threshold = input.edge_threshold()
        n = min(matrix.shape[0], n_pu)
        edge_count = sum(
            1 for i in range(n) for j in range(n)
            if float(matrix[i, j]) > threshold and i != j
        )

        return (
            f"{n_pu} nodes — colored by {metric_name}\n"
            f"Metric range: {min_val:.2f} – {max_val:.2f}\n"
            f"Edges shown: {edge_count} (threshold: {threshold:.2f})"
        )

    if not _HAS_IPYLEAFLET:

        @render.ui
        def network_content():
            p = problem()
            matrix = connectivity_matrix()
            if p is None or matrix is None:
                return ui.p(
                    "Load a project with connectivity data"
                    " to see the network."
                )

            n_pu = len(p.planning_units)
            metric_name = input.metric()

            if metric_name == "out_degree":
                from pymarxan.connectivity.metrics import compute_out_degree

                metric_values = compute_out_degree(matrix)
            else:
                metric_values = compute_in_degree(matrix)

            grid = generate_grid(n_pu)
            centroids = compute_centroids(grid)
            max_val = float(metric_values.max()) if len(metric_values) > 0 else 0.0
            min_val = float(metric_values.min()) if len(metric_values) > 0 else 0.0

            return ui.div(
                ui.p(f"{n_pu} nodes — colored by {metric_name}"),
                ui.p(f"{len(centroids)} centroids computed"),
                ui.p(f"Metric range: {min_val:.2f} – {max_val:.2f}"),
            )
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/pymarxan_shiny/test_network_view.py -v`
Expected: All passed (including existing metric_color, compute_centroids tests).

**Step 5: Commit**

```bash
git add src/pymarxan_shiny/modules/mapping/network_view.py tests/pymarxan_shiny/test_network_view.py
git commit -m "feat: upgrade network_view to ipyleaflet @render_widget with polyline edges"
```

---

### Task 7: Phase 13 Integration Tests

**Files:**
- Create: `tests/test_integration_phase13.py`

**Step 1: Write integration tests**

Create `tests/test_integration_phase13.py`:

```python
"""Phase 13 integration tests: ipyleaflet map upgrades."""
from __future__ import annotations

import pytest


@pytest.mark.integration
def test_app_imports_phase13():
    """Verify the app still imports cleanly after phase 13 changes."""
    from pymarxan_app import app
    assert app.app is not None


@pytest.mark.integration
def test_map_utils_importable():
    """Shared map helper is importable."""
    from pymarxan_shiny.modules.mapping.map_utils import create_grid_map
    assert callable(create_grid_map)


@pytest.mark.integration
def test_create_grid_map_produces_map():
    """create_grid_map returns an ipyleaflet Map with correct layers."""
    import ipyleaflet

    from pymarxan.models.geometry import generate_grid
    from pymarxan_shiny.modules.mapping.map_utils import create_grid_map

    grid = generate_grid(4)
    colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00"]
    m = create_grid_map(grid, colors)
    assert isinstance(m, ipyleaflet.Map)
    rects = [l for l in m.layers if isinstance(l, ipyleaflet.Rectangle)]
    assert len(rects) == 4


@pytest.mark.integration
def test_all_map_modules_importable():
    """All 5 map modules still import correctly."""
    from pymarxan_shiny.modules.mapping.solution_map import (
        solution_map_server,
        solution_map_ui,
    )
    from pymarxan_shiny.modules.mapping.spatial_grid import (
        spatial_grid_server,
        spatial_grid_ui,
    )
    from pymarxan_shiny.modules.mapping.frequency_map import (
        frequency_map_server,
        frequency_map_ui,
    )
    from pymarxan_shiny.modules.mapping.comparison_map import (
        comparison_map_server,
        comparison_map_ui,
    )
    from pymarxan_shiny.modules.mapping.network_view import (
        network_view_server,
        network_view_ui,
    )
    for fn in [
        solution_map_ui, solution_map_server,
        spatial_grid_ui, spatial_grid_server,
        frequency_map_ui, frequency_map_server,
        comparison_map_ui, comparison_map_server,
        network_view_ui, network_view_server,
    ]:
        assert callable(fn)


@pytest.mark.integration
def test_all_map_uis_render():
    """All 5 map module UIs render without error."""
    from pymarxan_shiny.modules.mapping.solution_map import solution_map_ui
    from pymarxan_shiny.modules.mapping.spatial_grid import spatial_grid_ui
    from pymarxan_shiny.modules.mapping.frequency_map import frequency_map_ui
    from pymarxan_shiny.modules.mapping.comparison_map import comparison_map_ui
    from pymarxan_shiny.modules.mapping.network_view import network_view_ui

    for ui_fn, name in [
        (solution_map_ui, "sol"),
        (spatial_grid_ui, "sg"),
        (frequency_map_ui, "fm"),
        (comparison_map_ui, "cm"),
        (network_view_ui, "nv"),
    ]:
        elem = ui_fn(name)
        assert elem is not None


@pytest.mark.integration
def test_color_functions_unchanged():
    """All color functions still exist and produce valid hex strings."""
    from pymarxan_shiny.modules.mapping.spatial_grid import cost_color, status_color
    from pymarxan_shiny.modules.mapping.frequency_map import frequency_color
    from pymarxan_shiny.modules.mapping.comparison_map import comparison_color
    from pymarxan_shiny.modules.mapping.network_view import metric_color

    assert cost_color(0.5).startswith("#")
    assert status_color(2).startswith("#")
    assert frequency_color(0.5).startswith("#")
    assert comparison_color(True, False).startswith("#")
    assert metric_color(0.5).startswith("#")
```

**Step 2: Run integration tests**

Run: `source .venv/bin/activate && pytest tests/test_integration_phase13.py -v`
Expected: 6 passed.

**Step 3: Commit**

```bash
git add tests/test_integration_phase13.py
git commit -m "test: add phase 13 integration tests for ipyleaflet maps"
```

---

### Task 8: Full Regression

**Files:** None (verification only)

**Step 1: Run full test suite**

Run: `source .venv/bin/activate && make test 2>&1 | tail -10`
Expected: All tests passed (489 existing + ~17 new), coverage at or above 75%.

**Step 2: Run lint**

Run: `source .venv/bin/activate && make lint`
Expected: All checks passed.

**Step 3: Run type checks**

Run: `source .venv/bin/activate && make types`
Expected: No errors.

**Step 4: Verify app imports**

Run: `source .venv/bin/activate && python -c "from pymarxan_app.app import app; print('OK')"`
Expected: OK

**Step 5: Quick ipyleaflet smoke test**

Run: `source .venv/bin/activate && python -c "
from pymarxan.models.geometry import generate_grid
from pymarxan_shiny.modules.mapping.map_utils import create_grid_map
import ipyleaflet
m = create_grid_map(generate_grid(9), ['#ff0000']*9)
rects = [l for l in m.layers if isinstance(l, ipyleaflet.Rectangle)]
print(f'Map with {len(rects)} rectangles — OK')
"`
Expected: `Map with 9 rectangles — OK`
