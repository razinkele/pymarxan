# Zonation Phase D — Shiny panel + wiring implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface Zonation in the Shiny app — a standalone "Zonation" tab (priority-rank choropleth + performance-curve plot) and a solver-picker entry so it's runnable in the main run flow.

**Architecture:** A `@module.ui`/`@module.server` panel (template: `rivers_panel`) reading the reactive `problem`; the map reuses `frequency_map`'s 0–1 choropleth (`frequency_color` + `create_geo_map`/`create_grid_map`, gated on `has_geometry`); the plot is server-side `@render.plot` matplotlib. Wiring: add `"zonation"` to the picker choices + a config panel, and a branch in `app.py::active_solver`. No new core deps.

**Tech Stack:** Shiny for Python, shinywidgets/ipyleaflet, matplotlib, pandas.

**Design spec:** `docs/plans/2026-07-14-zonation-phase-d-design.md` (read it first).

## Global Constraints

- Python 3.12+, `from __future__ import annotations`, full type hints on new module-level functions.
- Tests **must** run under the `shiny` micromamba env: `/opt/micromamba/envs/shiny/bin/pytest` (needs ipyleaflet/rasterio).
- Lint: ruff (E, F, I, UP; line length 99). Types: mypy clean. Coverage threshold 75% — `src/pymarxan_app/` is **excluded** from coverage, and Shiny module *servers* are lightly covered by convention; the **pure module-level helpers carry the real unit tests**.
- **Every plot is server-side `@render.plot` matplotlib** — never plotly-via-`render.ui` (produces blank charts; the hard-won lesson). Maps use `@render_widget` (shinywidgets/ipyleaflet), gated on `_HAS_IPYLEAFLET` + `has_geometry`.
- The bar before done: `make check` green + a Playwright live pass of the Zonation tab.
- Reused: `frequency_color` (`modules/mapping/frequency_map.py:22`), `create_geo_map`/`create_grid_map` (`modules/mapping/map_utils.py`), `generate_grid` (`models/geometry.py`), `has_geometry` (`models/problem.py:325`), `rank_removal`/`ZonationResult` (`pymarxan.zonation`).

## File Structure

- Create: `src/pymarxan_shiny/modules/zonation/__init__.py` (exports the ui/server).
- Create: `src/pymarxan_shiny/modules/zonation/zonation_panel.py` — panel + pure helpers.
- Modify: `src/pymarxan_shiny/modules/solver_config/solver_picker.py` — choice + config panel + config assembly.
- Modify: `src/pymarxan_app/app.py` — import, `active_solver` branch, nav_panel, server wiring.
- Create: `tests/pymarxan_shiny/test_zonation_panel.py`.
- Modify: `CHANGELOG.md`.

---

### Task 1: The Zonation panel module

**Files:**
- Create: `src/pymarxan_shiny/modules/zonation/__init__.py`
- Create: `src/pymarxan_shiny/modules/zonation/zonation_panel.py`
- Test: `tests/pymarxan_shiny/test_zonation_panel.py`

**Interfaces:**
- Produces: `zonation_panel_ui`, `zonation_panel_server` (module UI/server); pure helpers `rank_to_colors(priority_rank, pu_ids) -> list[str]` and `performance_curve_frame(result) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing tests**

Create `tests/pymarxan_shiny/test_zonation_panel.py`:

```python
"""Tests for the Zonation Shiny panel (Phase D)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pymarxan.models.problem import ConservationProblem
from pymarxan.zonation import rank_removal


def _problem():
    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [1.0, 1.0, 1.0], "status": [0, 0, 0]})
    feats = pd.DataFrame(
        {"id": [1, 2], "name": ["a", "b"], "target": [1.0, 1.0], "spf": [1.0, 1.0]}
    )
    pvf = pd.DataFrame(
        {"species": [1, 2, 1, 2], "pu": [1, 2, 3, 3], "amount": [10.0, 10.0, 5.0, 5.0]}
    )
    return ConservationProblem(pu, feats, pvf)


def test_rank_to_colors_aligned_and_hex():
    from pymarxan_shiny.modules.zonation.zonation_panel import rank_to_colors

    ranks = {1: 1 / 3, 2: 2 / 3, 3: 1.0}
    colors = rank_to_colors(ranks, [1, 2, 3])
    assert len(colors) == 3
    assert all(c.startswith("#") and len(c) == 7 for c in colors)
    # higher rank -> darker (lower luminance) via the ocean palette
    lum = [int(c[1:3], 16) + int(c[3:5], 16) + int(c[5:7], 16) for c in colors]
    assert lum[0] > lum[1] > lum[2]  # rank 1/3 lightest, rank 1.0 darkest


def test_rank_to_colors_missing_pu_defaults_light():
    from pymarxan_shiny.modules.zonation.zonation_panel import rank_to_colors

    colors = rank_to_colors({1: 1.0}, [1, 99])  # 99 not ranked
    assert colors[0] != colors[1]  # ranked vs default differ


def test_performance_curve_frame_columns():
    from pymarxan_shiny.modules.zonation.zonation_panel import performance_curve_frame

    res = rank_removal(_problem(), rule="caz")
    df = performance_curve_frame(res)
    assert "prop_landscape_remaining" in df.columns
    assert any(c.startswith("feat_") for c in df.columns)


def test_module_exposes_ui_and_server():
    from pymarxan_shiny.modules.zonation import (
        zonation_panel_server,
        zonation_panel_ui,
    )

    assert callable(zonation_panel_ui)
    assert callable(zonation_panel_server)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan_shiny/test_zonation_panel.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pymarxan_shiny.modules.zonation'`.

- [ ] **Step 3: Implement the module**

Create `src/pymarxan_shiny/modules/zonation/__init__.py`:

```python
"""Zonation priority-ranking Shiny panel."""
from __future__ import annotations

from pymarxan_shiny.modules.zonation.zonation_panel import (
    zonation_panel_server,
    zonation_panel_ui,
)

__all__ = ["zonation_panel_server", "zonation_panel_ui"]
```

Create `src/pymarxan_shiny/modules/zonation/zonation_panel.py`:

```python
"""Zonation priority-ranking Shiny panel (Phase D).

Given the reactive ``problem``, ranks every planning unit by iterative backward
removal (CAZ/ABF) and shows a priority-rank choropleth + per-feature performance
curves. Mirrors ``rivers_panel`` / ``frequency_map`` conventions; the plot is
server-side matplotlib (never plotly-via-render.ui).
"""
from __future__ import annotations

import pandas as pd
from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from pymarxan.models.geometry import generate_grid
from pymarxan.models.problem import has_geometry
from pymarxan.zonation import ZonationResult, rank_removal
from pymarxan_shiny.modules.mapping.frequency_map import frequency_color

try:
    from shinywidgets import output_widget, render_widget

    from pymarxan_shiny.modules.mapping.map_utils import (
        create_geo_map,
        create_grid_map,
    )

    _HAS_IPYLEAFLET = True
except ImportError:
    _HAS_IPYLEAFLET = False

_RULES = {
    "caz": "Core-area (CAZ) — rarity",
    "abf": "Additive benefit (ABF) — richness",
}


def rank_to_colors(priority_rank: dict[int, float], pu_ids: list[int]) -> list[str]:
    """Hex color per PU (aligned to ``pu_ids``) from its priority rank, via the
    ocean palette (high rank = dark navy). Unranked PUs default to rank 0."""
    return [frequency_color(priority_rank.get(int(pid), 0.0)) for pid in pu_ids]


def performance_curve_frame(result: ZonationResult) -> pd.DataFrame:
    """The performance-curve DataFrame the plot draws."""
    return result.performance_curves


@module.ui
def zonation_panel_ui():
    map_output = (
        output_widget("map") if _HAS_IPYLEAFLET else ui.output_ui("map_msg")
    )
    return ui.card(
        ui.card_header("Zonation (priority ranking)"),
        ui.p(
            "Rank every planning unit by iterative backward removal. CAZ favors "
            "rarity (protects every feature's core); ABF favors species-rich "
            "cells. Darker = higher priority. Click Rank to compute.",
            class_="text-muted small mb-3",
        ),
        ui.input_select("rule", "Removal rule", _RULES),
        ui.input_slider(
            "top_fraction", "Top priority fraction", min=0.0, max=1.0,
            value=0.3, step=0.05,
        ),
        ui.input_action_button("rank", "Rank", class_="btn-primary mb-3"),
        map_output,
        ui.output_plot("curves"),
        ui.output_text("summary"),
        ui.output_data_frame("top_table"),
    )


@module.server
def zonation_panel_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    problem: reactive.Value,
):
    _result: reactive.Value = reactive.value(None)

    @reactive.effect
    @reactive.event(input.rank)
    def _compute():
        p = problem()
        _result.set(rank_removal(p, rule=input.rule()) if p is not None else None)

    if _HAS_IPYLEAFLET:

        @render_widget
        def map():
            p = problem()
            res = _result()
            if p is None or res is None:
                return None
            pu_ids = [int(x) for x in p.planning_units["id"]]
            colors = rank_to_colors(res.priority_rank, pu_ids)
            if has_geometry(p):
                return create_geo_map(p.planning_units, colors)
            return create_grid_map(generate_grid(len(pu_ids)), colors)

    else:

        @render.ui
        def map_msg():
            return ui.p("Install ipyleaflet for the priority-rank map.")

    @render.plot
    def curves():
        import matplotlib.pyplot as plt

        res = _result()
        fig, ax = plt.subplots()
        if res is None:
            ax.text(0.5, 0.5, "Click Rank to compute.", ha="center", va="center")
            ax.axis("off")
            return fig
        df = performance_curve_frame(res)
        x = df["prop_landscape_remaining"]
        for col in (c for c in df.columns if c.startswith("feat_")):
            ax.plot(x, df[col], label=col)
        ax.set_xlabel("Proportion of landscape remaining")
        ax.set_ylabel("Proportion of feature retained")
        ax.invert_xaxis()  # remaining goes 1 -> 0 as worst cells are stripped
        ax.legend(fontsize="small")
        return fig

    @render.text
    def summary():
        res = _result()
        if res is None:
            return "Click Rank to compute the priority ranking."
        f = float(input.top_fraction())
        n_top = len(res.top_fraction(f)) if f > 0 else 0
        return (
            f"{res.rule.upper()}: top {f:.0%} priority = {n_top} of "
            f"{len(res.priority_rank)} planning units."
        )

    @render.data_frame
    def top_table():
        res = _result()
        if res is None:
            return pd.DataFrame({"pu_id": [], "priority_rank": []})
        df = res.to_dataframe().sort_values("priority_rank", ascending=False)
        return df[["pu_id", "priority_rank"]].head(20)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan_shiny/test_zonation_panel.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/pymarxan_shiny/modules/zonation/ tests/pymarxan_shiny/test_zonation_panel.py
git commit -m "feat(zonation): Shiny panel — priority-rank choropleth + performance-curve plot"
```

---

### Task 2: Solver-picker + app wiring

**Files:**
- Modify: `src/pymarxan_shiny/modules/solver_config/solver_picker.py`
- Modify: `src/pymarxan_app/app.py`
- Test: `tests/pymarxan_shiny/test_zonation_panel.py` (append the picker-choice test)

- [ ] **Step 1: Write the failing picker-choice test**

Append to `tests/pymarxan_shiny/test_zonation_panel.py`:

```python
def test_solver_picker_offers_zonation():
    # the picker's choices dict includes zonation
    import inspect

    from pymarxan_shiny.modules.solver_config import solver_picker

    src = inspect.getsource(solver_picker)
    assert '"zonation"' in src  # solver_choices entry present
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan_shiny/test_zonation_panel.py::test_solver_picker_offers_zonation -v`
Expected: FAIL — `"zonation"` not yet in the picker source.

- [ ] **Step 3: Wire the solver picker**

In `src/pymarxan_shiny/modules/solver_config/solver_picker.py`:

(a) After `solver_choices["pipeline"] = "Run Mode Pipeline"` (and the binary line), add:

```python
    solver_choices["zonation"] = "Zonation (rank-removal)"
```

(b) Add a config panel — near the other `ui.panel_conditional` blocks (e.g. after the `greedy` one):

```python
                ui.panel_conditional(
                    "input.solver_type === 'zonation'",
                    ui.input_select(
                        "zonation_rule", "Removal rule",
                        choices={
                            "caz": "Core-area (CAZ) — rarity",
                            "abf": "Additive benefit (ABF) — richness",
                        },
                        selected="caz",
                    ),
                    ui.input_numeric(
                        "zonation_top_fraction", "Top priority fraction",
                        value=0.3, min=0.0, max=1.0, step=0.05,
                    ),
                ),
```

(c) Add the two inputs to the `@reactive.event(...)` dependency list (before `ignore_init=False`):

```python
        input.zonation_rule, input.zonation_top_fraction,
```

(d) In `_update_config`, assemble the zonation keys (mirroring the `greedy` block):

```python
        if input.solver_type() == "zonation":
            config["zonation_rule"] = input.zonation_rule()
            config["zonation_top_fraction"] = float(input.zonation_top_fraction() or 0.3)
```

- [ ] **Step 4: Wire the run flow + nav tab in `app.py`**

In `src/pymarxan_app/app.py`:

(a) Add the import (near the rivers import, `app.py:79`):

```python
from pymarxan_shiny.modules.zonation import zonation_panel_server, zonation_panel_ui
```

(b) In `active_solver()` (the `reactive.calc`, app.py:255-276), add before the final `return MIPSolver()`:

```python
        elif st == "zonation":
            from pymarxan.solvers.zonation_solver import ZonationSolver
            return ZonationSolver(
                rule=config_dict.get("zonation_rule", "caz"),
                top_fraction=config_dict.get("zonation_top_fraction", 0.3),
            )
```

(c) Add a nav panel near the Rivers one (`app.py:191` context):

```python
    ui.nav_panel("Zonation", zonation_panel_ui("zonation")),
```

(d) Wire the panel server near the rivers server call (`app.py:234`):

```python
    zonation_panel_server("zonation", problem)
```

- [ ] **Step 5: Run the panel tests + import the app to verify wiring loads**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan_shiny/test_zonation_panel.py -v`
Expected: PASS (5 tests).

Run: `/opt/micromamba/envs/shiny/bin/python -c "import pymarxan_app.app"`
Expected: no exception (the app module imports with the new wiring).

- [ ] **Step 6: Commit**

```bash
git add src/pymarxan_shiny/modules/solver_config/solver_picker.py src/pymarxan_app/app.py tests/pymarxan_shiny/test_zonation_panel.py
git commit -m "feat(zonation): wire Zonation into the solver picker + a Zonation nav tab"
```

---

### Task 3: CHANGELOG + make check + live Playwright verification

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: CHANGELOG entry**

Under `## [Unreleased]` → `### Added` in `CHANGELOG.md`:

```markdown
- **Zonation Shiny tab + solver-picker entry (Phase D).** A "Zonation" tab shows
  the priority-rank choropleth + per-feature performance curves for the loaded
  problem (CAZ/ABF), and "Zonation (rank-removal)" is selectable in the solver
  picker — completing Zonation end-to-end across the stack. +5 tests.
```

- [ ] **Step 2: Full check**

Run: `PATH="/opt/micromamba/envs/shiny/bin:$HOME/.local/bin:$PWD/.venv/bin:$PATH" make check`
Expected: green — 0 ruff, 0 mypy, full suite + 5 new. (`test_solutions_are_different` flake → rerun once.)

- [ ] **Step 3: Live Playwright verification (required)**

Launch the app and drive it (the run-flow wiring is app-internal / coverage-excluded, so this is its real test):

```bash
# launch (run_in_background Bash, simple '&'+echo form — the nohup/pkill variants exit 144):
source .venv/bin/activate 2>/dev/null; \
  /opt/micromamba/envs/shiny/bin/shiny run src/pymarxan_app/app.py --port 8000 &
echo "launched"
```

Then with Playwright: navigate to `http://localhost:8000`, load the `tests/data/simple` project (or the demo), open the **Zonation** tab, pick a rule, click **Rank**, and confirm: 0 console errors, the performance-curve plot renders (matplotlib `<img>`), and the priority-rank map or its fallback message renders. Also select "Zonation (rank-removal)" in the solver picker, run, and confirm a reserve appears in Results. Screenshot to `docs/images/zonation_tab.png`. Stop the server when done.

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md docs/images/zonation_tab.png
git commit -m "docs(zonation): Phase D CHANGELOG + Zonation tab screenshot"
```

---

## Post-plan notes

- **Design review:** the user requested `multi-agent-design-review` before executing — worth a pass on the wiring (the `active_solver` branch, the picker config keys, the `reactive.event` compute pattern) and the map/plot reuse.
- **Parity:** pure UI + wiring, no solver/objective math. 35.0 anchor untouched; `make check` runs the parity tests.
- **Deferred:** smoothing controls in the UI (needs `SmoothingSpec.from_problem` geometry derivation). This is the **last** Zonation phase — after it, Zonation is complete end-to-end (core → solver → smoothing → UI).
