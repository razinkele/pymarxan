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
