"""Solution map Shiny module — ipyleaflet map of selected planning units."""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup
from pymarxan_shiny.modules.mapping.ocean_palette import MAP_SELECTED, MAP_NOT_SEL
from pymarxan.models.geometry import generate_grid
from pymarxan.models.problem import has_geometry

try:
    from shinywidgets import output_widget, render_widget

    from pymarxan_shiny.modules.mapping.map_utils import create_geo_map, create_grid_map

    _HAS_IPYLEAFLET = True
except ImportError:
    _HAS_IPYLEAFLET = False


@module.ui
def solution_map_ui():
    if _HAS_IPYLEAFLET:
        return ui.card(
            help_card_header("Solution Map"),
            ui.p(
                "Map of the current solution. Green planning units are selected "
                "for the reserve network; gray units are not selected. "
                "Run a solver to populate this map.",
                class_="text-muted small mb-3",
            ),
            output_widget("map"),
            ui.output_text_verbatim("map_summary"),
        )
    return ui.card(
        help_card_header("Solution Map"),
        ui.p(
            "Solution map (install ipyleaflet for interactive maps).",
            class_="text-muted small mb-3",
        ),
        ui.output_ui("map_content"),
    )


@module.server
def solution_map_server(
    input, output, session,
    problem: reactive.Value,
    solution: reactive.Value,
):
    help_server_setup(input, "solution_map")

    if _HAS_IPYLEAFLET:

        @render_widget
        def map():
            p = problem()
            s = solution()
            if p is None or s is None:
                return None

            n_pu = len(p.planning_units)
            colors = [
                MAP_SELECTED if s.selected[i] else MAP_NOT_SEL
                for i in range(n_pu)
            ]

            if has_geometry(p):
                return create_geo_map(p.planning_units, colors)

            grid = generate_grid(n_pu)
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
