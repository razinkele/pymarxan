"""Spatial grid Shiny module — PU map colored by cost or status."""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup
from pymarxan_shiny.modules.mapping.ocean_palette import (
    COST_LOW_RGB, COST_HIGH_RGB, MAP_AVAILABLE, MAP_LOCKED_IN,
    MAP_LOCKED_OUT, MAP_FALLBACK,
)
from pymarxan.models.geometry import generate_grid
from pymarxan.models.problem import has_geometry

try:
    from shinywidgets import output_widget, render_widget

    from pymarxan_shiny.modules.mapping.map_utils import create_geo_map, create_grid_map

    _HAS_IPYLEAFLET = True
except ImportError:
    _HAS_IPYLEAFLET = False


def cost_color(normalized: float) -> str:
    """Map a 0-1 normalized cost to a seafoam-to-coral hex color."""
    t = max(0.0, min(1.0, normalized))
    r = int(COST_LOW_RGB[0] * (1.0 - t) + COST_HIGH_RGB[0] * t)
    g = int(COST_LOW_RGB[1] * (1.0 - t) + COST_HIGH_RGB[1] * t)
    b = int(COST_LOW_RGB[2] * (1.0 - t) + COST_HIGH_RGB[2] * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def status_color(status: int) -> str:
    """Map a Marxan status code to a categorical color."""
    colors = {
        0: MAP_AVAILABLE,   # available — muted blue-gray
        1: MAP_AVAILABLE,   # available (initial included)
        2: MAP_LOCKED_IN,   # locked-in — kelp green
        3: MAP_LOCKED_OUT,  # locked-out — deep coral
    }
    return colors.get(status, MAP_FALLBACK)


@module.ui
def spatial_grid_ui():
    sidebar = ui.sidebar(
        ui.tooltip(
            ui.input_select(
                "color_by",
                "Color by",
                choices={"cost": "Cost", "status": "Status"},
                selected="cost",
            ),
            "Choose how to color planning units on the map. "
            "'Cost' shows a yellow-to-red gradient; 'Status' shows "
            "categorical colors (gray=available, green=locked-in, red=locked-out).",
        ),
        width=200,
    )
    if _HAS_IPYLEAFLET:
        return ui.card(
            help_card_header("Planning Unit Map"),
            ui.p(
                "Interactive map of all planning units colored by cost or status. "
                "Use this to visually inspect the spatial layout of your "
                "conservation planning region.",
                class_="text-muted small mb-3",
            ),
            ui.layout_sidebar(
                sidebar,
                ui.div(
                    output_widget("map"),
                    ui.output_text_verbatim("map_summary"),
                ),
            ),
        )
    return ui.card(
        help_card_header("Planning Unit Map"),
        ui.p(
            "Planning unit map (install ipyleaflet for interactive maps).",
            class_="text-muted small mb-3",
        ),
        ui.layout_sidebar(sidebar, ui.output_ui("grid_content")),
    )


@module.server
def spatial_grid_server(
    input,
    output,
    session,
    problem: reactive.Value,
):
    help_server_setup(input, "spatial_grid")

    if _HAS_IPYLEAFLET:

        @render_widget
        def map():
            p = problem()
            if p is None:
                return None

            n_pu = len(p.planning_units)
            color_mode = input.color_by()

            if color_mode == "status":
                statuses = p.planning_units["status"].tolist()
                colors = [status_color(s) for s in statuses]
            else:
                costs = p.planning_units["cost"].tolist()
                max_c = max(costs) if costs else 1.0
                colors = [
                    cost_color(c / max_c if max_c > 0 else 0.0) for c in costs
                ]

            if has_geometry(p):
                return create_geo_map(p.planning_units, colors)

            grid = generate_grid(n_pu)
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
