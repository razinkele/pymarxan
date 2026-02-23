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
