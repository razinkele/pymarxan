"""Spatial grid Shiny module — PU map colored by cost or status."""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan.models.geometry import generate_grid


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
    input,
    output,
    session,
    problem: reactive.Value,
):
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
