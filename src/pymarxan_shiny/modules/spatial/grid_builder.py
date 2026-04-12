"""Grid builder Shiny module — generate planning unit grids."""
from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from shiny import module, reactive, render, ui

from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup
from pymarxan.models.problem import ConservationProblem
from pymarxan.spatial.grid import compute_adjacency, generate_planning_grid

try:
    from shinywidgets import output_widget, render_widget

    from pymarxan_shiny.modules.mapping.map_utils import create_geo_map

    _HAS_IPYLEAFLET = True
except ImportError:  # pragma: no cover
    _HAS_IPYLEAFLET = False


@module.ui
def grid_builder_ui():
    map_output = (
        output_widget("grid_map")
        if _HAS_IPYLEAFLET
        else ui.output_ui("grid_map_text")
    )
    return ui.card(
        help_card_header("Generate Planning Grid"),
        ui.p(
            "Create a regular grid of planning units over a geographic extent. "
            "Planning units are the fundamental spatial units in Marxan — each "
            "cell is either selected or not in the final reserve design. Smaller "
            "cell sizes give finer spatial resolution but increase computation time.",
            class_="text-muted small mb-3",
        ),
        ui.layout_columns(
            ui.tooltip(
                ui.input_numeric("minx", "Min X (lon)", value=0.0),
                "Western longitude of the bounding box (decimal degrees, "
                "e.g. -180 to 180).",
            ),
            ui.tooltip(
                ui.input_numeric("miny", "Min Y (lat)", value=0.0),
                "Southern latitude of the bounding box (decimal degrees, "
                "e.g. -90 to 90).",
            ),
            ui.tooltip(
                ui.input_numeric("maxx", "Max X (lon)", value=1.0),
                "Eastern longitude of the bounding box.",
            ),
            ui.tooltip(
                ui.input_numeric("maxy", "Max Y (lat)", value=1.0),
                "Northern latitude of the bounding box.",
            ),
            col_widths=[3, 3, 3, 3],
        ),
        ui.layout_columns(
            ui.tooltip(
                ui.input_numeric("cell_size", "Cell Size", value=0.1, min=0.001),
                "Width/height of each grid cell in the same units as coordinates "
                "(degrees for geographic CRS). Smaller = more planning units.",
            ),
            ui.tooltip(
                ui.input_select(
                    "grid_type",
                    "Grid Type",
                    {"square": "Square", "hexagonal": "Hexagonal"},
                ),
                "Square grids are simpler; hexagonal grids reduce edge effects "
                "and provide more uniform neighbour distances.",
            ),
            col_widths=[6, 6],
        ),
        ui.tooltip(
            ui.input_checkbox("clip_gadm", "Clip to GADM boundary", value=False),
            "If checked, grid cells outside the GADM administrative boundary "
            "(fetched in the Spatial tab) will be removed.",
        ),
        ui.tooltip(
            ui.input_action_button("generate", "Generate Grid", class_="btn-primary"),
            "Generate the planning unit grid and compute adjacency (boundary) edges "
            "between neighbouring cells.",
        ),
        map_output,
        ui.output_text_verbatim("grid_info"),
    )


@module.server
def grid_builder_server(
    input, output, session, problem: reactive.Value,
    gadm_boundary: reactive.Value | Callable | None = None,
):
    help_server_setup(input, "grid_builder")

    @reactive.effect
    @reactive.event(input.generate)
    def _generate():
        vals = [input.minx(), input.miny(), input.maxx(), input.maxy(), input.cell_size()]
        if any(v is None for v in vals):
            ui.notification_show(
                "Please fill in all numeric fields.", type="warning"
            )
            return
        bounds = (input.minx(), input.miny(), input.maxx(), input.maxy())
        if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            ui.notification_show(
                "Min coordinates must be less than Max coordinates.",
                type="error",
            )
            return
        clip_to = None
        if gadm_boundary is not None and input.clip_gadm():
            boundary = gadm_boundary()
            if boundary is not None and len(boundary) > 0:
                clip_to = boundary.union_all()
        gdf = generate_planning_grid(
            bounds=bounds,
            cell_size=input.cell_size(),
            grid_type=input.grid_type(),
            clip_to=clip_to,
        )
        if len(gdf) == 0:
            ui.notification_show(
                "No cells generated. Check bounds and cell size.", type="warning"
            )
            return

        boundary = compute_adjacency(gdf)
        p = ConservationProblem(
            planning_units=gdf,
            features=pd.DataFrame({"id": [], "name": [], "target": [], "spf": []}),
            pu_vs_features=pd.DataFrame({"species": [], "pu": [], "amount": []}),
            boundary=boundary if len(boundary) > 0 else None,
        )
        problem.set(p)
        ui.notification_show(
            f"Generated {len(gdf)} planning units.", type="message"
        )

    @render.text
    def grid_info():
        p = problem()
        if p is None:
            return "No grid generated yet."
        n = len(p.planning_units)
        has_bnd = p.boundary is not None and len(p.boundary) > 0
        return f"{n} planning units | Boundary edges: {len(p.boundary) if has_bnd else 0}"

    if _HAS_IPYLEAFLET:  # pragma: no cover

        @render_widget
        def grid_map():
            p = problem()
            if p is None:
                return None
            colors = ["#3498db"] * len(p.planning_units)
            return create_geo_map(p.planning_units, colors)

    if not _HAS_IPYLEAFLET:  # pragma: no cover

        @render.ui
        def grid_map_text():
            p = problem()
            if p is None:
                return ui.p("Generate a grid to see the preview.")
            return ui.p(f"Grid preview: {len(p.planning_units)} planning units")
