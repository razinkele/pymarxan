"""Grid builder Shiny module — generate planning unit grids."""
from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from shiny import module, reactive, render, ui

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
        ui.card_header("Generate Planning Grid"),
        ui.layout_columns(
            ui.input_numeric("minx", "Min X (lon)", value=0.0),
            ui.input_numeric("miny", "Min Y (lat)", value=0.0),
            ui.input_numeric("maxx", "Max X (lon)", value=1.0),
            ui.input_numeric("maxy", "Max Y (lat)", value=1.0),
            col_widths=[3, 3, 3, 3],
        ),
        ui.layout_columns(
            ui.input_numeric("cell_size", "Cell Size", value=0.1, min=0.001),
            ui.input_select(
                "grid_type",
                "Grid Type",
                {"square": "Square", "hexagonal": "Hexagonal"},
            ),
            col_widths=[6, 6],
        ),
        ui.input_checkbox("clip_gadm", "Clip to GADM boundary", value=False),
        ui.input_action_button("generate", "Generate Grid", class_="btn-primary"),
        map_output,
        ui.output_text_verbatim("grid_info"),
    )


@module.server
def grid_builder_server(
    input, output, session, problem: reactive.Value,
    gadm_boundary: reactive.Value | Callable | None = None,
):
    @reactive.effect
    @reactive.event(input.generate)
    def _generate():
        bounds = (input.minx(), input.miny(), input.maxx(), input.maxy())
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
