"""GIS file import wizard Shiny module."""
from __future__ import annotations

import pandas as pd
from shiny import module, reactive, render, ui

from pymarxan.models.problem import ConservationProblem
from pymarxan.spatial.grid import compute_adjacency
from pymarxan.spatial.importers import import_planning_units


@module.ui
def import_wizard_ui():
    return ui.card(
        ui.card_header("Import Planning Units from GIS File"),
        ui.input_file(
            "pu_file",
            "Upload File (.shp, .geojson, .gpkg)",
            accept=[".shp", ".geojson", ".gpkg", ".json", ".zip"],
        ),
        ui.layout_columns(
            ui.input_text("id_col", "ID Column", value="id"),
            ui.input_text("cost_col", "Cost Column", value="cost"),
            ui.input_text(
                "status_col", "Status Column (optional)", value="status"
            ),
            col_widths=[4, 4, 4],
        ),
        ui.input_action_button(
            "import_btn", "Import", class_="btn-primary"
        ),
        ui.output_text_verbatim("import_info"),
    )


@module.server
def import_wizard_server(input, output, session, problem: reactive.Value):

    @reactive.effect
    @reactive.event(input.import_btn)
    def _import():
        file_info = input.pu_file()
        if not file_info:
            ui.notification_show(
                "Please upload a file first.", type="warning"
            )
            return

        path = file_info[0]["datapath"]
        status_col = input.status_col() or None

        try:
            gdf = import_planning_units(
                path,
                id_column=input.id_col(),
                cost_column=input.cost_col(),
                status_column=status_col,
            )
            boundary = compute_adjacency(gdf)
            p = ConservationProblem(
                planning_units=gdf,
                features=pd.DataFrame(
                    {"id": [], "name": [], "target": [], "spf": []}
                ),
                pu_vs_features=pd.DataFrame(
                    {"species": [], "pu": [], "amount": []}
                ),
                boundary=boundary if len(boundary) > 0 else None,
            )
            problem.set(p)
            ui.notification_show(
                f"Imported {len(gdf)} planning units.", type="message"
            )
        except Exception as exc:
            ui.notification_show(f"Import error: {exc}", type="error")

    @render.text
    def import_info():
        p = problem()
        if p is None:
            return "Upload a GIS file and click Import."
        return f"{len(p.planning_units)} planning units loaded"
