"""GIS file import wizard Shiny module."""
from __future__ import annotations

import pandas as pd
from shiny import module, reactive, render, ui

from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup
from pymarxan.models.problem import ConservationProblem
from pymarxan.spatial.grid import compute_adjacency
from pymarxan.spatial.importers import import_planning_units

_COMMON_GEOSPATIAL_EXTS = (".shp", ".geojson", ".gpkg", ".json", ".zip")


def _read_columns(path: str) -> list[str]:
    """Read column names from a geospatial file without loading all rows."""
    try:
        import geopandas as gpd
        gdf = gpd.read_file(path, rows=0)
        return [c for c in gdf.columns if c != "geometry"]
    except Exception:
        return []


@module.ui
def import_wizard_ui():
    return ui.card(
        help_card_header("Import Planning Units from GIS File"),
        ui.p(
            "Import planning units from a geospatial file (Shapefile, GeoJSON, "
            "or GeoPackage). Each polygon becomes one planning unit. Map the "
            "attribute columns to Marxan's required fields: a unique ID, a cost "
            "value, and an optional status code (0=available, 2=locked-in, "
            "3=locked-out).",
            class_="text-muted small mb-3",
        ),
        ui.tooltip(
            ui.input_file(
                "pu_file",
                "Upload File (.geojson, .gpkg, or .zip)",
                accept=[".geojson", ".gpkg", ".json", ".zip"],
            ),
            "Upload a geospatial vector file containing planning unit polygons. "
            "Supported formats: GeoJSON, GeoPackage (.gpkg), or a .zip archive "
            "containing a Shapefile bundle (.shp + .shx + .dbf + .prj).",
        ),
        ui.layout_columns(
            ui.tooltip(
                ui.input_selectize(
                    "id_col", "ID Column",
                    choices=["id"], selected="id",
                ),
                "Column containing the unique planning unit identifier. "
                "Corresponds to Marxan's PUID field.",
            ),
            ui.tooltip(
                ui.input_selectize(
                    "cost_col", "Cost Column",
                    choices=["cost"], selected="cost",
                ),
                "Column containing the cost of including each planning unit "
                "in the reserve network (e.g. area, land price, opportunity cost).",
            ),
            ui.tooltip(
                ui.input_selectize(
                    "status_col", "Status Column (optional)",
                    choices=["status", ""], selected="status",
                ),
                "Column with Marxan status codes: 0=available, 1=initial include, "
                "2=locked-in (must be selected), 3=locked-out (never selected). "
                "Leave blank if all PUs are available.",
            ),
            col_widths=[4, 4, 4],
        ),
        ui.tooltip(
            ui.input_action_button(
                "import_btn", "Import", class_="btn-primary"
            ),
            "Parse the uploaded file and create planning units with adjacency boundaries.",
        ),
        ui.output_text_verbatim("import_info"),
    )


@module.server
def import_wizard_server(input, output, session, problem: reactive.Value):
    help_server_setup(input, "import_wizard")

    @reactive.effect
    @reactive.event(input.pu_file)
    def _update_column_choices():
        """Populate column dropdowns from the uploaded file's headers."""
        file_info = input.pu_file()
        if not file_info:
            return
        cols = _read_columns(file_info[0]["datapath"])
        if not cols:
            return
        ui.update_selectize("id_col", choices=cols,
                            selected=cols[0] if "id" not in cols else "id")
        ui.update_selectize("cost_col", choices=cols,
                            selected="cost" if "cost" in cols else cols[0])
        status_choices = [""] + cols
        ui.update_selectize("status_col", choices=status_choices,
                            selected="status" if "status" in cols else "")

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
