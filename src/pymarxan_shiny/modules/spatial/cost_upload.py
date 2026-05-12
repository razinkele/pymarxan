"""Cost surface upload Shiny module."""
from __future__ import annotations

import copy

import geopandas as gpd
from shiny import module, reactive, render, ui

from pymarxan.models.problem import has_geometry
from pymarxan.spatial.cost_surface import apply_cost_from_vector
from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup


def _read_vector_columns(path: str) -> list[str]:
    """Read column names from a vector file header (no full read)."""
    try:
        gdf = gpd.read_file(path, rows=0)
        return [c for c in gdf.columns if c != "geometry"]
    except Exception:
        return []


@module.ui
def cost_upload_ui():
    return ui.card(
        help_card_header("Custom Cost Surface"),
        ui.p(
            "Upload a vector layer containing cost values to replace the default "
            "planning unit costs. Costs represent what is sacrificed by including "
            "a planning unit in the reserve (e.g. land price, opportunity cost, "
            "area). The cost layer is intersected with planning units using the "
            "chosen aggregation method.",
            class_="text-muted small mb-3",
        ),
        ui.tooltip(
            ui.input_file(
                "cost_file",
                "Upload Cost Layer (.geojson, .gpkg, or .zip)",
                accept=[".geojson", ".gpkg", ".json", ".zip"],
            ),
            "Upload a vector file with polygons containing a numeric cost attribute. "
            "Supported formats: GeoJSON, GeoPackage (.gpkg), or a .zip archive "
            "containing a Shapefile bundle (.shp + .shx + .dbf + .prj). "
            "The file is spatially joined with planning units.",
        ),
        ui.layout_columns(
            ui.tooltip(
                ui.input_selectize(
                    "cost_col", "Cost Column",
                    choices=["cost"], selected="cost",
                ),
                "Column in the uploaded file containing the numeric cost values "
                "to be applied to planning units.",
            ),
            ui.tooltip(
                ui.input_select(
                    "aggregation",
                    "Aggregation",
                    {
                        "area_weighted_mean": "Area-Weighted Mean",
                        "sum": "Sum",
                        "max": "Maximum",
                    },
                ),
                "How to combine cost values when a planning unit overlaps "
                "multiple cost polygons. Area-weighted mean is recommended "
                "for continuous cost surfaces.",
            ),
            col_widths=[6, 6],
        ),
        ui.tooltip(
            ui.input_action_button(
                "apply_cost", "Apply Cost Surface", class_="btn-primary"
            ),
            "Intersect the uploaded cost layer with planning units and update "
            "the cost column in the planning unit table.",
        ),
        ui.output_text_verbatim("cost_info"),
    )


@module.server
def cost_upload_server(input, output, session, problem: reactive.Value):
    help_server_setup(input, "cost_upload")

    @reactive.effect
    @reactive.event(input.cost_file)
    def _update_cost_columns():
        """Populate cost column dropdown from uploaded file."""
        file_info = input.cost_file()
        if not file_info:
            return
        cols = _read_vector_columns(file_info[0]["datapath"])
        if cols:
            ui.update_selectize(
                "cost_col", choices=cols,
                selected="cost" if "cost" in cols else cols[0],
            )

    @reactive.effect
    @reactive.event(input.apply_cost)
    def _apply():
        p = problem()
        if p is None:
            ui.notification_show("Load a project first.", type="warning")
            return
        if not has_geometry(p):
            ui.notification_show(
                "Planning units need geometry.", type="warning"
            )
            return

        file_info = input.cost_file()
        if not file_info:
            ui.notification_show(
                "Upload a cost layer file.", type="warning"
            )
            return

        try:
            cost_layer = gpd.read_file(file_info[0]["datapath"])
            col = input.cost_col()
            if col not in cost_layer.columns:
                ui.notification_show(
                    f"Column '{col}' not found. Available: {list(cost_layer.columns)}",
                    type="error",
                )
                return
            updated = apply_cost_from_vector(
                p.planning_units,
                cost_layer,
                cost_column=col,
                aggregation=input.aggregation(),
            )
            new_problem = copy.deepcopy(p)
            new_problem.planning_units = updated
            problem.set(new_problem)
            ui.notification_show("Cost surface applied.", type="message")
        except Exception as exc:
            ui.notification_show(f"Cost error: {exc}", type="error")

    @render.text
    def cost_info():
        p = problem()
        if p is None:
            return "Load a project to apply cost surfaces."
        costs = p.planning_units["cost"]
        return (
            f"Cost range: {costs.min():.2f} – {costs.max():.2f} "
            f"(mean: {costs.mean():.2f})"
        )
