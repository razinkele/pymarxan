"""Cost surface upload Shiny module."""
from __future__ import annotations

import copy

import geopandas as gpd
from shiny import module, reactive, render, ui

from pymarxan.models.problem import has_geometry
from pymarxan.spatial.cost_surface import apply_cost_from_vector


@module.ui
def cost_upload_ui():
    return ui.card(
        ui.card_header("Custom Cost Surface"),
        ui.input_file(
            "cost_file",
            "Upload Cost Layer (.shp, .geojson, .gpkg)",
            accept=[".shp", ".geojson", ".gpkg", ".json", ".zip"],
        ),
        ui.layout_columns(
            ui.input_text("cost_col", "Cost Column", value="cost"),
            ui.input_select(
                "aggregation",
                "Aggregation",
                {
                    "area_weighted_mean": "Area-Weighted Mean",
                    "sum": "Sum",
                    "max": "Maximum",
                },
            ),
            col_widths=[6, 6],
        ),
        ui.input_action_button(
            "apply_cost", "Apply Cost Surface", class_="btn-primary"
        ),
        ui.output_text_verbatim("cost_info"),
    )


@module.server
def cost_upload_server(input, output, session, problem: reactive.Value):

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
            updated = apply_cost_from_vector(
                p.planning_units,
                cost_layer,
                cost_column=input.cost_col(),
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
