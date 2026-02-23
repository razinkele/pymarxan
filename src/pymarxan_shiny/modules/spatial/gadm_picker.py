"""GADM boundary picker Shiny module."""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan.spatial.gadm import fetch_gadm, list_countries

try:
    from shinywidgets import output_widget, render_widget

    from pymarxan_shiny.modules.mapping.map_utils import create_geo_map

    _HAS_IPYLEAFLET = True
except ImportError:  # pragma: no cover
    _HAS_IPYLEAFLET = False


@module.ui
def gadm_picker_ui():
    countries = list_countries()
    choices = {c["iso3"]: f"{c['name']} ({c['iso3']})" for c in countries}
    map_output = (
        output_widget("boundary_map")
        if _HAS_IPYLEAFLET
        else ui.output_ui("boundary_text")
    )
    return ui.card(
        ui.card_header("Administrative Boundaries (GADM)"),
        ui.layout_columns(
            ui.input_selectize("country", "Country", choices=choices),
            ui.input_select(
                "admin_level",
                "Admin Level",
                {
                    "0": "Country (ADM0)",
                    "1": "State/Province (ADM1)",
                    "2": "District (ADM2)",
                },
            ),
            col_widths=[6, 6],
        ),
        ui.input_text("admin_name", "Region Name Filter (optional)", value=""),
        ui.input_action_button("fetch", "Fetch Boundary", class_="btn-primary"),
        map_output,
        ui.output_text_verbatim("boundary_info"),
    )


@module.server
def gadm_picker_server(input, output, session, boundary: reactive.Value):
    @reactive.effect
    @reactive.event(input.fetch)
    def _fetch():
        try:
            admin_name = input.admin_name() or None
            gdf = fetch_gadm(
                country_iso3=input.country(),
                admin_level=int(input.admin_level()),
                admin_name=admin_name,
            )
            boundary.set(gdf)
            ui.notification_show(
                f"Fetched {len(gdf)} boundary polygon(s).",
                type="message",
            )
        except Exception as e:
            ui.notification_show(f"Error fetching boundary: {e}", type="error")

    @render.text
    def boundary_info():
        gdf = boundary()
        if gdf is None:
            return "Select a country and fetch boundaries."
        return f"{len(gdf)} polygon(s) fetched"

    if _HAS_IPYLEAFLET:  # pragma: no cover

        @render_widget
        def boundary_map():
            gdf = boundary()
            if gdf is None:
                return None
            colors = ["#e74c3c"] * len(gdf)
            return create_geo_map(gdf, colors)

    if not _HAS_IPYLEAFLET:  # pragma: no cover

        @render.ui
        def boundary_text():
            gdf = boundary()
            if gdf is None:
                return ui.p("Fetch a boundary to see the preview.")
            return ui.p(f"Boundary: {len(gdf)} polygons")
