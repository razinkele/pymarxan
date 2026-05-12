"""Spatial export Shiny module — download solutions as GeoPackage or Shapefile."""
from __future__ import annotations

import tempfile
from pathlib import Path

from shiny import module, reactive, render, ui

from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup

FORMAT_CHOICES = {
    "GPKG": "GeoPackage (.gpkg)",
    "ESRI Shapefile": "Shapefile (.shp)",
}

FORMAT_EXTENSIONS = {
    "GPKG": ".gpkg",
    "ESRI Shapefile": ".shp",
}


@module.ui
def spatial_export_ui():
    return ui.card(
        help_card_header("Spatial Export"),
        ui.p(
            "Download the current solution or selection frequency map as a "
            "spatial file for use in GIS software. The output joins solver "
            "results to planning unit geometries. Requires that planning "
            "units have spatial geometry loaded.",
            class_="text-muted small mb-3",
        ),
        ui.tooltip(
            ui.input_select(
                "export_format",
                "Output Format",
                choices=FORMAT_CHOICES,
                selected="GPKG",
            ),
            "GeoPackage: modern, single-file format recommended for most "
            "workflows. Shapefile: legacy format compatible with all GIS "
            "software (creates multiple sidecar files).",
        ),
        ui.div(
            ui.tooltip(
                ui.download_button("download_solution_spatial", "Download Solution"),
                "Download the best solution joined to planning unit geometries. "
                "Includes a 'selected' column (1 = selected, 0 = not selected).",
            ),
            class_="mb-2",
        ),
        ui.div(
            ui.tooltip(
                ui.download_button("download_frequency_spatial", "Download Frequency Map"),
                "Download selection frequency across all runs joined to planning "
                "unit geometries. Includes 'count' and 'frequency' columns.",
            ),
            class_="mb-2",
        ),
        ui.output_ui("spatial_export_status"),
    )


@module.server
def spatial_export_server(
    input,
    output,
    session,
    problem: reactive.Value,
    solution: reactive.Value,
    all_solutions: reactive.Value,
):
    help_server_setup(input, "spatial_export")

    # Track temp files so they can be cleaned up when the session ends.
    # Spatial exports (GeoPackage/Shapefile) can be many MB each — without
    # cleanup, they accumulate in /tmp for the lifetime of the server.
    _temp_paths: list[Path] = []

    def _track_tempfile(suffix: str) -> tempfile._TemporaryFileWrapper:
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.close()
        _temp_paths.append(Path(tmp.name))
        return tmp

    @session.on_ended
    def _cleanup_temp_files() -> None:
        for path in _temp_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    @reactive.calc
    def _has_geometry():
        p = problem()
        if p is None:
            return False
        pu = p.planning_units
        return pu is not None and "geometry" in pu.columns

    def _solution_filename() -> str:
        return f"pymarxan_solution{FORMAT_EXTENSIONS.get(input.export_format(), '.gpkg')}"

    @render.download(filename=_solution_filename)
    def download_solution_spatial():
        p = problem()
        s = solution()
        if p is None or s is None or not _has_geometry():
            return
        from pymarxan.io.spatial_export import export_solution_spatial

        driver = input.export_format()
        ext = FORMAT_EXTENSIONS.get(driver, ".gpkg")
        tmp = _track_tempfile(ext)
        export_solution_spatial(p.planning_units, s, tmp.name, driver=driver)
        return tmp.name

    def _frequency_filename() -> str:
        return f"pymarxan_frequency{FORMAT_EXTENSIONS.get(input.export_format(), '.gpkg')}"

    @render.download(filename=_frequency_filename)
    def download_frequency_spatial():
        p = problem()
        sols = all_solutions()
        if p is None or sols is None or not _has_geometry():
            return
        from pymarxan.io.spatial_export import export_frequency_spatial

        driver = input.export_format()
        ext = FORMAT_EXTENSIONS.get(driver, ".gpkg")
        tmp = _track_tempfile(ext)
        export_frequency_spatial(p.planning_units, sols, tmp.name, driver=driver)
        return tmp.name

    @render.ui
    def spatial_export_status():
        p = problem()
        s = solution()
        if p is None:
            return ui.p(
                "⚠ No project loaded.",
                class_="text-warning mt-2",
            )
        if not _has_geometry():
            return ui.p(
                "⚠ Planning units have no geometry. Load spatial data first.",
                class_="text-warning mt-2",
            )
        if s is None:
            return ui.p(
                "⚠ No solution available. Run a solver first.",
                class_="text-warning mt-2",
            )
        return ui.p(
            f"✅ Ready — {s.n_selected} selected PUs, "
            f"format: {input.export_format()}",
            class_="text-success mt-2",
        )
