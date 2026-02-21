"""Results export Shiny module — download solutions as CSV."""
from __future__ import annotations

import tempfile

from shiny import module, reactive, render, ui

from pymarxan.io.exporters import export_solution_csv, export_summary_csv


@module.ui
def export_ui():
    return ui.card(
        ui.card_header("Export Results"),
        ui.download_button("download_solution", "Download Solution CSV"),
        ui.download_button("download_summary", "Download Target Summary CSV"),
    )


@module.server
def export_server(
    input, output, session,
    problem: reactive.Value,
    solution: reactive.Value,
):
    @render.download(filename="pymarxan_solution.csv")
    def download_solution():
        p = problem()
        s = solution()
        if p is None or s is None:
            return
        tmp = tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w",
        )
        export_solution_csv(p, s, tmp.name)
        return tmp.name

    @render.download(filename="pymarxan_target_summary.csv")
    def download_summary():
        p = problem()
        s = solution()
        if p is None or s is None:
            return
        tmp = tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w",
        )
        export_summary_csv(p, s, tmp.name)
        return tmp.name
