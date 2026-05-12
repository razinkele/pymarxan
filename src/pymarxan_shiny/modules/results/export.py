"""Results export Shiny module — download solutions as CSV."""
from __future__ import annotations

import tempfile
from pathlib import Path

from shiny import module, reactive, render, ui

from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup
from pymarxan.io.exporters import export_solution_csv, export_summary_csv


@module.ui
def export_ui():
    return ui.card(
        help_card_header("Export Results"),
        ui.p(
            "Download the solver results as CSV files for use in GIS software, "
            "reports, or further analysis. The solution CSV lists each planning "
            "unit and whether it was selected. The target summary CSV shows "
            "feature-level achievement details.",
            class_="text-muted small mb-3",
        ),
        ui.div(
            ui.tooltip(
                ui.download_button("download_solution", "Download Solution CSV"),
                "Download a CSV with columns: PU ID, selected (1/0), cost. "
                "One row per planning unit.",
            ),
            class_="mb-2",
        ),
        ui.div(
            ui.tooltip(
                ui.download_button("download_summary", "Download Target Summary CSV"),
                "Download a CSV summarising target achievement: feature ID, name, "
                "target, achieved amount, and whether the target is met.",
            ),
            class_="mb-2",
        ),
        ui.output_ui("export_status"),
    )


@module.server
def export_server(
    input, output, session,
    problem: reactive.Value,
    solution: reactive.Value,
):
    help_server_setup(input, "export")

    # Track temp file paths so they can be cleaned up when the session ends —
    # otherwise each download leaks a file in /tmp for the lifetime of the
    # server process (megabytes per export accumulate over long sessions).
    _temp_paths: list[Path] = []

    def _track_tempfile(suffix: str, mode: str = "w") -> tempfile._TemporaryFileWrapper:
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode=mode)
        _temp_paths.append(Path(tmp.name))
        return tmp

    @session.on_ended
    def _cleanup_temp_files() -> None:
        for path in _temp_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    @render.download(filename="pymarxan_solution.csv")
    def download_solution():
        p = problem()
        s = solution()
        if p is None or s is None:
            return
        tmp = _track_tempfile(".csv")
        export_solution_csv(p, s, tmp.name)
        return tmp.name

    @render.download(filename="pymarxan_target_summary.csv")
    def download_summary():
        p = problem()
        s = solution()
        if p is None or s is None:
            return
        tmp = _track_tempfile(".csv")
        export_summary_csv(p, s, tmp.name)
        return tmp.name

    @render.ui
    def export_status():
        p = problem()
        s = solution()
        if p is None or s is None:
            return ui.p(
                "\u26A0 No solution available to export. Run a solver first.",
                class_="text-warning mt-2",
            )
        return ui.p(
            f"\u2705 Ready — {s.n_selected} selected PUs, cost {s.cost:.2f}",
            class_="text-success mt-2",
        )
