"""Data upload Shiny module for loading Marxan project files."""
from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

from shiny import module, reactive, render, ui

from pymarxan.io.readers import load_project


@module.ui
def upload_ui():
    return ui.card(
        ui.card_header("Load Marxan Project"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_file("project_zip", "Upload Marxan project (.zip)",
                              accept=[".zip"], multiple=False),
                ui.hr(),
                ui.p("Or load from a local directory:"),
                ui.input_text("project_path", "Project directory path",
                              placeholder="/path/to/project"),
                ui.input_action_button("load_local", "Load from path"),
                width=350,
            ),
            ui.output_text_verbatim("project_summary"),
        ),
    )

@module.server
def upload_server(input, output, session, problem: reactive.Value):
    @reactive.effect
    @reactive.event(input.project_zip)
    def _handle_zip_upload():
        file_infos = input.project_zip()
        if not file_infos:
            return
        file_info = file_infos[0]
        uploaded_path = file_info["datapath"]
        with tempfile.TemporaryDirectory(prefix="pymarxan_upload_") as tmpdir:
            with zipfile.ZipFile(uploaded_path, "r") as zf:
                zf.extractall(tmpdir)
            tmpdir_path = Path(tmpdir)
            input_dat_files = list(tmpdir_path.rglob("input.dat"))
            if not input_dat_files:
                ui.notification_show("No input.dat found in uploaded ZIP", type="error")
                return
            project_dir = input_dat_files[0].parent
            try:
                loaded = load_project(project_dir)
                errors = loaded.validate()
                if errors:
                    msg = f"Validation warnings: {'; '.join(errors)}"
                    ui.notification_show(msg, type="warning")
                problem.set(loaded)
                ui.notification_show("Project loaded successfully!", type="message")
            except Exception as e:
                ui.notification_show(f"Error loading project: {e}", type="error")

    @reactive.effect
    @reactive.event(input.load_local)
    def _handle_local_load():
        path = input.project_path()
        if not path:
            return
        project_dir = Path(path)
        if not (project_dir / "input.dat").exists():
            ui.notification_show(
                f"No input.dat found in {project_dir}", type="error",
            )
            return
        try:
            loaded = load_project(project_dir)
            errors = loaded.validate()
            if errors:
                msg = f"Validation warnings: {'; '.join(errors)}"
                ui.notification_show(msg, type="warning")
            problem.set(loaded)
            ui.notification_show("Project loaded successfully!", type="message")
        except Exception as e:
            ui.notification_show(f"Error loading project: {e}", type="error")

    @render.text
    def project_summary():
        p = problem()
        if p is None:
            return "No project loaded. Upload a ZIP file or enter a directory path."
        return p.summary()
