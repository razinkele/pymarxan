"""Connectivity matrix upload Shiny module."""
from __future__ import annotations

from shiny import module, reactive, render, ui

FORMAT_LABELS = {
    "edge_list": "Edge List (id1, id2, value)",
    "full_matrix": "Full Matrix (NxN)",
}


def parse_format_label(fmt: str) -> str:
    """Return human-readable format label."""
    return FORMAT_LABELS.get(fmt, "Unknown")


@module.ui
def matrix_input_ui():
    return ui.card(
        ui.card_header("Connectivity Matrix"),
        ui.input_file("conn_file", "Upload CSV", accept=[".csv"]),
        ui.input_radio_buttons(
            "conn_format",
            "Format",
            choices={
                "edge_list": "Edge List (id1, id2, value)",
                "full_matrix": "Full Matrix (NxN)",
            },
            selected="edge_list",
        ),
        ui.output_text_verbatim("conn_preview"),
    )


@module.server
def matrix_input_server(
    input,
    output,
    session,
    problem: reactive.Value,
    connectivity_matrix: reactive.Value,
    connectivity_pu_ids: reactive.Value,
):
    @reactive.effect
    @reactive.event(input.conn_file)
    def _on_upload():
        file_info = input.conn_file()
        if file_info is None or len(file_info) == 0:
            return

        path = file_info[0]["datapath"]
        fmt = input.conn_format()
        p = problem()

        try:
            if fmt == "edge_list":
                from pymarxan.connectivity.io import (
                    read_connectivity_edgelist,
                )

                if p is None:
                    ui.notification_show(
                        "Load planning units first.",
                        type="warning",
                    )
                    return
                pu_ids = p.planning_units["id"].tolist()
                matrix = read_connectivity_edgelist(path, pu_ids)
                connectivity_pu_ids.set(pu_ids)
            else:
                from pymarxan.connectivity.io import (
                    read_connectivity_matrix,
                )

                matrix = read_connectivity_matrix(path)
                connectivity_pu_ids.set(
                    list(range(matrix.shape[0]))
                )

            connectivity_matrix.set(matrix)
            ui.notification_show(
                f"Loaded {matrix.shape[0]}x{matrix.shape[1]} matrix.",
                type="message",
            )
        except Exception as exc:
            ui.notification_show(f"Error: {exc}", type="error")

    @render.text
    def conn_preview():
        m = connectivity_matrix()
        if m is None:
            return "No connectivity matrix loaded."
        import numpy as np

        nonzero = int(np.count_nonzero(m))
        total = m.shape[0] * m.shape[1]
        density = (
            100.0 * nonzero / total if total > 0 else 0.0
        )
        return (
            f"Shape: {m.shape[0]} x {m.shape[1]}\n"
            f"Non-zero: {nonzero}\n"
            f"Density: {density:.1f}%"
        )
