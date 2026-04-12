"""Connectivity metrics visualization Shiny module.

Displays connectivity metrics (in-degree, out-degree, betweenness,
eigenvector centrality) for the current connectivity matrix.
"""
from __future__ import annotations

from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup
from pymarxan.connectivity.metrics import (
    compute_betweenness_centrality,
    compute_eigenvector_centrality,
    compute_in_degree,
    compute_out_degree,
)


@module.ui
def metrics_viz_ui():
    return ui.card(
        help_card_header("Connectivity Metrics"),
        ui.p(
            "Compute graph-theoretic connectivity metrics for the loaded "
            "connectivity matrix. These help identify critical planning units "
            "for maintaining ecological connectivity: high in-degree nodes receive "
            "many connections, high betweenness nodes bridge clusters, and high "
            "eigenvector centrality nodes are connected to other well-connected nodes.",
            class_="text-muted small mb-3",
        ),
        ui.layout_sidebar(
            ui.sidebar(
                ui.tooltip(
                    ui.input_action_button(
                        "compute_metrics", "Compute Metrics",
                        class_="btn-primary w-100",
                    ),
                    "Calculate in-degree, out-degree, betweenness centrality, "
                    "and eigenvector centrality for every planning unit.",
                ),
                ui.hr(),
                ui.output_text("metrics_status"),
                width=280,
            ),
            ui.output_data_frame("metrics_table"),
        ),
    )


@module.server
def metrics_viz_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    connectivity_matrix: reactive.Value,
    pu_ids: reactive.Value,
):
    help_server_setup(input, "metrics_viz")

    metrics_df: reactive.Value = reactive.value(None)

    @reactive.effect
    @reactive.event(input.compute_metrics)
    def _compute():
        import pandas as pd

        matrix = connectivity_matrix()
        ids = pu_ids()
        if matrix is None:
            ui.notification_show("No connectivity matrix loaded!", type="error")
            return

        in_deg = compute_in_degree(matrix)
        out_deg = compute_out_degree(matrix)
        bc = compute_betweenness_centrality(matrix)
        ec = compute_eigenvector_centrality(matrix)

        df = pd.DataFrame({
            "pu_id": ids if ids is not None else list(range(matrix.shape[0])),
            "in_degree": in_deg,
            "out_degree": out_deg,
            "betweenness": bc,
            "eigenvector": ec,
        })
        metrics_df.set(df)
        ui.notification_show("Metrics computed", type="message")

    @render.text
    def metrics_status():
        m = connectivity_matrix()
        if m is None:
            return "No connectivity matrix loaded"
        return f"Matrix: {m.shape[0]} nodes"

    @render.data_frame
    def metrics_table():
        return metrics_df()
