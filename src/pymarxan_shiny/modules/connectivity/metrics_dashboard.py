"""Connectivity metrics dashboard Shiny module (Phase 24 follow-up).

Surfaces all seven connectivity metrics in a single sortable table —
one row per PU, one column per metric. Lets users quickly identify
source / sink PUs and compare different centrality definitions side
by side.

Metrics shown (all from ``pymarxan.connectivity.metrics``):
- in-degree
- out-degree
- betweenness centrality
- eigenvector centrality
- PageRank
- donor (boolean)
- recipient (boolean)
"""
from __future__ import annotations

from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from pymarxan_shiny.modules.help.help_button import (
    help_card_header,
    help_server_setup,
)


@module.ui
def connectivity_metrics_dashboard_ui():
    return ui.card(
        help_card_header("Connectivity Metrics Dashboard"),
        ui.p(
            "Per-PU graph-theoretic metrics computed from the loaded "
            "connectivity matrix. In-degree and out-degree capture flow "
            "imbalance; betweenness, eigenvector, and PageRank capture "
            "centrality; donor / recipient flags identify net sources "
            "and sinks. Sort or filter to find priority PUs.",
            class_="text-muted small mb-3",
        ),
        ui.output_data_frame("connectivity_metrics_table"),
    )


@module.server
def connectivity_metrics_dashboard_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    problem: reactive.Value,
):
    help_server_setup(input, "connectivity_metrics_dashboard")

    @render.data_frame
    def connectivity_metrics_table():
        import pandas as pd

        from pymarxan.connectivity.io import connectivity_to_matrix
        from pymarxan.connectivity.metrics import (
            compute_betweenness_centrality,
            compute_donors,
            compute_eigenvector_centrality,
            compute_in_degree,
            compute_out_degree,
            compute_pagerank_centrality,
            compute_recipients,
        )

        p = problem()
        if p is None:
            return None
        if p.connectivity is None:
            # No connectivity matrix loaded — return an empty frame with
            # the expected column headers so the table renders cleanly.
            return pd.DataFrame(columns=[
                "PU", "in_degree", "out_degree", "betweenness",
                "eigenvector", "pagerank", "donor", "recipient",
            ])

        pu_ids = p.planning_units["id"].astype(int).tolist()
        matrix = connectivity_to_matrix(p.connectivity, pu_ids)

        rows = []
        in_deg = compute_in_degree(matrix)
        out_deg = compute_out_degree(matrix)
        betw = compute_betweenness_centrality(matrix)
        eig = compute_eigenvector_centrality(matrix)
        pagerank = compute_pagerank_centrality(matrix)
        donors = compute_donors(matrix)
        recipients = compute_recipients(matrix)

        for i, pid in enumerate(pu_ids):
            rows.append({
                "PU": int(pid),
                "in_degree": round(float(in_deg[i]), 4),
                "out_degree": round(float(out_deg[i]), 4),
                "betweenness": round(float(betw[i]), 4),
                "eigenvector": round(float(eig[i]), 4),
                "pagerank": round(float(pagerank[i]), 4),
                "donor": "yes" if bool(donors[i]) else "",
                "recipient": "yes" if bool(recipients[i]) else "",
            })
        return pd.DataFrame(rows)
