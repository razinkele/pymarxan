"""River barrier-restoration (DCI) Shiny panel.

Given a reactive ``RiverNetwork``, shows the budget–DCI efficiency frontier,
a before/after DCI readout for the chosen budget, and a barrier
selection-frequency table (robust no-regret picks). Mirrors the existing
results-module conventions.
"""
from __future__ import annotations

import pandas as pd
from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from pymarxan.rivers import (
    BarrierProblem,
    barrier_selection_frequency,
    budget_dci_frontier,
    optimize_barriers_greedy,
)

_FORMS = {"diadromous": "Diadromous (sea ↔ segment)", "potamodromous": "Potamodromous (all pairs)"}


@module.ui
def rivers_panel_ui():
    return ui.card(
        ui.card_header("River barriers (DCI restoration)"),
        ui.p(
            "Which barriers to remove, under a budget, to maximise the "
            "Dendritic Connectivity Index. The frontier shows DCI gained per "
            "unit budget; the table ranks barriers by how often they appear "
            "in good portfolios.",
            class_="text-muted small mb-3",
        ),
        ui.input_select("form", "DCI form", _FORMS),
        ui.input_numeric("budget", "Budget", value=1.0, min=0.0),
        ui.output_text("readout"),
        ui.output_ui("frontier_plot"),
        ui.output_data_frame("freq_table"),
    )


@module.server
def rivers_panel_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    network: reactive.Value,
):
    def _budget() -> float:
        b = input.budget()
        return 0.0 if b is None else float(b)

    @render.text
    def readout():
        net = network()
        if net is None:
            return "No river network loaded."
        sol = optimize_barriers_greedy(
            BarrierProblem(net, budget=_budget(), form=input.form())
        )
        return (
            f"DCI {sol.dci_before:.1f} → {sol.dci_after:.1f} "
            f"(gain {sol.gain:.1f}); {len(sol.removed)} barrier(s) removed, "
            f"cost {sol.cost:.1f}"
        )

    @render.ui
    def frontier_plot():
        net = network()
        if net is None:
            return ui.p("No river network loaded.", class_="text-muted")
        total = (
            float(net.barriers["removal_cost"].sum())
            if "removal_cost" in net.barriers.columns
            else float(net.n_barriers)
        )
        if total <= 0:
            return ui.p("No removable barriers.", class_="text-muted")
        budgets = [total * k / 10.0 for k in range(11)]
        df = budget_dci_frontier(net, budgets, form=input.form())
        try:
            import plotly.graph_objects as go

            fig = go.Figure(
                go.Scatter(x=df["budget"], y=df["dci_after"], mode="lines+markers")
            )
            fig.update_layout(
                xaxis_title="Budget",
                yaxis_title="DCI",
                margin={"l": 45, "r": 10, "t": 10, "b": 40},
                height=320,
            )
            return ui.HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))
        except ImportError:
            return ui.HTML(df.to_html(index=False))

    @render.data_frame
    def freq_table():
        net = network()
        if net is None:
            return None
        freq = barrier_selection_frequency(
            net, budget=_budget(), form=input.form(), n_runs=20
        )
        return pd.DataFrame(
            {
                "barrier_id": list(freq),
                "selection_frequency": [round(v, 3) for v in freq.values()],
            }
        )
