"""Representation (30x30 / GBF Target 3) Shiny module.

Shows, for the current solution, what fraction of each feature is
represented and whether it clears a user-set policy threshold.
"""
from __future__ import annotations

from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from pymarxan.analysis.representation import compute_representation
from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup


@module.ui
def representation_ui():
    return ui.card(
        help_card_header("Representation (30x30)"),
        ui.p(
            "Per-feature share of each feature's total amount captured by "
            "the current solution, and whether it clears the policy "
            "threshold (default 30% for the Global Biodiversity Framework "
            "Target 3).",
            class_="text-muted small mb-3",
        ),
        ui.input_slider(
            "threshold", "Representation threshold", min=0, max=100, value=30, post="%"
        ),
        ui.output_text("summary"),
        ui.output_data_frame("rep_table"),
    )


@module.server
def representation_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    problem: reactive.Value,
    solution: reactive.Value,
):
    help_server_setup(input, "representation")

    @reactive.Calc
    def _result():  # type: ignore[valid-type]
        p = problem()
        s = solution()
        if p is None or s is None:
            return None
        return compute_representation(p, s, threshold=input.threshold() / 100.0)

    @render.text
    def summary():
        r = _result()
        if r is None:
            return "No solution yet."
        return (
            f"{r.n_features_meeting}/{len(r.feature_ids)} features meet the "
            f"{r.threshold:.0%} threshold."
        )

    @render.data_frame
    def rep_table():
        r = _result()
        if r is None:
            return None
        return r.to_dataframe()
