"""Objective type selector Shiny module.

Extends solver configuration with objective type selection:
MinSet (default), MaxCoverage, MaxUtility, MinShortfall.
"""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup

OBJECTIVE_CHOICES = {
    "minset": "MinSet — Minimise cost meeting all targets",
    "maxcoverage": "MaxCoverage — Maximise features within budget",
    "maxutility": "MaxUtility — Maximise utility within budget",
    "minshortfall": "MinShortfall — Minimise target shortfall",
}

OBJECTIVE_DESCRIPTIONS = {
    "minset": (
        "Minimum Set (MinSet)\n"
        "--------------------\n"
        "Classic Marxan formulation: find the cheapest set of planning\n"
        "units that meets all conservation feature targets. This is the\n"
        "default and most widely used objective type."
    ),
    "maxcoverage": (
        "Maximum Coverage (MaxCoverage)\n"
        "------------------------------\n"
        "Maximise the number of conservation features represented in\n"
        "the reserve network, subject to a total budget constraint.\n"
        "Useful when funding is limited and not all targets can be met."
    ),
    "maxutility": (
        "Maximum Utility (MaxUtility)\n"
        "----------------------------\n"
        "Maximise total utility (weighted feature representation)\n"
        "subject to a budget constraint. Similar to MaxCoverage but\n"
        "allows differential weighting of features."
    ),
    "minshortfall": (
        "Minimum Shortfall (MinShortfall)\n"
        "--------------------------------\n"
        "Minimise the total shortfall across all features subject to\n"
        "a budget constraint. Shortfall is the gap between the target\n"
        "and achieved representation for each feature."
    ),
}


@module.ui
def objective_selector_ui():
    return ui.card(
        help_card_header("Objective Type"),
        ui.p(
            "Choose the optimisation objective for the reserve design problem. "
            "MinSet minimises cost while meeting targets. The budget-based "
            "objectives (MaxCoverage, MaxUtility, MinShortfall) require a "
            "total budget to be specified.",
            class_="text-muted small mb-3",
        ),
        ui.layout_sidebar(
            ui.sidebar(
                ui.tooltip(
                    ui.input_radio_buttons(
                        "objective_type",
                        "Objective",
                        choices=OBJECTIVE_CHOICES,
                        selected="minset",
                    ),
                    "MinSet: classic Marxan (minimise cost, meet targets). "
                    "MaxCoverage/MaxUtility/MinShortfall: budget-constrained "
                    "formulations that maximise representation or minimise "
                    "shortfall within a fixed budget.",
                ),
                ui.hr(),
                ui.panel_conditional(
                    "input.objective_type !== 'minset'",
                    ui.tooltip(
                        ui.input_numeric(
                            "budget",
                            "Total Budget",
                            value=1000.0,
                            min=0.0,
                            step=100.0,
                        ),
                        "Maximum total cost allowed for the reserve network. "
                        "Only used with budget-constrained objectives "
                        "(MaxCoverage, MaxUtility, MinShortfall).",
                    ),
                ),
                width=350,
            ),
            ui.output_text_verbatim("objective_info"),
        ),
    )


@module.server
def objective_selector_server(
    input,
    output,
    session,
    solver_config: reactive.Value,
):
    help_server_setup(input, "objective_selector")

    @reactive.effect
    @reactive.event(input.objective_type, input.budget)
    def _update_config():
        cfg = solver_config()
        if cfg is None:
            cfg = {}
        cfg = dict(cfg)
        cfg["objective_type"] = input.objective_type()
        if input.objective_type() != "minset":
            budget_val = input.budget()
            cfg["budget"] = float(budget_val) if budget_val is not None else 1000.0
        else:
            cfg.pop("budget", None)
        solver_config.set(cfg)

    @render.text
    def objective_info():
        obj = input.objective_type()
        return OBJECTIVE_DESCRIPTIONS.get(obj, "")
