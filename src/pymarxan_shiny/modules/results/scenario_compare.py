"""Scenario comparison Shiny module.

Lets users save solutions as named scenarios and compare them.
"""
from __future__ import annotations

from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from pymarxan.analysis.scenarios import ScenarioSet
from pymarxan.solvers.base import Solution


@module.ui
def scenario_compare_ui():
    return ui.card(
        ui.card_header("Scenario Comparison"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_text("scenario_name", "Scenario name", value=""),
                ui.input_action_button(
                    "save_scenario", "Save Current Solution",
                    class_="btn-primary w-100",
                ),
                ui.hr(),
                ui.output_text("scenario_count"),
                width=280,
            ),
            ui.output_data_frame("comparison_table"),
        ),
    )


@module.server
def scenario_compare_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    solution: reactive.Value,
    solver_config: reactive.Value,
):
    scenarios: reactive.Value[ScenarioSet] = reactive.value(ScenarioSet())

    @reactive.effect
    @reactive.event(input.save_scenario)
    def _save():
        sol = solution()
        if sol is None:
            ui.notification_show("Run a solver first!", type="error")
            return
        name = input.scenario_name() or f"scenario-{len(scenarios()) + 1}"
        ss = scenarios()
        ss.add(name, sol, dict(solver_config()))
        scenarios.set(ss)
        ui.notification_show(f"Saved scenario: {name}", type="message")

    @render.text
    def scenario_count():
        return f"{len(scenarios())} scenarios saved"

    @render.data_frame
    def comparison_table():
        ss = scenarios()
        if len(ss) == 0:
            return None
        return ss.compare()
