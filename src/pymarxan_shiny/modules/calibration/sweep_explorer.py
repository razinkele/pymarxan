"""Sweep explorer Shiny module.

Lets users configure and run a parameter sweep, then view results.
"""
from __future__ import annotations

from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from pymarxan.calibration.sweep import SweepConfig, SweepResult, run_sweep


@module.ui
def sweep_explorer_ui():
    return ui.card(
        ui.card_header("Parameter Sweep"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select(
                    "sweep_param",
                    "Parameter to sweep",
                    choices=["BLM", "NUMITNS", "NUMTEMP"],
                    selected="BLM",
                ),
                ui.input_numeric("sweep_min", "Min value", value=0.0),
                ui.input_numeric("sweep_max", "Max value", value=100.0),
                ui.input_numeric("sweep_steps", "Number of steps", value=10),
                ui.input_action_button(
                    "run_sweep", "Run Sweep", class_="btn-primary w-100"
                ),
                width=280,
            ),
            ui.output_data_frame("sweep_results_table"),
        ),
    )


@module.server
def sweep_explorer_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    problem: reactive.Value,
    solver: reactive.Calc,
):
    sweep_result: reactive.Value[SweepResult | None] = reactive.value(None)

    @reactive.effect
    @reactive.event(input.run_sweep)
    def _run():
        p = problem()
        if p is None:
            ui.notification_show("Load a project first!", type="error")
            return
        param_name = input.sweep_param()
        min_val = input.sweep_min()
        max_val = input.sweep_max()
        steps = int(input.sweep_steps())

        import numpy as np

        values = np.linspace(min_val, max_val, steps).tolist()
        config = SweepConfig(
            param_dicts=[{param_name: v} for v in values],
        )
        result = run_sweep(p, solver(), config)
        sweep_result.set(result)
        ui.notification_show(
            f"Sweep complete: {len(result.solutions)} points", type="message"
        )

    @render.data_frame
    def sweep_results_table():
        r = sweep_result()
        if r is None:
            return None
        return r.to_dataframe()
