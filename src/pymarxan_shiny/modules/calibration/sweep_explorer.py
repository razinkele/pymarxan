"""Sweep explorer Shiny module.

Lets users configure and run a parameter sweep, then view results.
"""
from __future__ import annotations

from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from pymarxan.calibration.sweep import SweepConfig, SweepResult, run_sweep
from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup


@module.ui
def sweep_explorer_ui():
    return ui.card(
        help_card_header("Parameter Sweep"),
        ui.p(
            "Run a systematic sweep over a Marxan parameter to understand its "
            "effect on solution quality. The solver is run once for each step "
            "in the parameter range, and results are displayed in a table. "
            "Useful for exploring BLM, NUMITNS, or NUMTEMP sensitivity.",
            class_="text-muted small mb-3",
        ),
        ui.layout_sidebar(
            ui.sidebar(
                ui.tooltip(
                    ui.input_select(
                        "sweep_param",
                        "Parameter to sweep",
                        choices=["BLM", "NUMITNS", "NUMTEMP"],
                        selected="BLM",
                    ),
                    "Marxan parameter to vary: BLM (Boundary Length Modifier), "
                    "NUMITNS (SA iterations), or NUMTEMP (temperature steps).",
                ),
                ui.tooltip(
                    ui.input_numeric("sweep_min", "Min value", value=0.0),
                    "Starting value for the sweep range.",
                ),
                ui.tooltip(
                    ui.input_numeric("sweep_max", "Max value", value=100.0),
                    "Ending value for the sweep range.",
                ),
                ui.tooltip(
                    ui.input_numeric("sweep_steps", "Number of steps", value=10),
                    "How many evenly-spaced values to test between min and max.",
                ),
                ui.tooltip(
                    ui.input_action_button(
                        "run_sweep", "Run Sweep", class_="btn-primary w-100"
                    ),
                    "Execute the solver for each parameter value and collect results.",
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
    help_server_setup(input, "sweep_explorer")

    sweep_result: reactive.Value[SweepResult | None] = reactive.value(None)

    @reactive.effect
    @reactive.event(input.run_sweep)
    def _run():
        p = problem()
        if p is None:
            ui.notification_show("Load a project first!", type="error")
            return
        try:
            param_name = input.sweep_param()
            min_val = input.sweep_min()
            max_val = input.sweep_max()
            steps = int(input.sweep_steps())

            if min_val >= max_val:
                ui.notification_show(
                    "Min value must be less than Max value.", type="error",
                )
                return

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
        except Exception as e:
            ui.notification_show(f"Sweep error: {e}", type="error")

    @render.data_frame
    def sweep_results_table():
        r = sweep_result()
        if r is None:
            return None
        return r.to_dataframe()
