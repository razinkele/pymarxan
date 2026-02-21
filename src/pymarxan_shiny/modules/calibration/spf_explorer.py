"""SPF calibration explorer Shiny module.

Lets users run iterative SPF calibration and view the history.
"""
from __future__ import annotations

from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from pymarxan.calibration.spf import SPFResult, calibrate_spf
from pymarxan.solvers.base import SolverConfig


@module.ui
def spf_explorer_ui():
    return ui.card(
        ui.card_header("SPF Calibration"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_numeric(
                    "max_iterations", "Max iterations", value=10, min=1, max=50,
                ),
                ui.input_numeric(
                    "multiplier", "SPF multiplier", value=2.0, min=1.1, max=10.0,
                ),
                ui.input_action_button(
                    "run_spf", "Run SPF Calibration", class_="btn-primary w-100",
                ),
                ui.hr(),
                ui.output_text("spf_status"),
                width=280,
            ),
            ui.output_data_frame("spf_history_table"),
        ),
    )


@module.server
def spf_explorer_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    problem: reactive.Value,
    solver: reactive.Calc,
):
    spf_result: reactive.Value[SPFResult | None] = reactive.value(None)

    @reactive.effect
    @reactive.event(input.run_spf)
    def _run():
        p = problem()
        if p is None:
            ui.notification_show("Load a project first!", type="error")
            return
        result = calibrate_spf(
            p,
            solver(),
            max_iterations=int(input.max_iterations()),
            multiplier=float(input.multiplier()),
            config=SolverConfig(num_solutions=1),
        )
        spf_result.set(result)
        ui.notification_show(
            f"SPF calibration done in {len(result.history)} iterations",
            type="message",
        )

    @render.text
    def spf_status():
        r = spf_result()
        if r is None:
            return "Not yet run"
        met = r.solution.all_targets_met
        return f"All targets met: {'Yes' if met else 'No'} ({len(r.history)} iterations)"

    @render.data_frame
    def spf_history_table():
        import pandas as pd

        r = spf_result()
        if r is None:
            return None
        rows = []
        for h in r.history:
            rows.append({
                "iteration": h["iteration"],
                "unmet_count": h["unmet_count"],
            })
        return pd.DataFrame(rows)
