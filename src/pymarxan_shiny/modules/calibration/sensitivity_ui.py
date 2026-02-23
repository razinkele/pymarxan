"""Sensitivity dashboard Shiny module — parameter sensitivity analysis."""
from __future__ import annotations

import numpy as np
from shiny import module, reactive, render, ui

from pymarxan.calibration.sensitivity import (
    SensitivityConfig,
    SensitivityResult,
    run_sensitivity,
)
from pymarxan.solvers.base import SolverConfig


def build_sensitivity_config(
    min_mult: float = 0.8,
    max_mult: float = 1.2,
    steps: int = 5,
    feature_ids: list[int] | None = None,
) -> SensitivityConfig:
    """Build a SensitivityConfig from UI parameters."""
    multipliers = list(np.linspace(min_mult, max_mult, steps))
    multipliers = [round(m, 4) for m in multipliers]
    return SensitivityConfig(
        feature_ids=feature_ids,
        multipliers=multipliers,
        solver_config=SolverConfig(num_solutions=1),
    )


@module.ui
def sensitivity_ui():
    return ui.card(
        ui.card_header("Target Sensitivity Analysis"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_slider(
                    "mult_range",
                    "Multiplier Range",
                    min=0.5,
                    max=2.0,
                    value=[0.8, 1.2],
                    step=0.1,
                ),
                ui.input_numeric(
                    "mult_steps",
                    "Steps",
                    value=5,
                    min=3,
                    max=20,
                ),
                ui.input_action_button(
                    "run_sensitivity",
                    "Run Sensitivity",
                    class_="btn-primary w-100",
                ),
                width=280,
            ),
            ui.div(
                ui.output_ui("sensitivity_chart"),
                ui.output_data_frame("sensitivity_table"),
            ),
        ),
    )


@module.server
def sensitivity_server(
    input,
    output,
    session,
    problem: reactive.Value,
    solver: reactive.Calc,
):
    result: reactive.Value[SensitivityResult | None] = reactive.value(None)

    @reactive.effect
    @reactive.event(input.run_sensitivity)
    def _run():
        p = problem()
        s = solver()
        if p is None or s is None:
            ui.notification_show(
                "Load a project and configure a solver first.",
                type="error",
            )
            return
        ui.notification_show(
            "Running sensitivity analysis...", type="message"
        )
        try:
            mult_range = input.mult_range()
            config = build_sensitivity_config(
                min_mult=float(mult_range[0]),
                max_mult=float(mult_range[1]),
                steps=int(input.mult_steps()),
            )
            res = run_sensitivity(p, s, config)
            result.set(res)
            ui.notification_show(
                "Sensitivity analysis complete!", type="message"
            )
        except Exception as e:
            ui.notification_show(
                f"Sensitivity error: {e}", type="error"
            )

    @render.ui
    def sensitivity_chart():
        res = result()
        if res is None:
            return ui.p("Run sensitivity analysis to see results.")
        try:
            import plotly.express as px

            df = res.to_dataframe()
            fig = px.scatter(
                df,
                x="multiplier",
                y="objective",
                color="feature_id",
                title="Objective vs Target Multiplier",
                labels={
                    "multiplier": "Target Multiplier",
                    "objective": "Objective",
                },
            )
            fig.update_layout(
                height=350, margin=dict(l=60, r=20, t=40, b=40)
            )
            return ui.HTML(
                fig.to_html(include_plotlyjs="cdn", full_html=False)
            )
        except ImportError:
            return ui.p("Install plotly for chart visualization.")

    @render.data_frame
    def sensitivity_table():
        res = result()
        if res is None:
            return None
        return res.to_dataframe()
