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
from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup


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
        help_card_header("Target Sensitivity Analysis"),
        ui.p(
            "Test how sensitive the reserve design is to changes in conservation "
            "targets. Each feature's target is scaled by a range of multipliers "
            "(e.g. 0.8\u20131.2 = 80\u2013120% of original target). This reveals "
            "which features drive the solution and how robust the design is to "
            "target uncertainty.",
            class_="text-muted small mb-3",
        ),
        ui.layout_sidebar(
            ui.sidebar(
                ui.tooltip(
                    ui.input_slider(
                        "mult_range",
                        "Multiplier Range",
                        min=0.5,
                        max=2.0,
                        value=[0.8, 1.2],
                        step=0.1,
                    ),
                    "Scales each feature's conservation target. "
                    "E.g. 0.8 = 80% of target, 1.2 = 120% of target.",
                ),
                ui.tooltip(
                    ui.input_numeric(
                        "mult_steps",
                        "Steps",
                        value=5,
                        min=3,
                        max=20,
                    ),
                    "Number of evenly-spaced multiplier values between the min "
                    "and max of the range. More steps = finer resolution.",
                ),
                ui.tooltip(
                    ui.input_action_button(
                        "run_sensitivity",
                        "Run Sensitivity",
                        class_="btn-primary w-100",
                    ),
                    "Run the solver once per multiplier value per feature and "
                    "plot objective cost vs. target multiplier.",
                ),
                width=280,
            ),
            ui.div(
                ui.output_plot("sensitivity_chart"),
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
    solver: reactive.Calc,  # type: ignore[valid-type]  # shiny stubs treat Calc as var
):
    help_server_setup(input, "sensitivity")

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

    @render.plot
    def sensitivity_chart():
        import matplotlib.pyplot as plt

        res = result()
        if res is None:
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.text(
                0.5, 0.5, "Run sensitivity analysis to see results.",
                ha="center", va="center", color="#666",
            )
            ax.axis("off")
            return fig

        # Rendered server-side with matplotlib (@render.plot) rather than plotly
        # via to_html(include_plotlyjs="cdn"), which fails to load under Shiny's
        # dynamic HTML injection ("Plotly is not defined").
        df = res.to_dataframe()
        fig, ax = plt.subplots(figsize=(7, 3.5))
        for fid, grp in df.groupby("feature_id"):
            grp = grp.sort_values("multiplier")
            ax.plot(grp["multiplier"], grp["objective"], "-o", label=f"feature {fid}")
        ax.set_xlabel("Target Multiplier")
        ax.set_ylabel("Objective")
        ax.set_title("Objective vs Target Multiplier")
        if df["feature_id"].nunique() <= 12:
            ax.legend(fontsize=8, loc="best")
        return fig

    @render.data_frame
    def sensitivity_table():
        res = result()
        if res is None:
            return None
        return res.to_dataframe()
