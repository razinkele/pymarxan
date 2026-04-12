"""Interactive BLM calibration Shiny module with cost-vs-boundary plot."""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup
from pymarxan.calibration.blm import BLMResult, calibrate_blm
from pymarxan.solvers.base import SolverConfig


@module.ui
def blm_explorer_ui():
    return ui.card(
        help_card_header("BLM Calibration"),
        ui.p(
            "Calibrate the Boundary Length Modifier (BLM) to find the best "
            "tradeoff between reserve cost and spatial compactness. The classic "
            "approach is to plot cost vs. boundary length across a range of BLM "
            "values and choose the 'elbow' point where increasing BLM yields "
            "diminishing compactness gains.",
            class_="text-muted small mb-3",
        ),
        ui.layout_sidebar(
            ui.sidebar(
                ui.tooltip(
                    ui.input_numeric(
                        "blm_min", "Min BLM", value=0, min=0, step=0.1,
                    ),
                    "Lower bound of BLM range to test. BLM controls the "
                    "tradeoff between cost and spatial compactness.",
                ),
                ui.tooltip(
                    ui.input_numeric(
                        "blm_max", "Max BLM", value=50, min=0, step=1,
                    ),
                    "Upper bound of BLM range. Higher BLM = more compact reserves.",
                ),
                ui.tooltip(
                    ui.input_numeric(
                        "blm_steps", "Steps", value=10, min=2, max=50,
                    ),
                    "Number of evenly-spaced BLM values to test between "
                    "Min and Max. More steps give a smoother curve but "
                    "take longer to compute.",
                ),
                ui.tooltip(
                    ui.input_action_button(
                        "run_calibration", "Run Calibration",
                        class_="btn-primary w-100",
                    ),
                    "Solve the problem at each BLM value and plot the "
                    "cost-vs-boundary tradeoff curve.",
                ),
                width=300,
            ),
            ui.div(
                ui.output_plot("blm_plot"),
                ui.output_text_verbatim("blm_table"),
            ),
        ),
    )


@module.server
def blm_explorer_server(
    input, output, session,
    problem: reactive.Value,
    solver: reactive.Calc,
):
    help_server_setup(input, "blm_explorer")

    result: reactive.Value[BLMResult | None] = reactive.value(None)

    @reactive.effect
    @reactive.event(input.run_calibration)
    def _run():
        p = problem()
        s = solver()
        if p is None or s is None:
            ui.notification_show(
                "Load a project and configure a solver first.",
                type="error",
            )
            return
        ui.notification_show("Running BLM calibration...", type="message")
        try:
            blm_min = float(input.blm_min())
            blm_max = float(input.blm_max())
            if blm_min >= blm_max:
                ui.notification_show(
                    "Min BLM must be less than Max BLM.", type="error",
                )
                return
            res = calibrate_blm(
                p, s,
                blm_min=blm_min,
                blm_max=blm_max,
                blm_steps=int(input.blm_steps()),
                config=SolverConfig(num_solutions=1),
            )
            result.set(res)
            ui.notification_show(
                "BLM calibration complete!", type="message",
            )
        except Exception as e:
            ui.notification_show(
                f"Calibration error: {e}", type="error",
            )

    @render.plot
    def blm_plot():
        res = result()
        if res is None:
            return None
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(res.blm_values, res.costs, "o-", color="#0fa3b1")
        ax1.set_xlabel("BLM")
        ax1.set_ylabel("Cost")
        ax1.set_title("Cost vs BLM")
        ax2.plot(res.costs, res.boundaries, "o-", color="#e07a5f")
        ax2.set_xlabel("Cost")
        ax2.set_ylabel("Boundary Length")
        ax2.set_title("Cost vs Boundary (find the elbow)")
        for i, blm in enumerate(res.blm_values):
            ax2.annotate(
                f"{blm:.1f}",
                (res.costs[i], res.boundaries[i]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
            )
        fig.tight_layout()
        return fig

    @render.text
    def blm_table():
        res = result()
        if res is None:
            return "Run calibration to see results."
        lines = [
            f"{'BLM':>8} {'Cost':>10} {'Boundary':>10} {'Objective':>12}",
        ]
        lines.append("-" * 44)
        for i in range(len(res.blm_values)):
            lines.append(
                f"{res.blm_values[i]:8.2f} {res.costs[i]:10.2f} "
                f"{res.boundaries[i]:10.2f} {res.objectives[i]:12.2f}"
            )
        return "\n".join(lines)
