"""Convergence plot Shiny module — SA objective over iterations."""
from __future__ import annotations

from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from pymarxan.solvers.base import Solution
from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup


def extract_history(
    solutions: list[Solution],
) -> list[dict]:
    """Extract iteration histories from solutions that have them.

    Returns a list of dicts, each with keys: run, iteration, objective,
    best_objective, temperature.
    """
    histories: list[dict] = []
    for sol in solutions:
        history = sol.metadata.get("history")
        if history and len(history.get("iteration", [])) > 0:
            histories.append({
                "run": sol.metadata.get("run", len(histories) + 1),
                **history,
            })
    return histories


@module.ui
def convergence_ui():
    return ui.card(
        help_card_header("SA Convergence"),
        ui.p(
            "Plot the simulated annealing convergence curve for each solver run. "
            "The current objective (blue) should decline towards the best objective "
            "(green) as iterations progress. Optionally overlay the temperature "
            "schedule (red dotted line). If the curves have not flattened, consider "
            "increasing NUMITNS or NUMTEMP.",
            class_="text-muted small mb-3",
        ),
        ui.layout_sidebar(
            ui.sidebar(
                ui.tooltip(
                    ui.input_select(
                        "run_select",
                        "Select Run",
                        choices={"1": "Run 1"},
                        selected="1",
                    ),
                    "Choose which solver run's convergence history to display.",
                ),
                ui.tooltip(
                    ui.input_checkbox(
                        "show_temperature",
                        "Show Temperature",
                        value=False,
                    ),
                    "Overlay the SA temperature on a secondary y-axis (log scale). "
                    "Useful for diagnosing cooling schedule issues.",
                ),
                width=200,
            ),
            ui.output_plot("convergence_plot"),
        ),
    )


@module.server
def convergence_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    all_solutions: reactive.Value,
):
    help_server_setup(input, "convergence")

    @reactive.calc
    def histories():
        solutions = all_solutions()
        if solutions is None:
            return []
        return extract_history(solutions)

    @reactive.effect
    def _update_run_choices():
        h = histories()
        if not h:
            return
        choices = {str(i + 1): f"Run {entry['run']}" for i, entry in enumerate(h)}
        ui.update_select("run_select", choices=choices, selected="1")

    @render.plot
    def convergence_plot():
        import matplotlib.pyplot as plt

        h = histories()
        if not h:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.text(
                0.5, 0.5, "No convergence data — run an SA solver first.",
                ha="center", va="center", color="#666",
            )
            ax.axis("off")
            return fig

        idx = int(input.run_select()) - 1
        if idx < 0 or idx >= len(h):
            idx = 0
        entry = h[idx]

        # Rendered server-side with matplotlib (@render.plot) rather than plotly
        # via to_html(include_plotlyjs="cdn"), which fails to load under Shiny's
        # dynamic HTML injection ("Plotly is not defined").
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(
            entry["iteration"], entry["objective"],
            color="#0fa3b1", lw=1, alpha=0.6, label="Current objective",
        )
        ax.plot(
            entry["iteration"], entry["best_objective"],
            color="#2d936c", lw=2, label="Best objective",
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective value")
        ax.set_title(f"SA Convergence — Run {entry['run']}")
        ax.legend(loc="upper right")

        if input.show_temperature():
            ax2 = ax.twinx()
            ax2.plot(
                entry["iteration"], entry["temperature"],
                color="#e07a5f", lw=1, ls=":", label="Temperature",
            )
            ax2.set_ylabel("Temperature (log)")
            ax2.set_yscale("log")
            ax2.legend(loc="lower right")

        return fig
