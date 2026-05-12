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
            ui.output_ui("convergence_plot"),
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

    @render.ui
    def convergence_plot():
        h = histories()
        if not h:
            return ui.p("No convergence data available. Run an SA solver first.")

        idx = int(input.run_select()) - 1
        if idx < 0 or idx >= len(h):
            idx = 0
        entry = h[idx]

        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=entry["iteration"],
                y=entry["objective"],
                mode="lines",
                name="Current Objective",
                line=dict(color="#0fa3b1", width=1),
                opacity=0.6,
            ))
            fig.add_trace(go.Scatter(
                x=entry["iteration"],
                y=entry["best_objective"],
                mode="lines",
                name="Best Objective",
                line=dict(color="#2d936c", width=2),
            ))

            if input.show_temperature():
                fig.add_trace(go.Scatter(
                    x=entry["iteration"],
                    y=entry["temperature"],
                    mode="lines",
                    name="Temperature",
                    line=dict(color="#e07a5f", width=1, dash="dot"),
                    yaxis="y2",
                ))
                fig.update_layout(
                    yaxis2=dict(
                        title="Temperature",
                        overlaying="y",
                        side="right",
                        type="log",
                    ),
                )

            fig.update_layout(
                xaxis_title="Iteration",
                yaxis_title="Objective Value",
                title=f"SA Convergence — Run {entry['run']}",
                height=400,
                margin=dict(l=60, r=60, t=40, b=40),
                legend=dict(x=0.7, y=0.95),
            )

            return ui.HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))
        except ImportError:
            # Fallback: text summary if plotly not available
            iters = entry["iteration"]
            bests = entry["best_objective"]
            lines = [f"Run {entry['run']} convergence:"]
            for i, b in zip(iters, bests):
                lines.append(f"  Iter {i:>8,}: best = {b:.2f}")
            return ui.pre("\n".join(lines))
