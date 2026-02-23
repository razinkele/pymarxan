"""Comparison map Shiny module -- side-by-side solution comparison."""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan.models.geometry import generate_grid

try:
    from shinywidgets import output_widget, render_widget

    from pymarxan_shiny.modules.mapping.map_utils import create_grid_map

    _HAS_IPYLEAFLET = True
except ImportError:
    _HAS_IPYLEAFLET = False


def comparison_color(in_a: bool, in_b: bool) -> str:
    """Return color based on which solutions include a PU.

    Green (#2ecc71) = both, Blue (#3498db) = A only,
    Orange (#e67e22) = B only, Gray (#bdc3c7) = neither.
    """
    if in_a and in_b:
        return "#2ecc71"  # green -- both
    elif in_a:
        return "#3498db"  # blue -- A only
    elif in_b:
        return "#e67e22"  # orange -- B only
    return "#bdc3c7"  # gray -- neither


@module.ui
def comparison_map_ui():
    sidebar = ui.sidebar(
        ui.input_select(
            "sol_a", "Solution A",
            choices={"0": "Run 1"}, selected="0",
        ),
        ui.input_select(
            "sol_b", "Solution B",
            choices={"1": "Run 2"}, selected="1",
        ),
        ui.div(
            ui.span("\u25a0", style="color:#2ecc71"), " Both  ",
            ui.span("\u25a0", style="color:#3498db"), " A only  ",
            ui.span("\u25a0", style="color:#e67e22"), " B only  ",
            ui.span("\u25a0", style="color:#bdc3c7"), " Neither",
        ),
        width=220,
    )
    if _HAS_IPYLEAFLET:
        return ui.card(
            ui.card_header("Solution Comparison"),
            ui.layout_sidebar(
                sidebar,
                ui.div(
                    output_widget("map"),
                    ui.output_text_verbatim("map_summary"),
                ),
            ),
        )
    return ui.card(
        ui.card_header("Solution Comparison"),
        ui.layout_sidebar(sidebar, ui.output_ui("cmp_content")),
    )


@module.server
def comparison_map_server(
    input,
    output,
    session,
    problem: reactive.Value,
    all_solutions: reactive.Value,
):
    @reactive.effect
    def _update_choices():
        sols = all_solutions()
        if sols is None or len(sols) < 2:
            return
        choices = {
            str(i): f"Run {i + 1}" for i in range(len(sols))
        }
        ui.update_select("sol_a", choices=choices, selected="0")
        ui.update_select("sol_b", choices=choices, selected="1")

    if _HAS_IPYLEAFLET:

        @render_widget
        def map():
            p = problem()
            sols = all_solutions()
            if p is None or sols is None or len(sols) < 2:
                return None

            idx_a = int(input.sol_a())
            idx_b = int(input.sol_b())
            if idx_a >= len(sols) or idx_b >= len(sols):
                return None

            sol_a = sols[idx_a]
            sol_b = sols[idx_b]
            n_pu = len(p.planning_units)
            grid = generate_grid(n_pu)

            colors = [
                comparison_color(sol_a.selected[i], sol_b.selected[i])
                for i in range(n_pu)
            ]
            return create_grid_map(grid, colors)

        @render.text
        def map_summary():
            p = problem()
            sols = all_solutions()
            if p is None or sols is None or len(sols) < 2:
                return "Run solver with 2+ solutions to compare."

            idx_a = int(input.sol_a())
            idx_b = int(input.sol_b())
            if idx_a >= len(sols) or idx_b >= len(sols):
                return "Invalid solution index."

            sol_a = sols[idx_a]
            sol_b = sols[idx_b]
            n_pu = len(p.planning_units)
            both = sum(
                1 for i in range(n_pu)
                if sol_a.selected[i] and sol_b.selected[i]
            )
            a_only = sum(
                1 for i in range(n_pu)
                if sol_a.selected[i] and not sol_b.selected[i]
            )
            b_only = sum(
                1 for i in range(n_pu)
                if not sol_a.selected[i] and sol_b.selected[i]
            )
            return f"Both: {both} | A only: {a_only} | B only: {b_only}"

    if not _HAS_IPYLEAFLET:

        @render.ui
        def cmp_content():
            p = problem()
            sols = all_solutions()
            if p is None or sols is None or len(sols) < 2:
                return ui.p("Run solver with 2+ solutions to compare.")

            idx_a = int(input.sol_a())
            idx_b = int(input.sol_b())
            if idx_a >= len(sols) or idx_b >= len(sols):
                return ui.p("Invalid solution index.")

            sol_a = sols[idx_a]
            sol_b = sols[idx_b]
            n_pu = len(p.planning_units)
            both = sum(
                1 for i in range(n_pu)
                if sol_a.selected[i] and sol_b.selected[i]
            )
            a_only = sum(
                1 for i in range(n_pu)
                if sol_a.selected[i] and not sol_b.selected[i]
            )
            b_only = sum(
                1 for i in range(n_pu)
                if not sol_a.selected[i] and sol_b.selected[i]
            )
            return ui.div(
                ui.p(f"Both: {both} | A only: {a_only} | B only: {b_only}"),
            )
