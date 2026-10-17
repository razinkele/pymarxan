"""Comparison map Shiny module -- side-by-side solution comparison."""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan.solvers.base import FractionalSolutionError, Solution, require_binary
from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup
from pymarxan_shiny.modules.mapping.map_utils import build_pu_map, too_large_for_map
from pymarxan_shiny.modules.mapping.ocean_palette import (
    CMP_A_ONLY,
    CMP_B_ONLY,
    CMP_BOTH,
    CMP_NEITHER,
)

try:
    from shinywidgets import output_widget, render_widget

    _HAS_IPYLEAFLET = True
except ImportError:
    _HAS_IPYLEAFLET = False


def comparison_color(in_a: bool, in_b: bool) -> str:
    """Return color based on which solutions include a PU.

    Teal = both, Ocean-blue = A only,
    Coral = B only, Steel-gray = neither.
    """
    if in_a and in_b:
        return CMP_BOTH      # teal — both
    elif in_a:
        return CMP_A_ONLY    # ocean-blue — A only
    elif in_b:
        return CMP_B_ONLY    # coral — B only
    return CMP_NEITHER       # steel-gray — neither


def _require_pair(sol_a: Solution, sol_b: Solution) -> None:
    """The four-colour comparison is categorical: binary plans only (Stage 4 §6.2)."""
    require_binary(sol_a, "comparison map")
    require_binary(sol_b, "comparison map")


def comparison_counts(sol_a: Solution, sol_b: Solution) -> tuple[int, int, int]:
    """(both, A only, B only) planning-unit counts of two binary plans."""
    _require_pair(sol_a, sol_b)
    a, b = sol_a.selected, sol_b.selected
    both = sum(1 for i in range(len(a)) if a[i] and b[i])
    a_only = sum(1 for i in range(len(a)) if a[i] and not b[i])
    b_only = sum(1 for i in range(len(a)) if not a[i] and b[i])
    return both, a_only, b_only


def comparison_summary(sol_a: Solution, sol_b: Solution) -> str:
    """The user-facing summary line; a fractional plan gets an explanation, not a crash."""
    try:
        both, a_only, b_only = comparison_counts(sol_a, sol_b)
    except FractionalSolutionError:
        return (
            "Comparison unavailable: this map compares whole-unit (binary) plans, but a "
            "selected solution holds proportion decisions. Re-run with Decisions = Binary, "
            "or convert with binarize_solution()."
        )
    return f"Both: {both} | A only: {a_only} | B only: {b_only}"


@module.ui
def comparison_map_ui():
    sidebar = ui.sidebar(
        ui.tooltip(
            ui.input_select(
                "sol_a", "Solution A",
                choices={"0": "Run 1"}, selected="0",
            ),
            "First solution to compare. Select from the available solver runs.",
        ),
        ui.tooltip(
            ui.input_select(
                "sol_b", "Solution B",
                choices={"1": "Run 2"}, selected="1",
            ),
            "Second solution to compare against Solution A.",
        ),
        ui.div(
            ui.span("\u25a0", style=f"color:{CMP_BOTH}"), " Both  ",
            ui.span("\u25a0", style=f"color:{CMP_A_ONLY}"), " A only  ",
            ui.span("\u25a0", style=f"color:{CMP_B_ONLY}"), " B only  ",
            ui.span("\u25a0", style=f"color:{CMP_NEITHER}"), " Neither",
        ),
        width=220,
    )
    if _HAS_IPYLEAFLET:
        return ui.card(
            help_card_header("Solution Comparison"),
            ui.p(
                "Compare two solutions side-by-side to see which planning units "
                "are shared vs. unique to each. Green = in both, blue = A only, "
                "orange = B only, gray = neither. Useful for assessing solution "
                "consistency.",
                class_="text-muted small mb-3",
            ),
            ui.layout_sidebar(
                sidebar,
                ui.div(
                    output_widget("map"),
                    ui.output_text_verbatim("map_summary"),
                ),
            ),
        )
    return ui.card(
        help_card_header("Solution Comparison"),
        ui.p(
            "Solution comparison (install ipyleaflet for interactive maps).",
            class_="text-muted small mb-3",
        ),
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
    help_server_setup(input, "comparison_map")

    @reactive.effect
    def _update_choices():
        sols = all_solutions()
        if sols is None or len(sols) < 2:
            return
        choices = {
            str(i): f"Run {i + 1}" for i in range(len(sols))
        }
        # Preserve the user's current selection if it still references a
        # valid run — previously every new solver run reset both dropdowns
        # back to Run 1 / Run 2.
        current_a = input.sol_a()
        current_b = input.sol_b()
        selected_a = current_a if current_a in choices else "0"
        selected_b = current_b if current_b in choices else "1"
        ui.update_select("sol_a", choices=choices, selected=selected_a)
        ui.update_select("sol_b", choices=choices, selected=selected_b)

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
            if sol_a.is_fractional or sol_b.is_fractional:
                return None  # map_summary explains why (comparison_summary)
            n_pu = len(p.planning_units)

            colors = [
                comparison_color(sol_a.selected[i], sol_b.selected[i])
                for i in range(n_pu)
            ]

            return build_pu_map(p, colors)

        @render.text
        def map_summary():
            p = problem()
            sols = all_solutions()
            if p is None or sols is None or len(sols) < 2:
                return "Run solver with 2+ solutions to compare."
            if too_large_for_map(p):
                return (
                    f"Grid too large to map ({p.n_planning_units} cells); "
                    "use the analysis/table views."
                )

            idx_a = int(input.sol_a())
            idx_b = int(input.sol_b())
            if idx_a >= len(sols) or idx_b >= len(sols):
                return "Invalid solution index."

            return comparison_summary(sols[idx_a], sols[idx_b])

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

            return ui.div(ui.p(comparison_summary(sols[idx_a], sols[idx_b])))
