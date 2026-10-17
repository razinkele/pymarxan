"""Solution map Shiny module — ipyleaflet map of selected planning units."""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution
from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup
from pymarxan_shiny.modules.mapping.map_utils import build_pu_map, too_large_for_map
from pymarxan_shiny.modules.mapping.ocean_palette import MAP_NOT_SEL, MAP_SELECTED

try:
    from shinywidgets import output_widget, render_widget

    _HAS_IPYLEAFLET = True
except ImportError:
    _HAS_IPYLEAFLET = False


def _blend(c0: str, c1: str, t: float) -> str:
    """Hex colour ``t`` of the way from ``c0`` to ``c1`` (``t`` clipped to [0, 1])."""
    t = min(1.0, max(0.0, float(t)))
    a = [int(c0[k:k + 2], 16) for k in (1, 3, 5)]
    b = [int(c1[k:k + 2], 16) for k in (1, 3, 5)]
    return "#" + "".join(f"{round(u + (v - u) * t):02x}" for u, v in zip(a, b, strict=True))


def solution_colors(solution: Solution) -> list[str]:
    """Per-PU map colours: selected / not selected, or interpolated by x for proportions."""
    if solution.is_fractional:
        return [_blend(MAP_NOT_SEL, MAP_SELECTED, x) for x in solution.selected]
    return [MAP_SELECTED if v else MAP_NOT_SEL for v in solution.selected]


def solution_listing(problem: ConservationProblem, solution: Solution) -> list[str]:
    """``"PU 7"`` per selected PU; with proportions ``"PU 7 (0.25)"``."""
    pu_ids = problem.planning_units["id"].tolist()
    if solution.is_fractional:
        return [
            f"PU {pid} ({float(solution.selected[i]):.4g})"
            for i, pid in enumerate(pu_ids) if solution.selected[i]
        ]
    return [f"PU {pid}" for i, pid in enumerate(pu_ids) if solution.selected[i]]


@module.ui
def solution_map_ui():
    if _HAS_IPYLEAFLET:
        return ui.card(
            help_card_header("Solution Map"),
            ui.p(
                "Map of the current solution. Green planning units are selected "
                "for the reserve network; gray units are not selected. "
                "Run a solver to populate this map.",
                class_="text-muted small mb-3",
            ),
            output_widget("map"),
            ui.output_text_verbatim("map_summary"),
        )
    return ui.card(
        help_card_header("Solution Map"),
        ui.p(
            "Solution map (install ipyleaflet for interactive maps).",
            class_="text-muted small mb-3",
        ),
        ui.output_ui("map_content"),
    )


@module.server
def solution_map_server(
    input, output, session,
    problem: reactive.Value,
    solution: reactive.Value,
):
    help_server_setup(input, "solution_map")

    if _HAS_IPYLEAFLET:

        @render_widget
        def map():
            p = problem()
            s = solution()
            if p is None or s is None:
                return None

            return build_pu_map(p, solution_colors(s))

        @render.text
        def map_summary():
            p = problem()
            s = solution()
            if p is None or s is None:
                return "Run a solver to see results here."
            if too_large_for_map(p):
                return (
                    f"Grid too large to map ({p.n_planning_units} cells); "
                    "use the analysis/table views."
                )

            n_pu = len(p.planning_units)
            targets_met = sum(s.targets_met.values())
            total_targets = len(s.targets_met)
            share = f" (Σx {s.selected_sum:.2f})" if s.is_fractional else ""
            return (
                f"Selected: {s.n_selected} / {n_pu} planning units{share}\n"
                f"Cost: {s.cost:.2f} | Boundary: {s.boundary:.2f}"
                f" | Objective: {s.objective:.2f}\n"
                f"Targets met: {targets_met} / {total_targets}"
            )

    if not _HAS_IPYLEAFLET:

        @render.ui
        def map_content():
            p = problem()
            s = solution()
            if p is None or s is None:
                return ui.p("Run a solver to see results here.")

            n_pu = len(p.planning_units)
            rows = solution_listing(p, s)
            return ui.div(
                ui.p(f"Selected: {s.n_selected} / {n_pu}"),
                ui.p(", ".join(rows) if rows else "No PUs selected."),
            )
