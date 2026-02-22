"""Solution map Shiny module — ipyleaflet map of selected planning units."""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan.models.geometry import generate_grid


@module.ui
def solution_map_ui():
    return ui.card(
        ui.card_header("Solution Map"),
        ui.output_ui("map_content"),
    )


@module.server
def solution_map_server(
    input, output, session,
    problem: reactive.Value,
    solution: reactive.Value,
):
    @render.ui
    def map_content():
        p = problem()
        s = solution()
        if p is None or s is None:
            return ui.p("Run a solver to see results here.")

        n_pu = len(p.planning_units)
        try:
            import ipyleaflet as _ipl  # noqa: F401

            # Compute grid bounds so the import is exercised
            grid = generate_grid(n_pu)
            all_lats = [b[0][0] for b in grid] + [b[1][0] for b in grid]
            all_lons = [b[0][1] for b in grid] + [b[1][1] for b in grid]
            bounds_info = (
                f"Grid bounds: [{min(all_lats):.4f}, {min(all_lons):.4f}]"
                f" to [{max(all_lats):.4f}, {max(all_lons):.4f}]"
            )

            cost_bnd = (
                f"Cost: {s.cost:.2f} | Boundary: {s.boundary:.2f}"
                f" | Objective: {s.objective:.2f}"
            )
            return ui.div(
                ui.h5("Solution Summary"),
                ui.p(f"Selected: {s.n_selected} / {n_pu} planning units"),
                ui.p(cost_bnd),
                ui.p(
                    f"Targets met: {sum(s.targets_met.values())}"
                    f" / {len(s.targets_met)}"
                ),
                ui.p(bounds_info),
            )
        except ImportError:
            pu_ids = p.planning_units["id"].tolist()
            rows = [
                f"PU {pid}" for i, pid in enumerate(pu_ids) if s.selected[i]
            ]
            return ui.div(
                ui.p(f"Selected: {s.n_selected} / {n_pu}"),
                ui.p(", ".join(rows) if rows else "No PUs selected."),
            )
