"""Frequency map Shiny module — selection frequency heatmap."""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan.analysis.selection_freq import compute_selection_frequency
from pymarxan.models.geometry import generate_grid


def frequency_color(freq: float) -> str:
    """Map a 0-1 frequency to a white-to-blue hex color.

    0.0 -> white (#ffffff), 1.0 -> dark blue (#2c3e50).
    """
    freq = max(0.0, min(1.0, freq))
    r = int(255 * (1.0 - freq) + 44 * freq)
    g = int(255 * (1.0 - freq) + 62 * freq)
    b = int(255 * (1.0 - freq) + 80 * freq)
    return f"#{r:02x}{g:02x}{b:02x}"


@module.ui
def frequency_map_ui():
    return ui.card(
        ui.card_header("Selection Frequency"),
        ui.output_ui("freq_content"),
    )


@module.server
def frequency_map_server(
    input, output, session,
    problem: reactive.Value,
    all_solutions: reactive.Value,
):
    @render.ui
    def freq_content():
        p = problem()
        sols = all_solutions()
        if p is None or sols is None or len(sols) == 0:
            return ui.p("Run solver with multiple solutions to see frequency map.")

        sf = compute_selection_frequency(sols)
        n_pu = len(p.planning_units)

        grid = generate_grid(n_pu)
        all_lats = [b[0][0] for b in grid] + [b[1][0] for b in grid]
        all_lons = [b[0][1] for b in grid] + [b[1][1] for b in grid]
        bounds_info = (
            f"Grid: [{min(all_lats):.4f}, {min(all_lons):.4f}]"
            f" to [{max(all_lats):.4f}, {max(all_lons):.4f}]"
        )
        return ui.div(
            ui.p(f"Frequency across {sf.n_solutions} solutions"),
            ui.p(bounds_info),
        )
