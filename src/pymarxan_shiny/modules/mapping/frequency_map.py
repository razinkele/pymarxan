"""Frequency map Shiny module — selection frequency heatmap."""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan.analysis.selection_freq import compute_selection_frequency
from pymarxan.models.geometry import generate_grid
from pymarxan.models.problem import has_geometry

try:
    from shinywidgets import output_widget, render_widget

    from pymarxan_shiny.modules.mapping.map_utils import create_geo_map, create_grid_map

    _HAS_IPYLEAFLET = True
except ImportError:
    _HAS_IPYLEAFLET = False


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
    if _HAS_IPYLEAFLET:
        return ui.card(
            ui.card_header("Selection Frequency"),
            output_widget("map"),
            ui.output_text_verbatim("map_summary"),
        )
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
    if _HAS_IPYLEAFLET:

        @render_widget
        def map():
            p = problem()
            sols = all_solutions()
            if p is None or sols is None or len(sols) == 0:
                return None

            sf = compute_selection_frequency(sols)
            n_pu = len(p.planning_units)
            colors = [frequency_color(sf.frequencies[i]) for i in range(n_pu)]

            if has_geometry(p):
                return create_geo_map(p.planning_units, colors)

            grid = generate_grid(n_pu)
            return create_grid_map(grid, colors)

        @render.text
        def map_summary():
            p = problem()
            sols = all_solutions()
            if p is None or sols is None or len(sols) == 0:
                return "Run solver with multiple solutions to see frequency map."

            sf = compute_selection_frequency(sols)
            return f"Frequency across {sf.n_solutions} solutions"

    if not _HAS_IPYLEAFLET:

        @render.ui
        def freq_content():
            p = problem()
            sols = all_solutions()
            if p is None or sols is None or len(sols) == 0:
                return ui.p(
                    "Run solver with multiple solutions to see frequency map."
                )

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
