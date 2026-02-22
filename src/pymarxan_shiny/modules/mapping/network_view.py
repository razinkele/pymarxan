"""Network view Shiny module — connectivity graph overlay on PU grid."""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan.connectivity.metrics import compute_in_degree
from pymarxan.models.geometry import generate_grid


def metric_color(normalized: float) -> str:
    """Map a 0-1 normalized metric to a yellow-to-purple hex color.

    0.0 -> yellow (#f1c40f), 1.0 -> purple (#8e44ad).
    """
    normalized = max(0.0, min(1.0, normalized))
    r = int(241 * (1.0 - normalized) + 142 * normalized)
    g = int(196 * (1.0 - normalized) + 68 * normalized)
    b = int(15 * (1.0 - normalized) + 173 * normalized)
    return f"#{r:02x}{g:02x}{b:02x}"


def compute_centroids(
    grid: list[tuple[tuple[float, float], tuple[float, float]]],
) -> list[tuple[float, float]]:
    """Compute the center point of each bounding box."""
    centroids: list[tuple[float, float]] = []
    for (s, w), (n, e) in grid:
        centroids.append(((s + n) / 2, (w + e) / 2))
    return centroids


@module.ui
def network_view_ui():
    return ui.card(
        ui.card_header("Connectivity Network"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select(
                    "metric",
                    "Color by Metric",
                    choices={
                        "in_degree": "In-Degree",
                        "out_degree": "Out-Degree",
                    },
                    selected="in_degree",
                ),
                ui.input_slider(
                    "edge_threshold",
                    "Min Edge Weight",
                    min=0.0,
                    max=1.0,
                    value=0.0,
                    step=0.01,
                ),
                width=220,
            ),
            ui.output_ui("network_content"),
        ),
    )


@module.server
def network_view_server(
    input,
    output,
    session,
    problem: reactive.Value,
    connectivity_matrix: reactive.Value,
    connectivity_pu_ids: reactive.Value,
):
    @render.ui
    def network_content():
        p = problem()
        matrix = connectivity_matrix()
        if p is None or matrix is None:
            return ui.p(
                "Load a project with connectivity data"
                " to see the network."
            )

        n_pu = len(p.planning_units)
        metric_name = input.metric()

        # Compute metric
        if metric_name == "out_degree":
            from pymarxan.connectivity.metrics import compute_out_degree

            metric_values = compute_out_degree(matrix)
        else:
            metric_values = compute_in_degree(matrix)

        grid = generate_grid(n_pu)
        centroids = compute_centroids(grid)

        max_val = float(metric_values.max()) if len(metric_values) > 0 else 0.0
        min_val = float(metric_values.min()) if len(metric_values) > 0 else 0.0

        return ui.div(
            ui.p(f"{n_pu} nodes — colored by {metric_name}"),
            ui.p(f"{len(centroids)} centroids computed"),
            ui.p(f"Metric range: {min_val:.2f} – {max_val:.2f}"),
        )
