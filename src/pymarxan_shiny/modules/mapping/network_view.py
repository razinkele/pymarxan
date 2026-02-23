"""Network view Shiny module — connectivity graph overlay on PU grid."""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan.connectivity.metrics import compute_in_degree
from pymarxan.models.geometry import generate_grid

try:
    import ipyleaflet
    from shinywidgets import output_widget, render_widget

    from pymarxan_shiny.modules.mapping.map_utils import create_grid_map

    _HAS_IPYLEAFLET = True
except ImportError:
    _HAS_IPYLEAFLET = False


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
    sidebar = ui.sidebar(
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
    )
    if _HAS_IPYLEAFLET:
        return ui.card(
            ui.card_header("Connectivity Network"),
            ui.layout_sidebar(
                sidebar,
                ui.div(
                    output_widget("map"),
                    ui.output_text_verbatim("map_summary"),
                ),
            ),
        )
    return ui.card(
        ui.card_header("Connectivity Network"),
        ui.layout_sidebar(sidebar, ui.output_ui("network_content")),
    )


@module.server
def network_view_server(
    input,
    output,
    session,
    problem: reactive.Value,
    connectivity_matrix: reactive.Value,
):
    if _HAS_IPYLEAFLET:

        @render_widget
        def map():
            p = problem()
            matrix = connectivity_matrix()
            if p is None or matrix is None:
                return None

            n_pu = len(p.planning_units)
            metric_name = input.metric()
            threshold = input.edge_threshold()

            if metric_name == "out_degree":
                from pymarxan.connectivity.metrics import compute_out_degree

                metric_values = compute_out_degree(matrix)
            else:
                metric_values = compute_in_degree(matrix)

            grid = generate_grid(n_pu)
            max_val = float(metric_values.max()) if len(metric_values) > 0 else 0.0
            min_val = float(metric_values.min()) if len(metric_values) > 0 else 0.0
            rng = max_val - min_val if max_val > min_val else 1.0

            colors = [
                metric_color((float(metric_values[i]) - min_val) / rng)
                if i < len(metric_values) else "#bdc3c7"
                for i in range(n_pu)
            ]

            m = create_grid_map(grid, colors)

            # Add polyline edges
            centroids = compute_centroids(grid)
            n = min(matrix.shape[0], n_pu)
            for i in range(n):
                for j in range(n):
                    weight = float(matrix[i, j])
                    if weight > threshold and i != j:
                        line = ipyleaflet.Polyline(
                            locations=[centroids[i], centroids[j]],
                            color="#3498db",
                            opacity=min(1.0, weight),
                            weight=2,
                        )
                        m.add(line)

            return m

    @render.text
    def map_summary():
        p = problem()
        matrix = connectivity_matrix()
        if p is None or matrix is None:
            return "Load a project with connectivity data to see the network."

        n_pu = len(p.planning_units)
        metric_name = input.metric()

        if metric_name == "out_degree":
            from pymarxan.connectivity.metrics import compute_out_degree

            metric_values = compute_out_degree(matrix)
        else:
            metric_values = compute_in_degree(matrix)

        max_val = float(metric_values.max()) if len(metric_values) > 0 else 0.0
        min_val = float(metric_values.min()) if len(metric_values) > 0 else 0.0

        threshold = input.edge_threshold()
        n = min(matrix.shape[0], n_pu)
        edge_count = sum(
            1 for i in range(n) for j in range(n)
            if float(matrix[i, j]) > threshold and i != j
        )

        return (
            f"{n_pu} nodes — colored by {metric_name}\n"
            f"Metric range: {min_val:.2f} – {max_val:.2f}\n"
            f"Edges shown: {edge_count} (threshold: {threshold:.2f})"
        )

    if not _HAS_IPYLEAFLET:

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
