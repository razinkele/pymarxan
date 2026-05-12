"""Network view Shiny module — connectivity graph overlay on PU grid."""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan.connectivity.metrics import compute_in_degree
from pymarxan.models.geometry import generate_grid
from pymarxan.models.problem import has_geometry
from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup
from pymarxan_shiny.modules.mapping.ocean_palette import (
    EDGE_COLOR,
    MAP_FALLBACK,
    METRIC_HIGH_RGB,
    METRIC_LOW_RGB,
)

try:
    import ipyleaflet
    from shinywidgets import output_widget, render_widget

    from pymarxan_shiny.modules.mapping.map_utils import create_geo_map, create_grid_map

    _HAS_IPYLEAFLET = True
except ImportError:
    _HAS_IPYLEAFLET = False

MAX_EDGES = 5000


def metric_color(normalized: float) -> str:
    """Map a 0-1 normalized metric to an aqua → deep-navy hex color.

    0.0 -> aqua, 1.0 -> deep navy.
    """
    normalized = max(0.0, min(1.0, normalized))
    r = int(METRIC_LOW_RGB[0] * (1.0 - normalized) + METRIC_HIGH_RGB[0] * normalized)
    g = int(METRIC_LOW_RGB[1] * (1.0 - normalized) + METRIC_HIGH_RGB[1] * normalized)
    b = int(METRIC_LOW_RGB[2] * (1.0 - normalized) + METRIC_HIGH_RGB[2] * normalized)
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
        ui.tooltip(
            ui.input_select(
                "metric",
                "Color by Metric",
                choices={
                    "in_degree": "In-Degree",
                    "out_degree": "Out-Degree",
                },
                selected="in_degree",
            ),
            "Connectivity metric to color nodes by. In-degree counts incoming "
            "connections; out-degree counts outgoing connections.",
        ),
        ui.tooltip(
            ui.input_slider(
                "edge_threshold",
                "Min Edge Weight",
                min=0.0,
                max=1.0,
                value=0.0,
                step=0.01,
            ),
            "Only display edges with weight above this threshold. "
            "Increase to reduce visual clutter for dense networks.",
        ),
        width=220,
    )
    if _HAS_IPYLEAFLET:
        return ui.card(
            help_card_header("Connectivity Network"),
            ui.p(
                "Visualise the connectivity network overlaid on planning units. "
                "Nodes are colored by the selected graph metric; edges show "
                "connections above the weight threshold. Useful for identifying "
                "key corridors and connectivity bottlenecks.",
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
        help_card_header("Connectivity Network"),
        ui.p(
            "Connectivity network view (install ipyleaflet for interactive maps).",
            class_="text-muted small mb-3",
        ),
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
    help_server_setup(input, "network_view")

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

            max_val = float(metric_values.max()) if len(metric_values) > 0 else 0.0
            min_val = float(metric_values.min()) if len(metric_values) > 0 else 0.0
            rng = max_val - min_val if max_val > min_val else 1.0

            colors = [
                metric_color((float(metric_values[i]) - min_val) / rng)
                if i < len(metric_values) else MAP_FALLBACK
                for i in range(n_pu)
            ]

            if has_geometry(p):
                m = create_geo_map(p.planning_units, colors)
                # Centroids must also be in EPSG:4326 (lat/lon) to line up
                # with the reprojected polygons drawn by create_geo_map.
                pus_for_centroids = p.planning_units
                if (
                    pus_for_centroids.crs is not None
                    and pus_for_centroids.crs.to_epsg() != 4326
                ):
                    pus_for_centroids = pus_for_centroids.to_crs("EPSG:4326")
                centroids = [
                    (geom.centroid.y, geom.centroid.x)
                    for geom in pus_for_centroids.geometry
                ]
            else:
                grid = generate_grid(n_pu)
                m = create_grid_map(grid, colors)
                centroids = compute_centroids(grid)

            # Add polyline edges (capped to prevent browser freeze)
            n = min(matrix.shape[0], n_pu)
            edge_count = 0
            for i in range(n):
                for j in range(n):
                    weight = float(matrix[i, j])
                    if weight > threshold and i != j:
                        if edge_count >= MAX_EDGES:
                            break
                        line = ipyleaflet.Polyline(
                            locations=[centroids[i], centroids[j]],
                            color=EDGE_COLOR,
                            opacity=min(1.0, weight),
                            weight=2,
                        )
                        m.add(line)
                        edge_count += 1
                if edge_count >= MAX_EDGES:
                    break

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
            edge_count_total = sum(
                1 for i in range(n) for j in range(n)
                if float(matrix[i, j]) > threshold and i != j
            )

            truncated = ""
            if edge_count_total > MAX_EDGES:
                truncated = (
                    f"\n\u26A0 Showing {MAX_EDGES} of {edge_count_total} edges "
                    f"(capped to prevent browser slowdown)"
                )

            return (
                f"{n_pu} nodes \u2014 colored by {metric_name}\n"
                f"Metric range: {min_val:.2f} \u2013 {max_val:.2f}\n"
                f"Edges shown: {min(edge_count_total, MAX_EDGES)} "
                f"(threshold: {threshold:.2f}){truncated}"
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
