"""Connectivity configuration Shiny module.

Extends the existing connectivity tab with decay function selection
and CONNECTIVITY_WEIGHT parameter.
"""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup


@module.ui
def connectivity_config_ui():
    return ui.card(
        help_card_header("Connectivity Configuration"),
        ui.p(
            "Configure how connectivity between planning units is modelled. "
            "Choose a distance-decay function to transform raw connectivity "
            "values and set the overall connectivity weight in the objective "
            "function.",
            class_="text-muted small mb-3",
        ),
        ui.layout_sidebar(
            ui.sidebar(
                ui.tooltip(
                    ui.input_file(
                        "conn_matrix_file",
                        "Upload Connectivity Matrix",
                        accept=[".csv", ".tsv", ".txt"],
                    ),
                    "Upload a CSV with connectivity values. Edge-list format "
                    "(id1, id2, value) or full N×N matrix.",
                ),
                ui.tooltip(
                    ui.input_radio_buttons(
                        "conn_format",
                        "Matrix Format",
                        choices={
                            "edge_list": "Edge List (id1, id2, value)",
                            "full_matrix": "Full Matrix (NxN)",
                        },
                        selected="edge_list",
                    ),
                    "Edge List: CSV with columns id1, id2, value. "
                    "Full Matrix: square N×N CSV.",
                ),
                ui.hr(),
                ui.h5("Decay Function"),
                ui.tooltip(
                    ui.input_select(
                        "decay_function",
                        "Distance Decay",
                        choices={
                            "none": "None (raw values)",
                            "exponential": "Negative Exponential",
                            "power": "Inverse Power",
                            "threshold": "Threshold",
                        },
                        selected="none",
                    ),
                    "Apply a decay function to distance-based connectivity. "
                    "Exponential: exp(-α·d), Power: d^(-β), "
                    "Threshold: 1 if d ≤ max_distance else 0.",
                ),
                ui.panel_conditional(
                    "input.decay_function === 'exponential'",
                    ui.tooltip(
                        ui.input_numeric(
                            "decay_alpha",
                            "Alpha (α)",
                            value=1.0,
                            min=0.01,
                            step=0.1,
                        ),
                        "Rate of exponential decay. Higher α means connectivity "
                        "drops off faster with distance.",
                    ),
                ),
                ui.panel_conditional(
                    "input.decay_function === 'power'",
                    ui.tooltip(
                        ui.input_numeric(
                            "decay_beta",
                            "Beta (β)",
                            value=1.0,
                            min=0.01,
                            step=0.1,
                        ),
                        "Exponent for inverse power decay. Higher β means "
                        "faster drop-off.",
                    ),
                ),
                ui.panel_conditional(
                    "input.decay_function === 'threshold'",
                    ui.tooltip(
                        ui.input_numeric(
                            "decay_max_dist",
                            "Max Distance",
                            value=100.0,
                            min=0.0,
                            step=10.0,
                        ),
                        "Maximum distance for connectivity. Planning units "
                        "beyond this distance are not connected.",
                    ),
                ),
                ui.hr(),
                ui.h5("Weight"),
                ui.tooltip(
                    ui.input_slider(
                        "connectivity_weight",
                        "CONNECTIVITY_WEIGHT",
                        min=0.0,
                        max=100.0,
                        value=1.0,
                        step=0.1,
                    ),
                    "Overall weight of the connectivity term in the objective "
                    "function. Higher values favour more connected reserve "
                    "networks.",
                ),
                width=350,
            ),
            ui.output_text_verbatim("conn_config_summary"),
        ),
    )


@module.server
def connectivity_config_server(
    input,
    output,
    session,
    problem: reactive.Value,
    connectivity_matrix: reactive.Value,
    connectivity_pu_ids: reactive.Value,
):
    help_server_setup(input, "connectivity_config")

    @reactive.effect
    @reactive.event(input.conn_matrix_file)
    def _on_upload():
        file_info = input.conn_matrix_file()
        if file_info is None or len(file_info) == 0:
            return

        path = file_info[0]["datapath"]
        fmt = input.conn_format()
        p = problem()

        try:
            if fmt == "edge_list":
                from pymarxan.connectivity.io import read_connectivity_edgelist

                if p is None:
                    ui.notification_show(
                        "Load planning units first.",
                        type="warning",
                    )
                    return
                pu_ids = p.planning_units["id"].tolist()
                matrix = read_connectivity_edgelist(path, pu_ids)
                connectivity_pu_ids.set(pu_ids)
            else:
                from pymarxan.connectivity.io import read_connectivity_matrix

                matrix = read_connectivity_matrix(path)
                connectivity_pu_ids.set(list(range(matrix.shape[0])))

            # Apply decay if selected
            decay = input.decay_function()
            if decay != "none":
                from pymarxan.connectivity.decay import apply_decay

                if decay == "exponential":
                    matrix = apply_decay(
                        matrix, "exponential", alpha=float(input.decay_alpha())
                    )
                elif decay == "power":
                    matrix = apply_decay(
                        matrix, "power", beta=float(input.decay_beta())
                    )
                elif decay == "threshold":
                    matrix = apply_decay(
                        matrix, "threshold", max_distance=float(input.decay_max_dist())
                    )

            connectivity_matrix.set(matrix)
            ui.notification_show(
                f"Loaded {matrix.shape[0]}×{matrix.shape[1]} connectivity matrix.",
                type="message",
            )
        except Exception as exc:
            ui.notification_show(f"Error: {exc}", type="error")

    @reactive.effect
    @reactive.event(input.connectivity_weight)
    def _update_weight():
        p = problem()
        if p is None:
            return
        p.parameters["CONNECTIVITY_WEIGHT"] = float(input.connectivity_weight())
        problem.set(p)

    @render.text
    def conn_config_summary():
        m = connectivity_matrix()
        if m is None:
            return "No connectivity matrix loaded."
        import numpy as np

        nonzero = int(np.count_nonzero(m))
        total = m.shape[0] * m.shape[1]
        density = 100.0 * nonzero / total if total > 0 else 0.0
        lines = [
            f"Matrix shape: {m.shape[0]} × {m.shape[1]}",
            f"Non-zero entries: {nonzero}",
            f"Density: {density:.1f}%",
            f"Value range: [{m.min():.4f}, {m.max():.4f}]",
            "",
            f"Decay function: {input.decay_function()}",
            f"CONNECTIVITY_WEIGHT: {input.connectivity_weight()}",
        ]
        return "\n".join(lines)
