"""Solver selection and configuration Shiny module."""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan.solvers.marxan_binary import MarxanBinarySolver


@module.ui
def solver_picker_ui():
    binary_available = MarxanBinarySolver().available()
    solver_choices = {"mip": "MIP Solver (exact, PuLP/CBC)"}
    solver_choices["sa"] = "Simulated Annealing (Python)"
    solver_choices["zone_sa"] = "Zone SA (Multi-Zone)"
    if binary_available:
        solver_choices["binary"] = "Marxan C++ Binary"
    return ui.card(
        ui.card_header("Solver Configuration"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_radio_buttons(
                    "solver_type", "Solver",
                    choices=solver_choices, selected="mip",
                ),
                ui.hr(),
                ui.h5("Parameters"),
                ui.input_numeric(
                    "blm", "Boundary Length Modifier (BLM)",
                    value=1.0, min=0, step=0.1,
                ),
                ui.input_numeric(
                    "num_solutions", "Number of solutions",
                    value=10, min=1, max=1000, step=1,
                ),
                ui.input_numeric(
                    "seed", "Random seed (optional)",
                    value=42, min=-1,
                ),
                ui.hr(),
                ui.panel_conditional(
                    "input.solver_type === 'binary'",
                    ui.input_numeric(
                        "num_iterations", "SA Iterations",
                        value=1000000, min=1000, step=100000,
                    ),
                    ui.input_numeric(
                        "num_temp", "Temperature steps",
                        value=10000, min=100, step=1000,
                    ),
                ),
                ui.panel_conditional(
                    "input.solver_type === 'sa'",
                    ui.input_numeric(
                        "sa_iterations", "SA Iterations",
                        value=1000000, min=1000, step=100000,
                    ),
                    ui.input_numeric(
                        "sa_temp_steps", "Temperature steps",
                        value=10000, min=100, step=1000,
                    ),
                ),
                ui.panel_conditional(
                    "input.solver_type === 'zone_sa'",
                    ui.p("Uses Zone SA solver for multi-zone problems. "
                         "Requires zone data to be loaded in the Zones tab."),
                    ui.input_numeric(
                        "zone_sa_iterations", "SA Iterations",
                        value=1000000, min=1000, step=100000,
                    ),
                    ui.input_numeric(
                        "zone_sa_temp_steps", "Temperature steps",
                        value=10000, min=100, step=1000,
                    ),
                ),
                ui.panel_conditional(
                    "input.solver_type === 'mip'",
                    ui.input_numeric(
                        "mip_time_limit", "Time Limit (seconds)",
                        value=300, min=10, step=30,
                    ),
                    ui.input_numeric(
                        "mip_gap", "Optimality Gap",
                        value=0.0, min=0.0, max=1.0, step=0.01,
                    ),
                    ui.input_checkbox(
                        "mip_verbose", "Verbose output", value=False,
                    ),
                ),
                width=350,
            ),
            ui.output_text_verbatim("solver_info"),
        ),
    )

@module.server
def solver_picker_server(input, output, session, solver_config: reactive.Value):
    @reactive.effect
    @reactive.event(
        input.solver_type, input.blm, input.num_solutions,
        input.seed, input.sa_iterations, input.sa_temp_steps,
        input.zone_sa_iterations, input.zone_sa_temp_steps,
        input.mip_time_limit, input.mip_gap, input.mip_verbose,
        ignore_init=False,
    )
    def _update_config():
        config = {
            "solver_type": input.solver_type(),
            "blm": float(input.blm()),
            "num_solutions": int(input.num_solutions()),
            "seed": int(input.seed()) if input.seed() and input.seed() > 0 else None,
        }
        if input.solver_type() == "binary":
            config["num_iterations"] = int(input.num_iterations() or 1000000)
            config["num_temp"] = int(input.num_temp() or 10000)
        if input.solver_type() == "sa":
            config["num_iterations"] = int(input.sa_iterations() or 1000000)
            config["num_temp"] = int(input.sa_temp_steps() or 10000)
        if input.solver_type() == "zone_sa":
            config["num_iterations"] = int(input.zone_sa_iterations() or 1000000)
            config["num_temp"] = int(input.zone_sa_temp_steps() or 10000)
        if input.solver_type() == "mip":
            config["mip_time_limit"] = int(input.mip_time_limit() or 300)
            config["mip_gap"] = float(
                input.mip_gap() if input.mip_gap() is not None else 0.0
            )
            config["mip_verbose"] = bool(input.mip_verbose())
        solver_config.set(config)

    @render.text
    def solver_info():
        st = input.solver_type()
        if st == "mip":
            return ("MIP Solver (PuLP/CBC)\n---------------------\n"
                    "Uses Mixed Integer Linear Programming to find the\n"
                    "mathematically optimal solution. Guaranteed to find\n"
                    "the minimum-cost reserve network that meets all targets.\n"
                    "Equivalent to prioritizr in R.")
        elif st == "binary":
            return ("Marxan C++ Binary\n-----------------\n"
                    "Wraps the original Marxan executable using simulated\n"
                    "annealing. Produces multiple solutions across repeat\n"
                    "runs. Heuristic — not guaranteed optimal but well-tested.")
        elif st == "sa":
            return ("Simulated Annealing (Python)\n----------------------------\n"
                    "Native Python implementation of the Marxan SA algorithm.\n"
                    "No external binary needed. Supports multiple independent\n"
                    "runs with adaptive cooling schedule.")
        elif st == "zone_sa":
            return ("Zone SA (Multi-Zone)\n--------------------\n"
                    "Simulated annealing for multi-zone conservation planning.\n"
                    "Each planning unit is assigned to a zone (or left unassigned).\n"
                    "Requires zone project data (zones, zone costs, etc.).")
        return ""
