"""Solver selection and configuration Shiny module."""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan.solvers.marxan_binary import MarxanBinarySolver
from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup


@module.ui
def solver_picker_ui():
    binary_available = MarxanBinarySolver().available()
    solver_choices = {"mip": "MIP Solver (exact, PuLP/CBC)"}
    solver_choices["sa"] = "Simulated Annealing (Python)"
    solver_choices["zone_sa"] = "Zone SA (Multi-Zone)"
    solver_choices["greedy"] = "Greedy Heuristic"
    solver_choices["iterative_improvement"] = "Iterative Improvement"
    solver_choices["pipeline"] = "Run Mode Pipeline"
    if binary_available:
        solver_choices["binary"] = "Marxan C++ Binary"
    return ui.card(
        help_card_header("Solver Configuration"),
        ui.p(
            "Select a solver algorithm and configure its parameters. Marxan supports "
            "multiple solution approaches: exact optimisation (MIP), simulated annealing "
            "(SA), greedy heuristics, iterative improvement, and pipeline combinations. "
            "The MIP solver guarantees optimality; SA and heuristics are faster for "
            "large problems but may not find the global optimum.",
            class_="text-muted small mb-3",
        ),
        ui.layout_sidebar(
            ui.sidebar(
                ui.tooltip(
                    ui.input_radio_buttons(
                        "solver_type", "Solver",
                        choices=solver_choices, selected="mip",
                    ),
                    "Choose the optimisation algorithm. MIP gives exact solutions; "
                    "SA explores the solution space stochastically; Greedy/II are "
                    "fast heuristics; Pipeline chains multiple methods.",
                ),
                ui.hr(),
                ui.h5("Parameters"),
                ui.tooltip(
                    ui.input_numeric(
                        "blm", "Boundary Length Modifier (BLM)",
                        value=1.0, min=0, step=0.1,
                    ),
                    "Controls the tradeoff between cost and spatial compactness. "
                    "Higher BLM = more compact reserves. Use the BLM Calibration tab "
                    "to find a good value.",
                ),
                ui.tooltip(
                    ui.input_numeric(
                        "num_solutions", "Number of solutions",
                        value=10, min=1, max=1000, step=1,
                    ),
                    "How many independent solver runs to perform. Multiple solutions "
                    "reveal selection frequency and solution variability. "
                    "Recommended: 10\u2013100 for SA, 1 for MIP (deterministic).",
                ),
                ui.tooltip(
                    ui.input_numeric(
                        "seed", "Random seed",
                        value=None, min=-1,
                    ),
                    "Set a specific seed for reproducible results. "
                    "Leave empty or set to -1 for random runs.",
                ),
                ui.hr(),
                ui.panel_conditional(
                    "input.solver_type === 'binary'",
                    ui.tooltip(
                        ui.input_numeric(
                            "num_iterations", "SA Iterations",
                            value=1000000, min=1000, step=100000,
                        ),
                        "Number of iterations per SA run (NUMITNS). More iterations "
                        "allow better exploration of the solution space. "
                        "Typical: 1,000,000 for moderate-sized problems.",
                    ),
                    ui.tooltip(
                        ui.input_numeric(
                            "num_temp", "Temperature steps",
                            value=10000, min=100, step=1000,
                        ),
                        "Number of temperature decreases during annealing (NUMTEMP). "
                        "Controls cooling schedule granularity. Typical: 10,000.",
                    ),
                ),
                ui.panel_conditional(
                    "input.solver_type === 'sa'",
                    ui.tooltip(
                        ui.input_numeric(
                            "sa_iterations", "SA Iterations",
                            value=1000000, min=1000, step=100000,
                        ),
                        "Number of iterations per SA run (NUMITNS). More iterations "
                        "allow better exploration. Typical: 1,000,000.",
                    ),
                    ui.tooltip(
                        ui.input_numeric(
                            "sa_temp_steps", "Temperature steps",
                            value=10000, min=100, step=1000,
                        ),
                        "Number of temperature decreases during annealing (NUMTEMP). "
                        "More steps = smoother cooling. Typical: 10,000.",
                    ),
                ),
                ui.panel_conditional(
                    "input.solver_type === 'zone_sa'",
                    ui.p("Uses Zone SA solver for multi-zone problems. "
                         "Requires zone data to be loaded in the Zones tab."),
                    ui.tooltip(
                        ui.input_numeric(
                            "zone_sa_iterations", "SA Iterations",
                            value=1000000, min=1000, step=100000,
                        ),
                        "Number of iterations per zone SA run. Each iteration "
                        "proposes a zone reassignment for one planning unit.",
                    ),
                    ui.tooltip(
                        ui.input_numeric(
                            "zone_sa_temp_steps", "Temperature steps",
                            value=10000, min=100, step=1000,
                        ),
                        "Number of temperature decreases for the zone SA cooling schedule.",
                    ),
                ),
                ui.panel_conditional(
                    "input.solver_type === 'mip'",
                    ui.tooltip(
                        ui.input_numeric(
                            "mip_time_limit", "Time Limit (seconds)",
                            value=300, min=10, step=30,
                        ),
                        "Maximum wall-clock time for the MIP solver. If the solver "
                        "hasn't proven optimality by this time, it returns the best "
                        "solution found so far.",
                    ),
                    ui.tooltip(
                        ui.input_numeric(
                            "mip_gap", "Optimality Gap",
                            value=0.0, min=0.0, max=1.0, step=0.01,
                        ),
                        "Acceptable gap between the best solution and the proven "
                        "lower bound. 0.0 = exact optimum; 0.01 = within 1% of optimal. "
                        "Setting a small gap can dramatically speed up large problems.",
                    ),
                    ui.tooltip(
                        ui.input_checkbox(
                            "mip_verbose", "Verbose output", value=False,
                        ),
                        "Print detailed solver progress to the console log.",
                    ),
                ),
                ui.panel_conditional(
                    "input.solver_type === 'greedy'",
                    ui.tooltip(
                        ui.input_select(
                            "heurtype", "Scoring Mode (HEURTYPE)",
                            choices={
                                "0": "0 - Richness",
                                "1": "1 - Greedy (cheapest)",
                                "2": "2 - Max Rarity (default)",
                                "3": "3 - Best Rarity/Cost",
                                "4": "4 - Average Rarity",
                                "5": "5 - Sum Rarity",
                                "6": "6 - Product Irreplaceability",
                                "7": "7 - Summation Irreplaceability",
                            },
                            selected="2",
                        ),
                        "Marxan HEURTYPE: how planning units are scored during greedy "
                        "selection. 0=richness (most features), 1=cheapest, 2=max rarity "
                        "(most irreplaceable first), 3=best rarity/cost ratio, "
                        "4\u20137=various irreplaceability measures.",
                    ),
                ),
                ui.panel_conditional(
                    "input.solver_type === 'iterative_improvement'",
                    ui.tooltip(
                        ui.input_select(
                            "itimptype", "Improvement Mode (ITIMPTYPE)",
                            choices={
                                "0": "0 - No improvement",
                                "1": "1 - Removal pass",
                                "2": "2 - Two-step (remove + add)",
                                "3": "3 - Swap",
                            },
                            selected="0",
                        ),
                        "Marxan ITIMPTYPE: local search strategy. 0=none, "
                        "1=try removing each selected PU, 2=remove then add pass, "
                        "3=try swapping selected/unselected PU pairs.",
                    ),
                ),
                ui.panel_conditional(
                    "input.solver_type === 'pipeline'",
                    ui.tooltip(
                        ui.input_select(
                            "runmode", "Pipeline Mode (RUNMODE)",
                            choices={
                                "0": "0 - SA only (default)",
                                "1": "1 - Heuristic only",
                                "2": "2 - SA + iterative improvement",
                                "3": "3 - Heuristic + iterative improvement",
                                "4": "4 - Heuristic + SA (pick best)",
                                "5": "5 - Heur + SA + improvement",
                                "6": "6 - Iterative improvement only",
                            },
                            selected="0",
                        ),
                        "Marxan RUNMODE: which combination of algorithms to run. "
                        "Mode 0 (SA only) is the classic Marxan default. Mode 5 "
                        "chains all three methods for highest solution quality.",
                    ),
                ),
                width=350,
            ),
            ui.output_text_verbatim("solver_info"),
        ),
    )

@module.server
def solver_picker_server(input, output, session, solver_config: reactive.Value):
    help_server_setup(input, "solver_picker")

    @reactive.effect
    @reactive.event(
        input.solver_type, input.blm, input.num_solutions,
        input.seed, input.sa_iterations, input.sa_temp_steps,
        input.zone_sa_iterations, input.zone_sa_temp_steps,
        input.mip_time_limit, input.mip_gap, input.mip_verbose,
        input.heurtype, input.itimptype, input.runmode,
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
        if input.solver_type() == "greedy":
            config["heurtype"] = int(input.heurtype() or 2)
        if input.solver_type() == "iterative_improvement":
            config["itimptype"] = int(input.itimptype() or 0)
        if input.solver_type() == "pipeline":
            config["runmode"] = int(input.runmode() or 0)
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
        elif st == "greedy":
            return (
                "Greedy Heuristic\n----------------\n"
                "Selects planning units one-by-one based on a scoring\n"
                "strategy (HEURTYPE 0-7). Fast baseline for comparison."
            )
        elif st == "iterative_improvement":
            return (
                "Iterative Improvement\n---------------------\n"
                "Refines an existing solution by trying removals,\n"
                "additions, or swaps (ITIMPTYPE 0-3)."
            )
        elif st == "pipeline":
            return (
                "Run Mode Pipeline\n-----------------\n"
                "Chains heuristic, SA, and iterative improvement\n"
                "in sequences matching Marxan RUNMODE 0-6."
            )
        return ""
