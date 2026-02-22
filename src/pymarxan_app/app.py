"""pymarxan: Assembled Shiny application for Marxan conservation planning.
Run with: shiny run src/pymarxan_app/app.py
"""
from __future__ import annotations

from shiny import App, Inputs, Outputs, Session, reactive, ui

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution
from pymarxan.solvers.marxan_binary import MarxanBinarySolver
from pymarxan.solvers.mip_solver import MIPSolver
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
from pymarxan.zones.solver import ZoneSASolver
from pymarxan_shiny.modules.calibration.blm_explorer import (
    blm_explorer_server,
    blm_explorer_ui,
)
from pymarxan_shiny.modules.calibration.spf_explorer import (
    spf_explorer_server,
    spf_explorer_ui,
)
from pymarxan_shiny.modules.calibration.sweep_explorer import (
    sweep_explorer_server,
    sweep_explorer_ui,
)
from pymarxan_shiny.modules.connectivity.metrics_viz import (
    metrics_viz_server,
    metrics_viz_ui,
)
from pymarxan_shiny.modules.data_input.upload import upload_server, upload_ui
from pymarxan_shiny.modules.mapping.solution_map import solution_map_server, solution_map_ui
from pymarxan_shiny.modules.results.convergence import convergence_server, convergence_ui
from pymarxan_shiny.modules.results.export import export_server, export_ui
from pymarxan_shiny.modules.results.scenario_compare import (
    scenario_compare_server,
    scenario_compare_ui,
)
from pymarxan_shiny.modules.results.summary_table import summary_table_server, summary_table_ui
from pymarxan_shiny.modules.results.target_met import (
    target_met_server,
    target_met_ui,
)
from pymarxan_shiny.modules.run_control.run_panel import run_panel_server, run_panel_ui
from pymarxan_shiny.modules.solver_config.solver_picker import (
    solver_picker_server,
    solver_picker_ui,
)
from pymarxan_shiny.modules.zones.zone_config import zone_config_server, zone_config_ui

app_ui = ui.page_navbar(
    ui.nav_panel(
        "Data",
        ui.layout_columns(upload_ui("upload"), col_widths=12),
    ),
    ui.nav_panel(
        "Configure",
        ui.layout_columns(solver_picker_ui("solver"), col_widths=12),
    ),
    ui.nav_panel(
        "Calibrate",
        ui.layout_columns(
            blm_explorer_ui("blm_cal"),
            spf_explorer_ui("spf_cal"),
            col_widths=[6, 6],
        ),
    ),
    ui.nav_panel(
        "Sweep",
        ui.layout_columns(sweep_explorer_ui("sweep"), col_widths=12),
    ),
    ui.nav_panel(
        "Connectivity",
        ui.layout_columns(metrics_viz_ui("connectivity"), col_widths=12),
    ),
    ui.nav_panel(
        "Zones",
        ui.layout_columns(zone_config_ui("zone_config"), col_widths=12),
    ),
    ui.nav_panel(
        "Run",
        run_panel_ui("run"),
    ),
    ui.nav_panel("Results", ui.layout_columns(
        solution_map_ui("solution_map"),
        summary_table_ui("summary"),
        target_met_ui("targets"),
        convergence_ui("convergence"),
        scenario_compare_ui("scenarios"),
        export_ui("export"),
        col_widths=[6, 6, 12, 12, 12, 12],
    )),
    title="pymarxan",
    id="navbar",
)

def server(input: Inputs, output: Outputs, session: Session):
    problem: reactive.Value[ConservationProblem | None] = reactive.value(None)
    solver_config: reactive.Value[dict] = reactive.value({
        "solver_type": "mip", "blm": 1.0, "num_solutions": 10, "seed": None,
    })
    current_solution: reactive.Value[Solution | None] = reactive.value(None)
    all_solutions: reactive.Value[list[Solution] | None] = reactive.value(None)
    zone_problem: reactive.Value = reactive.value(None)
    connectivity_matrix: reactive.Value = reactive.value(None)
    connectivity_pu_ids: reactive.Value = reactive.value(None)

    upload_server("upload", problem=problem)
    solver_picker_server("solver", solver_config=solver_config)
    solution_map_server("solution_map", problem=problem, solution=current_solution)
    summary_table_server("summary", problem=problem, solution=current_solution)
    zone_config_server("zone_config", zone_problem=zone_problem)

    @reactive.calc
    def active_solver():
        config_dict = solver_config()
        st = config_dict.get("solver_type", "mip")
        if st == "mip":
            return MIPSolver()
        elif st == "sa":
            return SimulatedAnnealingSolver()
        elif st == "binary":
            return MarxanBinarySolver()
        elif st == "zone_sa":
            return ZoneSASolver()
        elif st == "greedy":
            from pymarxan.solvers.heuristic import HeuristicSolver
            return HeuristicSolver()
        elif st == "iterative_improvement":
            from pymarxan.solvers.iterative_improvement import IterativeImprovementSolver
            return IterativeImprovementSolver()
        elif st == "pipeline":
            from pymarxan.solvers.run_mode import RunModePipeline
            return RunModePipeline()
        return MIPSolver()

    blm_explorer_server(
        "blm_cal", problem=problem, solver=active_solver,
    )
    spf_explorer_server("spf_cal", problem=problem, solver=active_solver)
    sweep_explorer_server("sweep", problem=problem, solver=active_solver)
    metrics_viz_server(
        "connectivity",
        connectivity_matrix=connectivity_matrix,
        pu_ids=connectivity_pu_ids,
    )
    target_met_server("targets", problem=problem, solution=current_solution)
    scenario_compare_server(
        "scenarios", solution=current_solution, solver_config=solver_config,
    )
    export_server("export", problem=problem, solution=current_solution)

    # Run panel with progress monitoring
    run_panel_server(
        "run",
        problem=problem,
        solver=active_solver,
        solver_config=solver_config,
        current_solution=current_solution,
        all_solutions=all_solutions,
    )

    # Convergence plot
    convergence_server("convergence", all_solutions=all_solutions)

app = App(app_ui, server)
