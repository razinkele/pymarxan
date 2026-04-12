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
from pymarxan_shiny.modules.calibration.sensitivity_ui import (
    sensitivity_server,
    sensitivity_ui,
)
from pymarxan_shiny.modules.calibration.spf_explorer import (
    spf_explorer_server,
    spf_explorer_ui,
)
from pymarxan_shiny.modules.calibration.sweep_explorer import (
    sweep_explorer_server,
    sweep_explorer_ui,
)
from pymarxan_shiny.modules.connectivity.matrix_input import (
    matrix_input_server,
    matrix_input_ui,
)
from pymarxan_shiny.modules.connectivity.metrics_viz import (
    metrics_viz_server,
    metrics_viz_ui,
)
from pymarxan_shiny.modules.data.feature_table import (
    feature_table_server,
    feature_table_ui,
)
from pymarxan_shiny.modules.data_input.upload import upload_server, upload_ui
from pymarxan_shiny.modules.mapping.comparison_map import (
    comparison_map_server,
    comparison_map_ui,
)
from pymarxan_shiny.modules.mapping.frequency_map import (
    frequency_map_server,
    frequency_map_ui,
)
from pymarxan_shiny.modules.mapping.network_view import (
    network_view_server,
    network_view_ui,
)
from pymarxan_shiny.modules.mapping.solution_map import solution_map_server, solution_map_ui
from pymarxan_shiny.modules.mapping.spatial_grid import (
    spatial_grid_server,
    spatial_grid_ui,
)
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
from pymarxan_shiny.modules.solver_config.objective_selector import (
    objective_selector_server,
    objective_selector_ui,
)
from pymarxan_shiny.modules.probability.probability_config import (
    probability_config_server,
    probability_config_ui,
)
from pymarxan_shiny.modules.connectivity.connectivity_config import (
    connectivity_config_server,
    connectivity_config_ui,
)
from pymarxan_shiny.modules.spatial_export.spatial_export import (
    spatial_export_server,
    spatial_export_ui,
)
from pymarxan_shiny.modules.spatial.cost_upload import cost_upload_server, cost_upload_ui
from pymarxan_shiny.modules.spatial.gadm_picker import gadm_picker_server, gadm_picker_ui
from pymarxan_shiny.modules.spatial.grid_builder import grid_builder_server, grid_builder_ui
from pymarxan_shiny.modules.spatial.import_wizard import import_wizard_server, import_wizard_ui
from pymarxan_shiny.modules.spatial.wdpa_overlay import wdpa_overlay_server, wdpa_overlay_ui
from pymarxan_shiny.modules.zones.zone_config import zone_config_server, zone_config_ui

app_ui = ui.page_navbar(
    ui.nav_panel(
        "Data",
        ui.layout_columns(
            upload_ui("upload"),
            import_wizard_ui("import_wiz"),
            grid_builder_ui("grid_gen"),
            gadm_picker_ui("gadm"),
            wdpa_overlay_ui("wdpa"),
            cost_upload_ui("cost"),
            feature_table_ui("features"),
            spatial_grid_ui("pu_grid"),
            col_widths=[12, 12, 12, 12, 12, 12, 12, 12],
        ),
    ),
    ui.nav_panel(
        "Configure",
        ui.layout_columns(
            solver_picker_ui("solver"),
            objective_selector_ui("objective"),
            col_widths=[12, 12],
        ),
    ),
    ui.nav_panel(
        "Probability",
        ui.layout_columns(probability_config_ui("probability"), col_widths=12),
    ),
    ui.nav_panel(
        "Calibrate",
        ui.layout_columns(
            blm_explorer_ui("blm_cal"),
            spf_explorer_ui("spf_cal"),
            sensitivity_ui("sensitivity"),
            col_widths=[6, 6, 12],
        ),
    ),
    ui.nav_panel(
        "Sweep",
        ui.layout_columns(sweep_explorer_ui("sweep"), col_widths=12),
    ),
    ui.nav_panel(
        "Connectivity",
        ui.layout_columns(
            matrix_input_ui("matrix_upload"),
            connectivity_config_ui("conn_config"),
            metrics_viz_ui("connectivity"),
            network_view_ui("network"),
            col_widths=[12, 12, 12, 12],
        ),
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
        frequency_map_ui("frequency"),
        comparison_map_ui("comparison"),
        target_met_ui("targets"),
        convergence_ui("convergence"),
        scenario_compare_ui("scenarios"),
        export_ui("export"),
        spatial_export_ui("spatial_export"),
        col_widths=[6, 6, 6, 6, 12, 12, 12, 12],
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
    gadm_boundary: reactive.Value = reactive.value(None)

    upload_server("upload", problem=problem)
    solver_picker_server("solver", solver_config=solver_config)
    objective_selector_server("objective", solver_config=solver_config)
    probability_config_server("probability", problem=problem)
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
    matrix_input_server(
        "matrix_upload",
        problem=problem,
        connectivity_matrix=connectivity_matrix,
        connectivity_pu_ids=connectivity_pu_ids,
    )
    connectivity_config_server(
        "conn_config",
        problem=problem,
        connectivity_matrix=connectivity_matrix,
        connectivity_pu_ids=connectivity_pu_ids,
    )
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
    spatial_export_server(
        "spatial_export",
        problem=problem,
        solution=current_solution,
        all_solutions=all_solutions,
    )

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

    # Spatial modules
    grid_builder_server("grid_gen", problem=problem, gadm_boundary=gadm_boundary)
    gadm_picker_server("gadm", boundary=gadm_boundary)
    wdpa_overlay_server("wdpa", problem=problem)
    import_wizard_server("import_wiz", problem=problem)
    cost_upload_server("cost", problem=problem)

    # Phase 9 modules
    feature_table_server("features", problem=problem)
    spatial_grid_server("pu_grid", problem=problem)
    frequency_map_server(
        "frequency", problem=problem, all_solutions=all_solutions,
    )
    comparison_map_server(
        "comparison", problem=problem, all_solutions=all_solutions,
    )
    sensitivity_server(
        "sensitivity", problem=problem, solver=active_solver,
    )
    network_view_server(
        "network",
        problem=problem,
        connectivity_matrix=connectivity_matrix,
    )

app = App(app_ui, server)
