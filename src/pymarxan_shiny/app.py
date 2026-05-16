"""
PyMarxan — Systematic Conservation Planning

Shiny application entry point. Composes all modules into a tabbed interface
and wires shared reactive state between them.
"""

import sys
from pathlib import Path

# Ensure the pymarxan core library is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from shiny import App, Inputs, Outputs, Session, reactive, ui

# Reuse pymarxan core's version (read from installed metadata, with a
# PEP 440 local-version fallback for source-checkout dev runs).
from pymarxan import __version__

# ---------------------------------------------------------------------------
# Static assets
# ---------------------------------------------------------------------------
_APP_DIR = Path(__file__).resolve().parent
_CSS_PATH = _APP_DIR / "www" / "ocean_theme.css"

# ---------------------------------------------------------------------------
# Module UI imports
# ---------------------------------------------------------------------------
from modules.calibration.blm_explorer import blm_explorer_server, blm_explorer_ui
from modules.calibration.sensitivity_ui import sensitivity_server, sensitivity_ui
from modules.calibration.spf_explorer import spf_explorer_server, spf_explorer_ui
from modules.calibration.sweep_explorer import sweep_explorer_server, sweep_explorer_ui
from modules.connectivity.connectivity_config import (
    connectivity_config_server,
    connectivity_config_ui,
)
from modules.connectivity.matrix_input import matrix_input_server, matrix_input_ui
from modules.connectivity.metrics_viz import metrics_viz_server, metrics_viz_ui
from modules.data.feature_table import feature_table_server, feature_table_ui
from modules.data_input.upload import upload_server, upload_ui
from modules.mapping.comparison_map import comparison_map_server, comparison_map_ui
from modules.mapping.frequency_map import frequency_map_server, frequency_map_ui
from modules.mapping.network_view import network_view_server, network_view_ui
from modules.mapping.solution_map import solution_map_server, solution_map_ui
from modules.mapping.spatial_grid import spatial_grid_server, spatial_grid_ui
from modules.probability.probability_config import probability_config_server, probability_config_ui
from modules.results.convergence import convergence_server, convergence_ui
from modules.results.export import export_server, export_ui
from modules.results.scenario_compare import scenario_compare_server, scenario_compare_ui
from modules.results.summary_table import summary_table_server, summary_table_ui
from modules.results.target_met import target_met_server, target_met_ui
from modules.run_control.run_panel import run_panel_server, run_panel_ui
from modules.solver_config.objective_selector import (
    objective_selector_server,
    objective_selector_ui,
)
from modules.solver_config.solver_picker import solver_picker_server, solver_picker_ui
from modules.spatial.cost_upload import cost_upload_server, cost_upload_ui
from modules.spatial.gadm_picker import gadm_picker_server, gadm_picker_ui
from modules.spatial.grid_builder import grid_builder_server, grid_builder_ui
from modules.spatial.import_wizard import import_wizard_server, import_wizard_ui
from modules.spatial.wdpa_overlay import wdpa_overlay_server, wdpa_overlay_ui
from modules.spatial_export.spatial_export import spatial_export_server, spatial_export_ui
from modules.zones.zone_config import zone_config_server, zone_config_ui

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
_head_content = ui.head_content(
    ui.tags.link(
        rel="preconnect",
        href="https://fonts.googleapis.com",
    ),
    ui.tags.link(
        rel="preconnect",
        href="https://fonts.gstatic.com",
        crossorigin="",
    ),
    ui.tags.link(
        rel="stylesheet",
        href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
    ),
    ui.tags.meta(name="theme-color", content="#0b2545"),
)

app_ui = ui.page_navbar(
    ui.nav_panel(
        "Data Input",
        ui.navset_tab(
            ui.nav_panel("Upload Project", upload_ui("upload")),
            ui.nav_panel("Import GIS", import_wizard_ui("import_wizard")),
            ui.nav_panel("Generate Grid", grid_builder_ui("grid_builder")),
        ),
    ),
    ui.nav_panel(
        "Spatial",
        ui.navset_tab(
            ui.nav_panel("GADM Boundaries", gadm_picker_ui("gadm_picker")),
            ui.nav_panel("Protected Areas", wdpa_overlay_ui("wdpa_overlay")),
            ui.nav_panel("Cost Surface", cost_upload_ui("cost_upload")),
        ),
    ),
    ui.nav_panel("Features", feature_table_ui("feature_table")),
    ui.nav_panel(
        "Probability",
        probability_config_ui("probability_config"),
    ),
    ui.nav_panel(
        "Connectivity",
        ui.navset_tab(
            ui.nav_panel("Matrix Input", matrix_input_ui("matrix_input")),
            ui.nav_panel("Configuration", connectivity_config_ui("connectivity_config")),
            ui.nav_panel("Metrics", metrics_viz_ui("metrics_viz")),
        ),
    ),
    ui.nav_panel(
        "Configure",
        ui.navset_tab(
            ui.nav_panel("Solver", solver_picker_ui("solver_picker")),
            ui.nav_panel("Objective", objective_selector_ui("objective_selector")),
            ui.nav_panel("Zones", zone_config_ui("zone_config")),
        ),
    ),
    ui.nav_panel(
        "Calibrate",
        ui.navset_tab(
            ui.nav_panel("BLM", blm_explorer_ui("blm_explorer")),
            ui.nav_panel("Sensitivity", sensitivity_ui("sensitivity")),
            ui.nav_panel("SPF", spf_explorer_ui("spf_explorer")),
            ui.nav_panel("Sweep", sweep_explorer_ui("sweep_explorer")),
        ),
    ),
    ui.nav_panel("Run", run_panel_ui("run_panel")),
    ui.nav_panel(
        "Maps",
        ui.navset_tab(
            ui.nav_panel("Planning Units", spatial_grid_ui("spatial_grid")),
            ui.nav_panel("Solution", solution_map_ui("solution_map")),
            ui.nav_panel("Frequency", frequency_map_ui("frequency_map")),
            ui.nav_panel("Comparison", comparison_map_ui("comparison_map")),
            ui.nav_panel("Network", network_view_ui("network_view")),
        ),
    ),
    ui.nav_panel(
        "Results",
        ui.navset_tab(
            ui.nav_panel("Summary", summary_table_ui("summary_table")),
            ui.nav_panel("Targets", target_met_ui("target_met")),
            ui.nav_panel("Convergence", convergence_ui("convergence")),
            ui.nav_panel("Scenarios", scenario_compare_ui("scenario_compare")),
            ui.nav_panel("Export", export_ui("export")),
            ui.nav_panel("Spatial Export", spatial_export_ui("spatial_export")),
        ),
    ),
    ui.nav_spacer(),
    ui.nav_control(
        ui.input_action_button(
            "about_btn",
            ui.tags.span("ⓘ About"),
            class_="btn btn-sm btn-outline-light py-1 px-3",
        ),
    ),
    title=ui.tags.span("🌊 PyMarxan", style="font-weight: 700;"),
    header=ui.TagList(_head_content, ui.include_css(_CSS_PATH)),
    window_title="PyMarxan — Conservation Planning",
)

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
def server(input: Inputs, output: Outputs, session: Session):
    # --- About Modal ---
    @reactive.effect
    @reactive.event(input.about_btn)
    def _show_about():
        m = ui.modal(
            ui.div(
                ui.tags.div(
                    ui.tags.span("🌊", style="font-size: 2.5rem;"),
                    style="text-align: center; margin-bottom: 0.5rem;",
                ),
                ui.tags.h4(
                    "PyMarxan",
                    style="text-align: center; margin: 0 0 0.25rem 0; font-weight: 700;",
                ),
                ui.tags.p(
                    f"Version {__version__}",
                    style=(
                        "text-align: center; color: var(--ocean-teal); "
                        "font-weight: 600; margin-bottom: 1rem;"
                    ),
                ),
                ui.tags.hr(style="border-color: rgba(15, 163, 177, 0.15);"),
                ui.tags.p(
                    "Systematic conservation planning tool powered by the Marxan "
                    "optimization framework. Design efficient reserve networks that "
                    "meet biodiversity targets while minimizing costs.",
                    style="line-height: 1.6; color: var(--ocean-mid);",
                ),
                ui.tags.h6("Key Features", style="margin-top: 1rem; font-weight: 600;"),
                ui.tags.ul(
                    ui.tags.li("Import & manage spatial planning units"),
                    ui.tags.li("Configure species & habitat feature targets"),
                    ui.tags.li("Connectivity matrix analysis & visualization"),
                    ui.tags.li("BLM, SPF & sensitivity calibration tools"),
                    ui.tags.li("Simulated annealing & exact solvers"),
                    ui.tags.li("Interactive maps with solution comparison"),
                    ui.tags.li("Export results in standard Marxan formats"),
                    style="color: var(--ocean-mid); line-height: 1.8; padding-left: 1.2rem;",
                ),
                ui.tags.hr(style="border-color: rgba(15, 163, 177, 0.15);"),
                ui.tags.p(
                    ui.tags.small(
                        "Built with Shiny for Python · Ocean Palette Theme · "
                        "© 2026 PyMarxan Project",
                        style="color: var(--ocean-mid); opacity: 0.7;",
                    ),
                    style="text-align: center; margin-bottom: 0;",
                ),
            ),
            title=ui.tags.span(
                "🌊 About PyMarxan",
                style="font-weight: 600;",
            ),
            easy_close=True,
            size="m",
        )
        ui.modal_show(m)

    # Shared reactive state
    problem = reactive.Value(None)
    current_solution = reactive.Value(None)
    all_solutions = reactive.Value(None)
    solver_config: reactive.Value[dict] = reactive.Value({})
    connectivity_matrix = reactive.Value(None)
    connectivity_pu_ids = reactive.Value(None)
    gadm_boundary = reactive.Value(None)
    zone_problem = reactive.Value(None)

    # Solver factory — returns the active solver based on current config
    @reactive.calc
    def solver():
        from pymarxan.solvers import get_solver
        cfg = solver_config()
        solver_type = cfg.get("solver_type", "sa")
        return get_solver(solver_type)

    # --- Data Input ---
    upload_server("upload", problem=problem)
    import_wizard_server("import_wizard", problem=problem)
    grid_builder_server("grid_builder", problem=problem, gadm_boundary=gadm_boundary)

    # --- Spatial ---
    gadm_picker_server("gadm_picker", boundary=gadm_boundary)
    wdpa_overlay_server("wdpa_overlay", problem=problem)
    cost_upload_server("cost_upload", problem=problem)

    # --- Features ---
    feature_table_server("feature_table", problem=problem)

    # --- Connectivity ---
    matrix_input_server(
        "matrix_input",
        problem=problem,
        connectivity_matrix=connectivity_matrix,
        connectivity_pu_ids=connectivity_pu_ids,
    )
    connectivity_config_server(
        "connectivity_config",
        problem=problem,
        connectivity_matrix=connectivity_matrix,
        connectivity_pu_ids=connectivity_pu_ids,
    )
    metrics_viz_server(
        "metrics_viz",
        connectivity_matrix=connectivity_matrix,
        pu_ids=connectivity_pu_ids,
    )

    # --- Probability ---
    probability_config_server("probability_config", problem=problem)

    # --- Configure ---
    solver_picker_server("solver_picker", solver_config=solver_config)
    objective_selector_server("objective_selector", solver_config=solver_config)
    zone_config_server("zone_config", zone_problem=zone_problem)

    # --- Calibrate ---
    blm_explorer_server("blm_explorer", problem=problem, solver=solver)
    sensitivity_server("sensitivity", problem=problem, solver=solver)
    spf_explorer_server("spf_explorer", problem=problem, solver=solver)
    sweep_explorer_server("sweep_explorer", problem=problem, solver=solver)

    # --- Run ---
    run_panel_server(
        "run_panel",
        problem=problem,
        solver=solver,
        solver_config=solver_config,
        current_solution=current_solution,
        all_solutions=all_solutions,
    )

    # --- Maps ---
    spatial_grid_server("spatial_grid", problem=problem)
    solution_map_server("solution_map", problem=problem, solution=current_solution)
    frequency_map_server("frequency_map", problem=problem, all_solutions=all_solutions)
    comparison_map_server("comparison_map", problem=problem, all_solutions=all_solutions)
    network_view_server(
        "network_view",
        problem=problem,
        connectivity_matrix=connectivity_matrix,
    )

    # --- Results ---
    summary_table_server("summary_table", problem=problem, solution=current_solution)
    target_met_server("target_met", problem=problem, solution=current_solution)
    convergence_server("convergence", all_solutions=all_solutions)
    scenario_compare_server(
        "scenario_compare",
        solution=current_solution,
        solver_config=solver_config,
    )
    export_server("export", problem=problem, solution=current_solution)
    spatial_export_server(
        "spatial_export",
        problem=problem,
        solution=current_solution,
        all_solutions=all_solutions,
    )


app = App(app_ui, server)
