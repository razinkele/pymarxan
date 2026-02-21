"""pymarxan: Assembled Shiny application for Marxan conservation planning.
Run with: shiny run src/pymarxan_app/app.py
"""
from __future__ import annotations

from shiny import App, Inputs, Outputs, Session, reactive, render, ui

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, SolverConfig
from pymarxan.solvers.marxan_binary import MarxanBinarySolver
from pymarxan.solvers.mip_solver import MIPSolver
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
from pymarxan.zones.solver import ZoneSASolver
from pymarxan_shiny.modules.calibration.blm_explorer import (
    blm_explorer_server,
    blm_explorer_ui,
)
from pymarxan_shiny.modules.calibration.sweep_explorer import (
    sweep_explorer_server,
    sweep_explorer_ui,
)
from pymarxan_shiny.modules.results.scenario_compare import (
    scenario_compare_server,
    scenario_compare_ui,
)
from pymarxan_shiny.modules.data_input.upload import upload_server, upload_ui
from pymarxan_shiny.modules.mapping.solution_map import solution_map_server, solution_map_ui
from pymarxan_shiny.modules.results.export import export_server, export_ui
from pymarxan_shiny.modules.results.summary_table import summary_table_server, summary_table_ui
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
        ui.layout_columns(blm_explorer_ui("blm_cal"), col_widths=12),
    ),
    ui.nav_panel(
        "Sweep",
        ui.layout_columns(sweep_explorer_ui("sweep"), col_widths=12),
    ),
    ui.nav_panel(
        "Zones",
        ui.layout_columns(zone_config_ui("zone_config"), col_widths=12),
    ),
    ui.nav_panel("Run", ui.card(
        ui.card_header("Run Solver"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_action_button(
                    "run_solver", "Run Solver",
                    class_="btn-primary btn-lg w-100",
                ),
                ui.hr(),
                ui.output_text_verbatim("run_log"),
                width=300,
            ),
            ui.output_text_verbatim("run_status"),
        ),
    )),
    ui.nav_panel("Results", ui.layout_columns(
        solution_map_ui("solution_map"),
        summary_table_ui("summary"),
        scenario_compare_ui("scenarios"),
        export_ui("export"),
        col_widths=[6, 6, 12, 12],
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
    zone_problem: reactive.Value = reactive.value(None)

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
        return MIPSolver()

    blm_explorer_server(
        "blm_cal", problem=problem, solver=active_solver,
    )
    sweep_explorer_server("sweep", problem=problem, solver=active_solver)
    scenario_compare_server(
        "scenarios", solution=current_solution, solver_config=solver_config,
    )
    export_server("export", problem=problem, solution=current_solution)

    @reactive.effect
    @reactive.event(input.run_solver)
    def _run_solver():
        p = problem()
        if p is None:
            ui.notification_show("Load a project first!", type="error")
            return
        config_dict = solver_config()
        solver_type = config_dict.get("solver_type", "mip")
        p.parameters["BLM"] = config_dict.get("blm", 1.0)
        config = SolverConfig(
            num_solutions=config_dict.get("num_solutions", 10),
            seed=config_dict.get("seed"),
            verbose=False,
        )
        if solver_type == "binary":
            p.parameters["NUMITNS"] = config_dict.get("num_iterations", 1000000)
            p.parameters["NUMTEMP"] = config_dict.get("num_temp", 10000)
        elif solver_type == "sa":
            p.parameters["NUMITNS"] = config_dict.get("num_iterations", 1000000)
            p.parameters["NUMTEMP"] = config_dict.get("num_temp", 10000)
        elif solver_type == "zone_sa":
            p.parameters["NUMITNS"] = config_dict.get("num_iterations", 1000000)
            p.parameters["NUMTEMP"] = config_dict.get("num_temp", 10000)
        if solver_type == "mip":
            solver = MIPSolver()
        elif solver_type == "binary":
            solver = MarxanBinarySolver()
        elif solver_type == "sa":
            solver = SimulatedAnnealingSolver()
        elif solver_type == "zone_sa":
            solver = ZoneSASolver()
        else:
            ui.notification_show(f"Unknown solver type: {solver_type}", type="error")
            return
        if not solver.available():
            ui.notification_show(f"Solver '{solver.name()}' is not available.", type="error")
            return
        ui.notification_show(f"Running {solver.name()}...", type="message")
        try:
            solutions = solver.solve(p, config)
            if solutions:
                best = min(solutions, key=lambda s: s.objective)
                current_solution.set(best)
                met = sum(best.targets_met.values())
                total = len(best.targets_met)
                ui.notification_show(
                    f"Done! Cost: {best.cost:.2f}, "
                    f"Targets met: {met}/{total}",
                    type="message",
                )
            else:
                ui.notification_show("Solver returned no solutions.", type="warning")
        except Exception as e:
            ui.notification_show(f"Solver error: {e}", type="error")

    @render.text
    def run_status():
        p = problem()
        s = current_solution()
        if p is None:
            return "Step 1: Go to 'Data' tab and load a Marxan project."
        if s is None:
            return (f"Project loaded: {p.n_planning_units} PUs, {p.n_features} features.\n"
                    f"Step 2: Configure solver in 'Configure' tab, then click 'Run Solver'.")
        all_met = "Yes" if s.all_targets_met else "No"
        return (
            f"Solution available!\n  Selected: {s.n_selected} PUs\n"
            f"  Cost: {s.cost:.2f}\n  Boundary: {s.boundary:.2f}\n"
            f"  Objective: {s.objective:.2f}\n"
            f"  All targets met: {all_met}\n\n"
            f"Go to 'Results' tab to explore the solution."
        )

    @render.text
    def run_log():
        s = current_solution()
        if s is None:
            return "No solver has been run yet."
        lines = ["Solver metadata:"]
        for k, v in s.metadata.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)

app = App(app_ui, server)
