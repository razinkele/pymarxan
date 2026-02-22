"""Run panel Shiny module — solver execution with progress monitoring."""
from __future__ import annotations

import threading

from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from pymarxan.solvers.base import SolverConfig
from pymarxan_shiny.modules.run_control.progress import SolverProgress


@module.ui
def run_panel_ui():
    return ui.card(
        ui.card_header("Run Solver"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_action_button(
                    "run_solver",
                    "Run Solver",
                    class_="btn-primary btn-lg w-100",
                ),
                ui.hr(),
                ui.output_ui("progress_bar"),
                ui.hr(),
                ui.output_text_verbatim("run_log"),
                width=300,
            ),
            ui.output_text_verbatim("run_status"),
        ),
    )


@module.server
def run_panel_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    problem: reactive.Value,
    solver: reactive.Calc,
    solver_config: reactive.Value,
    current_solution: reactive.Value,
    all_solutions: reactive.Value,
):
    progress = SolverProgress()
    solver_thread: reactive.Value[threading.Thread | None] = reactive.value(None)

    @reactive.effect
    @reactive.event(input.run_solver)
    def _run_solver():
        p = problem()
        if p is None:
            ui.notification_show("Load a project first!", type="error")
            return

        active = solver()
        if not active.available():
            ui.notification_show(
                f"Solver '{active.name()}' is not available.", type="error"
            )
            return

        config_dict = solver_config()
        p.parameters["BLM"] = config_dict.get("blm", 1.0)

        solver_type = config_dict.get("solver_type", "mip")
        if solver_type in ("binary", "sa", "zone_sa"):
            p.parameters["NUMITNS"] = config_dict.get(
                "num_iterations", 1000000
            )
            p.parameters["NUMTEMP"] = config_dict.get("num_temp", 10000)

        config = SolverConfig(
            num_solutions=config_dict.get("num_solutions", 10),
            seed=config_dict.get("seed"),
            verbose=False,
            metadata={"progress": progress},
        )

        progress.reset()
        ui.notification_show(f"Running {active.name()}...", type="message")

        def _run():
            try:
                solutions = active.solve(p, config)
                if solutions:
                    best = min(solutions, key=lambda s: s.objective)
                    current_solution.set(best)
                    all_solutions.set(solutions)
                    progress.status = "done"
                    progress.best_objective = best.objective
                    met = sum(best.targets_met.values())
                    total = len(best.targets_met)
                    progress.message = (
                        f"Done! Cost: {best.cost:.2f}, "
                        f"Targets met: {met}/{total}"
                    )
                else:
                    progress.status = "done"
                    progress.message = "Solver returned no solutions."
            except Exception as e:
                progress.status = "error"
                progress.error = str(e)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        solver_thread.set(thread)

    @render.ui
    def progress_bar():
        if progress.status == "running":
            reactive.invalidate_later(0.5)
        frac = progress.progress_fraction()
        pct = int(frac * 100)
        return ui.div(
            ui.div(
                ui.div(
                    f"{pct}%",
                    class_="progress-bar",
                    role="progressbar",
                    style=f"width: {pct}%",
                ),
                class_="progress",
            ),
            ui.p(progress.format_status(), class_="mt-2 text-muted small"),
        )

    @render.text
    def run_status():
        if progress.status == "running":
            reactive.invalidate_later(0.5)
        p = problem()
        s = current_solution()
        if p is None:
            return "Step 1: Go to 'Data' tab and load a Marxan project."
        if progress.status == "running":
            return progress.format_status()
        if progress.status == "error":
            return f"Error: {progress.error}"
        if s is None:
            return (
                f"Project loaded: {p.n_planning_units} PUs, "
                f"{p.n_features} features.\n"
                f"Step 2: Configure solver in 'Configure' tab, "
                f"then click 'Run Solver'."
            )
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
        if progress.status == "running":
            reactive.invalidate_later(0.5)
        s = current_solution()
        if s is None:
            return "No solver has been run yet."
        lines = ["Solver metadata:"]
        for k, v in s.metadata.items():
            if k == "history":
                lines.append(f"  history: {len(v.get('iteration', []))} data points")
            else:
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)
