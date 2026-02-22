"""Phase 8 integration tests: solver UX modules."""
from __future__ import annotations

from pathlib import Path

from pymarxan.io.readers import load_project
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
from pymarxan_shiny.modules.results.convergence import (
    convergence_ui,
    extract_history,
)
from pymarxan_shiny.modules.run_control.progress import SolverProgress
from pymarxan_shiny.modules.run_control.run_panel import run_panel_ui

DATA_DIR = Path(__file__).parent / "data" / "simple"


def test_app_imports():
    """Verify the app can import all new modules."""
    from pymarxan_app import app
    assert app.app is not None


def test_sa_with_progress_and_history():
    """End-to-end: SA solver produces progress updates and history."""
    problem = load_project(DATA_DIR)
    problem.parameters["NUMITNS"] = 5_000
    problem.parameters["NUMTEMP"] = 50

    progress = SolverProgress()
    config = SolverConfig(
        num_solutions=2, seed=42, metadata={"progress": progress}
    )

    solver = SimulatedAnnealingSolver()
    solutions = solver.solve(problem, config)

    # Progress was updated
    assert progress.status == "done"
    assert progress.current_run == 2

    # History was recorded
    for sol in solutions:
        assert "history" in sol.metadata
        h = sol.metadata["history"]
        assert len(h["iteration"]) > 0
        for i in range(1, len(h["best_objective"])):
            assert h["best_objective"][i] <= h["best_objective"][i - 1] + 1e-9

    # Convergence extraction works
    histories = extract_history(solutions)
    assert len(histories) == 2


def test_run_panel_ui_renders():
    ui_elem = run_panel_ui("test")
    assert ui_elem is not None


def test_convergence_ui_renders():
    ui_elem = convergence_ui("test")
    assert ui_elem is not None


def test_progress_lifecycle():
    """Test the full progress lifecycle: idle -> running -> done."""
    p = SolverProgress()
    assert p.status == "idle"
    assert p.progress_fraction() == 0.0

    p.status = "running"
    p.current_run = 1
    p.total_runs = 2
    p.iteration = 5000
    p.total_iterations = 10000
    assert 0.2 < p.progress_fraction() < 0.4
    assert "1" in p.format_status()
    assert "2" in p.format_status()

    p.status = "done"
    p.best_objective = 42.0
    assert p.progress_fraction() == 1.0
