"""Tests for SolverProgress dataclass."""
from __future__ import annotations

from pymarxan_shiny.modules.run_control.progress import SolverProgress


class TestSolverProgress:
    def test_default_construction(self):
        p = SolverProgress()
        assert p.status == "idle"
        assert p.current_run == 0
        assert p.total_runs == 0
        assert p.iteration == 0
        assert p.total_iterations == 0
        assert p.best_objective == float("inf")
        assert p.message == ""
        assert p.error is None

    def test_update_fields(self):
        p = SolverProgress()
        p.status = "running"
        p.current_run = 3
        p.total_runs = 10
        p.iteration = 50000
        p.total_iterations = 1000000
        p.best_objective = 1234.5
        p.message = "Run 3/10 — Iteration 50K/1M"
        assert p.status == "running"
        assert p.current_run == 3
        assert p.best_objective == 1234.5

    def test_reset(self):
        p = SolverProgress()
        p.status = "running"
        p.current_run = 5
        p.best_objective = 100.0
        p.reset()
        assert p.status == "idle"
        assert p.current_run == 0
        assert p.best_objective == float("inf")
        assert p.error is None

    def test_format_status_idle(self):
        p = SolverProgress()
        assert "idle" in p.format_status().lower() or "ready" in p.format_status().lower()

    def test_format_status_running(self):
        p = SolverProgress()
        p.status = "running"
        p.current_run = 2
        p.total_runs = 5
        p.iteration = 500000
        p.total_iterations = 1000000
        text = p.format_status()
        assert "2" in text
        assert "5" in text

    def test_format_status_done(self):
        p = SolverProgress()
        p.status = "done"
        p.best_objective = 42.0
        text = p.format_status()
        assert "42" in text or "done" in text.lower() or "complete" in text.lower()

    def test_format_status_error(self):
        p = SolverProgress()
        p.status = "error"
        p.error = "Solver crashed"
        text = p.format_status()
        assert "crash" in text.lower() or "error" in text.lower()

    def test_progress_fraction_idle(self):
        p = SolverProgress()
        assert p.progress_fraction() == 0.0

    def test_progress_fraction_running(self):
        p = SolverProgress()
        p.status = "running"
        p.current_run = 2
        p.total_runs = 4
        p.iteration = 500000
        p.total_iterations = 1000000
        # Run 1 done (25%) + half of run 2 (12.5%) = 37.5%
        frac = p.progress_fraction()
        assert 0.3 < frac < 0.45

    def test_progress_fraction_done(self):
        p = SolverProgress()
        p.status = "done"
        assert p.progress_fraction() == 1.0
