"""Thread-safe progress model for solver execution monitoring."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SolverProgress:
    """Tracks solver execution progress.

    All fields are simple Python types with atomic reads/writes on CPython.
    Shared between solver thread (writer) and UI thread (reader).
    """

    status: str = "idle"  # "idle" | "running" | "done" | "error"
    current_run: int = 0
    total_runs: int = 0
    iteration: int = 0
    total_iterations: int = 0
    best_objective: float = field(default=float("inf"))
    message: str = ""
    error: str | None = None

    def reset(self) -> None:
        """Reset to idle state."""
        self.status = "idle"
        self.current_run = 0
        self.total_runs = 0
        self.iteration = 0
        self.total_iterations = 0
        self.best_objective = float("inf")
        self.message = ""
        self.error = None

    def format_status(self) -> str:
        """Return a human-readable status string."""
        if self.status == "idle":
            return "Ready to run."
        elif self.status == "running":
            if self.total_iterations > 0:
                pct = (self.iteration / self.total_iterations) * 100
                return (
                    f"Run {self.current_run}/{self.total_runs} — "
                    f"Iteration {self.iteration:,}/{self.total_iterations:,} "
                    f"({pct:.0f}%) — Best: {self.best_objective:.2f}"
                )
            return f"Run {self.current_run}/{self.total_runs} — Running..."
        elif self.status == "done":
            return f"Complete! Best objective: {self.best_objective:.2f}"
        elif self.status == "error":
            return f"Error: {self.error}"
        return self.message

    def progress_fraction(self) -> float:
        """Return overall progress as a float in [0, 1]."""
        if self.status == "done":
            return 1.0
        if self.status != "running" or self.total_runs == 0:
            return 0.0
        run_fraction = (self.current_run - 1) / self.total_runs
        if self.total_iterations > 0:
            iter_fraction = self.iteration / self.total_iterations
        else:
            iter_fraction = 0.0
        return run_fraction + iter_fraction / self.total_runs
