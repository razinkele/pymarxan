"""Base solver interface and solution model for conservation planning."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from pymarxan.models.problem import ConservationProblem


@dataclass
class Solution:
    """Result of a conservation planning optimization run."""

    # One entry per planning unit, in planning_units order: a bool array, or float64
    # proportions, finite and in [0, 1], when solved with non-binary decisions. The dtype
    # is the discriminator; build_solution enforces this (int 0/1 arrays become bool).
    selected: np.ndarray
    cost: float
    boundary: float
    objective: float  # cost + BLM * boundary + penalties
    # Feature ID -> met. Zone solvers report the overall (contribution-weighted) feature
    # targets here and the per-zone targets in metadata["zone_targets_met"].
    targets_met: dict[int, bool] | dict[tuple[int, int], bool]
    penalty: float = 0.0  # Total SPF-weighted shortfall penalty
    shortfall: float = 0.0  # Total raw feature shortfall (sum of max(0, target - achieved))
    metadata: dict = field(default_factory=dict)
    zone_assignment: np.ndarray | None = None
    # PROBMODE 3 outputs — populated when the problem has PROBMODE=3 (and
    # any feature with ptarget > 0). Always None otherwise.
    prob_shortfalls: dict[int, float] | None = None  # feature_id -> max(0, ptarget − P)
    prob_penalty: float | None = None  # γ · Σ SPF · (ptarget − P) / ptarget
    # TARGET2 / clumping outputs — populated when the problem has any
    # feature with target2 > 0. Always None otherwise.
    # clump_shortfalls: feature_id -> max(0, T·MISSLEVEL − held_eff)
    # clump_penalty: Σ baseline · SPF · fractional_shortfall
    clump_shortfalls: dict[int, float] | None = None
    clump_penalty: float | None = None
    # SEPDISTANCE / SEPNUM outputs (Phase 20) — populated when any feature
    # is separation-active (sepdistance > 0 AND sepnum > 1). None otherwise.
    # sep_shortfalls: feature_id -> max(0, sepnum - count); integer-valued.
    # sep_penalty: Σ baseline · SPF · compute_sep_penalty(count, sepnum)
    sep_shortfalls: dict[int, int] | None = None
    sep_penalty: float | None = None
    # Space/adequacy outputs (raptr-style) — populated when any feature has
    # space_target > 0. Always None otherwise.
    # space_held: feature_id -> 1 − WSS/TSS proportion of attribute-space variation captured.
    # space_penalty: Σ space_spf · max(0, space_target − space_held)
    space_held: dict[int, float] | None = None
    space_penalty: float | None = None

    @property
    def all_targets_met(self) -> bool:
        return all(self.targets_met.values())

    @property
    def is_fractional(self) -> bool:
        """Solved with proportion / semicontinuous decisions: the dtype decides, not the
        values (an integral LP optimum is still fractional-typed)."""
        return self.selected.dtype.kind == "f"

    @property
    def n_selected(self) -> int:
        """Count of planning units with x > 0 (a count, not an area)."""
        return int(np.count_nonzero(self.selected))

    @property
    def selected_sum(self) -> float:
        """Σ x (prioritizr ``eval_n_summary``); equals ``n_selected`` for binary."""
        return float(self.selected.sum())


class FractionalSolutionError(ValueError):
    """A binary-only tool was given a solution holding proportion decisions."""


def require_binary_solver(solver: object, context: str) -> None:
    """``FractionalSolutionError`` up front for a solver configured with non-binary
    decisions (tools that re-solve and read 0/1 plans, spec §6.2)."""
    decisions = getattr(solver, "decisions", "binary")
    if decisions != "binary":
        raise FractionalSolutionError(
            f"{context} needs a binary solver, but this solver uses "
            f"decisions={decisions!r}. Use decisions='binary'."
        )


def require_binary(solution: Solution, context: str) -> None:
    """Raise ``FractionalSolutionError`` unless ``solution.selected`` is binary (spec §5.2)."""
    if solution.is_fractional:
        raise FractionalSolutionError(
            f"{context} needs a binary solution, but this solution holds proportion "
            "decisions. Convert explicitly with binarize_solution(problem, solution, "
            "threshold=...)."
        )


@dataclass
class SolverConfig:
    """Configuration for a solver run."""

    num_solutions: int = 10
    seed: int | None = None
    verbose: bool = False
    metadata: dict = field(default_factory=dict)


class Solver(ABC):
    """Abstract base class for conservation planning solvers."""

    @abstractmethod
    def solve(
        self, problem: ConservationProblem, config: SolverConfig | None = None
    ) -> list[Solution]:
        ...

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def supports_zones(self) -> bool:
        ...

    def available(self) -> bool:
        return True

    def supports_probmode3(self) -> bool:
        """Whether this solver supports PROBMODE 3 (Z-score chance constraints).

        Defaults to True. Solvers that genuinely cannot run under PROBMODE 3
        (no native or fallback path) override to False so dispatchers and UI
        can pre-filter. MIPSolver/ZoneMIPSolver keep this True because they
        fall back to ``mip_chance_strategy='drop'`` (deterministic solve,
        post-hoc chance-constraint reporting).
        """
        return True

    def supports_clumping(self) -> bool:
        """Whether this solver supports TARGET2 / CLUMPTYPE (Marxan type-4
        species / minimum-patch-size constraints).

        Defaults to True. MIPSolver falls back via ``mip_clump_strategy='drop'``
        (deterministic solve, post-hoc clump-shortfall reporting). The
        heuristic stays clumping-blind during scoring but reports the gap
        post-hoc through ``Solution.clump_shortfalls`` / ``clump_penalty``.
        SA and iterative-improvement honour clumping natively via the
        ``ClumpState`` companion to ``ProblemCache``.
        """
        return True

    def supports_separation(self) -> bool:
        """Whether this solver supports SEPDISTANCE / SEPNUM (Marxan
        geographic separation constraints).

        Defaults to True. MIPSolver falls back via ``mip_sep_strategy='drop'``
        (deterministic solve, post-hoc separation-shortfall reporting).
        The heuristic stays separation-blind during scoring but reports
        the gap post-hoc through ``Solution.sep_shortfalls`` /
        ``sep_penalty``. SA and iterative-improvement honour separation
        natively via the ``SepState`` companion to ``ProblemCache``. Zone
        solvers override this to False (per-zone separation deferred to
        v0.3).
        """
        return True

    def supports_space(self) -> bool:
        """Whether this solver optimizes raptr-style space/adequacy targets
        (``space_target`` feature column).

        Defaults to **False** — space targets are opt-in and only the SA and
        greedy solvers honour them (via the ``SpaceState`` companion / the
        two-phase greedy). Non-supporting solvers (MIP, zones) still *report*
        the space gap post-hoc through ``Solution.space_held`` /
        ``space_penalty`` — they simply do not optimize toward it.
        """
        return False
