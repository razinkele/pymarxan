"""Base solver interface and solution model for conservation planning."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from pymarxan.models.problem import ConservationProblem


@dataclass
class Solution:
    """Result of a conservation planning optimization run."""

    selected: np.ndarray  # Boolean array: is planning unit i selected?
    cost: float
    boundary: float
    objective: float  # cost + BLM * boundary + penalties
    # Single-zone solvers: feature ID -> met. Zone solvers: (zone, feature) -> met.
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

    @property
    def all_targets_met(self) -> bool:
        return all(self.targets_met.values())

    @property
    def n_selected(self) -> int:
        return int(self.selected.sum())


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
