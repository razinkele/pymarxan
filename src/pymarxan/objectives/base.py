"""Abstract base classes for optimization objectives."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pulp

    from pymarxan.models.problem import ConservationProblem
    from pymarxan.solvers.cache import ProblemCache


class Objective(ABC):
    """Base class for conservation planning objectives.

    All objectives use a **lower-is-better** (minimization) convention.
    Objectives that maximize a quantity (e.g., coverage) negate their
    score so the unified convention holds.
    """

    @abstractmethod
    def name(self) -> str:
        """Return a short identifier for this objective type."""

    @abstractmethod
    def compute_base_score(
        self,
        problem: ConservationProblem,
        selected: np.ndarray,
        effective_amounts: np.ndarray,
        pu_index: dict[int, int],
    ) -> float:
        """Compute the base objective score (lower-is-better).

        Args:
            problem: The conservation problem instance.
            selected: Boolean array of selected planning units.
            effective_amounts: PU-feature amount matrix (may be adjusted
                for PROBMODE=2).
            pu_index: Mapping from PU ID to array index.

        Returns:
            Base objective value. MaxCoverage/MaxUtility negate their
            scores for the minimization convention.
        """

    def uses_target_penalty(self) -> bool:
        """Whether classic target shortfall penalty applies.

        True for MinSet only. MaxCoverage/MaxUtility/MinShortfall
        disable penalty since targets are part of the objective, not
        constraints.
        """
        return True

    @abstractmethod
    def build_mip_objective(
        self,
        problem: ConservationProblem,
        model: pulp.LpProblem,
        x: dict[int, pulp.LpVariable],
        effective_amounts: np.ndarray,
        pu_index: dict[int, int],
    ) -> pulp.LpAffineExpression:
        """Return the MIP objective expression (must be linear)."""

    def build_mip_constraints(
        self,
        problem: ConservationProblem,
        model: pulp.LpProblem,
        x: dict[int, pulp.LpVariable],
        effective_amounts: np.ndarray,
        pu_index: dict[int, int],
    ) -> None:
        """Add objective-specific constraints to the MIP model.

        Default: no additional constraints (MinSet uses hard target
        constraints added by the solver itself).
        """

    def compute_delta(
        self,
        idx: int,
        selected: np.ndarray,
        cache: ProblemCache,
        state: Any,
    ) -> float | None:
        """Incremental delta for SA.

        Return ``None`` if not supported (forces full recomputation).
        """
        return None

    def init_state(
        self,
        problem: ConservationProblem,
        selected: np.ndarray,
        cache: ProblemCache,
    ) -> Any:
        """Initialize incremental state for SA."""
        return None

    def update_state(
        self,
        idx: int,
        selected: np.ndarray,
        state: Any,
        cache: ProblemCache,
    ) -> None:
        """Update state after accepting a move."""


class ZonalObjective(Objective):
    """Extension for zone-aware objectives."""

    @abstractmethod
    def compute_zone_score(
        self,
        problem: Any,  # ZonalProblem
        assignment: np.ndarray,
        effective_amounts: np.ndarray,
        pu_index: dict[int, int],
    ) -> float:
        """Compute zonal objective score (lower-is-better)."""

    @abstractmethod
    def build_zone_mip_objective(
        self,
        problem: Any,  # ZonalProblem
        model: pulp.LpProblem,
        x: dict[tuple[int, int], pulp.LpVariable],
        effective_amounts: np.ndarray,
        pu_index: dict[int, int],
    ) -> pulp.LpAffineExpression:
        """Return the zonal MIP objective expression."""

    def build_zone_mip_constraints(
        self,
        problem: Any,  # ZonalProblem
        model: pulp.LpProblem,
        x: dict[tuple[int, int], pulp.LpVariable],
        effective_amounts: np.ndarray,
        pu_index: dict[int, int],
    ) -> None:
        """Add objective-specific auxiliary constraints for zonal MIP.

        Default: no extra constraints (MinSet needs none).
        """

    def compute_zone_delta(
        self,
        idx: int,
        old_zone: int,
        new_zone: int,
        assignment: np.ndarray,
        zone_cache: Any,
        state: Any,
    ) -> float | None:
        """Incremental delta for zone SA reassignment."""
        return None

    def init_zone_state(
        self,
        problem: Any,
        assignment: np.ndarray,
        zone_cache: Any,
    ) -> Any:
        """Initialize zone-specific incremental state."""
        return None

    def update_zone_state(
        self,
        idx: int,
        old_zone: int,
        new_zone: int,
        assignment: np.ndarray,
        state: Any,
        zone_cache: Any,
    ) -> None:
        """Update zone state after accepting a reassignment."""
