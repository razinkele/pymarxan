"""Abstract base classes for conservation planning constraints.

Two constraint categories:

- **IncrementalConstraint** — participates in SA via O(degree) cached delta.
  Used for: linear constraints, neighbor constraints.
- **Constraint** (base) — MIP-only or post-hoc repair.
  Used for: contiguity, feature contiguity.

Zonal extensions:

- **ZonalConstraint** — zone-aware evaluation and MIP support.
- **IncrementalZonalConstraint** — zone SA participation via O(degree) delta.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pulp

    from pymarxan.models.problem import ConservationProblem
    from pymarxan.solvers.cache import ProblemCache
    from pymarxan.zones.cache import ZoneProblemCache
    from pymarxan.zones.model import ZonalProblem


@dataclass
class ConstraintResult:
    """Result of evaluating a constraint against a solution.

    Attributes
    ----------
    satisfied : bool
        Whether the constraint is satisfied.
    violation : float
        Magnitude of violation (0.0 if satisfied).
    description : str
        Human-readable description of the result.
    """

    satisfied: bool
    violation: float
    description: str


class Constraint(ABC):
    """Base class for conservation planning constraints.

    Constraints that only inherit from this base class are MIP-only or
    applied as post-hoc repair. They do NOT participate in SA's inner
    loop (too expensive for per-flip evaluation).
    """

    @abstractmethod
    def name(self) -> str:
        """Return a human-readable name for this constraint."""

    @abstractmethod
    def evaluate(
        self,
        problem: ConservationProblem,
        selected: np.ndarray,
    ) -> ConstraintResult:
        """Check if constraint is satisfied and return violation magnitude."""

    @abstractmethod
    def penalty(
        self,
        problem: ConservationProblem,
        selected: np.ndarray,
    ) -> float:
        """Return penalty value for objective function (0 if satisfied)."""

    def apply_to_mip(
        self,
        problem: ConservationProblem,
        model: pulp.LpProblem,
        x: dict[int, pulp.LpVariable],
    ) -> None:
        """Add constraint to MIP model.

        Override for MIP-compatible constraints. Default raises
        NotImplementedError.

        Parameters
        ----------
        problem : ConservationProblem
            The conservation problem.
        model : pulp.LpProblem
            The MIP model to add constraints to.
        x : dict[int, pulp.LpVariable]
            Decision variables keyed by planning unit ID.
        """
        raise NotImplementedError(
            f"{self.name()} does not support MIP formulation"
        )


class IncrementalConstraint(Constraint):
    """Constraint that supports O(degree) delta computation for SA.

    Only constraints inheriting this class participate in SA's inner
    loop. Others (e.g., contiguity) are MIP-only or applied post-hoc.

    Constraint states are tracked in the SA loop's mutable SolverState,
    NOT on ProblemCache (which is frozen).
    """

    @abstractmethod
    def init_state(
        self,
        problem: ConservationProblem,
        selected: np.ndarray,
        cache: ProblemCache,
    ) -> Any:
        """Initialize cached state for incremental updates.

        Returns
        -------
        Any
            Opaque state object passed to compute_delta/update_state.
        """

    @abstractmethod
    def compute_delta(
        self,
        idx: int,
        selected: np.ndarray,
        state: Any,
        cache: ProblemCache,
    ) -> float:
        """Return change in penalty if PU at *idx* is flipped.

        Must be O(degree) or better.

        Parameters
        ----------
        idx : int
            Array index of the planning unit to flip.
        selected : np.ndarray
            Current (n_pu,) boolean selection array.
        state : Any
            Opaque state from init_state/update_state.
        cache : ProblemCache
            Precomputed problem arrays.

        Returns
        -------
        float
            Change in penalty (positive = worse).
        """

    @abstractmethod
    def update_state(
        self,
        idx: int,
        selected: np.ndarray,
        state: Any,
        cache: ProblemCache,
    ) -> None:
        """Update cached state after accepting a flip.

        Called AFTER ``selected[idx]`` has been mutated.
        Must be O(degree) or better.
        """


class ZonalConstraint(ABC):
    """Mixin for constraints that support zone-aware evaluation.

    Concrete constraints can inherit both IncrementalConstraint and
    ZonalConstraint to support binary SA, MIP, AND zonal evaluation.
    """

    @abstractmethod
    def evaluate_zonal(
        self,
        problem: ZonalProblem,
        assignment: np.ndarray,
    ) -> ConstraintResult:
        """Evaluate constraint against a zonal assignment.

        Parameters
        ----------
        problem : ZonalProblem
            The zonal conservation problem.
        assignment : np.ndarray
            (n_pu,) int array of zone assignments (0 = unassigned).
        """

    @abstractmethod
    def penalty_zonal(
        self,
        problem: ZonalProblem,
        assignment: np.ndarray,
    ) -> float:
        """Return penalty value for zonal objective (0 if satisfied)."""

    def apply_to_zone_mip(
        self,
        problem: ZonalProblem,
        model: pulp.LpProblem,
        x_zone: dict[tuple[int, int], pulp.LpVariable],
    ) -> None:
        """Add constraint to zonal MIP model.

        Parameters
        ----------
        problem : ZonalProblem
            The zonal conservation problem.
        model : pulp.LpProblem
            The MIP model.
        x_zone : dict[tuple[int, int], pulp.LpVariable]
            Decision variables keyed by (pu_id, zone_id).
        """
        raise NotImplementedError(
            "Zonal MIP not supported for this constraint"
        )


class IncrementalZonalConstraint(ZonalConstraint):
    """Constraint supporting O(degree) delta for zone reassignment.

    These participate in zone SA's inner loop.
    """

    @abstractmethod
    def init_zone_state(
        self,
        problem: ZonalProblem,
        assignment: np.ndarray,
        zone_cache: ZoneProblemCache,
    ) -> Any:
        """Initialize cached state for incremental zone updates.

        Parameters
        ----------
        problem : ZonalProblem
            The zonal conservation problem.
        assignment : np.ndarray
            (n_pu,) int array of current zone assignments.
        zone_cache : ZoneProblemCache
            Precomputed zonal problem arrays.

        Returns
        -------
        Any
            Opaque state object.
        """

    @abstractmethod
    def compute_zone_delta(
        self,
        idx: int,
        old_zone: int,
        new_zone: int,
        assignment: np.ndarray,
        state: Any,
        zone_cache: ZoneProblemCache,
    ) -> float:
        """Return change in penalty for zone reassignment.

        Must be O(degree) or better.

        Parameters
        ----------
        idx : int
            Array index of the planning unit being reassigned.
        old_zone : int
            Current zone assignment for this PU.
        new_zone : int
            Proposed new zone assignment.
        assignment : np.ndarray
            Current (n_pu,) int zone assignment array.
        state : Any
            Opaque state from init_zone_state.
        zone_cache : ZoneProblemCache
            Precomputed zonal problem arrays.

        Returns
        -------
        float
            Change in penalty (positive = worse).
        """

    @abstractmethod
    def update_zone_state(
        self,
        idx: int,
        old_zone: int,
        new_zone: int,
        assignment: np.ndarray,
        state: Any,
        zone_cache: ZoneProblemCache,
    ) -> None:
        """Update cached state after accepting a zone reassignment.

        Called AFTER ``assignment[idx]`` has been mutated.
        Must be O(degree) or better.
        """
