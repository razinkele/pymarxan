"""Linear constraints for conservation planning problems.

A ``LinearConstraint`` enforces a bound on a weighted sum of PU
selection decisions: ``Σ coeff_i * x_i {<=, >=, ==} rhs``.

This is an ``IncrementalConstraint`` — delta for a single PU flip is
O(1) since only the flipped PU's coefficient matters.

When ``hard=True``, SA rejects infeasible moves (returns +inf delta)
using three-way logic to allow repair from infeasible initial states.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from pymarxan.constraints.base import IncrementalConstraint

if TYPE_CHECKING:
    import pulp

    from pymarxan.models.problem import ConservationProblem
    from pymarxan.solvers.cache import ProblemCache


@dataclass
class LinearConstraint(IncrementalConstraint):
    """Linear constraint: Σ coeff_i * x_i  {<=, >=, ==}  rhs.

    Parameters
    ----------
    coefficients : dict[int, float]
        Mapping from planning unit ID to coefficient.
    sense : str
        Comparison operator: ``"<="``, ``">="``, or ``"=="``.
    rhs : float
        Right-hand side value.
    penalty_weight : float
        Penalty multiplier for soft constraint in SA.
    hard : bool
        If True, SA rejects infeasible moves (budget constraint mode).
    label : str
        Human-readable name for this constraint.
    """

    coefficients: dict[int, float]
    sense: str
    rhs: float
    penalty_weight: float = 1000.0
    hard: bool = False
    label: str = "LinearConstraint"

    def __post_init__(self) -> None:
        if self.sense not in ("<=", ">=", "=="):
            raise ValueError(
                f"sense must be '<=', '>=', or '==', got '{self.sense}'"
            )

    def name(self) -> str:
        return self.label

    # ------------------------------------------------------------------
    # Violation helpers
    # ------------------------------------------------------------------

    def _compute_lhs(
        self, problem: ConservationProblem, selected: np.ndarray
    ) -> float:
        pu_ids = problem.planning_units["id"].values
        pu_id_to_idx = {int(pid): i for i, pid in enumerate(pu_ids)}
        total = 0.0
        for pid, coeff in self.coefficients.items():
            idx = pu_id_to_idx.get(pid)
            if idx is not None and selected[idx]:
                total += coeff
        return total

    def _is_violated(self, lhs: float) -> bool:
        if self.sense == "<=":
            return lhs > self.rhs
        elif self.sense == ">=":
            return lhs < self.rhs
        else:  # ==
            return abs(lhs - self.rhs) > 1e-10

    def _violation_amount(self, lhs: float) -> float:
        if self.sense == "<=":
            return max(0.0, lhs - self.rhs)
        elif self.sense == ">=":
            return max(0.0, self.rhs - lhs)
        else:  # ==
            return abs(lhs - self.rhs)

    # ------------------------------------------------------------------
    # Constraint interface
    # ------------------------------------------------------------------

    def evaluate(self, problem, selected):
        from pymarxan.constraints.base import ConstraintResult

        lhs = self._compute_lhs(problem, selected)
        violation = self._violation_amount(lhs)
        satisfied = violation < 1e-10
        return ConstraintResult(
            satisfied=satisfied,
            violation=violation,
            description=(
                f"{self.label}: LHS={lhs:.4f} {self.sense} {self.rhs}"
            ),
        )

    def penalty(self, problem, selected):
        lhs = self._compute_lhs(problem, selected)
        return self.penalty_weight * self._violation_amount(lhs)

    # ------------------------------------------------------------------
    # Incremental interface
    # ------------------------------------------------------------------

    def init_state(self, problem, selected, cache):
        lhs = 0.0
        for pid, coeff in self.coefficients.items():
            idx = cache.pu_id_to_idx.get(pid)
            if idx is not None and selected[idx]:
                lhs += coeff
        return {"lhs": lhs}

    def compute_delta(self, idx, selected, state, cache):
        # Reverse lookup: idx -> pu_id
        pu_id = self._idx_to_pu_id(cache, idx)
        coeff = self.coefficients.get(pu_id, 0.0)
        if coeff == 0.0:
            return 0.0

        sign = 1.0 if not selected[idx] else -1.0
        new_lhs = state["lhs"] + sign * coeff

        if self.hard:
            currently_violated = self._is_violated(state["lhs"])
            would_violate = self._is_violated(new_lhs)
            if would_violate and not currently_violated:
                return float("inf")
            if would_violate and currently_violated:
                old_v = self._violation_amount(state["lhs"])
                new_v = self._violation_amount(new_lhs)
                if new_v >= old_v:
                    return float("inf")
                # Strongly prefer repair moves
                return -1e6 * (old_v - new_v)
            # feasible→feasible or violated→feasible
            return 0.0

        old_violation = self._violation_amount(state["lhs"])
        new_violation = self._violation_amount(new_lhs)
        return self.penalty_weight * (new_violation - old_violation)

    def update_state(self, idx, selected, state, cache):
        # Called AFTER selected[idx] is mutated
        pu_id = self._idx_to_pu_id(cache, idx)
        coeff = self.coefficients.get(pu_id, 0.0)
        # selected[idx]=True means PU was just added → LHS increases
        sign = 1.0 if selected[idx] else -1.0
        state["lhs"] += sign * coeff

    # ------------------------------------------------------------------
    # MIP interface
    # ------------------------------------------------------------------

    def apply_to_mip(self, problem, model, x):
        import pulp

        expr = pulp.lpSum(
            coeff * x[pid]
            for pid, coeff in self.coefficients.items()
            if pid in x
        )
        constraint_name = self.label.replace(" ", "_")
        if self.sense == "<=":
            model += expr <= self.rhs, constraint_name
        elif self.sense == ">=":
            model += expr >= self.rhs, constraint_name
        else:
            model += expr == self.rhs, constraint_name

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _idx_to_pu_id(cache: ProblemCache, idx: int) -> int:
        """Reverse lookup from array index to PU ID."""
        # Build reverse mapping on first call (cached on cache object)
        if not hasattr(cache, "_idx_to_pu_id_map"):
            mapping = {v: k for k, v in cache.pu_id_to_idx.items()}
            object.__setattr__(cache, "_idx_to_pu_id_map", mapping)
        return cache._idx_to_pu_id_map.get(idx, -1)  # type: ignore[attr-defined]
