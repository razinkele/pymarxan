"""Maximum coverage objective: maximize feature coverage subject to budget."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from pymarxan.objectives.base import ZonalObjective

if TYPE_CHECKING:
    import pulp

    from pymarxan.models.problem import ConservationProblem


class MaxCoverageObjective(ZonalObjective):
    """Maximize total feature coverage subject to a budget constraint.

    Coverage for each feature is capped at its target. The score is
    negated for the lower-is-better convention:
    ``base = -Σ min(achieved_f, target_f)``.

    Requires ``BUDGET`` parameter in ``problem.parameters``.
    """

    def name(self) -> str:
        return "MaxCoverage"

    def uses_target_penalty(self) -> bool:
        return False

    def compute_base_score(
        self,
        problem: Any,
        selected: np.ndarray,
        effective_amounts: np.ndarray,
        pu_index: dict[int, int],
    ) -> float:
        feat_ids = problem.features["id"].values
        targets = dict(
            zip(problem.features["id"], problem.features["target"], strict=False),
        )
        achieved = self._compute_achieved(
            problem, selected, effective_amounts, pu_index,
        )
        total_coverage = 0.0
        for fid in feat_ids:
            t = float(targets.get(int(fid), 0.0))
            a = achieved.get(int(fid), 0.0)
            total_coverage += min(a, t) if t > 0 else 0.0
        return -total_coverage

    def build_mip_objective(
        self,
        problem: Any,
        model: pulp.LpProblem,
        x: dict[int, pulp.LpVariable],
        effective_amounts: np.ndarray,
        pu_index: dict[int, int],
    ) -> pulp.LpAffineExpression:
        import pulp as _pulp

        # y_f auxiliary variables (capped coverage per feature)
        # Negate for minimization: minimize -Σ y_f
        feat_ids = problem.features["id"].values
        y = {}
        for fid in feat_ids:
            fid_int = int(fid)
            t = float(problem.features.loc[
                problem.features["id"] == fid, "target"
            ].iloc[0])
            if t <= 0:
                continue
            y[fid_int] = _pulp.LpVariable(
                f"y_cov_{fid_int}", lowBound=0, upBound=t,
            )
        return _pulp.lpSum(-y[fid] for fid in y)

    def build_mip_constraints(
        self,
        problem: Any,
        model: pulp.LpProblem,
        x: dict[int, pulp.LpVariable],
        effective_amounts: np.ndarray,
        pu_index: dict[int, int],
    ) -> None:
        import pulp as _pulp

        feat_ids = problem.features["id"].values
        puvspr = problem.pu_vs_features

        for fid in feat_ids:
            fid_int = int(fid)
            t = float(problem.features.loc[
                problem.features["id"] == fid, "target"
            ].iloc[0])
            if t <= 0:
                continue

            rows = puvspr[puvspr["species"] == fid_int]
            achieved_expr = _pulp.lpSum(
                float(effective_amounts[pu_index[int(row["pu"])],
                      list(problem.features["id"]).index(fid_int)])
                * x[int(row["pu"])]
                for _, row in rows.iterrows()
                if int(row["pu"]) in x and int(row["pu"]) in pu_index
            )

            y_var = model.variablesDict().get(f"y_cov_{fid_int}")
            if y_var is not None:
                model += y_var <= achieved_expr, f"cov_cap_amount_{fid_int}"
                model += y_var <= t, f"cov_cap_target_{fid_int}"

        # Budget constraint
        budget = float(problem.parameters.get("BUDGET", float("inf")))
        if budget < float("inf"):
            cost_map = dict(
                zip(
                    problem.planning_units["id"],
                    problem.planning_units["cost"],
                    strict=False,
                ),
            )
            model += (
                _pulp.lpSum(
                    cost_map.get(pid, 0.0) * x[pid] for pid in x
                )
                <= budget
            ), "budget_constraint"

    # --- Zonal interface ---

    def compute_zone_score(
        self,
        problem: Any,
        assignment: np.ndarray,
        effective_amounts: np.ndarray,
        pu_index: dict[int, int],
    ) -> float:
        # Negate coverage for lower-is-better
        feat_ids = problem.features["id"].values
        targets = dict(
            zip(problem.features["id"], problem.features["target"], strict=False),
        )
        achieved = self._compute_zone_achieved(
            problem, assignment, effective_amounts, pu_index,
        )
        total_coverage = 0.0
        for fid in feat_ids:
            t = float(targets.get(int(fid), 0.0))
            a = achieved.get(int(fid), 0.0)
            total_coverage += min(a, t) if t > 0 else 0.0
        return -total_coverage

    def build_zone_mip_objective(
        self,
        problem: Any,
        model: pulp.LpProblem,
        x: dict[tuple[int, int], pulp.LpVariable],
        effective_amounts: np.ndarray,
        pu_index: dict[int, int],
    ) -> pulp.LpAffineExpression:
        import pulp as _pulp

        feat_ids = problem.features["id"].values
        y = {}
        for fid in feat_ids:
            fid_int = int(fid)
            t = float(problem.features.loc[
                problem.features["id"] == fid, "target"
            ].iloc[0])
            if t <= 0:
                continue
            y[fid_int] = _pulp.LpVariable(
                f"zy_cov_{fid_int}", lowBound=0, upBound=t,
            )
        return _pulp.lpSum(-y[fid] for fid in y)

    # --- Helpers ---

    @staticmethod
    def _compute_achieved(
        problem: Any,
        selected: np.ndarray,
        effective_amounts: np.ndarray,
        pu_index: dict[int, int],
    ) -> dict[int, float]:
        """Compute achieved amount per feature from selected PUs."""
        feat_ids = list(problem.features["id"])
        achieved: dict[int, float] = {int(f): 0.0 for f in feat_ids}
        puvspr = problem.pu_vs_features
        for _, row in puvspr.iterrows():
            pid = int(row["pu"])
            fid = int(row["species"])
            idx = pu_index.get(pid)
            if idx is not None and selected[idx]:
                fidx = feat_ids.index(fid)
                achieved[fid] += float(effective_amounts[idx, fidx])
        return achieved

    @staticmethod
    def _compute_zone_achieved(
        problem: Any,
        assignment: np.ndarray,
        effective_amounts: np.ndarray,
        pu_index: dict[int, int],
    ) -> dict[int, float]:
        """Compute achieved amount per feature from zone assignment."""
        feat_ids = list(problem.features["id"])
        achieved: dict[int, float] = {int(f): 0.0 for f in feat_ids}
        puvspr = problem.pu_vs_features
        zone_contributions = getattr(problem, "zone_contributions", {})
        for _, row in puvspr.iterrows():
            pid = int(row["pu"])
            fid = int(row["species"])
            idx = pu_index.get(pid)
            if idx is not None:
                z = int(assignment[idx])
                if z > 0:
                    fidx = feat_ids.index(fid)
                    contrib = float(
                        zone_contributions.get((z, fid), 1.0),
                    )
                    achieved[fid] += (
                        float(effective_amounts[idx, fidx]) * contrib
                    )
        return achieved
