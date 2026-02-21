"""Target sensitivity analysis for conservation planning.

Varies feature targets (e.g., +/-10%, +/-20%) and measures how the
optimal solution changes. Helps practitioners understand which
targets drive the solution.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solver, SolverConfig


@dataclass
class SensitivityConfig:
    """Configuration for a target sensitivity analysis."""

    feature_ids: list[int] | None = None
    multipliers: list[float] = field(
        default_factory=lambda: [0.8, 0.9, 1.0, 1.1, 1.2]
    )
    solver_config: SolverConfig | None = None


@dataclass
class SensitivityResult:
    """Results of a target sensitivity analysis."""

    runs: list[dict]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.runs)


def run_sensitivity(
    problem: ConservationProblem,
    solver: Solver,
    config: SensitivityConfig,
) -> SensitivityResult:
    """Run target sensitivity analysis."""
    solver_config = config.solver_config or SolverConfig(num_solutions=1)

    if config.feature_ids is not None:
        feature_ids = config.feature_ids
    else:
        feature_ids = problem.features["id"].tolist()

    runs: list[dict] = []

    for fid in feature_ids:
        original_target = float(
            problem.features.loc[problem.features["id"] == fid, "target"].iloc[0]
        )
        for mult in config.multipliers:
            features_df = problem.features.copy()
            features_df.loc[features_df["id"] == fid, "target"] = (
                original_target * mult
            )
            modified = ConservationProblem(
                planning_units=problem.planning_units,
                features=features_df,
                pu_vs_features=problem.pu_vs_features,
                boundary=problem.boundary,
                parameters=problem.parameters,
            )
            sols = solver.solve(modified, solver_config)
            best = min(sols, key=lambda s: s.objective)
            runs.append({
                "feature_id": fid,
                "multiplier": mult,
                "target": original_target * mult,
                "cost": best.cost,
                "boundary": best.boundary,
                "objective": best.objective,
                "n_selected": best.n_selected,
                "all_targets_met": best.all_targets_met,
            })

    return SensitivityResult(runs=runs)
