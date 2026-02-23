"""SPF (Species Penalty Factor) calibration for Marxan.

Iteratively adjusts SPF values to ensure all conservation targets are met.
Process: solve -> check unmet targets -> increase SPF for unmet features -> re-solve.
"""
from __future__ import annotations

from dataclasses import dataclass

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig


@dataclass
class SPFResult:
    """Results of an SPF calibration run."""
    final_spf: dict[int, float]
    solution: Solution | None
    history: list[dict]


def calibrate_spf(
    problem: ConservationProblem,
    solver: Solver,
    max_iterations: int = 10,
    multiplier: float = 2.0,
    config: SolverConfig | None = None,
) -> SPFResult:
    """Iteratively adjust SPF until all targets are met."""
    if config is None:
        config = SolverConfig(num_solutions=1)

    spf_values = {}
    for _, row in problem.features.iterrows():
        fid = int(row["id"])
        spf_values[fid] = float(row.get("spf", 1.0))

    history = []
    best_solution = None

    for iteration in range(max_iterations):
        features_df = problem.features.copy()
        for fid, spf in spf_values.items():
            features_df.loc[features_df["id"] == fid, "spf"] = spf

        modified = ConservationProblem(
            planning_units=problem.planning_units,
            features=features_df,
            pu_vs_features=problem.pu_vs_features,
            boundary=problem.boundary,
            parameters=problem.parameters,
        )

        sols = solver.solve(modified, config)
        if not sols:
            continue
        best = min(sols, key=lambda s: s.objective)
        best_solution = best

        unmet = [
            fid for fid, met in best.targets_met.items() if not met
        ]

        history.append({
            "iteration": iteration + 1,
            "unmet_count": len(unmet),
            "spf_values": dict(spf_values),
        })

        if not unmet:
            break

        for fid in unmet:
            spf_values[fid] *= multiplier

    return SPFResult(
        final_spf=spf_values,
        solution=best_solution,
        history=history,
    )
