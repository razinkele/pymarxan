"""Classic Marxan output file controller.

Manages which output files to write based on Marxan output control
parameters (SAVERUN, SAVEBEST, SAVESUMM, SAVETARGMET, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pymarxan.models.problem import ConservationProblem
    from pymarxan.solvers.base import Solution

# Marxan SAVE* parameter values:
#   0 = don't save
#   1 = save only if targets met
#   2 = save only if targets not met
#   3 = always save (default)
_SAVE_ALWAYS = 3
_SAVE_IF_MET = 1
_SAVE_IF_UNMET = 2
_SAVE_NEVER = 0


def _should_save(param_value: int, targets_met: bool) -> bool:
    """Decide whether to save based on SAVE* parameter and target status."""
    if param_value == _SAVE_ALWAYS:
        return True
    if param_value == _SAVE_NEVER:
        return False
    if param_value == _SAVE_IF_MET:
        return targets_met
    if param_value == _SAVE_IF_UNMET:
        return not targets_met
    return True


@dataclass
class OutputController:
    """Controls which Marxan output files are written.

    Parameters
    ----------
    params : dict
        Marxan parameters dict (from ``problem.parameters``).
    scenname : str
        Scenario name prefix for output files.
    """

    params: dict = field(default_factory=dict)
    scenname: str = "output"

    def __post_init__(self):
        self.scenname = str(
            self.params.get("SCENNAME", self.scenname)
        )

    def _param(self, key: str, default: int = 3) -> int:
        return int(self.params.get(key, default))

    def should_save_run(self, targets_met: bool = False) -> bool:
        """Whether to save per-run solution files (SAVERUN)."""
        return _should_save(self._param("SAVERUN"), targets_met)

    def should_save_best(self, targets_met: bool = False) -> bool:
        """Whether to save best solution file (SAVEBEST)."""
        return _should_save(self._param("SAVEBEST"), targets_met)

    def should_save_summary(self, targets_met: bool = False) -> bool:
        """Whether to save run summary file (SAVESUMM)."""
        return _should_save(self._param("SAVESUMM"), targets_met)

    def should_save_scenario(self, targets_met: bool = False) -> bool:
        """Whether to save scenario file (SAVESCEN)."""
        return _should_save(self._param("SAVESCEN"), targets_met)

    def should_save_targmet(self, targets_met: bool = False) -> bool:
        """Whether to save targets-met file (SAVETARGMET)."""
        return _should_save(self._param("SAVETARGMET"), targets_met)

    def should_save_solution_matrix(
        self, targets_met: bool = False,
    ) -> bool:
        """Whether to save solution matrix (SAVESOLUTIONSMATRIX)."""
        return _should_save(
            self._param("SAVESOLUTIONSMATRIX", 3), targets_met,
        )

    def write_run(
        self,
        problem: ConservationProblem,
        solution: Solution,
        run_number: int,
        output_dir: Path,
    ) -> None:
        """Write a per-run solution file (output_r001.csv)."""
        pu_ids = problem.planning_units["id"].tolist()
        rows = [
            {"Planning_Unit": pid, "Solution": int(solution.selected[i])}
            for i, pid in enumerate(pu_ids)
        ]
        fname = f"{self.scenname}_r{run_number:03d}.csv"
        pd.DataFrame(rows).to_csv(output_dir / fname, index=False)

    def write_best(
        self,
        problem: ConservationProblem,
        solution: Solution,
        output_dir: Path,
    ) -> None:
        """Write the best solution file (output_best.csv)."""
        pu_ids = problem.planning_units["id"].tolist()
        rows = [
            {"Planning_Unit": pid, "Solution": int(solution.selected[i])}
            for i, pid in enumerate(pu_ids)
        ]
        fname = f"{self.scenname}_best.csv"
        pd.DataFrame(rows).to_csv(output_dir / fname, index=False)

    def write_solution_matrix(
        self,
        problem: ConservationProblem,
        solutions: list[Solution],
        output_dir: Path,
    ) -> None:
        """Write the solution matrix (n_pu × n_runs)."""
        pu_ids = problem.planning_units["id"].tolist()
        matrix = np.column_stack(
            [sol.selected.astype(int) for sol in solutions]
        )
        cols = {f"S{i + 1}": matrix[:, i] for i in range(len(solutions))}
        df = pd.DataFrame({"Planning_Unit": pu_ids, **cols})
        fname = f"{self.scenname}_solutionsmatrix.csv"
        df.to_csv(output_dir / fname, index=False)

    def write_outputs(
        self,
        problem: ConservationProblem,
        solutions: list[Solution],
        output_dir: Path,
    ) -> None:
        """Write all enabled output files.

        Dispatches to existing write functions and new per-run files
        based on SAVE* parameters.
        """
        from pymarxan.io.writers import (
            write_mvbest,
            write_ssoln,
            write_sum,
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not solutions:
            return

        best = min(solutions, key=lambda s: s.objective)
        best_met = best.all_targets_met

        # Summary (output_sum.csv)
        if self.should_save_summary(best_met):
            write_sum(
                solutions, output_dir / f"{self.scenname}_sum.csv",
            )

        # Best solution (output_best.csv)
        if self.should_save_best(best_met):
            self.write_best(problem, best, output_dir)

        # Missing values for best (output_mvbest.csv)
        if self.should_save_best(best_met):
            write_mvbest(
                problem, best,
                output_dir / f"{self.scenname}_mvbest.csv",
            )

        # Summed solution (output_ssoln.csv)
        if self.should_save_summary(best_met):
            write_ssoln(
                problem, solutions,
                output_dir / f"{self.scenname}_ssoln.csv",
            )

        # Per-run files
        if self.should_save_run(best_met):
            for i, sol in enumerate(solutions, start=1):
                self.write_run(problem, sol, i, output_dir)

        # Targets met
        if self.should_save_targmet(best_met):
            self._write_targmet(
                problem, solutions, output_dir,
            )

        # Solution matrix
        if self.should_save_solution_matrix(best_met):
            self.write_solution_matrix(
                problem, solutions, output_dir,
            )

        # Scenario file
        if self.should_save_scenario(best_met):
            self._write_scenario(
                problem, solutions, output_dir,
            )

    def _write_targmet(
        self,
        problem: ConservationProblem,
        solutions: list[Solution],
        output_dir: Path,
    ) -> None:
        """Write per-run targets-met file."""
        feat_ids = problem.features["id"].values
        rows = []
        for i, sol in enumerate(solutions, start=1):
            for fid in feat_ids:
                # output.py is single-zone Marxan; targets_met is dict[int, bool]
                met = sol.targets_met.get(int(fid), False)  # type: ignore[call-overload]
                rows.append({
                    "Run": i,
                    "Feature_ID": int(fid),
                    "Target_Met": met,
                })
        fname = f"{self.scenname}_targmet.csv"
        pd.DataFrame(rows).to_csv(output_dir / fname, index=False)

    def _write_scenario(
        self,
        problem: ConservationProblem,
        solutions: list[Solution],
        output_dir: Path,
    ) -> None:
        """Write scenario summary file."""
        best = min(solutions, key=lambda s: s.objective)
        fname = f"{self.scenname}_sen.csv"
        data = {
            "Number_of_Runs": [len(solutions)],
            "Best_Score": [best.objective],
            "Best_Cost": [best.cost],
            "Best_Planning_Units": [best.n_selected],
            "Best_Boundary": [best.boundary],
            "Best_Penalty": [best.penalty],
            "Best_Shortfall": [best.shortfall],
        }
        pd.DataFrame(data).to_csv(output_dir / fname, index=False)
