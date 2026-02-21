"""Marxan C++ binary wrapper solver."""
from __future__ import annotations

import csv
import io
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from pymarxan.io.writers import save_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, Solver, SolverConfig

_BINARY_NAMES = ["Marxan_x64", "marxan", "Marxan", "marxan_x64"]


class MarxanBinarySolver(Solver):
    """Solver that wraps the Marxan C++ binary executable."""

    def __init__(self, binary_path: str | None = None) -> None:
        self._binary_path = binary_path

    def name(self) -> str:
        return "Marxan (C++ binary)"

    def supports_zones(self) -> bool:
        return False

    def available(self) -> bool:
        if self._binary_path is not None:
            return True
        return self._find_binary() is not None

    def solve(
        self, problem: ConservationProblem, config: SolverConfig | None = None
    ) -> list[Solution]:
        if config is None:
            config = SolverConfig()

        binary = self._binary_path or self._find_binary()
        if binary is None:
            raise RuntimeError(
                "Marxan binary not found. Install Marxan or provide binary_path."
            )

        blm = float(problem.parameters.get("BLM", 0.0))

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Override parameters for this run
            problem_copy = ConservationProblem(
                planning_units=problem.planning_units.copy(),
                features=problem.features.copy(),
                pu_vs_features=problem.pu_vs_features.copy(),
                boundary=problem.boundary.copy() if problem.boundary is not None else None,
                parameters=dict(problem.parameters),
            )
            problem_copy.parameters["NUMREPS"] = config.num_solutions
            if config.seed is not None:
                problem_copy.parameters["RANDSEED"] = config.seed
            problem_copy.parameters["VERBOSITY"] = 0 if not config.verbose else 2
            problem_copy.parameters["SAVERUN"] = 3  # CSV output
            problem_copy.parameters["SAVEBEST"] = 3
            problem_copy.parameters["OUTPUTDIR"] = "output"
            problem_copy.parameters["SCENNAME"] = "output"

            # Write project files
            save_project(problem_copy, tmp_path)

            # Create output directory
            output_dir = tmp_path / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Run Marxan binary
            input_dat_path = str(tmp_path / "input.dat")
            try:
                result = subprocess.run(
                    [binary, "-s", input_dat_path],
                    capture_output=True,
                    timeout=600,
                    cwd=str(tmp_path),
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("Marxan binary timed out after 600 seconds") from exc
            except FileNotFoundError as exc:
                raise RuntimeError(f"Marxan binary not found at: {binary}") from exc

            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="replace")
                raise RuntimeError(f"Marxan binary failed (rc={result.returncode}): {stderr}")

            # Parse output solutions
            pu_ids = problem.planning_units["id"].tolist()
            solutions = []

            for i in range(1, config.num_solutions + 1):
                csv_path = output_dir / f"output_r{i:04d}.csv"
                if csv_path.exists():
                    csv_content = csv_path.read_text()
                    selected = self._parse_solution_csv(csv_content, pu_ids)
                    sol = self._build_solution(problem, selected, blm)
                    solutions.append(sol)

            # If no individual solutions found, try best solution
            if not solutions:
                best_path = output_dir / "output_best.csv"
                if best_path.exists():
                    csv_content = best_path.read_text()
                    selected = self._parse_solution_csv(csv_content, pu_ids)
                    sol = self._build_solution(problem, selected, blm)
                    solutions.append(sol)

            return solutions

    @staticmethod
    def _find_binary() -> str | None:
        """Search for the Marxan binary on PATH."""
        for name in _BINARY_NAMES:
            path = shutil.which(name)
            if path is not None:
                return path
        return None

    @staticmethod
    def _parse_solution_csv(csv_content: str, pu_ids: list[int]) -> np.ndarray:
        """Parse a Marxan output CSV to a boolean selection array.

        Parameters
        ----------
        csv_content : str
            CSV text with columns ``planning_unit`` and ``solution``.
        pu_ids : list[int]
            Ordered list of planning unit IDs matching the problem.

        Returns
        -------
        np.ndarray
            Boolean array aligned with pu_ids.
        """
        reader = csv.DictReader(io.StringIO(csv_content))
        solution_map: dict[int, bool] = {}
        for row in reader:
            pu = int(row["planning_unit"])
            sol = int(row["solution"])
            solution_map[pu] = sol == 1

        selected = np.array(
            [solution_map.get(pid, False) for pid in pu_ids], dtype=bool
        )
        return selected

    @staticmethod
    def _build_solution(
        problem: ConservationProblem,
        selected: np.ndarray,
        blm: float,
    ) -> Solution:
        """Build a Solution object from a selection array.

        Parameters
        ----------
        problem : ConservationProblem
            The conservation problem.
        selected : np.ndarray
            Boolean selection array.
        blm : float
            Boundary length modifier.

        Returns
        -------
        Solution
            Computed solution with cost, boundary, and target information.
        """
        pu_ids = problem.planning_units["id"].tolist()
        pu_index = {pid: i for i, pid in enumerate(pu_ids)}

        # Compute cost
        costs = np.asarray(problem.planning_units["cost"].values)
        total_cost = float(np.sum(costs[selected]))

        # Compute boundary
        total_boundary = 0.0
        if problem.boundary is not None:
            for _, row in problem.boundary.iterrows():
                id1 = int(row["id1"])
                id2 = int(row["id2"])
                bval = float(row["boundary"])

                if id1 == id2:
                    idx = pu_index.get(id1)
                    if idx is not None and selected[idx]:
                        total_boundary += bval
                else:
                    idx1 = pu_index.get(id1)
                    idx2 = pu_index.get(id2)
                    if idx1 is not None and idx2 is not None:
                        if selected[idx1] != selected[idx2]:
                            total_boundary += bval

        # Check targets
        targets_met: dict[int, bool] = {}
        for _, feat_row in problem.features.iterrows():
            fid = int(feat_row["id"])
            target = float(feat_row["target"])
            feat_data = problem.pu_vs_features[
                problem.pu_vs_features["species"] == fid
            ]
            total_amount = 0.0
            for _, r in feat_data.iterrows():
                pu_id = int(r["pu"])
                idx = pu_index.get(pu_id)
                if idx is not None and selected[idx]:
                    total_amount += float(r["amount"])
            targets_met[fid] = total_amount >= target

        objective = total_cost + blm * total_boundary

        return Solution(
            selected=selected,
            cost=total_cost,
            boundary=total_boundary,
            objective=objective,
            targets_met=targets_met,
            metadata={"solver": "Marxan (C++ binary)"},
        )
