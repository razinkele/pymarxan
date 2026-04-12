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
from pymarxan.solvers.utils import build_solution

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
            run_params = dict(problem.parameters)
            run_params["NUMREPS"] = config.num_solutions
            if config.seed is not None:
                run_params["RANDSEED"] = config.seed
            run_params["VERBOSITY"] = 0 if not config.verbose else 2
            run_params["SAVERUN"] = 3  # CSV output
            run_params["SAVEBEST"] = 3
            run_params["OUTPUTDIR"] = "output"
            run_params["SCENNAME"] = "output"
            problem_copy = problem.copy_with(
                planning_units=problem.planning_units.copy(),
                features=problem.features.copy(),
                pu_vs_features=problem.pu_vs_features.copy(),
                boundary=problem.boundary.copy() if problem.boundary is not None else None,
                parameters=run_params,
            )

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
                    sol = build_solution(
                        problem, selected, blm,
                        metadata={"solver": "Marxan (C++ binary)"},
                    )
                    solutions.append(sol)

            # If no individual solutions found, try best solution
            if not solutions:
                best_path = output_dir / "output_best.csv"
                if best_path.exists():
                    csv_content = best_path.read_text()
                    selected = self._parse_solution_csv(csv_content, pu_ids)
                    sol = build_solution(
                        problem, selected, blm,
                        metadata={"solver": "Marxan (C++ binary)"},
                    )
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
