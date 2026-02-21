import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from pymarxan.io.readers import load_project
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.marxan_binary import MarxanBinarySolver

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"

class TestMarxanBinarySolver:
    def test_solver_name(self):
        solver = MarxanBinarySolver()
        assert solver.name() == "Marxan (C++ binary)"

    def test_does_not_support_zones(self):
        solver = MarxanBinarySolver()
        assert not solver.supports_zones()

    def test_available_when_binary_exists(self):
        with patch("shutil.which", return_value="/usr/bin/marxan"):
            solver = MarxanBinarySolver()
            assert solver.available()

    def test_not_available_when_binary_missing(self):
        with patch("shutil.which", return_value=None):
            solver = MarxanBinarySolver()
            assert not solver.available()

    def test_custom_binary_path(self):
        solver = MarxanBinarySolver(binary_path="/custom/path/marxan")
        assert solver._binary_path == "/custom/path/marxan"

    def test_solve_raises_when_unavailable(self):
        with patch("shutil.which", return_value=None):
            solver = MarxanBinarySolver()
            problem = load_project(DATA_DIR)
            with pytest.raises(RuntimeError, match="Marxan binary not found"):
                solver.solve(problem, SolverConfig(num_solutions=1))

    def test_parse_output_csv(self):
        solver = MarxanBinarySolver()
        csv_content = "planning_unit,solution\n1,1\n2,0\n3,1\n4,0\n5,1\n6,0\n"
        selected = solver._parse_solution_csv(csv_content, [1, 2, 3, 4, 5, 6])
        expected = np.array([True, False, True, False, True, False])
        np.testing.assert_array_equal(selected, expected)

    def test_parse_best_csv(self):
        solver = MarxanBinarySolver()
        csv_content = "planning_unit,solution\n1,0\n2,1\n3,1\n4,1\n5,0\n6,0\n"
        selected = solver._parse_solution_csv(csv_content, [1, 2, 3, 4, 5, 6])
        assert selected.sum() == 3
