"""Tests for MarxanBinarySolver — parsing and error paths."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pymarxan.solvers.marxan_binary import MarxanBinarySolver


class TestParseCSV:
    def test_parse_correct_csv(self):
        csv_content = "planning_unit,solution\n1,1\n2,0\n3,1\n"
        pu_ids = [1, 2, 3]
        result = MarxanBinarySolver._parse_solution_csv(csv_content, pu_ids)
        np.testing.assert_array_equal(result, [True, False, True])

    def test_parse_missing_pu(self):
        """PU not in CSV should default to False."""
        csv_content = "planning_unit,solution\n1,1\n3,1\n"
        pu_ids = [1, 2, 3]
        result = MarxanBinarySolver._parse_solution_csv(csv_content, pu_ids)
        np.testing.assert_array_equal(result, [True, False, True])

    def test_parse_empty_csv(self):
        csv_content = "planning_unit,solution\n"
        pu_ids = [1, 2]
        result = MarxanBinarySolver._parse_solution_csv(csv_content, pu_ids)
        np.testing.assert_array_equal(result, [False, False])


class TestAvailability:
    def test_not_available_without_binary(self):
        solver = MarxanBinarySolver(binary_path=None)
        assert isinstance(solver.available(), bool)

    def test_available_with_explicit_path(self):
        solver = MarxanBinarySolver(binary_path="/usr/bin/true")
        assert solver.available() is True

    def test_name(self):
        solver = MarxanBinarySolver()
        assert solver.name() == "Marxan (C++ binary)"

    def test_supports_zones(self):
        solver = MarxanBinarySolver()
        assert solver.supports_zones() is False


class TestSolveErrors:
    def test_missing_binary_raises(self):
        solver = MarxanBinarySolver(binary_path="/nonexistent/marxan")
        from pymarxan.io.readers import load_project
        DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"
        problem = load_project(DATA_DIR)
        with pytest.raises(RuntimeError, match="not found"):
            solver.solve(problem)
