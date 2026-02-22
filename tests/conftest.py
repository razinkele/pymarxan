"""Shared fixtures and marker registration for pymarxan tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem

SIMPLE_DATA = Path(__file__).parent / "data" / "simple"


@pytest.fixture
def tiny_problem() -> ConservationProblem:
    """Load the 6 PU / 3 feature test fixture from tests/data/simple/."""
    return load_project(SIMPLE_DATA)
