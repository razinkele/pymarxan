"""Objective types for conservation planning optimization."""

from __future__ import annotations

from pymarxan.objectives.base import Objective, ZonalObjective
from pymarxan.objectives.minset import MinSetObjective
from pymarxan.objectives.max_coverage import MaxCoverageObjective
from pymarxan.objectives.max_utility import MaxUtilityObjective
from pymarxan.objectives.min_shortfall import MinShortfallObjective

__all__ = [
    "Objective",
    "ZonalObjective",
    "MinSetObjective",
    "MaxCoverageObjective",
    "MaxUtilityObjective",
    "MinShortfallObjective",
]
