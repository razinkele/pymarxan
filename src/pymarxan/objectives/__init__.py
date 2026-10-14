"""Objective types for conservation planning optimization."""

from __future__ import annotations

import sys
import warnings

from pymarxan.objectives.base import Objective, ZonalObjective
from pymarxan.objectives.max_coverage import MaxCoverageObjective
from pymarxan.objectives.max_utility import MaxUtilityObjective
from pymarxan.objectives.min_shortfall import MinShortfallObjective
from pymarxan.objectives.minset import MinSetObjective

_DEPRECATION = (
    "pymarxan.objectives is deprecated and will be removed in v0.42; use "
    "MIPSolver(objective=...) / ZoneMIPSolver(objective=...)"
)
if sys.version_info >= (3, 12):
    import os

    # Attribute to the importer's line (stacklevel would land in importlib frames).
    warnings.warn(
        _DEPRECATION,
        DeprecationWarning,
        skip_file_prefixes=(os.path.dirname(__file__), "<frozen importlib"),
    )
else:  # 3.11: skip_file_prefixes does not exist; the importlib frame is acceptable
    warnings.warn(_DEPRECATION, DeprecationWarning, stacklevel=2)

__all__ = [
    "Objective",
    "ZonalObjective",
    "MinSetObjective",
    "MaxCoverageObjective",
    "MaxUtilityObjective",
    "MinShortfallObjective",
]
