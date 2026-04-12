"""Constraint framework for conservation planning problems."""

from pymarxan.constraints.base import (
    Constraint,
    ConstraintResult,
    IncrementalConstraint,
    IncrementalZonalConstraint,
    ZonalConstraint,
)
from pymarxan.constraints.contiguity import ContiguityConstraint
from pymarxan.constraints.feature_contiguity import FeatureContiguityConstraint

__all__ = [
    "Constraint",
    "ConstraintResult",
    "ContiguityConstraint",
    "FeatureContiguityConstraint",
    "IncrementalConstraint",
    "IncrementalZonalConstraint",
    "ZonalConstraint",
]
