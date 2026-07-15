"""raptr-style space/adequacy targets (Hanson et al. 2018, doi:10.1111/2041-210x.12862)."""
from __future__ import annotations

from pymarxan.adequacy.model import SpaceSpec, pu_attribute_space
from pymarxan.adequacy.space import compute_space_held, evaluate_solution_space

__all__ = [
    "SpaceSpec",
    "compute_space_held",
    "evaluate_solution_space",
    "pu_attribute_space",
]
