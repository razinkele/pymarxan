"""raptr-style space/adequacy targets (Hanson et al. 2018, doi:10.1111/2041-210x.12862)."""
from __future__ import annotations

from pymarxan.adequacy.model import SpaceSpec, pu_attribute_space
from pymarxan.adequacy.space import compute_space_held

__all__ = ["SpaceSpec", "compute_space_held", "pu_attribute_space"]
