"""Automatic target-setting rules.

Set each feature's representation target by a rule rather than by hand.
Each rule returns a ``{feature_id: target_amount}`` mapping in the feature's
native units; :func:`apply_targets` writes it onto a problem's features.
Features a rule does not return keep their existing target, so methods are
mixed across feature groups with ``features=`` plus repeated
:func:`apply_targets` calls.

Area-based methods take a required ``km2_per_unit`` (a factor or a
``{feature_id: factor}`` mapping): feature totals are converted to km², the
published rule is applied, and the target is converted back. There is no
default unit.

Mirrors prioritizr 9.0.1's target methods (Hanson et al. (2024),
*Conservation Biology*, https://doi.org/10.1111/cobi.14376); deviations
from prioritizr are listed in the CHANGELOG for v0.40.0.
"""
from __future__ import annotations

from pymarxan.targets._basic import (
    apply_targets,
    group_targets,
    loglinear_targets,
    relative_targets,
)
from pymarxan.targets._interp import (
    interp_absolute_targets,
    interp_area_targets,
    polak_targets,
    rodrigues_targets,
    sreekar_targets,
    watson_targets,
)
from pymarxan.targets._redlist import (
    jung_targets,
    rl_ecosystem_targets,
    rl_species_targets,
    ward_targets,
)
from pymarxan.targets._units import PerFeatureFloat, PerFeatureStr

__all__ = [
    "PerFeatureFloat",
    "PerFeatureStr",
    "apply_targets",
    "group_targets",
    "interp_absolute_targets",
    "interp_area_targets",
    "jung_targets",
    "loglinear_targets",
    "polak_targets",
    "relative_targets",
    "rl_ecosystem_targets",
    "rl_species_targets",
    "rodrigues_targets",
    "sreekar_targets",
    "ward_targets",
    "watson_targets",
]
