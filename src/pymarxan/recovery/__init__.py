"""Nature recovery planning: protect, restore actively, leave to passive recovery or take no
action, as a Marxan-with-Zones problem with expected restored amounts (spec 2026-11-03).

``build_recovery_problem(RecoveryInputs) -> (ZonalProblem, RecoveryMapping)``; every zone
solver works on the result.
"""
from __future__ import annotations

from pymarxan.recovery.builder import (
    ACTION_NAMES,
    SPF_SCALE,
    ZONE_NO_ACTION,
    ZONE_PROTECT,
    ZONE_RESTORE_ACTIVE,
    ZONE_RESTORE_PASSIVE,
    RecoveryCheck,
    RecoveryInputError,
    RecoveryInputWarning,
    RecoveryMapping,
    build_recovery_problem,
    check_recovery_inputs,
    existing_feature_id,
    potential_feature_id,
    scaled_spf,
)
from pymarxan.recovery.inputs import PassiveAction, RecoveryInputs
from pymarxan.recovery.solve import (
    RECOVERY_MIP_SIZE_WARNING,
    RecoverySizeWarning,
    solve_recovery,
)

__all__ = [
    "ACTION_NAMES",
    "RECOVERY_MIP_SIZE_WARNING",
    "SPF_SCALE",
    "ZONE_NO_ACTION",
    "ZONE_PROTECT",
    "ZONE_RESTORE_ACTIVE",
    "ZONE_RESTORE_PASSIVE",
    "PassiveAction",
    "RecoveryCheck",
    "RecoveryInputError",
    "RecoveryInputWarning",
    "RecoveryInputs",
    "RecoveryMapping",
    "RecoverySizeWarning",
    "build_recovery_problem",
    "check_recovery_inputs",
    "existing_feature_id",
    "potential_feature_id",
    "scaled_spf",
    "solve_recovery",
]
