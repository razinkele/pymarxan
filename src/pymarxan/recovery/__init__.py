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
from pymarxan.recovery.evaluate import (
    DEFAULT_MULTIPLIERS,
    NO_TIME_LAG,
    RECOVERY_CAVEAT,
    RecoveryResult,
    evaluate_recovery,
    recovery_caveat,
    success_sensitivity,
)
from pymarxan.recovery.inputs import PassiveAction, RecoveryInputs
from pymarxan.recovery.io import (
    RecoveryProject,
    load_recovery_project,
    read_recovery_inputs,
    save_recovery_project,
    write_recovery_inputs,
)
from pymarxan.recovery.solve import (
    RECOVERY_MIP_SIZE_WARNING,
    RecoverySizeWarning,
    solve_recovery,
)

__all__ = [
    "ACTION_NAMES",
    "DEFAULT_MULTIPLIERS",
    "NO_TIME_LAG",
    "PassiveAction",
    "RECOVERY_CAVEAT",
    "RECOVERY_MIP_SIZE_WARNING",
    "RecoveryCheck",
    "RecoveryInputError",
    "RecoveryInputWarning",
    "RecoveryInputs",
    "RecoveryMapping",
    "RecoveryProject",
    "RecoveryResult",
    "RecoverySizeWarning",
    "SPF_SCALE",
    "ZONE_NO_ACTION",
    "ZONE_PROTECT",
    "ZONE_RESTORE_ACTIVE",
    "ZONE_RESTORE_PASSIVE",
    "build_recovery_problem",
    "check_recovery_inputs",
    "evaluate_recovery",
    "existing_feature_id",
    "load_recovery_project",
    "potential_feature_id",
    "read_recovery_inputs",
    "recovery_caveat",
    "save_recovery_project",
    "scaled_spf",
    "solve_recovery",
    "success_sensitivity",
    "write_recovery_inputs",
]
