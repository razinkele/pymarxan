"""Target tolerance for fractional (proportion) plans (Stage 4 spec §5.3). Stdlib only, so
``solvers.utils`` and ``solvers._mip_objectives`` (pulp + stdlib) can both import it.

An LP optimum leaves every active target row at ``R − O(1e-9·|R|)`` (CBC), so an exact
``held >= R`` flips on solver noise. Binary plans keep the exact comparison. The tolerance
is purely relative: an absolute floor would let a plan holding nothing "meet" any target at
or below that floor (review R1). A target R = 0 is met by every plan.
"""
from __future__ import annotations

TARGET_REL_TOL = 1e-7


def target_tol(v: float) -> float:
    """``TARGET_REL_TOL·|v|``, scale-free across amount magnitudes."""
    return TARGET_REL_TOL * abs(v)
