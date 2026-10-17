"""Decision types for ``MIPSolver`` / ``MultiObjectiveMIP`` (Stage 4): validation and the
extraction of ``Solution.selected`` from solver values. numpy / stdlib only, so the solvers
and ``presolve_check`` share it without import cycles (review R6).
"""
from __future__ import annotations

import math
import numbers
from collections.abc import Sequence

import numpy as np

DECISIONS = ("binary", "proportion", "semicontinuous")
ZERO_TOL = 1e-9  # extracted x <= ZERO_TOL is solver dust -> 0 (spec §4)
SNAP_TOL = 1e-6  # semicontinuous: x within SNAP_TOL below lower snaps up to lower


def _bound(name: str, value: object) -> float:
    if (
        not isinstance(value, numbers.Real) or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _validate_decisions(
    decisions: str, upper: float, lower: float,
) -> tuple[str, float, float]:
    """``(decisions, upper, lower)`` validated per the spec §2 table; ``ValueError`` naming
    the offending argument."""
    if decisions not in DECISIONS:
        raise ValueError(
            "decisions must be one of 'binary', 'proportion', 'semicontinuous', "
            f"got {decisions!r}"
        )
    up, lo = _bound("upper", upper), _bound("lower", lower)
    if decisions != "semicontinuous" and lo != 0.0:
        raise ValueError(f"lower only applies to decisions='semicontinuous', got {lower!r}")
    if decisions == "binary":
        if up != 1.0:
            raise ValueError(
                "upper only applies to decisions='proportion' or 'semicontinuous', "
                f"got {upper!r}"
            )
        return decisions, up, lo
    if not 0.0 < up <= 1.0:
        raise ValueError(f"upper must be in (0, 1], got {upper!r}")
    if decisions == "semicontinuous" and not 0.0 < lo <= up:
        raise ValueError(f"lower must be in (0, upper], got {lower!r} (upper {up!r})")
    return decisions, up, lo


def _extract_selected(
    x_values: Sequence[float | None],
    s_values: Sequence[float | None],
    decisions: str,
    upper: float | Sequence[float],
    lower: float,
) -> np.ndarray:
    """PU-ordered ``varValue``s -> ``Solution.selected`` (spec §4, review R5).

    Binary: ``bool(round(v))``. Otherwise float64 clipped to ``[0, upper]`` (a scalar, or
    one bound per PU: 1.0 for locked-in units); ``x <= ZERO_TOL`` -> 0. Semicontinuous:
    ``lower − SNAP_TOL <= x < lower`` -> ``lower``, and ``round(s) == 0`` -> 0 (``s``
    ``None``: the PU has no on/off variable, e.g. it is locked). Nothing else moves, so a
    value is never lowered except dust, ``s = 0``, or ``x`` above its bound.
    """
    if decisions == "binary":
        return np.array([bool(round(v or 0.0)) for v in x_values], dtype=bool)
    ub = np.broadcast_to(np.asarray(upper, dtype=np.float64), (len(x_values),))
    out = np.clip(np.array([v or 0.0 for v in x_values], dtype=np.float64), 0.0, ub)
    out[out <= ZERO_TOL] = 0.0
    if decisions == "semicontinuous":
        out[(out >= lower - SNAP_TOL) & (out < lower)] = lower
        off = np.array([v is not None and round(v) == 0 for v in s_values], dtype=bool)
        out[off] = 0.0
    return out
