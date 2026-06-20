"""Shared PuLP MIP backend factory.

Extracted from ``mip_solver.py`` (Phase 21) so multiple subpackages — the
core MIP solver, the zone MIP solver, and the river barrier MIP — build their
PuLP solver instance the same way without importing each other's privates.

PuLP supports several solver backends; pymarxan exposes three by name
(cbc, highs, gurobi) plus ``"auto"``. CBC ships with PuLP — always available.
HiGHS is opt-in via a pip-installable binary (or system PATH). Gurobi requires
``gurobipy`` (the ``pymarxan[gurobi]`` extra).
"""
from __future__ import annotations

import pulp

_MIP_BACKEND_NAMES = ("auto", "cbc", "highs", "gurobi")


def _available_backends() -> dict[str, bool]:
    """Detect which MIP backends can actually run a solve right now.

    Returns ``{backend_name: bool}``. CBC is always True (ships with PuLP);
    HiGHS / Gurobi True iff the respective binary or Python package is
    discoverable. Callers use this to surface 'why-not' status in the UI
    or pick the best backend under ``"auto"``.
    """
    result: dict[str, bool] = {}
    try:
        result["cbc"] = bool(pulp.PULP_CBC_CMD(msg=False).available())
    except Exception:
        result["cbc"] = False
    try:
        result["highs"] = bool(pulp.HiGHS_CMD(msg=False).available())
    except Exception:
        result["highs"] = False
    try:
        result["gurobi"] = bool(pulp.GUROBI_CMD(msg=False).available())
    except Exception:
        result["gurobi"] = False
    return result


def _make_pulp_solver(
    backend: str,
    *,
    time_limit: int,
    gap: float,
    verbose: bool,
):
    """Construct the PuLP solver instance for a given backend name.

    Parameters
    ----------
    backend
        One of ``"auto"`` (pick the fastest available — HiGHS > CBC),
        ``"cbc"``, ``"highs"``, ``"gurobi"``.
    time_limit, gap, verbose
        Forwarded as ``timeLimit`` / ``gapRel`` / ``msg`` kwargs.

    Returns
    -------
    pulp.LpSolver_CMD
        A PuLP solver instance ready to be passed to ``model.solve(...)``.

    Raises
    ------
    ValueError
        For unknown backend names. Use ``_available_backends()`` for the
        runtime availability check.
    """
    if backend not in _MIP_BACKEND_NAMES:
        raise ValueError(
            f"mip_backend must be one of {_MIP_BACKEND_NAMES}, got "
            f"{backend!r}."
        )
    if backend == "auto":
        # Prefer HiGHS when available (5-50× faster than CBC on large MIPs
        # per Phase 21 design rationale). Fall back to CBC, which ships
        # with PuLP and is therefore always present. We skip the CBC
        # availability probe to avoid an extra PULP_CBC_CMD construction
        # under "auto" — keeps call patterns clean for tests that mock the
        # CBC backend.
        try:
            highs_ok = bool(pulp.HiGHS_CMD(msg=False).available())
        except Exception:
            highs_ok = False
        backend = "highs" if highs_ok else "cbc"

    kwargs = {"msg": int(verbose), "timeLimit": time_limit, "gapRel": gap}
    if backend == "cbc":
        return pulp.PULP_CBC_CMD(**kwargs)
    if backend == "highs":
        return pulp.HiGHS_CMD(**kwargs)
    if backend == "gurobi":
        return pulp.GUROBI_CMD(**kwargs)
    # Unreachable: _MIP_BACKEND_NAMES is exhaustive.
    raise AssertionError(f"unreachable backend branch: {backend!r}")
