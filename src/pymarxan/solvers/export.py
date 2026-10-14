"""Export a MIP solver's model to LP / MPS (prioritizr ``write_problem``).

Deliberate exception to the ``read_*`` / ``write_*`` pairing convention: nothing reads a
model back into a problem. Time limit and gap are solver settings, not part of the model;
no-good cuts are included only when that ``config`` is passed. ``solver`` is a pymarxan
Solver instance (prioritizr's ``solver`` picks a backend).
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pymarxan.models.problem import ConservationProblem
    from pymarxan.solvers.base import Solver, SolverConfig

_WRITERS = {".lp": "writeLP", ".mps": "writeMPS"}


def write_problem(
    problem: ConservationProblem,
    path: str | Path,
    *,
    solver: Solver | None = None,
    config: SolverConfig | None = None,
) -> Path:
    """Build ``solver``'s MIP for ``problem`` and write it to ``path`` (.lp or .mps)."""
    from pymarxan.zones.model import ZonalProblem

    out = Path(path)
    suffix = out.suffix.lower()
    if suffix not in _WRITERS:
        raise ValueError(f"write_problem: path must end in .lp or .mps, got {out.name!r}")
    zonal = isinstance(problem, ZonalProblem)
    if solver is None:
        if zonal:
            from pymarxan.zones.mip_solver import ZoneMIPSolver
            solver = ZoneMIPSolver()
        else:
            from pymarxan.solvers.mip_solver import MIPSolver
            solver = MIPSolver()
    build: Any = getattr(solver, "build_model", None)
    if build is None:
        raise TypeError(
            f"write_problem: {type(solver).__name__} has no build_model; use MIPSolver or "
            "ZoneMIPSolver"
        )
    if solver.supports_zones() != zonal:
        detail = (
            "a ZonalProblem needs ZoneMIPSolver" if zonal
            else "ZonalProblem solvers need a ZonalProblem"
        )
        raise TypeError(
            f"write_problem: {type(solver).__name__} does not match the problem type ({detail})"
        )
    model = build(problem, config)
    constant = float(model.objective.constant) if model.objective is not None else 0.0
    if constant != 0.0:
        warnings.warn(
            f"write_problem: the objective constant {constant} is dropped by LP/MPS writers",
            UserWarning,
            stacklevel=2,
        )
    getattr(model, _WRITERS[suffix])(str(out))
    return out
