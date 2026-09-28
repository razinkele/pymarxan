"""Marxan-with-Zones (MarZone) support: multi-zone problem model, solvers, objective, I/O.

Names are re-exported from their submodules; no re-export shares a submodule's name, so
``importlib.import_module("pymarxan.zones.readers")`` and friends keep working.
"""
from __future__ import annotations

from pymarxan.zones.cache import ZoneHeld
from pymarxan.zones.heuristic import ZoneHeuristicSolver
from pymarxan.zones.iterative_improvement import ZoneIISolver
from pymarxan.zones.mip_solver import ZoneMIPSolver
from pymarxan.zones.model import ZonalProblem
from pymarxan.zones.objective import (
    build_zone_solution,
    check_overall_targets,
    check_zone_targets,
    compute_overall_achieved,
    compute_overall_shortfalls,
    compute_zone_objective,
    compute_zone_shortfall,
    compute_zone_shortfalls,
)
from pymarxan.zones.readers import (
    load_zone_project,
    read_zone_boundary_costs,
    read_zone_contributions,
    read_zone_costs,
    read_zone_targets,
    read_zones,
    resolve_zone_target_types,
)
from pymarxan.zones.solver import ZoneSASolver
from pymarxan.zones.writers import (
    write_zone_boundary_costs,
    write_zone_contributions,
    write_zone_costs,
    write_zone_solution,
    write_zone_summary,
    write_zone_targets,
    write_zones,
)

__all__ = [
    "ZonalProblem",
    "ZoneHeld",
    "ZoneHeuristicSolver",
    "ZoneIISolver",
    "ZoneMIPSolver",
    "ZoneSASolver",
    "build_zone_solution",
    "check_overall_targets",
    "check_zone_targets",
    "compute_overall_achieved",
    "compute_overall_shortfalls",
    "compute_zone_objective",
    "compute_zone_shortfall",
    "compute_zone_shortfalls",
    "load_zone_project",
    "read_zone_boundary_costs",
    "read_zone_contributions",
    "read_zone_costs",
    "read_zone_targets",
    "read_zones",
    "resolve_zone_target_types",
    "write_zone_boundary_costs",
    "write_zone_contributions",
    "write_zone_costs",
    "write_zone_solution",
    "write_zone_summary",
    "write_zone_targets",
    "write_zones",
]
