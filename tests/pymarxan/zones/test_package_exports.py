"""Issue #3: pymarxan.zones and pymarxan.analysis re-export their public API."""
from __future__ import annotations

import importlib

import pytest

ZONES_EXPORTS = {
    "ZonalProblem": "pymarxan.zones.model",
    "ZoneMIPSolver": "pymarxan.zones.mip_solver",
    "ZoneSASolver": "pymarxan.zones.solver",
    "ZoneHeuristicSolver": "pymarxan.zones.heuristic",
    "ZoneIISolver": "pymarxan.zones.iterative_improvement",
    "check_zone_targets": "pymarxan.zones.objective",
    "compute_zone_objective": "pymarxan.zones.objective",
    "compute_zone_shortfall": "pymarxan.zones.objective",
    "load_zone_project": "pymarxan.zones.readers",
    "read_zones": "pymarxan.zones.readers",
    "write_zone_solution": "pymarxan.zones.writers",
    "ZoneHeld": "pymarxan.zones.cache",
    "build_zone_solution": "pymarxan.zones.objective",
    "check_overall_targets": "pymarxan.zones.objective",
    "compute_overall_achieved": "pymarxan.zones.objective",
    "compute_overall_shortfalls": "pymarxan.zones.objective",
    "compute_zone_shortfalls": "pymarxan.zones.objective",
    "resolve_zone_target_types": "pymarxan.zones.readers",
}

ANALYSIS_EXPORTS = {
    "compute_equity": "pymarxan.analysis.equity",
    "compute_ferrier_importance": "pymarxan.analysis.ferrier_importance",
    "compute_gap_analysis": "pymarxan.analysis.gap_analysis",
    "compute_irreplaceability": "pymarxan.analysis.irreplaceability",
    "compute_rank_importance": "pymarxan.analysis.rank_importance",
    "compute_replacement_cost": "pymarxan.analysis.replacement_cost",
    "compute_representation": "pymarxan.analysis.representation",
    "compute_selection_frequency": "pymarxan.analysis.selection_freq",
    "compute_solution_clusters": "pymarxan.analysis.posthoc_clusters",
    "generate_portfolio_cuts": "pymarxan.analysis.portfolio_cuts",
    "minimax_regret": "pymarxan.analysis.robustness",
    "evaluate_plans_across_scenarios": "pymarxan.analysis.robustness",
    "selection_frequency": "pymarxan.analysis.portfolio",
    "best_solution": "pymarxan.analysis.portfolio",
    "gap_filter": "pymarxan.analysis.portfolio",
    "solution_diversity": "pymarxan.analysis.portfolio",
    "summary_statistics": "pymarxan.analysis.portfolio",
}


@pytest.mark.parametrize("name,source", sorted(ZONES_EXPORTS.items()))
def test_zones_reexports_are_the_source_objects(name: str, source: str) -> None:
    import pymarxan.zones as pkg

    assert name in pkg.__all__
    assert getattr(pkg, name) is getattr(importlib.import_module(source), name)


@pytest.mark.parametrize("name,source", sorted(ANALYSIS_EXPORTS.items()))
def test_analysis_reexports_are_the_source_objects(name: str, source: str) -> None:
    import pymarxan.analysis as pkg

    assert name in pkg.__all__
    assert getattr(pkg, name) is getattr(importlib.import_module(source), name)


def test_reexports_do_not_shadow_submodules() -> None:
    """A re-exported name equal to a submodule filename would break importlib users."""
    import pymarxan.analysis as analysis
    import pymarxan.zones as zones

    for pkg in (zones, analysis):
        for sub in ("model", "solver", "mip_solver", "objective", "readers", "writers",
                    "portfolio", "robustness", "selection_freq"):
            assert sub not in pkg.__all__
