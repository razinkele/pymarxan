"""Post-hoc analysis of solutions: importance, representation, robustness, portfolios.

Names are re-exported from their submodules; no re-export shares a submodule's name.
"""
from __future__ import annotations

from pymarxan.analysis.equity import compute_equity
from pymarxan.analysis.ferrier_importance import compute_ferrier_importance
from pymarxan.analysis.gap_analysis import compute_gap_analysis
from pymarxan.analysis.irreplaceability import compute_irreplaceability
from pymarxan.analysis.portfolio import (
    best_solution,
    gap_filter,
    selection_frequency,
    solution_diversity,
    summary_statistics,
)
from pymarxan.analysis.portfolio_cuts import generate_portfolio_cuts
from pymarxan.analysis.posthoc_clusters import compute_solution_clusters
from pymarxan.analysis.rank_importance import compute_rank_importance
from pymarxan.analysis.replacement_cost import compute_replacement_cost
from pymarxan.analysis.representation import compute_representation
from pymarxan.analysis.robustness import evaluate_plans_across_scenarios, minimax_regret
from pymarxan.analysis.selection_freq import compute_selection_frequency

__all__ = [
    "best_solution",
    "compute_equity",
    "compute_ferrier_importance",
    "compute_gap_analysis",
    "compute_irreplaceability",
    "compute_rank_importance",
    "compute_replacement_cost",
    "compute_representation",
    "compute_selection_frequency",
    "compute_solution_clusters",
    "evaluate_plans_across_scenarios",
    "gap_filter",
    "generate_portfolio_cuts",
    "minimax_regret",
    "selection_frequency",
    "solution_diversity",
    "summary_statistics",
]
