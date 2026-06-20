"""Project prioritization (oppr-equivalent): prioritize management actions.

Spatially-implicit complement to site selection — fund a budget-constrained set
of actions to maximise weighted expected feature persistence, where projects
(bundles of shared actions) secure features. See ``model.py`` for the problem
definition and the Joseph et al. (2009) / Hanson et al. (2019) references.
"""
from __future__ import annotations

from pymarxan.projects.model import ProjectProblem, ProjectSolution
from pymarxan.projects.optimize import (
    evaluate_projects,
    prioritize_projects_greedy,
    prioritize_projects_mip,
)

__all__ = [
    "ProjectProblem",
    "ProjectSolution",
    "evaluate_projects",
    "prioritize_projects_greedy",
    "prioritize_projects_mip",
]
