"""Project-prioritization data model (oppr-equivalent).

The "Project Prioritization Protocol" (Joseph, Maloney & Possingham 2009,
doi:10.1111/j.1523-1739.2008.01124.x; exact form: Hanson et al. 2019,
doi:10.1111/2041-210X.13264) prioritizes *management actions* rather than
sites. Fund a set of actions under a budget; a **project** is completed iff all
its required actions are funded; each completed project secures a set of
**features** with a persistence probability. Maximise the weighted expected
persistence ``Σ_f w_f · max_{funded project p benefiting f} persistence_{f,p}``.

Shared actions across projects make simple cost-effectiveness ranking
suboptimal — exactly the complementarity the exact MIP captures (and the
project's "a heuristic below the exact optimum is a bug" anchor applies).

This is *spatially implicit*: it complements, rather than replaces, the
site-selection ``ConservationProblem``. A free "baseline" project (no actions)
gives each feature its do-nothing persistence; features with no funded project
default to persistence 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ProjectProblem:
    """A project-prioritization instance.

    Parameters
    ----------
    features
        Columns: ``id``, optional ``weight`` (defaults to 1.0).
    actions
        Columns: ``id``, ``cost``.
    projects
        Columns: ``id`` (optional ``name``). A project with no rows in
        ``project_actions`` is a free, always-available baseline.
    project_actions
        Long form: ``project``, ``action`` — the actions each project requires.
    project_features
        Long form: ``project``, ``feature``, ``persistence`` — the persistence
        probability a completed project confers on a feature.
    budget
        Total cost cap on the funded action set. ``None`` means unconstrained.
    """

    features: pd.DataFrame
    actions: pd.DataFrame
    projects: pd.DataFrame
    project_actions: pd.DataFrame
    project_features: pd.DataFrame
    budget: float | None = None

    def __post_init__(self) -> None:
        if self.budget is not None and self.budget < 0:
            raise ValueError("budget must be non-negative")
        for col in ("id",):
            for name, df in (
                ("features", self.features),
                ("actions", self.actions),
                ("projects", self.projects),
            ):
                if col not in df.columns:
                    raise ValueError(f"{name} must have an '{col}' column")
        if "cost" not in self.actions.columns:
            raise ValueError("actions must have a 'cost' column")
        for c in ("project", "feature", "persistence"):
            if c not in self.project_features.columns:
                raise ValueError(f"project_features must have a '{c}' column")
        for c in ("project", "action"):
            if c not in self.project_actions.columns:
                raise ValueError(f"project_actions must have a '{c}' column")
        ids = self.actions["id"].astype(int)
        if len(ids.unique()) != len(ids):
            raise ValueError("action ids must be unique")

    # --- derived lookups -------------------------------------------------

    def action_cost(self) -> dict[int, float]:
        return {
            int(a): float(c)
            for a, c in zip(self.actions["id"], self.actions["cost"])
        }

    def feature_weight(self) -> dict[int, float]:
        if "weight" in self.features.columns:
            return {
                int(f): float(w)
                for f, w in zip(self.features["id"], self.features["weight"])
            }
        return {int(f): 1.0 for f in self.features["id"]}

    def project_action_map(self) -> dict[int, set[int]]:
        """project id → set of required action ids (empty for baselines)."""
        out: dict[int, set[int]] = {int(p): set() for p in self.projects["id"]}
        for p, a in zip(self.project_actions["project"], self.project_actions["action"]):
            out.setdefault(int(p), set()).add(int(a))
        return out

    def feature_project_persistence(self) -> dict[int, dict[int, float]]:
        """feature id → {project id: persistence}."""
        out: dict[int, dict[int, float]] = {}
        for p, f, v in zip(
            self.project_features["project"],
            self.project_features["feature"],
            self.project_features["persistence"],
        ):
            out.setdefault(int(f), {})[int(p)] = float(v)
        return out


@dataclass
class ProjectSolution:
    """Outcome of a project-prioritization solve.

    ``benefit`` is ``Σ_f weight_f · feature_persistence[f]``. ``optimal`` is
    True only from the exact MIP.
    """

    funded_actions: set[int]
    funded_projects: set[int]
    cost: float
    benefit: float
    feature_persistence: dict[int, float]
    optimal: bool = False
    metadata: dict = field(default_factory=dict)
