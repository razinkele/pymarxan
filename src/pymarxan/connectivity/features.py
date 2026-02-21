"""Convert connectivity metrics into Marxan features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pymarxan.models.problem import ConservationProblem


def metric_to_feature(
    pu_ids: list[int],
    metric_values: np.ndarray,
    feature_id: int,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Convert a per-PU metric array into puvspr rows."""
    rows = []
    for i, pid in enumerate(pu_ids):
        val = float(metric_values[i])
        if threshold is not None and val < threshold:
            continue
        rows.append({"species": feature_id, "pu": pid, "amount": val})
    return pd.DataFrame(rows)


def add_connectivity_features(
    problem: ConservationProblem,
    metrics: dict[str, np.ndarray],
    targets: dict[str, float],
    start_feature_id: int | None = None,
    threshold: float | None = None,
) -> ConservationProblem:
    """Add connectivity metrics as synthetic features to a problem."""
    pu_ids = problem.planning_units["id"].tolist()
    existing_max = int(problem.features["id"].max())
    if start_feature_id is None:
        start_feature_id = existing_max + 100

    new_features = problem.features.copy()
    new_puvspr = problem.pu_vs_features.copy()

    fid = start_feature_id
    for name, values in metrics.items():
        target = targets.get(name, 0.0)
        feat_row = pd.DataFrame({
            "id": [fid],
            "name": [f"conn_{name}"],
            "target": [target],
            "spf": [1.0],
        })
        new_features = pd.concat(
            [new_features, feat_row], ignore_index=True,
        )
        puvspr_rows = metric_to_feature(
            pu_ids, values, fid, threshold=threshold,
        )
        new_puvspr = pd.concat(
            [new_puvspr, puvspr_rows], ignore_index=True,
        )
        fid += 1

    return ConservationProblem(
        planning_units=problem.planning_units,
        features=new_features,
        pu_vs_features=new_puvspr,
        boundary=problem.boundary,
        parameters=problem.parameters,
    )
