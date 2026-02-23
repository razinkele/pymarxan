"""Cost surface processing for conservation planning."""
from __future__ import annotations

import geopandas as gpd
import numpy as np


def apply_cost_from_vector(
    planning_units: gpd.GeoDataFrame,
    cost_layer: gpd.GeoDataFrame,
    cost_column: str,
    aggregation: str = "area_weighted_mean",
) -> gpd.GeoDataFrame:
    """Compute cost from vector overlay.

    Parameters
    ----------
    planning_units : gpd.GeoDataFrame
        Must have ``id``, ``cost``, and ``geometry`` columns.
    cost_layer : gpd.GeoDataFrame
        Vector layer with cost values.
    cost_column : str
        Column in cost_layer containing cost values.
    aggregation : str
        ``"area_weighted_mean"`` | ``"sum"`` | ``"max"``.

    Returns
    -------
    gpd.GeoDataFrame
        Copy with updated cost column. Original PU cost preserved where
        no overlay exists.
    """
    result = planning_units.copy()

    # Reproject if needed
    if cost_layer.crs != planning_units.crs and cost_layer.crs is not None:
        cost_layer = cost_layer.to_crs(planning_units.crs)

    overlay = gpd.overlay(
        planning_units[["id", "geometry"]],
        cost_layer[[cost_column, "geometry"]],
        how="intersection",
    )

    if len(overlay) == 0:
        return result

    overlay["_intersection_area"] = overlay.geometry.area

    new_costs: dict[int, float] = {}
    pu_area_by_id = dict(zip(
        planning_units["id"].values,
        planning_units.geometry.area,
    ))
    for pu_id, group in overlay.groupby("id"):
        pu_area = pu_area_by_id.get(pu_id, 0.0)
        if aggregation == "area_weighted_mean":
            weighted = (group[cost_column] * group["_intersection_area"]).sum()
            total_area = group["_intersection_area"].sum()
            if total_area > 0 and total_area >= pu_area * 0.01:
                new_costs[pu_id] = weighted / total_area
        elif aggregation == "sum":
            new_costs[pu_id] = group[cost_column].sum()
        elif aggregation == "max":
            new_costs[pu_id] = group[cost_column].max()

    for pu_id, cost in new_costs.items():
        result.loc[result["id"] == pu_id, "cost"] = cost

    return result


def combine_cost_layers(
    planning_units: gpd.GeoDataFrame,
    layers: list[tuple[str, np.ndarray]],
    weights: list[float] | None = None,
) -> gpd.GeoDataFrame:
    """Combine multiple cost arrays with optional weighting.

    Parameters
    ----------
    planning_units : gpd.GeoDataFrame
        Must have ``id`` and ``cost`` columns.
    layers : list of (name, array) tuples
        Each array has one value per PU.
    weights : list of float or None
        Per-layer weights. If None, equal weighting.

    Returns
    -------
    gpd.GeoDataFrame
        Copy with updated cost column (weighted sum of normalized layers).
    """
    result = planning_units.copy()
    n_layers = len(layers)
    if n_layers == 0:
        return result

    if weights is None:
        weights = [1.0 / n_layers] * n_layers

    combined = np.zeros(len(planning_units), dtype=float)
    for (name, values), w in zip(layers, weights):
        arr = np.asarray(values, dtype=float)
        # Min-max normalize
        vmin, vmax = arr.min(), arr.max()
        if vmax > vmin:
            normalized = (arr - vmin) / (vmax - vmin)
        else:
            normalized = np.zeros_like(arr)
        combined += w * normalized

    result["cost"] = combined
    return result
