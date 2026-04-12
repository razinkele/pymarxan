"""Cost surface processing for conservation planning."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask as rasterio_mask


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
    if (
        cost_layer.crs is not None
        and planning_units.crs is not None
        and cost_layer.crs != planning_units.crs
    ):
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

    if new_costs:
        id_to_idx = {pid: i for i, pid in enumerate(result["id"].values)}
        cost_arr = result["cost"].values.copy()
        for pu_id, cost in new_costs.items():
            idx = id_to_idx.get(pu_id)
            if idx is not None:
                cost_arr[idx] = cost
        result["cost"] = cost_arr

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
    elif len(weights) != n_layers:
        raise ValueError(
            f"len(weights)={len(weights)} != len(layers)={n_layers}"
        )

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


_AGG_FUNCS = {
    "mean": np.nanmean,
    "sum": np.nansum,
    "max": np.nanmax,
    "min": np.nanmin,
    "median": np.nanmedian,
}


def apply_cost_from_raster(
    planning_units: gpd.GeoDataFrame,
    raster_path: str | Path,
    aggregation: str = "mean",
    band: int = 1,
    nodata_value: float | None = None,
) -> gpd.GeoDataFrame:
    """Extract raster values for each PU polygon and update cost column.

    Uses ``rasterio.mask`` to clip raster data per planning unit geometry
    and applies zonal statistics to compute the cost value.

    Parameters
    ----------
    planning_units : gpd.GeoDataFrame
        Must have ``id``, ``cost``, and ``geometry`` columns.
    raster_path : str or Path
        Path to a raster file (GeoTIFF, etc.).
    aggregation : str
        One of ``"mean"``, ``"sum"``, ``"max"``, ``"min"``, ``"median"``.
    band : int
        Raster band to read (1-indexed).
    nodata_value : float or None
        Override raster nodata. If None, uses the raster's own nodata value.

    Returns
    -------
    gpd.GeoDataFrame
        Copy with updated ``cost`` column. Original cost preserved where
        raster has no valid data.
    """
    if aggregation not in _AGG_FUNCS:
        raise ValueError(
            f"Unknown aggregation '{aggregation}'. "
            f"Choose from: {sorted(_AGG_FUNCS)}"
        )
    agg_fn = _AGG_FUNCS[aggregation]
    result = planning_units.copy()

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        nd = nodata_value if nodata_value is not None else src.nodata

        # Reproject PUs to raster CRS if needed
        pus = planning_units
        if (
            raster_crs is not None
            and planning_units.crs is not None
            and planning_units.crs != raster_crs
        ):
            pus = planning_units.to_crs(raster_crs)

        new_costs: dict[int, float] = {}
        for idx in range(len(pus)):
            geom = pus.geometry.iloc[idx]
            pu_id = int(pus["id"].iloc[idx])
            try:
                out_image, _ = rasterio_mask(
                    src, [geom], crop=True, filled=True, band_indexes=[band]
                )
            except ValueError:
                # Geometry doesn't overlap raster
                continue

            data = out_image[0].astype(float)
            if nd is not None:
                data = np.where(data == nd, np.nan, data)

            valid = data[~np.isnan(data)]
            if len(valid) > 0:
                new_costs[pu_id] = float(agg_fn(valid))

    if new_costs:
        id_to_idx = {pid: i for i, pid in enumerate(result["id"].values)}
        cost_arr = result["cost"].values.copy().astype(float)
        for pu_id, cost in new_costs.items():
            idx = id_to_idx.get(pu_id)
            if idx is not None:
                cost_arr[idx] = cost
        result["cost"] = cost_arr

    return result
