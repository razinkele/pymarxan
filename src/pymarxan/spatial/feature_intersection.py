"""Feature intersection for conservation planning."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rasterio_mask

_AGG_FUNCS = {
    "mean": np.nanmean,
    "sum": np.nansum,
    "max": np.nanmax,
    "min": np.nanmin,
    "median": np.nanmedian,
}


def intersect_vector_features(
    planning_units: gpd.GeoDataFrame,
    feature_layers: dict[int, gpd.GeoDataFrame],
    method: str = "area",
) -> pd.DataFrame:
    """Compute feature amounts per planning unit from vector layers.

    Parameters
    ----------
    planning_units : gpd.GeoDataFrame
        Must have ``id`` and ``geometry`` columns.
    feature_layers : dict[int, gpd.GeoDataFrame]
        Mapping from feature ID to a GeoDataFrame of feature geometries.
    method : str
        ``"area"`` — intersection area as amount.
        ``"binary"`` — 1.0 if any overlap, else 0.
        ``"proportion"`` — fraction of PU area covered.

    Returns
    -------
    pd.DataFrame
        Columns: ``species``, ``pu``, ``amount``.
    """
    if method not in ("area", "binary", "proportion"):
        raise ValueError(
            f"Unknown method '{method}'. Choose from: 'area', 'binary', 'proportion'"
        )

    rows: list[dict] = []
    pu_areas = dict(zip(planning_units["id"].values, planning_units.geometry.area))

    for feature_id, layer in feature_layers.items():
        # Reproject if needed
        if (
            layer.crs is not None
            and planning_units.crs is not None
            and layer.crs != planning_units.crs
        ):
            layer = layer.to_crs(planning_units.crs)

        overlay = gpd.overlay(
            planning_units[["id", "geometry"]],
            layer[["geometry"]],
            how="intersection",
        )
        if len(overlay) == 0:
            continue

        overlay["_area"] = overlay.geometry.area
        grouped = overlay.groupby("id")["_area"].sum()

        for pu_id, area in grouped.items():
            if area <= 0:
                continue
            pu_id_int = int(pu_id)
            if method == "area":
                amount = float(area)
            elif method == "binary":
                amount = 1.0
            else:  # proportion
                pu_area = pu_areas.get(pu_id_int, 0.0)
                amount = float(area / pu_area) if pu_area > 0 else 0.0
            rows.append({"species": feature_id, "pu": pu_id_int, "amount": amount})

    return pd.DataFrame(rows, columns=["species", "pu", "amount"])


def intersect_raster_features(
    planning_units: gpd.GeoDataFrame,
    raster_paths: dict[int, Path],
    aggregation: str = "sum",
) -> pd.DataFrame:
    """Compute feature amounts per planning unit from raster files.

    Uses ``rasterio.mask`` for zonal statistics.

    Parameters
    ----------
    planning_units : gpd.GeoDataFrame
        Must have ``id`` and ``geometry`` columns.
    raster_paths : dict[int, Path]
        Mapping from feature ID to raster file path.
    aggregation : str
        One of ``"mean"``, ``"sum"``, ``"max"``, ``"min"``, ``"median"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``species``, ``pu``, ``amount``.
    """
    if aggregation not in _AGG_FUNCS:
        raise ValueError(
            f"Unknown aggregation '{aggregation}'. Choose from: {sorted(_AGG_FUNCS)}"
        )
    agg_fn = _AGG_FUNCS[aggregation]
    rows: list[dict] = []

    for feature_id, raster_path in raster_paths.items():
        with rasterio.open(raster_path) as src:
            nd = src.nodata
            raster_crs = src.crs

            pus = planning_units
            if (
                raster_crs is not None
                and planning_units.crs is not None
                and planning_units.crs != raster_crs
            ):
                pus = planning_units.to_crs(raster_crs)

            for idx in range(len(pus)):
                geom = pus.geometry.iloc[idx]
                pu_id = int(pus["id"].iloc[idx])
                try:
                    out_image, _ = rasterio_mask(
                        src, [geom], crop=True, filled=True, indexes=[1]
                    )
                except ValueError:
                    continue

                data = out_image[0].astype(float)
                if nd is not None:
                    data = np.where(data == nd, np.nan, data)

                valid = data[~np.isnan(data)]
                if len(valid) > 0:
                    amount = float(agg_fn(valid))
                    if amount > 0:
                        rows.append({
                            "species": feature_id,
                            "pu": pu_id,
                            "amount": amount,
                        })

    return pd.DataFrame(rows, columns=["species", "pu", "amount"])
