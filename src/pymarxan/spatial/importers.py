"""Import planning units and features from GIS files."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def import_planning_units(
    path: str | Path,
    id_column: str = "id",
    cost_column: str = "cost",
    status_column: str | None = "status",
) -> gpd.GeoDataFrame:
    """Import planning units from shapefile, GeoJSON, or GeoPackage.

    Parameters
    ----------
    path : str or Path
        Path to the GIS file.
    id_column : str
        Column name to use as planning unit ID.
    cost_column : str
        Column name to use as cost. If missing, defaults to 1.0.
    status_column : str or None
        Column name for status. If None or missing, defaults to 0.

    Returns
    -------
    gpd.GeoDataFrame
        Columns: id, cost, status, geometry. CRS preserved from input.
    """
    gdf = gpd.read_file(path)

    if id_column not in gdf.columns:
        raise ValueError(
            f"ID column '{id_column}' not found in file. "
            f"Available columns: {list(gdf.columns)}"
        )

    data = {"id": gdf[id_column].astype(int).values}

    if cost_column in gdf.columns:
        data["cost"] = gdf[cost_column].astype(float).values
    else:
        data["cost"] = np.ones(len(gdf))

    if status_column is not None and status_column in gdf.columns:
        data["status"] = gdf[status_column].astype(int).values
    else:
        data["status"] = np.zeros(len(gdf), dtype=int)

    result = gpd.GeoDataFrame(data, geometry=gdf.geometry.values, crs=gdf.crs)
    return result


def import_features_from_vector(
    path: str | Path,
    planning_units: gpd.GeoDataFrame,
    feature_name: str,
    feature_id: int,
    amount_column: str | None = None,
) -> pd.DataFrame:
    """Compute feature amounts per PU via spatial overlay.

    Parameters
    ----------
    path : str or Path
        Path to feature vector file.
    planning_units : gpd.GeoDataFrame
        Must have ``id`` and ``geometry`` columns.
    feature_name : str
        Name for the feature (for display).
    feature_id : int
        Numeric ID for the feature.
    amount_column : str or None
        Column in the feature file to sum per PU.
        If None, uses area of intersection as amount.

    Returns
    -------
    pd.DataFrame
        Columns: species, pu, amount.
    """
    features_gdf = gpd.read_file(path)

    # Reproject if CRS differs
    if (features_gdf.crs is not None
            and planning_units.crs is not None
            and features_gdf.crs != planning_units.crs):
        features_gdf = features_gdf.to_crs(planning_units.crs)

    overlay = gpd.overlay(
        planning_units[["id", "geometry"]], features_gdf, how="intersection"
    )

    rows = []
    if amount_column and amount_column in overlay.columns:
        grouped = overlay.groupby("id")[amount_column].sum()
        for pu_id, amount in grouped.items():
            if amount > 0:
                rows.append(
                    {"species": feature_id, "pu": int(pu_id), "amount": float(amount)}
                )
    else:
        # Use intersection area
        overlay["_area"] = overlay.geometry.area
        grouped = overlay.groupby("id")["_area"].sum()
        for pu_id, area in grouped.items():
            if area > 0:
                rows.append(
                    {"species": feature_id, "pu": int(pu_id), "amount": float(area)}
                )

    return pd.DataFrame(rows, columns=["species", "pu", "amount"])
