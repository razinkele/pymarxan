"""Import planning units and features from GIS files."""
from __future__ import annotations

import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import geopandas as gpd
import numpy as np
import pandas as pd


@contextmanager
def _resolve_gis_path(path: str | Path) -> Iterator[Path]:
    """Resolve a user-uploaded GIS file path to a readable file path.

    Handles two awkward cases that surface from Shiny ``input_file`` uploads:

    * ZIP archives are extracted to a temporary directory and the first ``.shp``
      (or ``.geojson``/``.gpkg``) inside is returned.
    * A bare ``.shp`` path with no sidecar files (``.shx``/``.dbf``) raises a
      clear error instructing the user to upload a ZIP archive — Shiny strips
      sidecars when delivering a single-file upload.
    """
    p = Path(path)

    if p.suffix.lower() == ".zip":
        tmpdir = tempfile.TemporaryDirectory(prefix="pymarxan_gis_")
        try:
            with zipfile.ZipFile(p, "r") as zf:
                zf.extractall(tmpdir.name)
            extracted = Path(tmpdir.name)
            # Prefer .shp, then .gpkg, then .geojson found anywhere in the archive
            for ext in (".shp", ".gpkg", ".geojson", ".json"):
                matches = list(extracted.rglob(f"*{ext}"))
                if matches:
                    yield matches[0]
                    return
            raise ValueError(
                "ZIP archive did not contain a supported GIS file (.shp, .gpkg, .geojson)."
            )
        finally:
            tmpdir.cleanup()
        return

    if p.suffix.lower() == ".shp":
        # Without the .shx/.dbf sidecars, Fiona/pyogrio raises a generic
        # DataSourceError that hides the real problem from end users.
        required_sidecars = [p.with_suffix(".shx"), p.with_suffix(".dbf")]
        if not all(s.exists() for s in required_sidecars):
            raise ValueError(
                f"Shapefile '{p.name}' is missing required sidecar files "
                "(.shx and .dbf). Upload the shapefile as a .zip archive "
                "containing all sidecar files (.shp, .shx, .dbf, .prj)."
            )

    yield p


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
        Path to the GIS file. ZIP archives containing a shapefile bundle are
        also accepted and extracted automatically.
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
    with _resolve_gis_path(path) as resolved:
        gdf = gpd.read_file(resolved)

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
    with _resolve_gis_path(path) as resolved:
        features_gdf = gpd.read_file(resolved)

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
