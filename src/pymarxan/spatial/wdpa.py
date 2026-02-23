"""WDPA protected area integration."""
from __future__ import annotations

import copy

import geopandas as gpd
import requests
from shapely.geometry import box as shapely_box
from shapely.geometry import shape

from pymarxan.models.problem import STATUS_LOCKED_IN, ConservationProblem

_WDPA_API = "https://api.protectedplanet.net/v3"


def fetch_wdpa(
    bounds: tuple[float, float, float, float],
    country_iso3: str | None = None,
    api_token: str | None = None,
) -> gpd.GeoDataFrame:
    """Fetch protected areas from Protected Planet API.

    Parameters
    ----------
    bounds : tuple
        (minx, miny, maxx, maxy) bounding box.
    country_iso3 : str or None
        Optional country filter.
    api_token : str or None
        Protected Planet API token. Required for authenticated access.

    Returns
    -------
    gpd.GeoDataFrame
        Columns: name, desig, iucn_cat, geometry.
    """
    params: dict = {
        "with_geometry": "true",
        "per_page": 50,
    }
    if api_token:
        params["token"] = api_token
    if country_iso3:
        params["country"] = country_iso3

    url = f"{_WDPA_API}/protected_areas/search"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    pa_list = data.get("protected_areas", [])

    bounds_box = shapely_box(*bounds)
    rows = []
    geometries = []
    for pa in pa_list:
        geojson = pa.get("geojson")
        if geojson is None:
            continue
        geom = shape(geojson)
        if geom.is_empty:
            continue
        if not geom.intersects(bounds_box):
            continue
        rows.append({
            "name": pa.get("name", ""),
            "desig": pa.get("designation", ""),
            "iucn_cat": pa.get("iucn_category", ""),
        })
        geometries.append(geom)

    if not rows:
        return gpd.GeoDataFrame(
            {"name": [], "desig": [], "iucn_cat": [], "geometry": []},
            crs="EPSG:4326",
        )

    return gpd.GeoDataFrame(rows, geometry=geometries, crs="EPSG:4326")


def apply_wdpa_status(
    problem: ConservationProblem,
    wdpa: gpd.GeoDataFrame,
    overlap_threshold: float = 0.5,
    status: int = STATUS_LOCKED_IN,
) -> ConservationProblem:
    """Set PU status for units overlapping protected areas.

    Parameters
    ----------
    problem : ConservationProblem
        Must have GeoDataFrame planning_units.
    wdpa : gpd.GeoDataFrame
        Protected area polygons.
    overlap_threshold : float
        Minimum fraction of PU area that must be covered.
    status : int
        Status to assign (default STATUS_LOCKED_IN=2).

    Returns
    -------
    ConservationProblem
        New problem with updated statuses (does not mutate input).
    """
    result = copy.deepcopy(problem)
    pu_gdf = result.planning_units

    wdpa_union = wdpa.geometry.union_all()

    new_statuses = pu_gdf["status"].values.copy()
    for idx in range(len(pu_gdf)):
        pu_geom = pu_gdf.geometry.iloc[idx]
        pu_area = pu_geom.area
        if pu_area <= 0:
            continue
        intersection = pu_geom.intersection(wdpa_union)
        overlap_ratio = intersection.area / pu_area
        if overlap_ratio >= overlap_threshold:
            new_statuses[idx] = status

    result.planning_units["status"] = new_statuses
    return result
