"""Fetch administrative boundaries from geoBoundaries API."""
from __future__ import annotations

import geopandas as gpd
import requests

_GEOBOUNDARIES_API = "https://www.geoboundaries.org/api/current/gbOpen"

# Common countries (ISO 3166-1 alpha-3). Full list at geoboundaries.org.
_COUNTRIES = [
    {"iso3": "AFG", "name": "Afghanistan"},
    {"iso3": "ALB", "name": "Albania"},
    {"iso3": "DZA", "name": "Algeria"},
    {"iso3": "AGO", "name": "Angola"},
    {"iso3": "ARG", "name": "Argentina"},
    {"iso3": "AUS", "name": "Australia"},
    {"iso3": "AUT", "name": "Austria"},
    {"iso3": "BGD", "name": "Bangladesh"},
    {"iso3": "BEL", "name": "Belgium"},
    {"iso3": "BOL", "name": "Bolivia"},
    {"iso3": "BRA", "name": "Brazil"},
    {"iso3": "KHM", "name": "Cambodia"},
    {"iso3": "CMR", "name": "Cameroon"},
    {"iso3": "CAN", "name": "Canada"},
    {"iso3": "CHL", "name": "Chile"},
    {"iso3": "CHN", "name": "China"},
    {"iso3": "COL", "name": "Colombia"},
    {"iso3": "COD", "name": "Congo (DRC)"},
    {"iso3": "CRI", "name": "Costa Rica"},
    {"iso3": "CUB", "name": "Cuba"},
    {"iso3": "DNK", "name": "Denmark"},
    {"iso3": "ECU", "name": "Ecuador"},
    {"iso3": "EGY", "name": "Egypt"},
    {"iso3": "ETH", "name": "Ethiopia"},
    {"iso3": "FIN", "name": "Finland"},
    {"iso3": "FRA", "name": "France"},
    {"iso3": "DEU", "name": "Germany"},
    {"iso3": "GHA", "name": "Ghana"},
    {"iso3": "GRC", "name": "Greece"},
    {"iso3": "GTM", "name": "Guatemala"},
    {"iso3": "HND", "name": "Honduras"},
    {"iso3": "IND", "name": "India"},
    {"iso3": "IDN", "name": "Indonesia"},
    {"iso3": "IRN", "name": "Iran"},
    {"iso3": "IRQ", "name": "Iraq"},
    {"iso3": "IRL", "name": "Ireland"},
    {"iso3": "ISR", "name": "Israel"},
    {"iso3": "ITA", "name": "Italy"},
    {"iso3": "JPN", "name": "Japan"},
    {"iso3": "KEN", "name": "Kenya"},
    {"iso3": "MDG", "name": "Madagascar"},
    {"iso3": "MYS", "name": "Malaysia"},
    {"iso3": "MEX", "name": "Mexico"},
    {"iso3": "MAR", "name": "Morocco"},
    {"iso3": "MOZ", "name": "Mozambique"},
    {"iso3": "MMR", "name": "Myanmar"},
    {"iso3": "NPL", "name": "Nepal"},
    {"iso3": "NLD", "name": "Netherlands"},
    {"iso3": "NZL", "name": "New Zealand"},
    {"iso3": "NGA", "name": "Nigeria"},
    {"iso3": "NOR", "name": "Norway"},
    {"iso3": "PAK", "name": "Pakistan"},
    {"iso3": "PAN", "name": "Panama"},
    {"iso3": "PER", "name": "Peru"},
    {"iso3": "PHL", "name": "Philippines"},
    {"iso3": "POL", "name": "Poland"},
    {"iso3": "PRT", "name": "Portugal"},
    {"iso3": "ROU", "name": "Romania"},
    {"iso3": "RUS", "name": "Russia"},
    {"iso3": "ZAF", "name": "South Africa"},
    {"iso3": "KOR", "name": "South Korea"},
    {"iso3": "ESP", "name": "Spain"},
    {"iso3": "LKA", "name": "Sri Lanka"},
    {"iso3": "SWE", "name": "Sweden"},
    {"iso3": "CHE", "name": "Switzerland"},
    {"iso3": "TZA", "name": "Tanzania"},
    {"iso3": "THA", "name": "Thailand"},
    {"iso3": "TUR", "name": "Turkey"},
    {"iso3": "UGA", "name": "Uganda"},
    {"iso3": "UKR", "name": "Ukraine"},
    {"iso3": "GBR", "name": "United Kingdom"},
    {"iso3": "USA", "name": "United States"},
    {"iso3": "VEN", "name": "Venezuela"},
    {"iso3": "VNM", "name": "Vietnam"},
    {"iso3": "ZMB", "name": "Zambia"},
    {"iso3": "ZWE", "name": "Zimbabwe"},
]


def list_countries() -> list[dict[str, str]]:
    """Return list of available countries with ISO3 codes."""
    return list(_COUNTRIES)


def fetch_gadm(
    country_iso3: str,
    admin_level: int = 0,
    admin_name: str | None = None,
) -> gpd.GeoDataFrame:
    """Fetch administrative boundary from geoBoundaries API.

    Parameters
    ----------
    country_iso3 : str
        ISO 3166-1 alpha-3 country code (e.g. "USA", "GBR").
    admin_level : int
        0 = country, 1 = state/province, 2 = district.
    admin_name : str or None
        Filter to a specific admin region by name.

    Returns
    -------
    gpd.GeoDataFrame
        Boundary polygon(s) with CRS EPSG:4326.
    """
    url = f"{_GEOBOUNDARIES_API}/{country_iso3}/ADM{admin_level}/"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    meta = resp.json()
    geojson_url = meta.get("gjDownloadURL")
    if geojson_url is None:
        raise ValueError(
            f"geoBoundaries API response missing 'gjDownloadURL' key. "
            f"Response keys: {list(meta.keys())}"
        )

    geojson_resp = requests.get(geojson_url, timeout=60)
    geojson_resp.raise_for_status()

    gdf = gpd.GeoDataFrame.from_features(
        geojson_resp.json()["features"],
        crs="EPSG:4326",
    )

    if admin_name is not None:
        name_col = "shapeName" if "shapeName" in gdf.columns else gdf.columns[0]
        gdf = gdf[gdf[name_col].str.contains(admin_name, case=False, na=False)]
        gdf = gdf.reset_index(drop=True)

    return gdf
