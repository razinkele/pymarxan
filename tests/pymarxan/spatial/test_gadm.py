"""Tests for GADM admin boundary fetching."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest

from pymarxan.spatial.gadm import fetch_gadm, list_countries

# Sample GeoJSON response from geoBoundaries API
_SAMPLE_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"shapeName": "TestCountry"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            },
        }
    ],
}

_SAMPLE_API_RESPONSE = {
    "boundarySource-1": "geoBoundaries",
    "gjDownloadURL": "https://example.com/geojson.geojson",
}


class TestFetchGadm:
    @patch("pymarxan.spatial.gadm.requests.get")
    def test_fetch_returns_geodataframe(self, mock_get):
        # First call: API metadata -> returns gjDownloadURL
        # Second call: GeoJSON download
        meta_resp = MagicMock()
        meta_resp.status_code = 200
        meta_resp.json.return_value = _SAMPLE_API_RESPONSE

        geojson_resp = MagicMock()
        geojson_resp.status_code = 200
        geojson_resp.json.return_value = _SAMPLE_GEOJSON

        mock_get.side_effect = [meta_resp, geojson_resp]

        gdf = fetch_gadm("TST", admin_level=0)
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 1
        assert gdf.geometry.iloc[0].is_valid

    @patch("pymarxan.spatial.gadm.requests.get")
    def test_fetch_with_admin_name_filters(self, mock_get):
        multi_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"shapeName": "RegionA"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                    },
                },
                {
                    "type": "Feature",
                    "properties": {"shapeName": "RegionB"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[1, 0], [2, 0], [2, 1], [1, 1], [1, 0]]],
                    },
                },
            ],
        }

        meta_resp = MagicMock()
        meta_resp.status_code = 200
        meta_resp.json.return_value = _SAMPLE_API_RESPONSE

        geojson_resp = MagicMock()
        geojson_resp.status_code = 200
        geojson_resp.json.return_value = multi_geojson

        mock_get.side_effect = [meta_resp, geojson_resp]

        gdf = fetch_gadm("TST", admin_level=1, admin_name="RegionA")
        assert len(gdf) == 1

    @patch("pymarxan.spatial.gadm.requests.get")
    def test_fetch_whitespace_admin_name_treated_as_no_filter(self, mock_get):
        """A whitespace-only admin_name must not filter via str.contains(' ').

        Previously a stray space would match every row containing any space
        in its name — silently returning the full country dataset rather
        than indicating no filter was applied.
        """
        multi_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"shapeName": "Region A"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                    },
                },
                {
                    "type": "Feature",
                    "properties": {"shapeName": "RegionB"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[1, 0], [2, 0], [2, 1], [1, 1], [1, 0]]],
                    },
                },
            ],
        }
        meta_resp = MagicMock()
        meta_resp.status_code = 200
        meta_resp.json.return_value = _SAMPLE_API_RESPONSE
        geojson_resp = MagicMock()
        geojson_resp.status_code = 200
        geojson_resp.json.return_value = multi_geojson
        mock_get.side_effect = [meta_resp, geojson_resp]

        # Whitespace-only filter — must NOT silently match "Region A"
        gdf = fetch_gadm("TST", admin_level=1, admin_name="   ")
        assert len(gdf) == 2  # both rows returned, like no filter

    @patch("pymarxan.spatial.gadm.requests.get")
    def test_fetch_invalid_country_raises(self, mock_get):
        resp = MagicMock()
        resp.status_code = 404
        resp.raise_for_status.side_effect = Exception("404 Not Found")
        mock_get.return_value = resp

        with pytest.raises(Exception):
            fetch_gadm("ZZZ", admin_level=0)


class TestListCountries:
    def test_returns_list_of_dicts(self):
        countries = list_countries()
        assert isinstance(countries, list)
        assert len(countries) > 0
        assert "iso3" in countries[0]
        assert "name" in countries[0]

    def test_contains_common_countries(self):
        countries = list_countries()
        iso3s = {c["iso3"] for c in countries}
        assert "USA" in iso3s
        assert "GBR" in iso3s
        assert "AUS" in iso3s
