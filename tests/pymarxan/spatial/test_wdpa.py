"""Tests for WDPA protected area integration."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from pymarxan.models.problem import STATUS_INITIAL_INCLUDE, STATUS_LOCKED_IN, ConservationProblem
from pymarxan.spatial.wdpa import apply_wdpa_status, fetch_wdpa

_SAMPLE_WDPA_RESPONSE = {
    "protected_areas": [
        {
            "id": 1,
            "name": "Test Reserve",
            "designation": "National Park",
            "iucn_category": "II",
            "reported_area": 100.0,
            "geojson": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [0.6, 0], [0.6, 0.6], [0, 0.6], [0, 0]]],
            },
        }
    ]
}


def _make_geo_problem():
    """Create a problem with 4 PUs as a 2x2 grid."""
    gdf = gpd.GeoDataFrame(
        {
            "id": [1, 2, 3, 4],
            "cost": [1.0, 2.0, 3.0, 4.0],
            "status": [0, 0, 0, 0],
        },
        geometry=[
            box(0, 0, 0.5, 0.5),
            box(0.5, 0, 1.0, 0.5),
            box(0, 0.5, 0.5, 1.0),
            box(0.5, 0.5, 1.0, 1.0),
        ],
        crs="EPSG:4326",
    )
    return ConservationProblem(
        planning_units=gdf,
        features=pd.DataFrame({"id": [1], "name": ["f1"], "target": [10.0], "spf": [1.0]}),
        pu_vs_features=pd.DataFrame({"species": [1], "pu": [1], "amount": [20.0]}),
    )


class TestApplyWdpaStatus:
    def test_marks_overlapping_pus(self):
        problem = _make_geo_problem()
        # WDPA polygon covers PU 1 fully (box 0,0,0.5,0.5)
        wdpa = gpd.GeoDataFrame(
            {"name": ["Reserve"], "desig": ["NP"]},
            geometry=[box(0, 0, 0.6, 0.6)],
            crs="EPSG:4326",
        )
        result = apply_wdpa_status(problem, wdpa, overlap_threshold=0.5)
        # PU 1 should be locked in (>50% overlap)
        assert result.planning_units.loc[
            result.planning_units["id"] == 1, "status"
        ].iloc[0] == STATUS_LOCKED_IN

    def test_does_not_mutate_original(self):
        problem = _make_geo_problem()
        wdpa = gpd.GeoDataFrame(
            {"name": ["Reserve"]},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326",
        )
        apply_wdpa_status(problem, wdpa)
        assert all(problem.planning_units["status"] == 0)

    def test_high_threshold_excludes_partial_overlaps(self):
        problem = _make_geo_problem()
        # WDPA covers only a corner of PU 1
        wdpa = gpd.GeoDataFrame(
            {"name": ["Reserve"]},
            geometry=[box(0, 0, 0.2, 0.2)],
            crs="EPSG:4326",
        )
        result = apply_wdpa_status(problem, wdpa, overlap_threshold=0.5)
        assert result.planning_units.loc[
            result.planning_units["id"] == 1, "status"
        ].iloc[0] == 0  # Not locked — overlap < 50%

    def test_custom_status_value(self):
        problem = _make_geo_problem()
        wdpa = gpd.GeoDataFrame(
            {"name": ["Reserve"]},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326",
        )
        result = apply_wdpa_status(problem, wdpa, status=STATUS_INITIAL_INCLUDE)
        assert all(result.planning_units["status"] == STATUS_INITIAL_INCLUDE)


    def test_apply_wdpa_status_handles_invalid_wdpa_union(self):
        """Invalid (self-intersecting) WDPA geometry must not crash.

        Complex coastlines often produce self-intersecting unions; the
        cleanup with buffer(0) keeps apply_wdpa_status usable instead of
        raising a raw GEOSException to the Shiny UI.
        """
        from shapely.geometry import Polygon

        # Bowtie polygon — invalid, self-intersecting
        bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
        pu_gdf = gpd.GeoDataFrame(
            {"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]},
            geometry=[box(0, 0, 0.5, 0.5), box(2, 2, 3, 3)],
            crs="EPSG:4326",
        )
        problem = ConservationProblem(
            planning_units=pu_gdf,
            features=pd.DataFrame(
                {"id": [1], "name": ["f"], "target": [1.0], "spf": [1.0]}
            ),
            pu_vs_features=pd.DataFrame(
                {"species": [1], "pu": [1], "amount": [1.0]}
            ),
        )
        wdpa = gpd.GeoDataFrame(
            {"name": ["bow"], "desig": ["?"], "iucn_cat": ["?"]},
            geometry=[bowtie],
            crs="EPSG:4326",
        )
        # Must not raise
        result = apply_wdpa_status(problem, wdpa, overlap_threshold=0.1)
        assert "status" in result.planning_units.columns

    def test_apply_wdpa_status_reprojects_crs(self):
        """apply_wdpa_status must reproject WDPA to PU CRS before intersection."""
        # PUs in a projected CRS (EPSG:32610 — UTM zone 10N)
        pu_gdf = gpd.GeoDataFrame(
            {"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]},
            geometry=[
                box(500000, 5000000, 501000, 5001000),
                box(501000, 5000000, 502000, 5001000),
            ],
            crs="EPSG:32610",
        )
        problem = ConservationProblem(
            planning_units=pu_gdf,
            features=pd.DataFrame(
                {"id": [1], "name": ["f"], "target": [1.0], "spf": [1.0]}
            ),
            pu_vs_features=pd.DataFrame(
                {"species": [1], "pu": [1], "amount": [1.0]}
            ),
        )

        # WDPA polygon covering PU 1, but in EPSG:4326
        pu1_4326 = gpd.GeoDataFrame(
            geometry=[box(500000, 5000000, 501000, 5001000)],
            crs="EPSG:32610",
        ).to_crs("EPSG:4326")
        wdpa = gpd.GeoDataFrame(
            {"name": ["PA1"], "desig": ["NP"], "iucn_cat": ["II"]},
            geometry=pu1_4326.geometry.values,
            crs="EPSG:4326",
        )

        result = apply_wdpa_status(problem, wdpa, overlap_threshold=0.5)
        assert result.planning_units.iloc[0]["status"] == 2
        assert result.planning_units.iloc[1]["status"] == 0


class TestFetchWdpa:
    @patch("pymarxan.spatial.wdpa.requests.get")
    def test_fetch_returns_geodataframe(self, mock_get):
        resp_page1 = MagicMock()
        resp_page1.status_code = 200
        resp_page1.json.return_value = _SAMPLE_WDPA_RESPONSE

        resp_page2 = MagicMock()
        resp_page2.status_code = 200
        resp_page2.json.return_value = {"protected_areas": []}

        mock_get.side_effect = [resp_page1, resp_page2]

        gdf = fetch_wdpa(bounds=(0, 0, 1, 1), api_token="test-token")
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) >= 1
        assert "name" in gdf.columns

    @patch("pymarxan.spatial.wdpa.requests.get")
    def test_fetch_without_token_raises(self, mock_get):
        resp = MagicMock()
        resp.status_code = 401
        resp.raise_for_status.side_effect = Exception("401 Unauthorized")
        mock_get.return_value = resp

        with pytest.raises(Exception):
            fetch_wdpa(bounds=(0, 0, 1, 1))

    @patch("pymarxan.spatial.wdpa.requests.get")
    def test_fetch_paginates_multiple_pages(self, mock_get):
        """fetch_wdpa should paginate through all API result pages."""
        pa_page1 = {
            "protected_areas": [{
                "id": 1, "name": "Reserve A", "designation": "NP",
                "iucn_category": "II",
                "geojson": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [0.3, 0], [0.3, 0.3], [0, 0.3], [0, 0]]],
                },
            }]
        }
        pa_page2 = {
            "protected_areas": [{
                "id": 2, "name": "Reserve B", "designation": "NP",
                "iucn_category": "III",
                "geojson": {
                    "type": "Polygon",
                    "coordinates": [[[0.5, 0.5], [0.8, 0.5], [0.8, 0.8], [0.5, 0.8], [0.5, 0.5]]],
                },
            }]
        }
        pa_page3 = {"protected_areas": []}

        resp1 = MagicMock(); resp1.json.return_value = pa_page1
        resp2 = MagicMock(); resp2.json.return_value = pa_page2
        resp3 = MagicMock(); resp3.json.return_value = pa_page3
        mock_get.side_effect = [resp1, resp2, resp3]

        gdf = fetch_wdpa(bounds=(0, 0, 1, 1), api_token="tok")
        assert len(gdf) == 2
        assert set(gdf["name"]) == {"Reserve A", "Reserve B"}
        # Should have made 3 API calls (pages 1, 2, 3=empty)
        assert mock_get.call_count == 3
