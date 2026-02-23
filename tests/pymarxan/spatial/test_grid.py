"""Tests for planning unit grid generation."""
from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Polygon, box

from pymarxan.spatial.grid import compute_adjacency, generate_planning_grid


class TestSquareGrid:
    def test_basic_square_grid(self):
        gdf = generate_planning_grid(
            bounds=(0.0, 0.0, 1.0, 1.0),
            cell_size=0.5,
            grid_type="square",
        )
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 4  # 2x2 grid
        assert set(gdf.columns) >= {"id", "cost", "status", "geometry"}
        assert gdf["cost"].tolist() == [1.0] * 4
        assert gdf["status"].tolist() == [0] * 4

    def test_square_grid_ids_are_sequential(self):
        gdf = generate_planning_grid(
            bounds=(0.0, 0.0, 1.5, 1.0),
            cell_size=0.5,
        )
        assert gdf["id"].tolist() == list(range(1, len(gdf) + 1))

    def test_square_grid_crs(self):
        gdf = generate_planning_grid(
            bounds=(0.0, 0.0, 1.0, 1.0),
            cell_size=0.5,
            crs="EPSG:4326",
        )
        assert gdf.crs is not None
        assert gdf.crs.to_epsg() == 4326

    def test_square_grid_geometries_are_polygons(self):
        gdf = generate_planning_grid(
            bounds=(0.0, 0.0, 1.0, 1.0),
            cell_size=0.5,
        )
        for geom in gdf.geometry:
            assert isinstance(geom, Polygon)
            assert geom.is_valid

    def test_square_grid_no_overlaps(self):
        gdf = generate_planning_grid(
            bounds=(0.0, 0.0, 1.0, 1.0),
            cell_size=0.5,
        )
        for i in range(len(gdf)):
            for j in range(i + 1, len(gdf)):
                overlap = gdf.geometry.iloc[i].intersection(gdf.geometry.iloc[j]).area
                assert overlap < 1e-10

    def test_empty_bounds_returns_empty(self):
        gdf = generate_planning_grid(
            bounds=(0.0, 0.0, 0.0, 0.0),
            cell_size=0.5,
        )
        assert len(gdf) == 0


class TestHexGrid:
    def test_basic_hex_grid(self):
        gdf = generate_planning_grid(
            bounds=(0.0, 0.0, 2.0, 2.0),
            cell_size=0.5,
            grid_type="hexagonal",
        )
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) > 0
        for geom in gdf.geometry:
            assert isinstance(geom, Polygon)
            # Hexagons have 6 vertices (+ closing vertex = 7 coords)
            assert len(geom.exterior.coords) == 7

    def test_hex_grid_tessellates(self):
        """Hex cells must share edges with neighbors (no gaps)."""
        grid = generate_planning_grid(
            bounds=(0, 0, 10, 10), cell_size=2.0, grid_type="hexagonal",
        )
        assert len(grid) > 4, "Need enough hexes to test tessellation"

        geoms = grid.geometry.values
        found_shared_edge = False
        for i in range(len(geoms)):
            for j in range(i + 1, len(geoms)):
                intersection = geoms[i].intersection(geoms[j])
                if intersection.length > 1e-10:
                    found_shared_edge = True
                    break
            if found_shared_edge:
                break
        assert found_shared_edge, "No hex pair shares an edge — tessellation is broken"

    def test_hex_grid_no_large_overlaps(self):
        gdf = generate_planning_grid(
            bounds=(0.0, 0.0, 1.0, 1.0),
            cell_size=0.3,
            grid_type="hexagonal",
        )
        for i in range(len(gdf)):
            for j in range(i + 1, len(gdf)):
                overlap = gdf.geometry.iloc[i].intersection(gdf.geometry.iloc[j]).area
                assert overlap < 0.01 * gdf.geometry.iloc[i].area


class TestClipping:
    def test_clip_to_polygon(self):
        clip_poly = box(0.0, 0.0, 0.7, 0.7)
        gdf = generate_planning_grid(
            bounds=(0.0, 0.0, 1.0, 1.0),
            cell_size=0.5,
            clip_to=clip_poly,
        )
        for geom in gdf.geometry:
            assert clip_poly.contains(geom.centroid)

    def test_clip_removes_cells_outside(self):
        clip_poly = box(0.0, 0.0, 0.3, 0.3)
        gdf_full = generate_planning_grid(
            bounds=(0.0, 0.0, 1.0, 1.0),
            cell_size=0.5,
        )
        gdf_clipped = generate_planning_grid(
            bounds=(0.0, 0.0, 1.0, 1.0),
            cell_size=0.5,
            clip_to=clip_poly,
        )
        assert len(gdf_clipped) < len(gdf_full)


class TestAdjacency:
    def test_square_grid_adjacency(self):
        gdf = generate_planning_grid(
            bounds=(0.0, 0.0, 1.0, 1.0),
            cell_size=0.5,
        )
        adj = compute_adjacency(gdf)
        assert set(adj.columns) == {"id1", "id2", "boundary"}
        # 2x2 grid: 4 shared edges (right, down for each applicable cell)
        assert len(adj) == 4
        assert all(adj["boundary"] > 0)

    def test_adjacency_ids_match_grid(self):
        gdf = generate_planning_grid(
            bounds=(0.0, 0.0, 1.0, 1.0),
            cell_size=0.5,
        )
        adj = compute_adjacency(gdf)
        all_ids = set(gdf["id"])
        adj_ids = set(adj["id1"]) | set(adj["id2"])
        assert adj_ids <= all_ids

    def test_single_cell_no_adjacency(self):
        gdf = generate_planning_grid(
            bounds=(0.0, 0.0, 0.5, 0.5),
            cell_size=0.5,
        )
        assert len(gdf) == 1
        adj = compute_adjacency(gdf)
        assert len(adj) == 0


class TestInvalidInput:
    def test_invalid_grid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown grid_type"):
            generate_planning_grid(
                bounds=(0.0, 0.0, 1.0, 1.0),
                cell_size=0.5,
                grid_type="triangle",
            )
