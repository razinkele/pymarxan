"""Phase 9 integration tests: all new Shiny modules wired into app."""
from __future__ import annotations

import pytest


@pytest.mark.integration
def test_app_imports_phase9():
    """Verify the app can import all phase 9 modules."""
    from pymarxan_app import app
    assert app.app is not None


@pytest.mark.integration
def test_geometry_importable():
    """Geometry generator is accessible from models."""
    from pymarxan.models.geometry import generate_grid
    grid = generate_grid(4)
    assert len(grid) == 4


@pytest.mark.integration
def test_all_map_modules_importable():
    """All map modules can be imported."""
    from pymarxan_shiny.modules.mapping.comparison_map import comparison_map_ui
    from pymarxan_shiny.modules.mapping.frequency_map import frequency_map_ui
    from pymarxan_shiny.modules.mapping.network_view import network_view_ui
    from pymarxan_shiny.modules.mapping.solution_map import solution_map_ui
    from pymarxan_shiny.modules.mapping.spatial_grid import spatial_grid_ui
    assert all(callable(fn) for fn in [
        solution_map_ui, spatial_grid_ui, frequency_map_ui,
        comparison_map_ui, network_view_ui,
    ])


@pytest.mark.integration
def test_feature_table_importable():
    """Feature table module can be imported."""
    from pymarxan_shiny.modules.data.feature_table import feature_table_ui
    assert callable(feature_table_ui)


@pytest.mark.integration
def test_sensitivity_ui_importable():
    """Sensitivity dashboard module can be imported."""
    from pymarxan_shiny.modules.calibration.sensitivity_ui import sensitivity_ui
    assert callable(sensitivity_ui)


@pytest.mark.integration
def test_all_ui_elements_render():
    """All new UI elements render without error."""
    from pymarxan_shiny.modules.calibration.sensitivity_ui import sensitivity_ui
    from pymarxan_shiny.modules.data.feature_table import feature_table_ui
    from pymarxan_shiny.modules.mapping.comparison_map import comparison_map_ui
    from pymarxan_shiny.modules.mapping.frequency_map import frequency_map_ui
    from pymarxan_shiny.modules.mapping.network_view import network_view_ui
    from pymarxan_shiny.modules.mapping.spatial_grid import spatial_grid_ui

    for fn, name in [
        (spatial_grid_ui, "grid"),
        (frequency_map_ui, "freq"),
        (comparison_map_ui, "cmp"),
        (network_view_ui, "net"),
        (feature_table_ui, "ft"),
        (sensitivity_ui, "sens"),
    ]:
        elem = fn(name)
        assert elem is not None, f"{fn.__name__} returned None"
