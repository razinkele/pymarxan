"""Phase 13 integration tests: ipyleaflet map upgrades."""
from __future__ import annotations

import pytest


@pytest.mark.integration
def test_app_imports_phase13():
    """Verify the app still imports cleanly after phase 13 changes."""
    from pymarxan_app import app
    assert app.app is not None


@pytest.mark.integration
def test_map_utils_importable():
    """Shared map helper is importable."""
    from pymarxan_shiny.modules.mapping.map_utils import create_grid_map
    assert callable(create_grid_map)


@pytest.mark.integration
def test_all_map_modules_importable():
    """All 5 map modules still import correctly."""
    from pymarxan_shiny.modules.mapping.comparison_map import (
        comparison_map_server,
        comparison_map_ui,
    )
    from pymarxan_shiny.modules.mapping.frequency_map import (
        frequency_map_server,
        frequency_map_ui,
    )
    from pymarxan_shiny.modules.mapping.network_view import (
        network_view_server,
        network_view_ui,
    )
    from pymarxan_shiny.modules.mapping.solution_map import (
        solution_map_server,
        solution_map_ui,
    )
    from pymarxan_shiny.modules.mapping.spatial_grid import (
        spatial_grid_server,
        spatial_grid_ui,
    )
    for fn in [
        solution_map_ui, solution_map_server,
        spatial_grid_ui, spatial_grid_server,
        frequency_map_ui, frequency_map_server,
        comparison_map_ui, comparison_map_server,
        network_view_ui, network_view_server,
    ]:
        assert callable(fn)


@pytest.mark.integration
def test_all_map_uis_render():
    """All 5 map module UIs render without error."""
    from pymarxan_shiny.modules.mapping.comparison_map import comparison_map_ui
    from pymarxan_shiny.modules.mapping.frequency_map import frequency_map_ui
    from pymarxan_shiny.modules.mapping.network_view import network_view_ui
    from pymarxan_shiny.modules.mapping.solution_map import solution_map_ui
    from pymarxan_shiny.modules.mapping.spatial_grid import spatial_grid_ui

    for ui_fn, name in [
        (solution_map_ui, "sol"),
        (spatial_grid_ui, "sg"),
        (frequency_map_ui, "fm"),
        (comparison_map_ui, "cm"),
        (network_view_ui, "nv"),
    ]:
        elem = ui_fn(name)
        assert elem is not None


@pytest.mark.integration
def test_color_functions_unchanged():
    """All color functions still exist and produce valid hex strings."""
    from pymarxan_shiny.modules.mapping.comparison_map import comparison_color
    from pymarxan_shiny.modules.mapping.frequency_map import frequency_color
    from pymarxan_shiny.modules.mapping.network_view import metric_color
    from pymarxan_shiny.modules.mapping.spatial_grid import cost_color, status_color

    assert cost_color(0.5).startswith("#")
    assert status_color(2).startswith("#")
    assert frequency_color(0.5).startswith("#")
    assert comparison_color(True, False).startswith("#")
    assert metric_color(0.5).startswith("#")
