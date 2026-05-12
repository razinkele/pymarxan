"""Tests for results/export Shiny module."""
from pathlib import Path

import pymarxan_shiny.modules.results.export as export_module
import pymarxan_shiny.modules.spatial_export.spatial_export as spatial_export_module
from pymarxan_shiny.modules.results.export import export_server, export_ui


def test_export_ui_callable():
    assert callable(export_ui)


def test_export_server_callable():
    assert callable(export_server)


def test_export_module_registers_tempfile_cleanup():
    """Download handlers must register session.on_ended cleanup for tempfiles.

    Each download creates ``NamedTemporaryFile(delete=False)`` and returns its
    path. Without an explicit unlink on session end, every export leaks a
    file in /tmp for the server's lifetime.
    """
    src = Path(export_module.__file__).read_text()
    assert "session.on_ended" in src, "missing session-end cleanup registration"
    assert ".unlink" in src, "cleanup must actually unlink the tempfiles"


def test_spatial_export_module_registers_tempfile_cleanup():
    """Same cleanup contract for the spatial export module."""
    src = Path(spatial_export_module.__file__).read_text()
    assert "session.on_ended" in src
    assert ".unlink" in src
