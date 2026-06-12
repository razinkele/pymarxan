"""Smoke test: the representation Shiny module imports and exposes ui/server."""
from __future__ import annotations


def test_module_exposes_ui_and_server():
    from pymarxan_shiny.modules.results.representation import (
        representation_server,
        representation_ui,
    )

    assert callable(representation_ui)
    assert callable(representation_server)
