"""Shiny module for river barrier-restoration (DCI) analysis."""
from __future__ import annotations

from pymarxan_shiny.modules.rivers.rivers_panel import (
    rivers_panel_server,
    rivers_panel_ui,
)

__all__ = ["rivers_panel_server", "rivers_panel_ui"]
