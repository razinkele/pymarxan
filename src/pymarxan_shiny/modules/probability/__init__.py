"""Probability configuration Shiny module."""

from pymarxan_shiny.modules.probability.probability_config import (
    probability_config_server as server,
)
from pymarxan_shiny.modules.probability.probability_config import (
    probability_config_ui as ui,
)

__all__ = ["ui", "server"]
