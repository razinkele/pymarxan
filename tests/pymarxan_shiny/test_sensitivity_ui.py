"""Tests for sensitivity dashboard Shiny module."""
from __future__ import annotations

from pymarxan_shiny.modules.calibration.sensitivity_ui import (
    build_sensitivity_config,
    sensitivity_server,
    sensitivity_ui,
)


def test_sensitivity_ui_returns_tag():
    ui_elem = sensitivity_ui("test_sens")
    assert ui_elem is not None


def test_sensitivity_server_callable():
    assert callable(sensitivity_server)


def test_build_config_defaults():
    """Default config has 5 multipliers centered on 1.0."""
    config = build_sensitivity_config(min_mult=0.8, max_mult=1.2, steps=5)
    assert len(config.multipliers) == 5
    assert 1.0 in config.multipliers
    assert config.multipliers[0] == 0.8
    assert config.multipliers[-1] == 1.2


def test_build_config_custom_range():
    """Custom multiplier range with 3 steps."""
    config = build_sensitivity_config(min_mult=0.5, max_mult=1.5, steps=3)
    assert len(config.multipliers) == 3
    assert config.multipliers[0] == 0.5
    assert config.multipliers[-1] == 1.5
