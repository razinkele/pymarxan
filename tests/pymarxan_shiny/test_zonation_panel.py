"""Tests for the Zonation Shiny panel (Phase D)."""
from __future__ import annotations

import pandas as pd

from pymarxan.models.problem import ConservationProblem
from pymarxan.zonation import rank_removal


def _problem():
    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [1.0, 1.0, 1.0], "status": [0, 0, 0]})
    feats = pd.DataFrame(
        {"id": [1, 2], "name": ["a", "b"], "target": [1.0, 1.0], "spf": [1.0, 1.0]}
    )
    pvf = pd.DataFrame(
        {"species": [1, 2, 1, 2], "pu": [1, 2, 3, 3], "amount": [10.0, 10.0, 5.0, 5.0]}
    )
    return ConservationProblem(pu, feats, pvf)


def test_rank_to_colors_aligned_and_hex():
    from pymarxan_shiny.modules.zonation.zonation_panel import rank_to_colors

    ranks = {1: 1 / 3, 2: 2 / 3, 3: 1.0}
    colors = rank_to_colors(ranks, [1, 2, 3])
    assert len(colors) == 3
    assert all(c.startswith("#") and len(c) == 7 for c in colors)
    # higher rank -> darker (lower luminance) via the ocean palette
    lum = [int(c[1:3], 16) + int(c[3:5], 16) + int(c[5:7], 16) for c in colors]
    assert lum[0] > lum[1] > lum[2]  # rank 1/3 lightest, rank 1.0 darkest


def test_rank_to_colors_missing_pu_defaults_light():
    from pymarxan_shiny.modules.zonation.zonation_panel import rank_to_colors

    colors = rank_to_colors({1: 1.0}, [1, 99])  # 99 not ranked
    assert colors[0] != colors[1]  # ranked vs default differ


def test_performance_curve_frame_columns():
    from pymarxan_shiny.modules.zonation.zonation_panel import performance_curve_frame

    res = rank_removal(_problem(), rule="caz")
    df = performance_curve_frame(res)
    assert "prop_landscape_remaining" in df.columns
    assert any(c.startswith("feat_") for c in df.columns)


def test_module_exposes_ui_and_server():
    from pymarxan_shiny.modules.zonation import (
        zonation_panel_server,
        zonation_panel_ui,
    )

    assert callable(zonation_panel_ui)
    assert callable(zonation_panel_server)


def test_zonation_solver_from_config_maps_and_clamps():
    from pymarxan.solvers.zonation_solver import ZonationSolver
    from pymarxan_shiny.modules.zonation.zonation_panel import (
        zonation_solver_from_config,
    )

    s = zonation_solver_from_config(
        {"zonation_rule": "abf", "zonation_top_fraction": 0.5}
    )
    assert isinstance(s, ZonationSolver)
    assert s.rule == "abf"
    assert s.top_fraction == 0.5
    # defaults when keys absent
    d = zonation_solver_from_config({})
    assert d.rule == "caz" and d.top_fraction == 0.3
    # a typed out-of-range top_fraction is clamped, not raised
    assert zonation_solver_from_config({"zonation_top_fraction": 1.5}).top_fraction == 1.0


def test_solver_picker_offers_zonation():
    # the choice is actually rendered into the picker UI (@module.ui renders
    # session-free), not merely present in the source text.
    from pymarxan_shiny.modules.solver_config.solver_picker import solver_picker_ui

    assert "Zonation (rank-removal)" in str(solver_picker_ui("t"))
