"""Phase 18 Batch 4 — Shiny UI surface tests.

These are file-based smoke tests (matching the Review 6 H4 / H5 pattern):
verify the right strings appear in the right modules. We can't run a real
Shiny session here, but we can pin that the integration points the
implementation plan called out actually got touched.
"""
from __future__ import annotations

from pathlib import Path

import pymarxan_shiny.modules.help.help_content as help_content_mod
import pymarxan_shiny.modules.probability.probability_config as probability_config_mod
import pymarxan_shiny.modules.results.target_met as target_met_mod
import pymarxan_shiny.modules.run_control.run_panel as run_panel_mod


def test_probability_config_offers_mode_3():
    """probability_config.py — the dedicated PROBMODE radio — must
    include choice "3" (Z-score chance constraints) after Phase 18."""
    src = Path(probability_config_mod.__file__).read_text()
    assert '"3"' in src
    assert "Z-score" in src
    # Sanity: the previous modes still listed
    assert '"1"' in src and '"2"' in src


def test_target_met_module_references_zscore_columns():
    """target_met.py shows ptarget / P(met) / prob_gap columns when
    PROBMODE 3 and the Solution carries prob_shortfalls."""
    src = Path(target_met_mod.__file__).read_text()
    assert "prob_shortfalls" in src
    assert "ptarget" in src
    assert "PROBMODE" in src


def test_run_panel_shows_mip_drop_notice():
    """run_panel.py must conditionally show a notice when MIP is the
    active solver under PROBMODE 3 — the "drop" strategy isn't visible
    to the user without it."""
    src = Path(run_panel_mod.__file__).read_text()
    assert "probmode3_mip_notice" in src
    assert "deterministic relaxation" in src
    assert "PROBMODE" in src


def test_help_content_documents_probmode3():
    """The target_met help entry explains the new PROBMODE 3 columns
    and points users at the citations and the MIP caveat."""
    src = Path(help_content_mod.__file__).read_text()
    assert "PROBMODE 3" in src
    # At least one citation is mentioned
    assert "Game 2008" in src or "Tulloch 2013" in src
    # The MIP caveat appears
    assert "post-hoc" in src.lower()
