"""Shiny probability panel for Probability 1D (v0.48.0, spec §3.5 / §4 "Shiny").

There is no module-session harness, so the panel's decisions live in pure helpers that
are unit-tested here; source-text smoke tests pin the wiring. Shiny cannot tell a user's
radio change from ``ui.update_radio_buttons``, so the mode effect writes ``PROBMODE``
only when the radio differs from the mode in force (``mode_to_write``), and the weight
slider has its own effect that writes only ``PROBABILITYWEIGHTING``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import pymarxan_shiny.modules.data_input.upload as upload_mod
import pymarxan_shiny.modules.probability.probability_config as pc
from pymarxan.models.problem import ConservationProblem
from pymarxan_shiny.modules.data.feature_table import validate_feature_edit
from pymarxan_shiny.modules.help.help_content import HELP_CONTENT


def _problem(params=None, *, pt1d=0.9) -> ConservationProblem:
    return ConservationProblem(
        planning_units=pd.DataFrame({"id": [1, 2], "cost": [1.0, 1.0], "status": 0,
                                     "prob": [0.1, 0.2]}),
        features=pd.DataFrame({"id": [1], "name": ["a"], "target": [1.0], "spf": 1.0,
                               "ptarget1d": [pt1d]}),
        pu_vs_features=pd.DataFrame({"species": [1, 1], "pu": [1, 2], "amount": [1.0, 2.0]}),
        parameters=dict(params or {}),
    )


class TestModeHelpers:
    def test_implicit_4_initialises_to_4_and_writes_nothing(self):
        p = _problem()
        assert pc.initial_mode_choice(p) == "4"
        assert pc.mode_to_write(p, "4") is None

    def test_mode_0_problem_initialises_to_0(self):
        assert pc.initial_mode_choice(_problem(pt1d=-1.0)) == "0"

    def test_a_real_change_is_written(self):
        assert pc.mode_to_write(_problem(), "3") == 3

    def test_explicit_mode_initialises_to_itself(self):
        assert pc.initial_mode_choice(_problem({"PROBMODE": 2})) == "2"

    def test_radio_offers_0_to_5(self):
        assert list(pc.MODE_CHOICES) == ["0", "1", "2", "3", "4", "5"]


class TestZeroVarHelpers:
    def test_default_is_0_and_writes_nothing(self):
        p = _problem()
        assert pc.initial_zero_var_choice(p) == "0"
        assert pc.zero_var_to_write(p, "0") is None

    def test_change_is_written(self):
        assert pc.zero_var_to_write(_problem(), "2") == 2
        p = _problem({"PROBZEROVAR": 1})
        assert pc.initial_zero_var_choice(p) == "1"
        assert pc.zero_var_to_write(p, "1") is None

    def test_choices_are_0_1_2(self):
        assert list(pc.ZERO_VAR_CHOICES) == ["0", "1", "2"]


def test_mode_effect_uses_mode_to_write_and_slider_writes_only_the_weight():
    src = Path(pc.__file__).read_text()
    src = src[src.index("def probability_config_server"):]
    mode_block = src[src.index("def _update_mode"):src.index("def _update_weight")]
    weight_block = src[src.index("def _update_weight"):]
    weight_block = weight_block[:weight_block.index("problem.set")]
    assert "mode_to_write(" in mode_block
    assert "PROBABILITYWEIGHTING" in weight_block and "PROBMODE" not in weight_block
    assert "zero_var_to_write(" in src
    assert "initial_mode_choice(" in src  # the radio follows the loaded problem


def test_panel_text_no_longer_says_1d_is_missing():
    src = Path(pc.__file__).read_text()
    assert "not yet implemented" not in src
    assert "PROBZEROVAR" in src
    assert "restoration" in src  # the 1D prob read as the chance an action fails


@pytest.mark.parametrize("column", ["ptarget", "ptarget1d"])
@pytest.mark.parametrize("value, ok", [
    ("-1", -1.0), ("0.5", 0.5), ("1", 1.0), ("0", None), ("1.2", None), ("-0.5", None),
    ("abc", None),
])
def test_feature_table_probability_targets(column, value, ok):
    assert validate_feature_edit(column, value) == ok


def test_upload_shows_probability_diagnostics():
    src = Path(upload_mod.__file__).read_text()
    assert src.count("_probability_diagnostics(") >= 2  # zip and local loads


def test_help_documents_probability_1d():
    text = str(HELP_CONTENT["probability_config"])
    for needle in ("PROBMODE 4", "PROBMODE 5", "ptarget1d", "PROBZEROVAR", "restoration"):
        assert needle in text
    assert "Probability 1D" in str(HELP_CONTENT["target_met"])
