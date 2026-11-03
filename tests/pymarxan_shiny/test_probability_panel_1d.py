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
    weight_block = weight_block[:weight_block.index("@render.text")]
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


class TestPanelWritesANewProblem:
    """Review (2026-11-03, HIGH): the handlers edited ``p`` in place and called
    ``problem.set(p)``; Shiny's ``Value.set`` ignores the same object, so nothing that
    reads ``problem()`` re-ran (the preview, the radios, the run panel's MIP notice)."""

    def test_with_parameter_returns_a_new_problem_and_leaves_the_old_one(self):
        p = _problem({"PROBMODE": 0})
        q = pc.with_parameter(p, "PROBMODE", 4)
        assert q is not p
        assert q.parameters["PROBMODE"] == 4
        assert p.parameters["PROBMODE"] == 0
        assert q.parameters is not p.parameters

    def test_with_probability_returns_a_new_problem(self):
        p = _problem()
        df = pd.DataFrame({"pu": [1, 2], "probability": [0.1, 0.2]})
        q = pc.with_probability(p, df)
        assert q is not p
        assert q.probability is df
        assert p.probability is None

    def test_handlers_do_not_mutate_the_problem_in_place(self):
        src = Path(pc.__file__).read_text()
        assert "p.parameters[" not in src
        assert "p.probability =" not in src


class TestWeightFollowsTheProject:
    """Review (MEDIUM): the slider started at 1.0 whatever the project said, and the
    preview printed the slider rather than the weighting in force."""

    def test_initial_weight_reads_the_parameter_with_the_solvers_default(self):
        assert pc.initial_weight(_problem()) == 1.0
        assert pc.initial_weight(_problem({"PROBABILITYWEIGHTING": 10})) == 10.0

    def test_weight_to_write_is_none_when_already_in_force(self):
        p = _problem({"PROBABILITYWEIGHTING": 10})
        assert pc.weight_to_write(p, 10.0) is None
        assert pc.weight_to_write(p, 2.5) == 2.5
        assert pc.weight_to_write(_problem(), 1.0) is None

    def test_preview_reports_the_weighting_in_force_not_the_slider(self):
        src = Path(pc.__file__).read_text()
        assert 'f"PROBABILITYWEIGHTING: {input.prob_weight()}"' not in src
        assert "update_slider" in src


class TestFeatureEditsAreEnforced:
    """Review (MEDIUM): an edit the validator rejected was stored anyway."""

    def test_rejected_probability_target_raises(self):
        from pymarxan_shiny.modules.data.feature_table import patched_value

        with pytest.raises(ValueError, match="ptarget1d"):
            patched_value("ptarget1d", "1.5")
        with pytest.raises(ValueError, match="target"):
            patched_value("target", "-3")

    def test_accepted_edits_return_the_validated_value(self):
        from pymarxan_shiny.modules.data.feature_table import patched_value

        assert patched_value("ptarget1d", "-1") == -1.0
        assert patched_value("ptarget", "0.95") == 0.95
        assert patched_value("target", "12") == 12.0

    def test_unvalidated_columns_pass_through(self):
        from pymarxan_shiny.modules.data.feature_table import patched_value

        assert patched_value("name", "renamed") == "renamed"
