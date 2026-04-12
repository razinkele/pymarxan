"""Feature table editor Shiny module — editable target and SPF values."""
from __future__ import annotations

import copy

from shiny import module, reactive, render, ui

from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup

_COLUMN_ORDER = ["id", "name", "target", "spf"]


def validate_feature_edit(column: str, value: str) -> float | None:
    """Validate an edit to a feature table cell.

    Returns validated float, or None if edit is rejected.
    """
    if column not in ("target", "spf"):
        return None
    try:
        val = float(value)
    except (ValueError, TypeError):
        return None
    if val < 0:
        return None
    return val


@module.ui
def feature_table_ui():
    return ui.card(
        help_card_header("Feature Targets & SPF"),
        ui.p(
            "Edit conservation targets and Species Penalty Factors (SPF) for each "
            "feature. The target is the minimum amount of each feature that must be "
            "represented in the reserve network. SPF controls how heavily Marxan "
            "penalises solutions that fail to meet a feature's target \u2014 higher SPF "
            "makes it more expensive to miss that target. Click a cell to edit, then "
            "Save Changes to apply.",
            class_="text-muted small mb-3",
        ),
        ui.output_data_frame("feature_grid"),
        ui.tooltip(
            ui.input_action_button(
                "save_changes", "Save Changes", class_="btn-warning w-100 mt-2"
            ),
            "Write edited target and SPF values back to the conservation problem. "
            "Changes take effect immediately for subsequent solver runs.",
        ),
    )


@module.server
def feature_table_server(
    input,
    output,
    session,
    problem: reactive.Value,
):
    help_server_setup(input, "feature_table")

    @render.data_frame
    def feature_grid():
        p = problem()
        if p is None:
            return None
        df = p.features[["id", "name", "target", "spf"]].copy()
        return render.DataGrid(df, editable=True)

    @feature_grid.set_patch_fn
    def _(*, patch):
        col_idx = patch["column_index"]
        col = _COLUMN_ORDER[col_idx] if col_idx < len(_COLUMN_ORDER) else ""
        validated = validate_feature_edit(col, str(patch["value"]))
        if validated is not None:
            return validated
        return patch["value"]

    @reactive.effect
    @reactive.event(input.save_changes)
    def _save():
        p = problem()
        if p is None:
            return
        df = feature_grid.data_view()
        updated = copy.deepcopy(p)
        # Join on id to handle user-sorted/filtered views correctly
        edits = df.set_index("id")[["target", "spf"]]
        for fid in edits.index:
            mask = updated.features["id"] == fid
            updated.features.loc[mask, "target"] = float(edits.at[fid, "target"])
            updated.features.loc[mask, "spf"] = float(edits.at[fid, "spf"])
        problem.set(updated)
        ui.notification_show("Feature targets saved.", type="message")
