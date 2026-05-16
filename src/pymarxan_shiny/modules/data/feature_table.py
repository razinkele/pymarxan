"""Feature table editor Shiny module — editable target and SPF values."""
from __future__ import annotations

import copy

from shiny import module, reactive, render, ui

from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup

_COLUMN_ORDER = ["id", "name", "target", "spf", "ptarget", "target2", "clumptype"]
_EDITABLE_FLOAT_COLUMNS = ("target", "spf", "target2")
_EDITABLE_INT_COLUMNS = ("clumptype",)


def validate_feature_edit(column: str, value: str) -> float | int | None:
    """Validate an edit to a feature table cell.

    Returns the validated value (float for target/spf/target2, int for
    clumptype), or None if the edit should be rejected.

    Phase 19 columns:
      - target2 : float ≥ 0  (0 disables clumping for this feature)
      - clumptype : int ∈ {0, 1, 2}  (Marxan PartialPen4 semantics)
    """
    if column in _EDITABLE_FLOAT_COLUMNS:
        try:
            val = float(value)
        except (ValueError, TypeError):
            return None
        if val < 0:
            return None
        return val
    if column in _EDITABLE_INT_COLUMNS:
        try:
            ival = int(value)
        except (ValueError, TypeError):
            return None
        if ival not in (0, 1, 2):
            return None
        return ival
    return None


@module.ui
def feature_table_ui():
    return ui.card(
        help_card_header("Feature Targets & SPF"),
        ui.p(
            "Edit conservation targets and Species Penalty Factors (SPF) for each "
            "feature. The target is the minimum amount of each feature that must be "
            "represented in the reserve network. SPF controls how heavily Marxan "
            "penalises solutions that fail to meet a feature's target \u2014 higher SPF "
            "makes it more expensive to miss that target. ",
            ui.tags.strong("target2 / clumptype"),
            " are optional Marxan TARGET2 / CLUMPTYPE columns: set target2 > 0 to "
            "require the feature in a contiguous patch of at least that occupancy, "
            "scored by CLUMPTYPE 0 (binary), 1 (half-amount sub-target), or 2 "
            "(quadratic sub-target). Leave target2 = 0 to disable clumping. ",
            "Click a cell to edit, then Save Changes to apply.",
            class_="text-muted small mb-3",
        ),
        ui.output_data_frame("feature_grid"),
        ui.tooltip(
            ui.input_action_button(
                "save_changes", "Save Changes", class_="btn-warning w-100 mt-2"
            ),
            "Write edited target / SPF / ptarget / target2 / clumptype values back "
            "to the conservation problem. Changes take effect immediately for "
            "subsequent solver runs.",
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
        # Build the display columns dynamically — only show optional Phase 18/19
        # columns when they exist on the problem, so legacy projects without
        # ptarget / target2 / clumptype don't see empty columns in the editor.
        cols = ["id", "name", "target", "spf"]
        for opt in ("ptarget", "target2", "clumptype"):
            if opt in p.features.columns:
                cols.append(opt)
        df = p.features[cols].copy()
        return render.DataGrid(df, editable=True)

    @feature_grid.set_patch_fn
    def _(*, patch):
        col_idx = patch["column_index"]
        # `_COLUMN_ORDER` is the canonical full ordering; the actual rendered
        # grid omits columns missing from p.features, so we need to map the
        # patch's column_index through the currently rendered column list.
        p = problem()
        if p is None:
            return patch["value"]
        cols = ["id", "name", "target", "spf"]
        for opt in ("ptarget", "target2", "clumptype"):
            if opt in p.features.columns:
                cols.append(opt)
        col = cols[col_idx] if col_idx < len(cols) else ""
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
        # Edited columns: target, spf, plus any of ptarget/target2/clumptype
        # that are present on the data view (which mirrors p.features).
        edit_cols = ["target", "spf"]
        for opt in ("ptarget", "target2", "clumptype"):
            if opt in df.columns:
                edit_cols.append(opt)
        # Join on id to handle user-sorted/filtered views correctly
        edits = df.set_index("id")[edit_cols]
        for fid in edits.index:
            mask = updated.features["id"] == fid
            for col in edit_cols:
                if col == "clumptype":
                    updated.features.loc[mask, col] = int(edits.at[fid, col])
                else:
                    updated.features.loc[mask, col] = float(edits.at[fid, col])
        problem.set(updated)
        ui.notification_show("Feature targets saved.", type="message")
