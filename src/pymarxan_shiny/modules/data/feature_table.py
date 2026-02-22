"""Feature table editor Shiny module — editable target and SPF values."""
from __future__ import annotations

from shiny import module, reactive, render, ui


def validate_feature_edit(column: str, value: str) -> float | None:
    """Validate an edit to a feature table cell.

    Parameters
    ----------
    column : str
        Column name being edited.
    value : str
        New value as string.

    Returns
    -------
    float | None
        Validated float value, or None if the edit is rejected.
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
        ui.card_header("Feature Targets & SPF"),
        ui.output_data_frame("feature_grid"),
    )


@module.server
def feature_table_server(
    input,
    output,
    session,
    problem: reactive.Value,
):
    @render.data_frame
    def feature_grid():
        p = problem()
        if p is None:
            return None
        df = p.features[["id", "name", "target", "spf"]].copy()
        return render.DataGrid(df, editable=True)

    @feature_grid.set_patch_fn
    def _(*, patch):
        col = patch["column_id"]
        validated = validate_feature_edit(col, str(patch["value"]))
        if validated is not None:
            return validated
        # Reject edit by returning original value
        return patch["prev_value"]
