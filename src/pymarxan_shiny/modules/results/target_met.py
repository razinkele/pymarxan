"""Target achievement Shiny module.

Shows which conservation targets are met/unmet for the current solution.
"""
from __future__ import annotations

from shiny import Inputs, Outputs, Session, module, reactive, render, ui


@module.ui
def target_met_ui():
    return ui.card(
        ui.card_header("Target Achievement"),
        ui.output_data_frame("target_table"),
    )


@module.server
def target_met_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    problem: reactive.Value,
    solution: reactive.Value,
):
    @render.data_frame
    def target_table():
        import pandas as pd

        p = problem()
        s = solution()
        if p is None or s is None:
            return None

        rows = []
        for _, row in p.features.iterrows():
            fid = int(row["id"])
            met = s.targets_met.get(fid, False)
            rows.append({
                "feature_id": fid,
                "name": row["name"],
                "target": float(row["target"]),
                "met": "Yes" if met else "No",
            })
        return pd.DataFrame(rows)
