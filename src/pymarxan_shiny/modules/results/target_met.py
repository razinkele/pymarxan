"""Target achievement Shiny module.

Shows which conservation targets are met/unmet for the current solution.
"""
from __future__ import annotations

from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup


@module.ui
def target_met_ui():
    return ui.card(
        help_card_header("Target Achievement"),
        ui.p(
            "Data table of conservation targets and whether each feature is "
            "adequately represented in the current solution. Sortable and "
            "filterable for quick inspection of individual features.",
            class_="text-muted small mb-3",
        ),
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
    help_server_setup(input, "target_met")

    @render.data_frame
    def target_table():
        import pandas as pd

        p = problem()
        s = solution()
        if p is None or s is None:
            return None

        # Show probability-target columns only when PROBMODE 3 is active
        # and the solution actually carries a Z-score evaluation. This
        # makes the table self-explanatory without clutter for legacy runs.
        is_probmode3 = (
            int(p.parameters.get("PROBMODE", 0)) == 3
            and s.prob_shortfalls is not None
        )

        rows = []
        for _, row in p.features.iterrows():
            fid = int(row["id"])
            met = s.targets_met.get(fid, False)
            r = {
                "feature_id": fid,
                "name": row["name"],
                "target": float(row["target"]),
                "met": "Yes" if met else "No",
            }
            if is_probmode3:
                ptarget = (
                    float(row["ptarget"])
                    if "ptarget" in p.features.columns
                    else -1.0
                )
                if ptarget > 0:
                    shortfall = s.prob_shortfalls.get(fid, 0.0)
                    # P(target met) = ptarget - shortfall (clamped at 0)
                    prob_met = max(0.0, ptarget - shortfall)
                    r["ptarget"] = ptarget
                    r["P(met)"] = round(prob_met, 4)
                    r["prob_gap"] = round(shortfall, 4)
                else:
                    r["ptarget"] = "—"
                    r["P(met)"] = "—"
                    r["prob_gap"] = "—"
            rows.append(r)
        return pd.DataFrame(rows)
