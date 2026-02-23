"""Results summary Shiny module — target achievement table."""
from __future__ import annotations

from shiny import module, reactive, render, ui


@module.ui
def summary_table_ui():
    return ui.card(ui.card_header("Target Achievement"), ui.output_ui("target_table"))

@module.server
def summary_table_server(
    input, output, session, problem: reactive.Value, solution: reactive.Value,
):
    @render.ui
    def target_table():
        p = problem()
        s = solution()
        if p is None or s is None:
            return ui.p("No solution available. Run a solver first.")
        pu_ids = p.planning_units["id"].values
        id_to_idx = {int(pid): i for i, pid in enumerate(pu_ids)}

        # Vectorized: map PU IDs to indices, filter selected, groupby feature
        pvf = p.pu_vs_features.copy()
        pvf["_idx"] = pvf["pu"].map(id_to_idx)
        pvf = pvf.dropna(subset=["_idx"])
        pvf["_idx"] = pvf["_idx"].astype(int)
        pvf["_selected"] = pvf["_idx"].map(lambda i: bool(s.selected[i]))
        achieved_by_feat = (
            pvf[pvf["_selected"]].groupby("species")["amount"].sum()
        )

        rows = []
        for _, frow in p.features.iterrows():
            fid = int(frow["id"])
            fname = frow.get("name", f"Feature {fid}")
            target = float(frow.get("target", 0.0))
            achieved = float(achieved_by_feat.get(fid, 0.0))
            met = achieved >= target
            pct = (achieved / target * 100) if target > 0 else 100.0
            rows.append({
                "id": fid, "name": fname, "target": target,
                "achieved": achieved, "pct": pct, "met": met,
            })
        table_rows = [
            ui.tags.tr(
                ui.tags.td(str(r["id"])), ui.tags.td(r["name"]),
                ui.tags.td(f"{r['target']:.1f}"), ui.tags.td(f"{r['achieved']:.1f}"),
                ui.tags.td(f"{r['pct']:.1f}%"),
                ui.tags.td("Met" if r["met"] else "NOT MET",
                           style=f"color: {'green' if r['met'] else 'red'}; font-weight: bold"),
            ) for r in rows
        ]
        return ui.tags.table(
            ui.tags.thead(ui.tags.tr(
                ui.tags.th("ID"), ui.tags.th("Feature"), ui.tags.th("Target"),
                ui.tags.th("Achieved"), ui.tags.th("%"), ui.tags.th("Status"),
            )),
            ui.tags.tbody(*table_rows),
            class_="table table-striped",
        )
