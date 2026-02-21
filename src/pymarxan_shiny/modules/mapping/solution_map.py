"""Solution map Shiny module — displays selected planning units."""
from __future__ import annotations

from shiny import module, reactive, render, ui


@module.ui
def solution_map_ui():
    return ui.card(ui.card_header("Solution Map"), ui.output_ui("map_or_table"))

@module.server
def solution_map_server(input, output, session, problem: reactive.Value, solution: reactive.Value):
    @render.ui
    def map_or_table():
        p = problem()
        s = solution()
        if p is None or s is None:
            return ui.p("Run a solver to see results here.")
        pu_ids = p.planning_units["id"].tolist()
        costs = p.planning_units["cost"].tolist()
        rows = [
            {"pu": pid, "cost": cost}
            for i, (pid, cost) in enumerate(zip(pu_ids, costs))
            if s.selected[i]
        ]
        if not rows:
            return ui.p("No planning units selected.")
        header = ui.div(
            ui.h5("Solution Summary"),
            ui.p(f"Selected: {s.n_selected} / {len(pu_ids)} planning units"),
            ui.p(f"Cost: {s.cost:.2f}"),
            ui.p(f"Boundary: {s.boundary:.2f}"),
            ui.p(f"Objective: {s.objective:.2f}"),
            ui.p(f"Targets met: {sum(s.targets_met.values())} / {len(s.targets_met)}"),
        )
        table_rows = [
            ui.tags.tr(ui.tags.td(str(r["pu"])), ui.tags.td(f"{r['cost']:.1f}"))
            for r in rows
        ]
        table = ui.tags.table(
            ui.tags.thead(ui.tags.tr(ui.tags.th("Planning Unit"), ui.tags.th("Cost"))),
            ui.tags.tbody(*table_rows),
            class_="table table-striped table-sm",
        )
        return ui.div(header, table)
