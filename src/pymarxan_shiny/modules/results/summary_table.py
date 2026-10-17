"""Results summary Shiny module — target achievement table."""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution
from pymarxan.solvers.utils import check_targets, compute_held
from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup


def _target_rows(problem: ConservationProblem, solution: Solution) -> list[dict]:
    """One row per feature: held (Σ r·x for proportions) and met per ``check_targets``
    (MISSLEVEL applied; tolerant for fractional plans, Stage 4 §6.1 / §6.5)."""
    pu_index = {int(pid): i for i, pid in enumerate(problem.planning_units["id"].values)}
    held = compute_held(problem, solution.selected, pu_index)
    met = (
        check_targets(problem, solution.selected, pu_index, _achieved=held)
        if "target" in problem.features.columns
        else {}
    )
    rows = []
    for _, frow in problem.features.iterrows():
        fid = int(frow["id"])
        fname = frow.get("name", f"Feature {fid}")
        target = float(frow.get("target", 0.0))
        achieved = float(held.get(fid, 0.0))
        pct = (achieved / target * 100) if target > 0 else 100.0
        rows.append({
            "id": fid, "name": fname, "target": target,
            "achieved": achieved, "pct": pct, "met": bool(met.get(fid, True)),
        })
    return rows


@module.ui
def summary_table_ui():
    return ui.card(
        help_card_header("Target Achievement"),
        ui.p(
            "Summary table showing whether each conservation feature's target "
            "is met by the current solution. The 'target' column is the minimum "
            "amount required; 'achieved' is what the selected reserve network "
            "provides. Green \u2713 = target met; red \u2717 = target not met.",
            class_="text-muted small mb-3",
        ),
        ui.output_ui("target_table"),
    )

@module.server
def summary_table_server(
    input, output, session, problem: reactive.Value, solution: reactive.Value,
):
    help_server_setup(input, "summary_table")

    @render.ui
    def target_table():
        p = problem()
        s = solution()
        if p is None or s is None:
            return ui.p("No solution available. Run a solver first.")
        rows = _target_rows(p, s)
        met_count = sum(1 for r in rows if r["met"])
        total = len(rows)
        summary_line = ui.p(
            f"{met_count} of {total} targets met",
            class_="fw-bold mb-2",
        )
        table_rows = [
            ui.tags.tr(
                ui.tags.td(str(r["id"])), ui.tags.td(r["name"]),
                ui.tags.td(f"{r['target']:.1f}"), ui.tags.td(f"{r['achieved']:.1f}"),
                ui.tags.td(f"{r['pct']:.1f}%"),
                ui.tags.td(
                    "\u2713 Met" if r["met"] else "\u2717 NOT MET",
                    style=f"color: {'#2d936c' if r['met'] else '#c1440e'}; font-weight: bold",
                ),
            ) for r in rows
        ]
        return ui.TagList(
            summary_line,
            ui.tags.table(
                ui.tags.caption("Feature target achievement summary"),
                ui.tags.thead(ui.tags.tr(
                    ui.tags.th("ID", scope="col"),
                    ui.tags.th("Feature", scope="col"),
                    ui.tags.th("Target", scope="col"),
                    ui.tags.th("Achieved", scope="col"),
                    ui.tags.th("%", scope="col"),
                    ui.tags.th("Status", scope="col"),
                )),
                ui.tags.tbody(*table_rows),
                class_="table table-striped",
            ),
        )
