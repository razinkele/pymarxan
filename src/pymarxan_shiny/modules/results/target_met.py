"""Target achievement Shiny module.

Shows which conservation targets are met/unmet for the current solution.
"""
from __future__ import annotations

from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from pymarxan.solvers.utils import _active_chance_terms
from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup

# One column group per chance term: (term, ptarget column, Solution gaps attribute,
# table column names). 2D (PROBMODE 3 / 5) keeps its Phase 18 names.
_CHANCE_GROUPS = (
    ("2d", "ptarget", "prob_shortfalls", ("ptarget", "P(met)", "prob_gap")),
    ("1d", "ptarget1d", "prob1d_shortfalls", ("ptarget1d", "P1D(met)", "prob1d_gap")),
)


def chance_columns(problem, solution, fid: int) -> dict[str, object]:
    """The probability-target columns of feature ``fid``'s row (pure; tested directly).

    One group per **active** chance term (``_active_chance_terms``: PROBMODE 3 → 2D,
    4 → 1D, 5 → both, each with its data) whose gaps the solution carries
    (``prob_shortfalls`` / ``prob1d_shortfalls``). P(met) = ptarget − gap (clamped at 0);
    a feature without that term's target (ptarget ≤ 0) shows "—".
    """
    active = _active_chance_terms(problem)
    feats = problem.features
    row = feats.loc[feats["id"] == fid]
    out: dict[str, object] = {}
    for term, pt_col, attr, (c_pt, c_met, c_gap) in _CHANCE_GROUPS:
        gaps = getattr(solution, attr, None)
        if term not in active or gaps is None:
            continue
        pt = float(row[pt_col].iloc[0]) if pt_col in feats.columns and len(row) else -1.0
        if pt > 0:
            gap = float(gaps.get(fid, 0.0))
            out[c_pt] = pt
            out[c_met] = round(max(0.0, pt - gap), 4)
            out[c_gap] = round(gap, 4)
        else:
            out[c_pt] = out[c_met] = out[c_gap] = "—"
    return out


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

        # Probability-target columns (chance_columns): one group per active chance term
        # (PROBMODE 3 → 2D, 4 → 1D, 5 → both) whose Z-score gaps the solution carries. Show
        # clumping columns only when any feature has target2 > 0 and the
        # solution carries a clump evaluation. Same for separation when
        # any feature is sep-active. Keeps the table tidy for legacy runs.
        has_clumping = (
            "target2" in p.features.columns
            and (p.features["target2"] > 0).any()
            and s.clump_shortfalls is not None
        )
        has_separation = (
            "sepnum" in p.features.columns
            and "sepdistance" in p.features.columns
            and (
                (p.features["sepnum"] > 1) & (p.features["sepdistance"] > 0)
            ).any()
            and s.sep_shortfalls is not None
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
            r.update(chance_columns(p, s, fid))
            if has_clumping:
                t2 = float(row.get("target2", 0.0))
                ct = int(row.get("clumptype", 0))
                if t2 > 0:
                    clump_short = s.clump_shortfalls.get(fid, 0.0)
                    r["target2"] = t2
                    r["clumptype"] = ct
                    r["clump_short"] = round(clump_short, 4)
                else:
                    r["target2"] = "—"
                    r["clumptype"] = "—"
                    r["clump_short"] = "—"
            if has_separation:
                sd = float(row.get("sepdistance", 0.0))
                sn = int(row.get("sepnum", 1))
                if sd > 0 and sn > 1:
                    sep_short = s.sep_shortfalls.get(fid, 0)
                    r["sepdistance"] = sd
                    r["sepnum"] = sn
                    r["sep_short"] = int(sep_short)
                else:
                    r["sepdistance"] = "—"
                    r["sepnum"] = "—"
                    r["sep_short"] = "—"
            rows.append(r)
        return pd.DataFrame(rows)
