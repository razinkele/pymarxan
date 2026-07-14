"""Zonation priority-ranking Shiny panel (Phase D).

Given the reactive ``problem``, ranks every planning unit by iterative backward
removal (CAZ/ABF) and shows a priority-rank choropleth + per-feature performance
curves. Mirrors ``rivers_panel`` / ``frequency_map`` conventions; the plot is
server-side matplotlib (never plotly-via-render.ui).
"""
from __future__ import annotations

import pandas as pd
from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from pymarxan.models.geometry import generate_grid
from pymarxan.models.problem import has_geometry
from pymarxan.solvers.zonation_solver import ZonationSolver
from pymarxan.zonation import ZonationResult, rank_removal
from pymarxan_shiny.modules.mapping.ocean_palette import frequency_color

try:
    from shinywidgets import output_widget, render_widget

    from pymarxan_shiny.modules.mapping.map_utils import (
        create_geo_map,
        create_grid_map,
    )

    _HAS_IPYLEAFLET = True
except ImportError:
    _HAS_IPYLEAFLET = False

_RULES = {
    "caz": "Core-area (CAZ) — rarity",
    "abf": "Additive benefit (ABF) — richness",
}


def rank_to_colors(priority_rank: dict[int, float], pu_ids: list[int]) -> list[str]:
    """Hex color per PU (aligned to ``pu_ids``) from its priority rank, via the
    ocean palette (high rank = dark navy). Unranked PUs default to rank 0."""
    return [frequency_color(priority_rank.get(int(pid), 0.0)) for pid in pu_ids]


def performance_curve_frame(result: ZonationResult) -> pd.DataFrame:
    """The performance-curve DataFrame the plot draws."""
    return result.performance_curves


def zonation_solver_from_config(config_dict: dict) -> ZonationSolver:
    """Build a ``ZonationSolver`` from a solver-picker config dict (top_fraction
    clamped to (0, 1] so a typed out-of-range value can't raise mid-reactive)."""
    top = float(config_dict.get("zonation_top_fraction", 0.3) or 0.3)
    return ZonationSolver(
        rule=config_dict.get("zonation_rule", "caz"),
        top_fraction=min(1.0, max(0.05, top)),
    )


@module.ui
def zonation_panel_ui():
    map_output = (
        output_widget("map") if _HAS_IPYLEAFLET else ui.output_ui("map_msg")
    )
    return ui.card(
        ui.card_header("Zonation (priority ranking)"),
        ui.p(
            "Rank every planning unit by iterative backward removal. CAZ favors "
            "rarity (protects every feature's core); ABF favors species-rich "
            "cells. Darker = higher priority. Click Rank to compute. (For a "
            "target-meeting reserve, run 'Zonation' from the solver picker.)",
            class_="text-muted small mb-3",
        ),
        ui.input_select("rule", "Removal rule", _RULES),
        ui.input_action_button("rank", "Rank", class_="btn-primary mb-3"),
        map_output,
        ui.output_plot("curves"),
        ui.output_text("summary"),
        ui.output_data_frame("top_table"),
    )


@module.server
def zonation_panel_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    problem: reactive.Value,
):
    _result: reactive.Value = reactive.value(None)

    @reactive.effect
    @reactive.event(problem)
    def _reset_on_problem():
        # A newly-loaded project invalidates the previous ranking; drop it so the
        # map/curves return to the "click Rank" state instead of painting new PUs
        # against the old ranks.
        _result.set(None)

    @reactive.effect
    @reactive.event(input.rank)
    def _compute():
        p = problem()
        if p is None:
            _result.set(None)
            return
        try:
            _result.set(rank_removal(p, rule=input.rule()))
        except Exception as e:  # e.g. use_cost with a zero-cost PU
            _result.set(None)
            ui.notification_show(f"Ranking failed: {e}", type="error")

    if _HAS_IPYLEAFLET:

        @render_widget
        def map():
            p = problem()
            res = _result()
            if p is None or res is None:
                return None
            pu_ids = [int(x) for x in p.planning_units["id"]]
            colors = rank_to_colors(res.priority_rank, pu_ids)
            if has_geometry(p):
                return create_geo_map(p.planning_units, colors)
            return create_grid_map(generate_grid(len(pu_ids)), colors)

    else:

        @render.ui
        def map_msg():
            return ui.p("Install ipyleaflet for the priority-rank map.")

    @render.plot
    def curves():
        import matplotlib.pyplot as plt

        res = _result()
        fig, ax = plt.subplots()
        if res is None:
            ax.text(0.5, 0.5, "Click Rank to compute.", ha="center", va="center")
            ax.axis("off")
            return fig
        df = performance_curve_frame(res)
        x = df["prop_landscape_remaining"]
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        for col in feat_cols:
            ax.plot(x, df[col], label=col)
        ax.set_xlabel("Proportion of landscape remaining")
        ax.set_ylabel("Proportion of feature retained")
        ax.set_title(f"Performance curves ({res.rule.upper()})")
        ax.invert_xaxis()  # remaining goes 1 -> 0 as worst cells are stripped
        if feat_cols:
            ax.legend(fontsize="small")
        return fig

    @render.text
    def summary():
        res = _result()
        if res is None:
            return "Click Rank to compute the priority ranking."
        return (
            f"{res.rule.upper()}: ranked {len(res.priority_rank)} planning "
            "units (darker = higher priority)."
        )

    @render.data_frame
    def top_table():
        res = _result()
        if res is None:
            return pd.DataFrame({"pu_id": [], "priority_rank": []})
        df = res.to_dataframe().sort_values("priority_rank", ascending=False)
        return df[["pu_id", "priority_rank"]].head(20)
