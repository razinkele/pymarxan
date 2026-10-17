"""Probability data upload and configuration Shiny module.

Allows users to upload a prob.dat file, set PROBABILITYWEIGHTING and PROBMODE,
and preview loaded probability data.
"""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup


@module.ui
def probability_config_ui():
    return ui.card(
        help_card_header("Probability Configuration"),
        ui.p(
            "Upload a prob.dat file containing per-planning-unit probability "
            "values (columns: pu, probability). These represent the likelihood "
            "of each planning unit persisting in a viable state. Marxan uses "
            "these values with the PROBABILITYWEIGHTING parameter to incorporate "
            "risk into the objective function.",
            class_="text-muted small mb-3",
        ),
        ui.layout_sidebar(
            ui.sidebar(
                ui.tooltip(
                    ui.input_file(
                        "prob_file",
                        "Upload prob.dat",
                        accept=[".dat", ".csv", ".tsv", ".txt"],
                    ),
                    "Columns 'pu' and 'probability': the chance (0–1) that the "
                    "planning unit is lost (not that it persists). CSV, TSV, or "
                    "Marxan .dat format.",
                ),
                ui.hr(),
                ui.h5("Parameters"),
                ui.tooltip(
                    ui.input_slider(
                        "prob_weight",
                        "PROBABILITYWEIGHTING",
                        min=0.0,
                        max=100.0,
                        value=1.0,
                        step=0.1,
                    ),
                    "Weight on the PROBMODE 1 risk premium (probability of loss × "
                    "cost) and on the PROBMODE 3 probability shortfall. Not used by "
                    "PROBMODE 2.",
                ),
                ui.tooltip(
                    ui.input_radio_buttons(
                        "prob_mode",
                        "PROBMODE",
                        choices={
                            "1": "Mode 1 — Risk premium on cost (expected cost lost)",
                            "2": "Mode 2 — Expected (persistence-adjusted) amounts",
                            "3": "Mode 3 — Z-score probability penalty (soft chance "
                            "target, based on Marxan's Probability 2D)",
                        },
                        selected="1",
                    ),
                    "Modes 1 and 2 are pymarxan extensions over per-PU "
                    "probabilities (prob.dat); Marxan's Probability 1D is not yet "
                    "implemented. Mode 1 (the default with a probability table): "
                    "adds a risk premium probability × cost × weight to the "
                    "objective. Mode 2: each amount counts as amount × (1 − "
                    "probability), and targets are judged on these expected "
                    "amounts by every single-zone solver and report. This is an average, not a "
                    "guarantee: a target met on expected amounts has roughly even "
                    "odds of being met. "
                    "Mode 3: per-(PU, feature) probability of presence via the "
                    "optional `prob` column on puvspr.dat and a per-feature "
                    "`ptarget` on spec.dat (Marxan's column is `ptarget2d`); a soft "
                    "penalty on the probability shortfall, not a hard constraint. "
                    "SA / heuristic / iterative-improvement optimise it, MIP drops "
                    "the term and reports the gap post-hoc.",
                ),
                width=350,
            ),
            ui.output_text_verbatim("prob_preview"),
        ),
    )


@module.server
def probability_config_server(
    input,
    output,
    session,
    problem: reactive.Value,
):
    help_server_setup(input, "probability_config")

    @reactive.effect
    @reactive.event(input.prob_file)
    def _on_upload():
        file_info = input.prob_file()
        if file_info is None or len(file_info) == 0:
            return

        path = file_info[0]["datapath"]
        p = problem()
        if p is None:
            ui.notification_show(
                "Load a project first before uploading probability data.",
                type="warning",
            )
            return

        try:
            from pymarxan.io.readers import read_probability

            prob_df = read_probability(path)
            # Attach to the problem
            p.probability = prob_df
            problem.set(p)
            ui.notification_show(
                f"Loaded probability data for {len(prob_df)} planning units.",
                type="message",
            )
        except Exception as exc:
            ui.notification_show(f"Error reading prob.dat: {exc}", type="error")

    @reactive.effect
    @reactive.event(input.prob_weight, input.prob_mode)
    def _update_params():
        p = problem()
        if p is None:
            return
        p.parameters["PROBABILITYWEIGHTING"] = float(input.prob_weight())
        p.parameters["PROBMODE"] = int(input.prob_mode())
        problem.set(p)

    @render.text
    def prob_preview():
        p = problem()
        if p is None or p.probability is None:
            return "No probability data loaded."
        df = p.probability
        n = len(df)
        prob_vals = df["probability"]
        return (
            f"Planning units with probability data: {n}\n"
            f"Min probability:  {prob_vals.min():.4f}\n"
            f"Max probability:  {prob_vals.max():.4f}\n"
            f"Mean probability: {prob_vals.mean():.4f}\n"
            f"\nPROBMODE: {input.prob_mode()}\n"
            f"PROBABILITYWEIGHTING: {input.prob_weight()}"
        )
