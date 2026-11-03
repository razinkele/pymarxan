"""Probability data upload and configuration Shiny module.

Allows users to upload a prob.dat file, set PROBABILITYWEIGHTING, PROBMODE (0–5) and
PROBZEROVAR, and preview the loaded probability data (prob.dat and the pu.dat ``prob``
column of Probability 1D).

The panel's decisions are pure helpers (unit-tested; there is no module-session
harness): Shiny cannot tell a user's radio change from ``ui.update_radio_buttons``, so
the radios are initialised from the problem (``initial_mode_choice`` /
``initial_zero_var_choice``) and an effect writes a parameter only when the radio differs
from the value in force (``mode_to_write`` / ``zero_var_to_write``). The weight slider
follows the project the same way (``initial_weight`` / ``weight_to_write``) and has its own
effect, which writes only ``PROBABILITYWEIGHTING``.

Every write builds a new problem (``with_parameter`` / ``with_probability``): Shiny's
``reactive.Value.set`` ignores the object it already holds, so editing the problem in place
and setting it again would notify nothing that reads it.
"""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan.models.problem import ConservationProblem, _probzerovar_value
from pymarxan.solvers.utils import _probmode
from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup

MODE_CHOICES: dict[str, str] = {
    "0": "Mode 0 — None",
    "1": "Mode 1 — Risk premium on cost (prob.dat)",
    "2": "Mode 2 — Expected (persistence-adjusted) amounts (prob.dat)",
    "3": "Mode 3 — Probability 2D (puvspr prob, ptarget)",
    "4": "Mode 4 — Probability 1D (pu.dat prob or prob.dat, ptarget1d)",
    "5": "Mode 5 — Probability 1D + 2D",
}

ZERO_VAR_CHOICES: dict[str, str] = {
    "0": "0 — Marxan (Z = 4)",
    "1": "1 — Counts as met",
    "2": "2 — Exact (certain amount)",
}


DEFAULT_WEIGHT = 1.0  # the solvers' default PROBABILITYWEIGHTING


def with_parameter(
    problem: ConservationProblem, key: str, value: object
) -> ConservationProblem:
    """A copy of ``problem`` with ``parameters[key] = value``; ``problem`` is unchanged."""
    return problem.copy_with(parameters={**problem.parameters, key: value})


def with_probability(problem: ConservationProblem, probability) -> ConservationProblem:
    """A copy of ``problem`` with the prob.dat table replaced; ``problem`` is unchanged."""
    return problem.copy_with(probability=probability)


def initial_weight(problem: ConservationProblem) -> float:
    """The weight slider's value: PROBABILITYWEIGHTING in force (the solvers' default 1.0
    when unset)."""
    return float(problem.parameters.get("PROBABILITYWEIGHTING", DEFAULT_WEIGHT))


def weight_to_write(problem: ConservationProblem, slider: float) -> float | None:
    """The PROBABILITYWEIGHTING to write for ``slider``, or None when it is in force."""
    value = float(slider)
    return None if value == initial_weight(problem) else value


def initial_mode_choice(problem: ConservationProblem) -> str:
    """The PROBMODE radio's value for ``problem``: the mode in force (explicit, or
    selected from the data, ``_probmode``)."""
    return str(_probmode(problem))


def mode_to_write(problem: ConservationProblem, radio: str) -> int | None:
    """The PROBMODE to write for radio value ``radio``, or None to write nothing (the
    radio already shows the mode in force, e.g. after ``update_radio_buttons``)."""
    mode = int(radio)
    return None if mode == _probmode(problem) else mode


def initial_zero_var_choice(problem: ConservationProblem) -> str:
    """The PROBZEROVAR radio's value: the parameter's value (0 when unset or invalid)."""
    value = _probzerovar_value(problem.parameters.get("PROBZEROVAR", 0))
    return str(value if value is not None else 0)


def zero_var_to_write(problem: ConservationProblem, radio: str) -> int | None:
    """The PROBZEROVAR to write for ``radio``, or None when it is already in force."""
    value = int(radio)
    return None if radio == initial_zero_var_choice(problem) else value


@module.ui
def probability_config_ui():
    return ui.card(
        help_card_header("Probability Configuration"),
        ui.p(
            "Per-planning-unit probabilities of loss come from a prob.dat upload "
            "(columns: pu, probability) or from a 'prob' column in pu.dat (Marxan's "
            "Probability 1D). Per-cell probabilities of presence come from a 'prob' "
            "column in puvspr.dat (Probability 2D). PROBMODE chooses how they enter "
            "the objective; PROBABILITYWEIGHTING scales the penalty.",
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
                    "Marxan .dat format. A 'prob' column in pu.dat takes precedence "
                    "for Probability 1D. The probability can also be read as the chance "
                    "that the unit's contribution is lost, e.g. a restoration action "
                    "fails or its funding ends.",
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
                    "cost) and on the chance-term penalties of PROBMODE 3–5. Not used "
                    "by PROBMODE 2.",
                ),
                ui.tooltip(
                    ui.input_radio_buttons(
                        "prob_mode",
                        "PROBMODE",
                        choices=MODE_CHOICES,
                        selected="0",
                    ),
                    "Initialised from the project: an explicit PROBMODE, or the mode its "
                    "data selects (pu.dat prob or prob.dat with ptarget1d > 0 → 4; "
                    "puvspr prob with ptarget > 0 → 3; both → 5). "
                    "Modes 1 and 2 are pymarxan extensions over prob.dat. Mode 1 adds a "
                    "risk premium probability × cost × weight. Mode 2 counts each "
                    "amount as amount × (1 − probability) — an average, not a "
                    "guarantee. Modes 3–5 are Marxan's Z-score chance terms, a soft "
                    "penalty on the probability shortfall per feature, with no cost "
                    "premium and raw amounts: 3 = Probability 2D (per-cell presence, "
                    "spec ptarget2d), 4 = Probability 1D (per-unit loss probability, "
                    "spec ptarget1d), 5 = both. Simulated annealing and iterative "
                    "improvement optimise them; the greedy heuristic and Zonation "
                    "ignore them while selecting; MIP drops them and reports the gap "
                    "post-hoc.",
                ),
                ui.tooltip(
                    ui.input_radio_buttons(
                        "prob_zero_var",
                        "PROBZEROVAR",
                        choices=ZERO_VAR_CHOICES,
                        selected="0",
                    ),
                    "How the chance terms score a feature whose selected cells carry no "
                    "variance. 0 (Marxan): Z = 4, so it fails any ptarget above "
                    "3.2e-5. 1: counts as met. 2: exact — its certain amount meets the "
                    "target (no penalty) or misses it (full penalty).",
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
    def _sync_radios():
        # Follow the loaded problem. The resulting input change reaches _update_mode /
        # _update_zero_var, which write nothing because the value is already in force.
        p = problem()
        if p is None:
            return
        ui.update_radio_buttons("prob_mode", selected=initial_mode_choice(p))
        ui.update_radio_buttons("prob_zero_var", selected=initial_zero_var_choice(p))
        ui.update_slider("prob_weight", value=initial_weight(p))

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
            problem.set(with_probability(p, prob_df))
            ui.notification_show(
                f"Loaded probability data for {len(prob_df)} planning units.",
                type="message",
            )
        except Exception as exc:
            ui.notification_show(f"Error reading prob.dat: {exc}", type="error")

    @reactive.effect
    @reactive.event(input.prob_mode)
    def _update_mode():
        p = problem()
        if p is None:
            return
        mode = mode_to_write(p, input.prob_mode())
        if mode is None:
            return
        problem.set(with_parameter(p, "PROBMODE", mode))

    @reactive.effect
    @reactive.event(input.prob_zero_var)
    def _update_zero_var():
        p = problem()
        if p is None:
            return
        value = zero_var_to_write(p, input.prob_zero_var())
        if value is None:
            return
        problem.set(with_parameter(p, "PROBZEROVAR", value))

    @reactive.effect
    @reactive.event(input.prob_weight)
    def _update_weight():
        p = problem()
        if p is None:
            return
        value = weight_to_write(p, input.prob_weight())
        if value is None:
            return
        problem.set(with_parameter(p, "PROBABILITYWEIGHTING", value))

    @render.text
    def prob_preview():
        p = problem()
        if p is None:
            return "No project loaded."
        lines: list[str] = []
        if p.probability is not None:
            vals = p.probability["probability"]
            lines += [
                f"prob.dat planning units: {len(vals)}",
                f"  min / max / mean: {vals.min():.4f} / {vals.max():.4f} / "
                f"{vals.mean():.4f}",
            ]
        if "prob" in p.planning_units.columns:
            vals = p.planning_units["prob"].astype(float)
            lines += [
                "pu.dat prob (Probability 1D loss):",
                f"  min / max / mean: {vals.min():.4f} / {vals.max():.4f} / "
                f"{vals.mean():.4f}",
            ]
        if not lines:
            lines.append("No per-unit probability data loaded.")
        lines += [
            "",
            f"PROBMODE in force: {_probmode(p)}"
            + ("" if "PROBMODE" in p.parameters else " (selected from the data)"),
            f"PROBZEROVAR: {initial_zero_var_choice(p)}",
            f"PROBABILITYWEIGHTING: {initial_weight(p)}",
        ]
        return "\n".join(lines)
