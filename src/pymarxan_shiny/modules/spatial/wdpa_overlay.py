"""WDPA protected area overlay Shiny module."""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup
from pymarxan.models.problem import STATUS_INITIAL_INCLUDE, STATUS_LOCKED_IN, has_geometry
from pymarxan.spatial.wdpa import apply_wdpa_status, fetch_wdpa


@module.ui
def wdpa_overlay_ui():
    return ui.card(
        help_card_header("Protected Areas (WDPA)"),
        ui.p(
            "Overlay existing protected areas from the World Database on Protected "
            "Areas (WDPA) onto your planning units. Planning units that overlap "
            "sufficiently with protected areas can be locked in (status=2) or "
            "given initial-include status (status=1), reflecting existing conservation.",
            class_="text-muted small mb-3",
        ),
        ui.tooltip(
            ui.input_password("api_token", "API Token (optional)", value=""),
            "Protected Planet API token for accessing WDPA data. "
            "Optional — a default public endpoint is used if left blank. "
            "Get a token at protectedplanet.net/en/thematic-areas/wdpa.",
        ),
        ui.layout_columns(
            ui.tooltip(
                ui.input_slider(
                    "threshold", "Overlap Threshold",
                    min=0.1, max=1.0, value=0.5, step=0.1,
                ),
                "Minimum fraction of a planning unit's area that must overlap "
                "with a protected area to be marked as protected.",
            ),
            ui.tooltip(
                ui.input_select(
                    "status",
                    "Set Status",
                    {
                        str(STATUS_LOCKED_IN): "Locked In (2)",
                        str(STATUS_INITIAL_INCLUDE): "Initial Include (1)",
                    },
                ),
                "Marxan status to assign to protected planning units. "
                "'Locked In' (2) forces inclusion; 'Initial Include' (1) "
                "starts them in the solution but allows removal.",
            ),
            col_widths=[6, 6],
        ),
        ui.tooltip(
            ui.input_action_button(
                "fetch_wdpa", "Fetch & Apply Protected Areas", class_="btn-primary",
            ),
            "Download WDPA polygons for the planning unit extent, compute "
            "overlap, and update planning unit status codes.",
        ),
        ui.output_text_verbatim("wdpa_info"),
    )


@module.server
def wdpa_overlay_server(input, output, session, problem: reactive.Value):
    help_server_setup(input, "wdpa_overlay")

    @reactive.effect
    @reactive.event(input.fetch_wdpa)
    def _fetch_and_apply():
        p = problem()
        if p is None:
            ui.notification_show("Load a project first.", type="warning")
            return
        if not has_geometry(p):
            ui.notification_show(
                "Planning units need geometry. Generate a grid first.",
                type="warning",
            )
            return

        try:
            bounds = tuple(p.planning_units.total_bounds)
            token = input.api_token() or None
            wdpa = fetch_wdpa(bounds=bounds, api_token=token)
            if len(wdpa) == 0:
                ui.notification_show(
                    "No protected areas found in this region.", type="warning",
                )
                return

            status_val = int(input.status())
            result = apply_wdpa_status(
                p,
                wdpa,
                overlap_threshold=input.threshold(),
                status=status_val,
            )
            n_marked = int(
                (result.planning_units["status"] != p.planning_units["status"]).sum()
            )
            problem.set(result)
            ui.notification_show(f"Marked {n_marked} PUs as protected.", type="message")
        except Exception as e:
            ui.notification_show(f"WDPA error: {e}", type="error")

    @render.text
    def wdpa_info():
        p = problem()
        if p is None:
            return "Load a project to use WDPA overlay."
        n_locked = int((p.planning_units["status"] == STATUS_LOCKED_IN).sum())
        return f"Currently {n_locked} PUs locked in"
