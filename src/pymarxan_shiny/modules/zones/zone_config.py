"""Zone configuration Shiny module for multi-zone problems."""
from __future__ import annotations

from shiny import module, reactive, render, ui

from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup


@module.ui
def zone_config_ui():
    return ui.card(
        help_card_header("Zone Configuration"),
        ui.p(
            "View the zone definitions and costs for multi-zone Marxan problems. "
            "In multi-zone planning, each planning unit can be assigned to one of "
            "several management zones (e.g. no-take, buffer, sustainable use). "
            "Zone data is loaded automatically from the project files.",
            class_="text-muted small mb-3",
        ),
        ui.layout_sidebar(
            ui.sidebar(
                ui.p("Zone project data is loaded via the Data tab."),
                ui.hr(),
                ui.output_text_verbatim("zone_summary"),
                width=300,
            ),
            ui.div(
                ui.output_text_verbatim("zone_details"),
                ui.output_text_verbatim("zone_cost_summary"),
            ),
        ),
    )


@module.server
def zone_config_server(
    input, output, session,
    zone_problem: reactive.Value,
):
    help_server_setup(input, "zone_config")

    @render.text
    def zone_summary():
        zp = zone_problem()
        if zp is None:
            return "No zone project loaded."
        return (
            f"Zones: {zp.n_zones}\n"
            f"Planning Units: {zp.n_planning_units}\n"
            f"Features: {zp.n_features}"
        )

    @render.text
    def zone_details():
        zp = zone_problem()
        if zp is None:
            return ""
        lines = ["Zone Definitions:"]
        for _, row in zp.zones.iterrows():
            lines.append(f"  Zone {int(row['id'])}: {row['name']}")
        return "\n".join(lines)

    @render.text
    def zone_cost_summary():
        zp = zone_problem()
        if zp is None:
            return ""
        lines = ["Zone Costs (avg per PU):"]
        for _, zrow in zp.zones.iterrows():
            zid = int(zrow["id"])
            zname = zrow["name"]
            costs = zp.zone_costs[zp.zone_costs["zone"] == zid]["cost"]
            avg = costs.mean() if len(costs) > 0 else 0.0
            lines.append(f"  {zname}: {avg:.2f}")
        return "\n".join(lines)
