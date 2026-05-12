"""Reusable help button for Shiny card headers.

Usage inside a @module.ui / @module.server pair:

    from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup

    @module.ui
    def my_ui():
        return ui.card(
            help_card_header("Page Title"),
            ...
        )

    @module.server
    def my_server(input, output, session, ...):
        help_server_setup(input, "my_help_key")
        ...
"""
from __future__ import annotations

from shiny import reactive, ui
from shiny.ui import CardItem


def help_card_header(title: str) -> CardItem:
    """Return a card header containing *title* and a small Help button.

    The button has id ``"help_btn"`` which is automatically namespaced by
    the enclosing ``@module.ui``.
    """
    return ui.card_header(
        ui.div(
            ui.span(title),
            ui.input_action_button(
                "help_btn",
                ui.span("\u2139\ufe0e Help"),
                class_="btn btn-sm btn-outline-info py-0 px-2",
            ),
            class_="d-flex justify-content-between align-items-center w-100",
        )
    )


def help_server_setup(input, help_key: str) -> None:
    """Create a reactive effect that opens a help modal when *help_btn* is
    clicked.  Call this once at the top of your ``@module.server`` function.

    Parameters
    ----------
    input : shiny.Inputs
        The module-namespaced input object.
    help_key : str
        Key into :data:`modules.help.help_content.HELP_CONTENT`.
    """
    from pymarxan_shiny.modules.help.help_content import HELP_CONTENT

    @reactive.effect
    @reactive.event(input.help_btn)
    def _show_help():
        content = HELP_CONTENT.get(
            help_key,
            ui.p("No help content is available for this page yet."),
        )
        m = ui.modal(
            content,
            title=ui.span(
                "\u2139\ufe0e Help",
                class_="text-info",
            ),
            easy_close=True,
            size="l",
        )
        ui.modal_show(m)
