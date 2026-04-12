"""Reusable server-side directory browser Shiny module.

Provides a text display of the selected path plus a "Browse…" button that
opens a modal dialog letting users navigate the server filesystem and pick
a directory.
"""
from __future__ import annotations

from pathlib import Path

from shiny import module, reactive, render, ui


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_directory(path: Path) -> list[dict[str, str]]:
    """Return sorted child entries of *path* that are directories.

    Each entry is ``{"name": ..., "path": ...}``.  Hidden dirs (starting
    with ``'.'``) are excluded.  Returns an empty list when *path* is not a
    readable directory.
    """
    try:
        entries = sorted(
            (
                {"name": d.name, "path": str(d)}
                for d in path.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ),
            key=lambda e: e["name"].lower(),
        )
    except (PermissionError, FileNotFoundError, OSError):
        entries = []
    return entries


def _looks_like_marxan_project(path: Path) -> bool:
    """Quick heuristic: does *path* contain an ``input.dat`` file?"""
    return (path / "input.dat").is_file()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

@module.ui
def directory_browser_ui(
    label: str = "Project directory",
    button_label: str = "Browse\u2026",
):
    """Render a read-only text display + browse button.

    Parameters
    ----------
    label:
        Label shown above the selected-path display.
    button_label:
        Text on the button that opens the browser modal.
    """
    return ui.TagList(
        ui.output_text_verbatim("selected_path_display"),
        ui.input_action_button("open_browser", button_label,
                               class_="btn-outline-secondary btn-sm mt-1"),
    )


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

@module.server
def directory_browser_server(
    input,
    output,
    session,
    *,
    start_path: str | Path | None = None,
    selected_path: reactive.Value | None = None,
):
    """Drive the directory-browser modal.

    Parameters
    ----------
    start_path:
        Initial directory shown when the modal opens for the first time.
        Defaults to the user's home directory.
    selected_path:
        An *external* ``reactive.Value[str | None]`` that will be updated
        when the user confirms a selection.  If ``None`` a private reactive
        is created (retrieve it via the return value).
    """
    if start_path is None:
        start_path = str(Path.home())
    start_path = str(start_path)

    # Internal reactive holding the *browsing* path (what the modal shows).
    _browsing = reactive.Value(start_path)

    # The confirmed selected path.
    if selected_path is None:
        selected_path = reactive.Value(None)

    # ------ Display in the main UI ------------------------------------

    @render.text
    def selected_path_display():
        p = selected_path()
        if p is None:
            return "(no directory selected)"
        return str(p)

    # ------ Modal construction ----------------------------------------

    def _build_modal():
        """Build the modal dialog showing the current browsing directory."""
        cur = Path(_browsing())
        entries = _list_directory(cur)
        is_project = _looks_like_marxan_project(cur)

        dir_buttons = [
            ui.div(
                ui.input_action_button(
                    f"goto_{i}",
                    f"\U0001F4C1 {entry['name']}",
                    class_="btn btn-link p-0 text-start",
                ),
                class_="mb-1",
            )
            for i, entry in enumerate(entries)
        ]
        if not dir_buttons:
            dir_buttons = [ui.p("(no sub-directories)", class_="text-muted")]

        # Breadcrumb-style path segments
        parts = cur.parts
        breadcrumbs = []
        for idx in range(len(parts)):
            segment_path = str(Path(*parts[: idx + 1]))
            breadcrumbs.append(
                ui.input_action_button(
                    f"bc_{idx}",
                    parts[idx] or "/",
                    class_="btn btn-link btn-sm p-0",
                )
            )
            if idx < len(parts) - 1:
                breadcrumbs.append(ui.span(" / "))

        project_badge = (
            ui.span(
                "\u2705 Marxan project detected (input.dat present)",
                class_="text-success fw-bold",
            )
            if is_project
            else ui.span()
        )

        return ui.modal(
            ui.div(
                ui.div(*breadcrumbs, class_="mb-2"),
                ui.hr(),
                project_badge,
                ui.div(
                    *dir_buttons,
                    style="max-height:350px;overflow-y:auto;",
                    class_="mt-2",
                ),
            ),
            title="Select Directory",
            easy_close=True,
            footer=ui.TagList(
                ui.input_action_button(
                    "nav_up", "\u2B06 Parent",
                    class_="btn btn-outline-secondary me-auto",
                ),
                ui.input_action_button(
                    "select_dir", "Select this directory",
                    class_="btn btn-primary",
                ),
            ),
        )

    # ------ Open modal ------------------------------------------------

    @reactive.effect
    @reactive.event(input.open_browser)
    def _open_modal():
        ui.modal_show(_build_modal())

    # ------ Navigate upward -------------------------------------------

    @reactive.effect
    @reactive.event(input.nav_up)
    def _go_up():
        cur = Path(_browsing())
        parent = cur.parent
        if parent != cur:
            _browsing.set(str(parent))
            ui.modal_show(_build_modal())

    # ------ Navigate into a sub-directory or breadcrumb ---------------

    @reactive.effect
    def _navigate():
        """Watch all ``goto_*`` and ``bc_*`` action buttons."""
        cur = Path(_browsing())
        entries = _list_directory(cur)

        # Sub-directory buttons
        for i, entry in enumerate(entries):
            btn_id = f"goto_{i}"
            try:
                val = input[btn_id]()
            except Exception:
                continue
            if val and val > 0:
                _browsing.set(entry["path"])
                ui.modal_show(_build_modal())
                return

        # Breadcrumb buttons
        parts = cur.parts
        for idx in range(len(parts)):
            btn_id = f"bc_{idx}"
            try:
                val = input[btn_id]()
            except Exception:
                continue
            if val and val > 0:
                target = str(Path(*parts[: idx + 1]))
                _browsing.set(target)
                ui.modal_show(_build_modal())
                return

    # ------ Select / confirm ------------------------------------------

    @reactive.effect
    @reactive.event(input.select_dir)
    def _select():
        selected_path.set(_browsing())
        ui.modal_remove()
        ui.notification_show(
            f"Selected: {_browsing()}", type="message", duration=3,
        )

    return selected_path
