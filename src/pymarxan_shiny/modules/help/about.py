"""The About modal: version, authorship and how pymarxan compares with Marxan and prioritizr.

``COMPARISON_ROWS`` mirrors the "Side by side" table in ``docs/USER_MANUAL.md``
(section "pymarxan, Marxan and prioritizr"); keep the two in step.
"""
from __future__ import annotations

from shiny import ui

AUTHOR = "Artūras Razinkovas-Baziukas"
INSTITUTION = "Marine Research Institute, Klaipėda University"

# (aspect, Marxan, prioritizr, pymarxan)
COMPARISON_ROWS: tuple[tuple[str, str, str, str], ...] = (
    ("Form", "Stand-alone program, text files", "R package",
     "Python package and web application"),
    ("Solving", "Simulated annealing, heuristics", "Exact ILP", "Both"),
    ("Optimality", "Unknown quality", "Proven (within a gap)",
     "Proven for the MIP; heuristics checked against it"),
    ("Marxan project files", "Native", "Importable", "Native, read and write"),
    ("Zones", "Marxan with Zones (separate program)", "Yes", "Yes, exact and heuristic"),
    ("Clumping, separation", "Yes", "No (not linear)", "Yes (heuristic solvers)"),
    ("Probability (chance) terms", "Marxan 3: 1D and 2D", "Not a built-in term",
     "2D now; 1D in v0.48.0"),
    ("Interactive interface", "Separate tools", "Separate applications built on it",
     "Built-in Shiny application"),
)

_MUTED = "color: var(--ocean-mid);"
_RULE = ui.tags.hr(style="border-color: rgba(15, 163, 177, 0.15);")


def _comparison_table() -> ui.Tag:
    head = ui.tags.thead(
        ui.tags.tr(ui.tags.th(""), ui.tags.th("Marxan"), ui.tags.th("prioritizr"),
                   ui.tags.th("pymarxan"))
    )
    body = ui.tags.tbody(
        *(
            ui.tags.tr(
                ui.tags.th(aspect, scope="row", style="font-weight: 600;"),
                ui.tags.td(marxan),
                ui.tags.td(prioritizr),
                ui.tags.td(pymarxan, style="background: var(--ocean-mist);"),
            )
            for aspect, marxan, prioritizr, pymarxan in COMPARISON_ROWS
        )
    )
    return ui.div(
        ui.tags.table(head, body, class_="table table-sm table-bordered align-middle mb-0",
                      style="font-size: 0.85rem;"),
        class_="table-responsive",
    )


# (text before the link, DOI URL) — the User Manual's reference list, same wording.
ACKNOWLEDGED_WORKS: tuple[tuple[str, str], ...] = (
    ("Ball, I. R., Possingham, H. P., & Watts, M. (2009). Marxan and relatives: software for "
     "spatial conservation prioritisation. In A. Moilanen, K. A. Wilson, & H. P. Possingham "
     "(Eds.), Spatial conservation prioritisation: quantitative methods and computational "
     "tools (pp. 185–195). Oxford University Press.", ""),
    ("Watts, M. E., Ball, I. R., Stewart, R. S., et al. (2009). Marxan with Zones: software "
     "for optimal conservation based land- and sea-use zoning. Environmental Modelling & "
     "Software, 24(12), 1513–1521.", "https://doi.org/10.1016/j.envsoft.2009.06.005"),
    ("Hanson, J. O., Schuster, R., Strimas-Mackey, M., et al. (2024). Systematic conservation "
     "prioritization with the prioritizr R package. Conservation Biology, 39(1).",
     "https://doi.org/10.1111/cobi.14376"),
)


def _acknowledgements() -> ui.Tag:
    items = [
        ui.tags.li(
            text,
            *((" ", ui.tags.a(doi, href=doi, target="_blank", rel="noopener")) if doi else ()),
        )
        for text, doi in ACKNOWLEDGED_WORKS
    ]
    return ui.div(
        ui.tags.h6("Acknowledgements", style="margin-top: 1rem; font-weight: 600;"),
        ui.tags.p(
            "pymarxan stands on two bodies of work. Marxan, by Ian Ball, Hugh Possingham, "
            "Matthew Watts and colleagues, defined the problem formulation, file formats and "
            "algorithms that pymarxan reproduces. prioritizr, by Jeffrey O. Hanson, Richard "
            "Schuster, Matthew Strimas-Mackey and colleagues, showed how to solve these "
            "problems exactly with integer programming; pymarxan's exact solvers, objectives "
            "and target methods follow it. We thank their authors. If pymarxan helps your "
            "work, please cite them too:",
            style=f"line-height: 1.6; {_MUTED}",
        ),
        ui.tags.ul(*items, style=f"{_MUTED} font-size: 0.85rem; padding-left: 1.2rem;"),
        ui.tags.p(
            ui.tags.small(
                "pymarxan is an independent project, not affiliated with or endorsed by the "
                "Marxan or prioritizr developers."
            ),
            style=f"{_MUTED} opacity: 0.8;",
        ),
    )


def about_modal(version: str) -> ui.Tag:
    """Build the About dialog for pymarxan ``version``."""
    return ui.modal(
        ui.div(
            ui.tags.div(ui.tags.span("🌊", style="font-size: 2.5rem;"),
                        style="text-align: center; margin-bottom: 0.5rem;"),
            ui.tags.h4("PyMarxan",
                       style="text-align: center; margin: 0 0 0.25rem 0; font-weight: 700;"),
            ui.tags.p(
                f"Version {version}",
                style=("text-align: center; color: var(--ocean-teal); "
                       "font-weight: 600; margin-bottom: 0.25rem;"),
            ),
            ui.tags.p(
                AUTHOR, ui.tags.br(), INSTITUTION,
                style=f"text-align: center; {_MUTED} margin-bottom: 1rem;",
            ),
            _RULE,
            ui.tags.p(
                "Systematic conservation planning in Python: run Marxan projects as Marxan "
                "would, solve the same problems exactly with integer programming, and use "
                "the analyses that grew up around Marxan and prioritizr.",
                style=f"line-height: 1.6; {_MUTED}",
            ),
            ui.tags.h6("pymarxan, Marxan and prioritizr",
                       style="margin-top: 1rem; font-weight: 600;"),
            _comparison_table(),
            ui.tags.p(
                ui.tags.small(
                    "Details: User Manual, section “pymarxan, Marxan and prioritizr”."
                ),
                style=f"{_MUTED} opacity: 0.8; margin-top: 0.5rem;",
            ),
            _acknowledgements(),
            _RULE,
            ui.tags.p(
                ui.tags.small(
                    f"© 2026 {AUTHOR}, {INSTITUTION} · Built with Shiny for Python",
                    style=f"{_MUTED} opacity: 0.7;",
                ),
                style="text-align: center; margin-bottom: 0;",
            ),
        ),
        title="🌊 About PyMarxan",
        easy_close=True,
        size="l",
    )
