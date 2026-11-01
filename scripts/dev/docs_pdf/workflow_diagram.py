"""Draw docs/images/pymarxan_workflow.svg: pymarxan's core workflow in five columns.

usage: python scripts/dev/docs_pdf/workflow_diagram.py docs/images/pymarxan_workflow.svg
(then e.g. ``inkscape --export-type=png --export-dpi=150`` for a PNG).
"""

from __future__ import annotations

import sys
from html import escape

W, H = 1600, 840
COL_W, GAP, TOP = 252, 54, 118
X0 = (W - 5 * COL_W - 4 * GAP) // 2
GREEN, DARK, LIGHT, PALE = "#2f7d5b", "#1f4d3a", "#eef6f1", "#f6faf7"
AMBER, AMBER_BG, INK, MUTED = "#c9932e", "#fff6e5", "#1d2a24", "#5b6f64"

# (title, [(heading, detail lines, style)]) per column; style: "box" | "key" | "note" | "dash"
COLUMNS = [
    (
        "1  Inputs",
        [
            ("Marxan project", ["input.dat · pu.dat · spec.dat", "puvspr.dat · bound.dat"], "box"),
            ("Marxan with Zones", ["zones · zone costs", "contributions · targets"], "box"),
            (
                "GIS layers",
                ["vector planning units", "rasters (from_rasters)", "square / hex grids"],
                "box",
            ),
            ("Spatial data", ["GADM · WDPA", "cost surfaces"], "box"),
            ("River networks", ["HydroRIVERS · barriers"], "box"),
        ],
    ),
    (
        "2  Problem",
        [
            (
                "ConservationProblem",
                [
                    "planning units · features",
                    "amounts · boundary",
                    "probability · connectivity",
                    "parameters (BLM, SPF, …)",
                ],
                "key",
            ),
            ("ZonalProblem", ["zones · costs · targets", "→ zone solvers"], "box"),
            ("RiverNetwork", ["segments · barriers", "→ barrier optimisation"], "box"),
            ("validate() · presolve", ["errors before solving"], "note"),
        ],
    ),
    (
        "3  Prepare",
        [
            ("Targets", ["absolute · relative", "published methods"], "box"),
            ("Connectivity", ["metrics · circuit theory", "penalties"], "box"),
            ("Constraints", ["locks · contiguity", "budget · linear"], "box"),
            ("Probability", ["PROBMODE 1–3", "PROBZEROVAR"], "box"),
            ("Calibration", ["BLM / SPF sweeps", "sensitivity"], "box"),
        ],
    ),
    (
        "4  Solve",
        [
            (
                "Exact MIP",
                ["CBC · HiGHS · Gurobi", "min-set · coverage · shortfall", "multi-objective"],
                "box",
            ),
            (
                "Marxan heuristics",
                ["simulated annealing", "greedy · iterative improvement", "RUNMODE pipelines"],
                "box",
            ),
            ("Zone solvers", ["SA · heuristic · II · MIP", "for ZonalProblem"], "box"),
            ("Zonation ranking", ["CAZ / ABF priority maps"], "box"),
            ("Barrier optimisation", ["greedy · SA · MIP (DCI)", "for RiverNetwork"], "box"),
            ("Marxan C++ binary", ["optional"], "dash"),
        ],
    ),
    (
        "5  Results",
        [
            (
                "list[Solution]",
                ["selection · cost · boundary", "objective · targets met", "chance penalty"],
                "key",
            ),
            (
                "Analysis",
                ["selection frequency", "portfolios · replacement cost", "rank importance · gaps"],
                "box",
            ),
            ("Modern planning", ["30×30 representation", "equity · robustness"], "box"),
            ("Outputs", ["Marxan files (best, mvbest)", "GeoPackage · Shapefile · maps"], "box"),
        ],
    ),
]


def text(
    x: float,
    y: float,
    s: str,
    size: int,
    fill: str,
    weight: str = "normal",
    anchor: str = "middle",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
        f'font-weight="{weight}" text-anchor="{anchor}">{escape(s)}</text>'
    )


def box(x: float, y: float, w: float, h: float, head: str, lines: list[str], style: str) -> str:
    fill, stroke, hc, lc, dash = LIGHT, GREEN, DARK, INK, ""
    if style == "key":
        fill, stroke, hc, lc = GREEN, GREEN, "white", "#e6f2eb"
    elif style == "note":
        fill, stroke, hc, lc = AMBER_BG, AMBER, "#7a5512", "#7a5512"
    elif style == "dash":
        fill, dash = "white", ' stroke-dasharray="5 4"'
    out = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.4"{dash}/>'
    ]
    cy = y + 21
    out.append(text(x + w / 2, cy, head, 14, hc, "bold"))
    for ln in lines:
        cy += 16
        out.append(text(x + w / 2, cy, ln, 11.5, lc))
    return "\n".join(out)


def arrow(x1: float, y1: float, x2: float, y2: float, dashed: bool = False) -> str:
    d = ' stroke-dasharray="6 5"' if dashed else ""
    return (
        f'<path d="M{x1:.1f},{y1:.1f} C{(x1 + x2) / 2:.1f},{y1:.1f} {(x1 + x2) / 2:.1f},{y2:.1f} '
        f'{x2:.1f},{y2:.1f}" fill="none" stroke="{MUTED}" stroke-width="1.6"{d} '
        f'marker-end="url(#arr)"/>'
    )


def main(out_path: str) -> None:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
        f'height="{H}" font-family="DejaVu Sans, Arial, sans-serif">',
        f'<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/>'
        f"</marker></defs>",
        f'<rect width="{W}" height="{H}" fill="white"/>',
        text(W / 2, 44, "pymarxan — core workflow", 26, DARK, "bold"),
        text(
            W / 2,
            72,
            "Marxan-compatible data in, exact or heuristic solutions out, with "
            "analysis and reporting",
            14,
            MUTED,
        ),
    ]
    centers: dict[str, tuple[float, float, float, float]] = {}
    for c, (title, items) in enumerate(COLUMNS):
        x = X0 + c * (COL_W + GAP)
        heights = [42 + 16 * len(lines) for _, lines, _ in items]
        inner_gap = 14
        total = sum(heights) + inner_gap * (len(items) - 1)
        panel_h = H - TOP - 60
        parts.append(
            f'<rect x="{x - 12}" y="{TOP}" width="{COL_W + 24}" height="{panel_h}" rx="14" '
            f'fill="{PALE}" stroke="#9cc5ae" stroke-width="1.2"/>'
        )
        parts.append(text(x + COL_W / 2, TOP + 28, title, 16, DARK, "bold"))
        y = TOP + 46 + (panel_h - 46 - total) / 2
        for (head, lines, style), h in zip(items, heights):
            parts.append(box(x, y, COL_W, h, head, lines, style))
            centers[head] = (x, y, x + COL_W, y + h)
            y += h + inner_gap

    def right(n: str) -> tuple[float, float]:
        x0, y0, x1, y1 = centers[n]
        return x1, (y0 + y1) / 2

    def left(n: str) -> tuple[float, float]:
        x0, y0, x1, y1 = centers[n]
        return x0, (y0 + y1) / 2

    links = [
        ("Marxan project", "ConservationProblem", False),
        ("GIS layers", "ConservationProblem", False),
        ("Spatial data", "ConservationProblem", True),
        ("Marxan with Zones", "ZonalProblem", False),
        ("River networks", "RiverNetwork", False),
        ("ConservationProblem", "Targets", False),
        ("ConservationProblem", "Connectivity", False),
        ("ConservationProblem", "Constraints", False),
        ("ConservationProblem", "Probability", False),
        ("ConservationProblem", "Calibration", False),
        ("Targets", "Exact MIP", False),
        ("Constraints", "Exact MIP", False),
        ("Connectivity", "Marxan heuristics", False),
        ("Probability", "Marxan heuristics", False),
        ("Calibration", "Marxan heuristics", True),
        ("Exact MIP", "list[Solution]", False),
        ("Marxan heuristics", "list[Solution]", False),
        ("Zone solvers", "list[Solution]", False),
        ("Zonation ranking", "list[Solution]", False),
        ("Marxan C++ binary", "list[Solution]", True),
    ]
    for a, b, dashed in links:
        (x1, y1), (x2, y2) = right(a), left(b)
        parts.append(arrow(x1, y1, x2 - 2, y2, dashed))
    # list[Solution] fans out downwards inside its column
    sx0, sy0, sx1, sy1 = centers["list[Solution]"]
    for n in ("Analysis", "Modern planning", "Outputs"):
        bx0, by0, bx1, by1 = centers[n]
        parts.append(
            f'<path d="M{sx1},{(sy0 + sy1) / 2} L{sx1 + 9},{(sy0 + sy1) / 2} '
            f'L{sx1 + 9},{(by0 + by1) / 2} L{bx1 + 2},{(by0 + by1) / 2}" fill="none" '
            f'stroke="{MUTED}" stroke-width="1.4" marker-end="url(#arr)"/>'
        )
    # round trip: Outputs back to Marxan project files
    ox, oy = left("Outputs")[0], centers["Outputs"][3]
    mx, my = left("Marxan project")[0], centers["Marxan project"][1]
    by = H - 34
    parts.append(
        f'<path d="M{ox + COL_W / 2},{oy} L{ox + COL_W / 2},{by} L{mx - 22},{by} '
        f'L{mx - 22},{my + 18} L{mx - 2},{my + 18}" fill="none" stroke="{GREEN}" '
        f'stroke-width="1.5" stroke-dasharray="6 5" marker-end="url(#arr)"/>'
    )
    parts.append(
        text(
            W / 2,
            by - 8,
            "save_project: results and edited projects round-trip to Marxan's file format",
            12,
            GREEN,
        )
    )
    parts.append(
        text(
            W - 24,
            H - 10,
            "Validation anchor: the MIP reaches the hand-verified "
            "optimum 35.0 on tests/data/simple; heuristics never report less.",
            11,
            AMBER,
            anchor="end",
        )
    )
    parts.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main(sys.argv[1])
