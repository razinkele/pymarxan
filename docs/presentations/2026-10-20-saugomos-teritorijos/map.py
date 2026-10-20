"""Curonian Lagoon case-study map for the stakeholder deck (slide 6).

Solves ``examples/baltic_marine_planning.py`` exactly (MIP) and heuristically (SA, seed 7, the
example's own settings) and draws cost / MIP reserve / SA reserve side by side with Lithuanian
labels. Run under the shiny env: ``python map.py`` (writes ``baltic_map.png`` next to this file).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from pymarxan.solvers.base import SolverConfig  # noqa: E402
from pymarxan.solvers.mip_solver import MIPSolver  # noqa: E402
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "examples"))
from baltic_marine_planning import _cell_centroids, build_baltic_problem  # noqa: E402

INK, NONE, PROPOSED, LOCKED = "#16323A", "#E8EEF0", "#2E8B7A", "#C8553D"


def main() -> None:
    problem, grid = build_baltic_problem()
    mip = MIPSolver().solve(problem, SolverConfig(num_solutions=1))[0]
    problem.parameters["NUMITNS"] = 20_000
    problem.parameters["NUMTEMP"] = 100
    sa = SimulatedAnnealingSolver().solve(problem, SolverConfig(num_solutions=1, seed=7))[0]
    print(f"MIP cost {mip.cost:.2f} ({mip.selected.sum()} cells), "
          f"SA cost {sa.cost:.2f} ({sa.selected.sum()} cells)")

    cx, cy = _cell_centroids(grid)
    ux, uy = np.unique(cx), np.unique(cy)
    ix, iy = np.searchsorted(ux, cx), np.searchsorted(uy, cy)

    def to_grid(values: np.ndarray) -> np.ndarray:
        out = np.full((len(uy), len(ux)), np.nan)
        out[iy, ix] = values
        return out

    status = problem.planning_units["status"].to_numpy()
    cost = problem.planning_units["cost"].to_numpy()

    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.8), dpi=200)
    im = axs[0].imshow(to_grid(cost), origin="lower", cmap="YlOrBr")
    axs[0].set_title("Sąnaudos (veiklos spaudimas)", color=INK, fontsize=13)
    fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.03).outline.set_visible(False)

    cmap = ListedColormap([NONE, PROPOSED, LOCKED])
    for ax, sol, title in [(axs[1], mip, "Tikslusis optimumas (MIP)"),
                           (axs[2], sa, "Klasikinė euristika (SA)")]:
        # 2 = already protected (locked in), 1 = proposed, 0 = not selected
        ax.imshow(to_grid(np.where(status == 2, 2, sol.selected.astype(int))),
                  origin="lower", cmap=cmap, vmin=0, vmax=2)
        ax.set_title(title, color=INK, fontsize=13)
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.legend(
        handles=[Patch(color=PROPOSED, label="Siūloma saugoti"),
                 Patch(color=LOCKED, label="Jau saugoma (užfiksuota)"),
                 Patch(color=NONE, label="Nesaugoma")],
        loc="lower center", ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(0.62, -0.01),
    )
    plt.tight_layout(rect=(0, 0.07, 1, 0.97))
    plt.savefig(HERE / "baltic_map.png", facecolor="white")


if __name__ == "__main__":
    main()
