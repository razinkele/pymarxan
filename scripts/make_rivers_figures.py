"""Generate the river-feature figures used in README.md.

Reproducible: builds a small dendritic river network, computes the DCI, the
budget-constrained optimal barrier removal, and the budget-DCI frontier, then
renders two PNGs into ``docs/images/``.

Run: ``python scripts/make_rivers_figures.py``
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from pymarxan.rivers import (  # noqa: E402
    BarrierProblem,
    RiverNetwork,
    budget_dci_frontier,
    dci_diadromous,
    optimize_barriers_mip,
    segment_connectivity,
)

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"

# A small dendritic network (mouth at the bottom). Each segment is drawn from
# its upstream end to its downstream end; a barrier sits at the downstream end.
# down_id = -1 marks the outlet (S1, the river mouth).
COORDS = {
    1: ((0.0, 1.0), (0.0, 0.0)),       # outlet (no barrier)
    2: ((-1.0, 2.0), (0.0, 1.0)),
    3: ((1.0, 2.0), (0.0, 1.0)),
    4: ((-1.6, 3.0), (-1.0, 2.0)),
    5: ((-0.4, 3.0), (-1.0, 2.0)),
    6: ((0.4, 3.0), (1.0, 2.0)),
    7: ((1.6, 3.0), (1.0, 2.0)),
}
DOWN = {1: -1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3}
BARRIER_SEGMENTS = [2, 3, 4, 5, 6, 7]   # one impassable barrier per non-outlet segment

_BLUE, _GREY, _RED, _GREEN, _SEA = "#1f77b4", "#c9ccd1", "#d62728", "#2ca02c", "#17becf"


def _network() -> RiverNetwork:
    segs = pd.DataFrame(
        {"id": list(COORDS), "length": [1.0] * len(COORDS), "down_id": [DOWN[s] for s in COORDS]}
    )
    bars = pd.DataFrame(
        {
            "id": BARRIER_SEGMENTS,            # barrier id == its segment id here
            "segment": BARRIER_SEGMENTS,
            "pass_up": [0.0] * len(BARRIER_SEGMENTS),
            "pass_down": [0.0] * len(BARRIER_SEGMENTS),
            "removal_cost": [1.0] * len(BARRIER_SEGMENTS),
            "status": [0] * len(BARRIER_SEGMENTS),
        }
    )
    return RiverNetwork(segments=segs, barriers=bars)


def _draw(ax, net, removed, title):
    conn = segment_connectivity(net, {b: 1.0 for b in removed}, form="diadromous")
    for seg, ((xu, yu), (xd, yd)) in COORDS.items():
        reachable = conn[seg] > 0.5
        ax.plot(
            [xu, xd], [yu, yd],
            color=_BLUE if reachable else _GREY,
            lw=4, solid_capstyle="round", zorder=1,
        )
    for seg in BARRIER_SEGMENTS:
        (xu, yu), (xd, yd) = COORDS[seg]
        # Place the barrier a little up its own segment from the downstream
        # end so sibling barriers at a shared confluence don't overlap.
        t = 0.28
        bx, by = xd + t * (xu - xd), yd + t * (yu - yd)
        gone = seg in removed
        ax.scatter(
            [bx], [by], marker="s", s=110,
            c=_GREEN if gone else _RED, edgecolors="white", linewidths=1.4, zorder=3,
        )
    ax.scatter([0.0], [0.0], marker="v", s=190, c=_SEA, edgecolors="white", zorder=4)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-0.4, 3.3)
    ax.set_aspect("equal")
    ax.axis("off")


def make_network_figure(net: RiverNetwork) -> Path:
    before = dci_diadromous(net)
    sol = optimize_barriers_mip(BarrierProblem(net, budget=3.0))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    _draw(axes[0], net, set(), f"Before — DCI {before:.0f}")
    _draw(axes[1], net, sol.removed, f"After (budget 3) — DCI {sol.dci_after:.0f}")
    def _sq(color, label):
        return plt.Line2D(
            [], [], marker="s", color="w", markerfacecolor=color, markersize=11, label=label
        )

    handles = [
        plt.Line2D([], [], color=_BLUE, lw=4, label="connected to sea"),
        plt.Line2D([], [], color=_GREY, lw=4, label="disconnected"),
        _sq(_RED, "barrier (kept)"),
        _sq(_GREEN, "barrier (removed)"),
        plt.Line2D(
            [], [], marker="v", color="w", markerfacecolor=_SEA, markersize=13, label="river mouth"
        ),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=5, frameon=False,
        fontsize=9, bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Barrier-removal optimization (Dendritic Connectivity Index)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    path = OUT / "river_network_optimization.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def make_frontier_figure(net: RiverNetwork) -> Path:
    budgets = [float(k) for k in range(len(BARRIER_SEGMENTS) + 1)]
    df = budget_dci_frontier(net, budgets, optimizer="mip")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(df["budget"], df["dci_after"], "-o", color=_BLUE, lw=2.5, markersize=7)
    ax.fill_between(df["budget"], df["dci_after"], color=_BLUE, alpha=0.10)
    ax.set_xlabel("Budget (number of barriers removed)", fontsize=11)
    ax.set_ylabel("Dendritic Connectivity Index", fontsize=11)
    ax.set_title("DCI gained per unit budget (exact MIP)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    path = OUT / "dci_budget_frontier.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    net = _network()
    p1 = make_network_figure(net)
    p2 = make_frontier_figure(net)
    print(f"wrote {p1}")
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
