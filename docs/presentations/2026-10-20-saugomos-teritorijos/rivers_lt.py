"""Lithuanian versions of the README river figures, for the stakeholder deck.

Reuses the network and drawing helpers from ``scripts/make_rivers_figures.py``;
only the labels differ. Run under the shiny env: ``python rivers_lt.py``.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "make_rivers_figures", HERE.parents[2] / "scripts" / "make_rivers_figures.py"
)
mrf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mrf)


def network_figure(net) -> Path:
    before = mrf.dci_diadromous(net)
    sol = mrf.optimize_barriers_mip(mrf.BarrierProblem(net, budget=3.0))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    mrf._draw(axes[0], net, set(), f"Prieš – DCI {before:.0f}")
    mrf._draw(axes[1], net, sol.removed, f"Po (biudžetas 3) – DCI {sol.dci_after:.0f}")

    def _sq(color, label):
        return plt.Line2D([], [], marker="s", color="w", markerfacecolor=color, markersize=11,
                          label=label)

    handles = [
        plt.Line2D([], [], color=mrf._BLUE, lw=4, label="sujungta su jūra"),
        plt.Line2D([], [], color=mrf._GREY, lw=4, label="atskirta"),
        _sq(mrf._RED, "barjeras (paliktas)"),
        _sq(mrf._GREEN, "barjeras (pašalintas)"),
        plt.Line2D([], [], marker="v", color="w", markerfacecolor=mrf._SEA, markersize=13,
                   label="upės žiotys"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Barjerų šalinimo optimizavimas (dendritinio jungtumo indeksas)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    path = HERE / "river_network_optimization_lt.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def frontier_figure(net) -> Path:
    budgets = [float(k) for k in range(len(mrf.BARRIER_SEGMENTS) + 1)]
    df = mrf.budget_dci_frontier(net, budgets, optimizer="mip")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(df["budget"], df["dci_after"], "-o", color=mrf._BLUE, lw=2.5, markersize=7)
    ax.fill_between(df["budget"], df["dci_after"], color=mrf._BLUE, alpha=0.10)
    ax.set_xlabel("Biudžetas (pašalintų barjerų skaičius)", fontsize=11)
    ax.set_ylabel("Dendritinio jungtumo indeksas", fontsize=11)
    ax.set_title("DCI prieaugis pagal biudžetą (tikslusis MIP)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    path = HERE / "dci_budget_frontier_lt.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    net = mrf._network()
    print("wrote", network_figure(net))
    print("wrote", frontier_figure(net))
