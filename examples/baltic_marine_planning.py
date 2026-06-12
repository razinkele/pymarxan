"""Baltic marine spatial planning — an end-to-end worked example.

The :doc:`tutorial <../docs/TUTORIAL.md>` shows the API one feature at a
time on a 12-cell toy problem. This script is the opposite: a single,
domain-flavoured scenario carried all the way from "raw region" to
"defensible reserve design", the way an analyst actually uses the
package.

Scenario
--------
A planner in the south-eastern Baltic (the Curonian Lagoon and the
adjacent coastal shelf off Klaipėda, Lithuania) must expand a small
existing protected-area network to cover four conservation features
while keeping the reserve cheap to establish. We:

1.  Lay a square planning grid over the region.
2.  Assign a *cost* surface (a proxy for human-use pressure that peaks
    near the port) and four feature distributions.
3.  Lock in the two cells that are *already* protected.
4.  Set each feature's target to a fraction of its regional total.
5.  Solve the minimum-set problem exactly (MIP) and heuristically (SA),
    and compare the two reserves.
6.  Run a gap analysis (before vs. after) and rank cells by
    irreplaceability.

Everything is deterministic (fixed RNG seed, analytic spatial fields)
and needs no network access, so it doubles as a regression fixture —
see ``tests/test_examples.py``.

Run it::

    python examples/baltic_marine_planning.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# South-eastern Baltic: Curonian Lagoon + Klaipėda coastal shelf.
# (minx, miny, maxx, maxy) in EPSG:4326 — roughly the lagoon mouth.
REGION_BOUNDS = (20.90, 55.30, 21.30, 55.70)
CELL_SIZE = 0.025  # ~1.6 km cells → a 16×16 grid

FEATURES = [
    # id, name, the spatial field that generates its amount
    (1, "reedbeds"),           # eastern lagoon shore
    (2, "submerged_macrophytes"),  # shallow margins
    (3, "pikeperch_spawning"),     # central deep channel
    (4, "migratory_waterbirds"),   # northern flats
]


@dataclass
class CompareResult:
    """Side-by-side summary of the exact vs. heuristic reserve."""

    mip_cost: float
    sa_cost: float
    mip_n_selected: int
    sa_n_selected: int
    overlap: int  # cells both solutions agree to protect
    mip_targets_met: bool
    sa_targets_met: bool


def _cell_centroids(grid) -> tuple[np.ndarray, np.ndarray]:
    """Centroids from cell bounds — avoids the geographic-CRS centroid
    warning and is exact for axis-aligned square cells."""
    b = grid.geometry.bounds  # minx, miny, maxx, maxy
    cx = ((b["minx"] + b["maxx"]) / 2.0).to_numpy()
    cy = ((b["miny"] + b["maxy"]) / 2.0).to_numpy()
    return cx, cy


def _feature_fields(nx: np.ndarray, ny: np.ndarray) -> dict[int, np.ndarray]:
    """Analytic abundance fields in [0, 1] keyed by feature id.

    ``nx``/``ny`` are grid-normalised centroid coordinates (0 = west/south,
    1 = east/north). Each field is a smooth bump placed where that feature
    realistically concentrates."""
    def bump(cx0: float, cy0: float, sx: float, sy: float) -> np.ndarray:
        return np.exp(-(((nx - cx0) / sx) ** 2 + ((ny - cy0) / sy) ** 2))

    return {
        1: bump(0.85, 0.50, 0.25, 0.60),  # reedbeds — eastern shore band
        2: 1.0 - bump(0.50, 0.50, 0.45, 0.45),  # macrophytes — shallow rim
        3: bump(0.45, 0.45, 0.22, 0.30),  # pikeperch — central channel
        4: bump(0.50, 0.95, 0.50, 0.22),  # waterbirds — northern flats
    }


def build_baltic_problem(
    target_fraction: float = 0.3,
    blm: float = 0.0,
    spf: float = 100.0,
    seed: int = 42,
):
    """Construct the Baltic :class:`ConservationProblem` and its grid.

    Returns ``(problem, grid)`` where ``grid`` is the GeoDataFrame (kept
    so callers can map or export the reserve).

    ``spf`` is the species penalty factor. The exact MIP treats targets as
    hard constraints and ignores it, but the heuristic SA needs the
    target-miss penalty to dominate cost or it will happily under-protect;
    a high SPF is the standard Marxan way to enforce feasibility.
    """
    from pymarxan.models.problem import ConservationProblem
    from pymarxan.spatial.grid import compute_adjacency, generate_planning_grid

    grid = generate_planning_grid(REGION_BOUNDS, CELL_SIZE, grid_type="square")
    n = len(grid)
    rng = np.random.default_rng(seed)

    cx, cy = _cell_centroids(grid)
    nx = (cx - cx.min()) / (cx.max() - cx.min())
    ny = (cy - cy.min()) / (cy.max() - cy.min())

    # Cost surface: human-use pressure peaking at the Klaipėda port in the
    # north-west (nx≈0, ny≈1). Cheaper to protect the quiet south-east.
    port_dist = np.hypot(nx - 0.0, ny - 1.0)
    cost = 1.0 + 5.0 * np.exp(-((port_dist / 0.40) ** 2)) + 0.2 * rng.random(n)

    # Two cells are already protected (status 2 = locked in). Pick the
    # cells nearest two fixed points so the example is stable.
    def nearest_cell(px: float, py: float) -> int:
        return int(np.argmin(np.hypot(nx - px, ny - py)))

    status = np.zeros(n, dtype=int)
    status[nearest_cell(0.80, 0.55)] = 2  # an existing lagoon reserve
    status[nearest_cell(0.50, 0.90)] = 2  # an existing bird sanctuary

    planning_units = pd.DataFrame(
        {"id": grid["id"].to_numpy(), "cost": cost, "status": status}
    )

    # Feature amounts: analytic field × patchy noise, kept only where the
    # feature is actually present (amount > 0).
    fields = _feature_fields(nx, ny)
    pv_rows: list[dict] = []
    feat_rows: list[dict] = []
    pu_ids = grid["id"].to_numpy()
    for fid, name in FEATURES:
        field = fields[fid]
        amount = np.clip(field * (0.4 + 0.6 * rng.random(n)) - 0.15, 0.0, None)
        if fid == 3:
            # Pikeperch spawning aggregates on one key ground. Spike the
            # peak cell so it holds a dominant share — this is what makes a
            # cell *irreplaceable* once targets get ambitious.
            peak = int(np.argmax(field))
            amount[peak] += 6.0 * amount.sum()
        present = amount > 1e-6
        total = float(amount[present].sum())
        feat_rows.append(
            {
                "id": fid,
                "name": name,
                "target": target_fraction * total,
                "spf": spf,
            }
        )
        for pid, amt in zip(pu_ids[present], amount[present]):
            pv_rows.append({"species": fid, "pu": int(pid), "amount": float(amt)})

    features = pd.DataFrame(feat_rows)
    pu_vs_features = pd.DataFrame(pv_rows)
    boundary = compute_adjacency(grid)

    problem = ConservationProblem(
        planning_units=planning_units,
        features=features,
        pu_vs_features=pu_vs_features,
        boundary=boundary,
        parameters={"BLM": blm},
    )
    return problem, grid


def solve_and_compare(problem) -> CompareResult:
    """Solve the problem exactly (MIP) and heuristically (SA), compare."""
    from pymarxan.solvers.base import SolverConfig
    from pymarxan.solvers.mip_solver import MIPSolver
    from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver

    mip = MIPSolver().solve(problem, SolverConfig(num_solutions=1))[0]

    # SA needs an iteration budget; set it on the problem parameters.
    problem.parameters["NUMITNS"] = 20_000
    problem.parameters["NUMTEMP"] = 100
    sa = SimulatedAnnealingSolver().solve(
        problem, SolverConfig(num_solutions=1, seed=7)
    )[0]

    overlap = int(np.sum(mip.selected & sa.selected))
    return CompareResult(
        mip_cost=mip.cost,
        sa_cost=sa.cost,
        mip_n_selected=int(mip.selected.sum()),
        sa_n_selected=int(sa.selected.sum()),
        overlap=overlap,
        mip_targets_met=mip.all_targets_met,
        sa_targets_met=sa.all_targets_met,
    )


def gap_before_after(problem, selected_ids: list[int]) -> pd.DataFrame:
    """Gap analysis with only the existing reserves vs. with the proposed
    reserve added. Returns a tidy table for printing."""
    from pymarxan.analysis.gap_analysis import compute_gap_analysis

    before = compute_gap_analysis(problem).to_dataframe()

    # Apply the proposed reserve as locked-in, recompute.
    after_problem, _ = build_baltic_problem()
    pu = after_problem.planning_units
    pu.loc[pu["id"].isin(selected_ids), "status"] = 2
    after = compute_gap_analysis(after_problem).to_dataframe()

    return pd.DataFrame(
        {
            "feature": before["feature_name"],
            "pct_before": before["percent_protected"].round(1),
            "pct_after": after["percent_protected"].round(1),
            "met_after": after["target_met"],
        }
    )


def main() -> None:
    from pymarxan.analysis.irreplaceability import compute_irreplaceability
    from pymarxan.solvers.base import SolverConfig
    from pymarxan.solvers.mip_solver import MIPSolver

    print("Baltic marine spatial planning — Curonian Lagoon / Klaipėda shelf")
    print("=" * 66)

    problem, grid = build_baltic_problem()
    print(
        f"Grid: {problem.n_planning_units} planning units, "
        f"{problem.n_features} features"
    )
    locked = int((problem.planning_units["status"] == 2).sum())
    print(f"Existing protected cells (locked in): {locked}")

    cmp = solve_and_compare(problem)
    print("\nReserve comparison (minimum-set, BLM=0):")
    print(
        f"  MIP (exact):     cost={cmp.mip_cost:7.2f}  "
        f"cells={cmp.mip_n_selected:3d}  targets_met={cmp.mip_targets_met}"
    )
    print(
        f"  SA  (heuristic): cost={cmp.sa_cost:7.2f}  "
        f"cells={cmp.sa_n_selected:3d}  targets_met={cmp.sa_targets_met}"
    )
    gap = 100.0 * (cmp.sa_cost - cmp.mip_cost) / cmp.mip_cost if cmp.mip_cost else 0.0
    print(f"  SA is {gap:+.1f}% above the exact optimum; {cmp.overlap} cells agree.")

    # Recover the exact reserve's selected ids for the gap story.
    mip = MIPSolver().solve(problem, SolverConfig(num_solutions=1))[0]
    selected_ids = [
        int(i) for i, s in zip(problem.planning_units["id"], mip.selected) if s
    ]

    print("\nGap analysis — existing reserves vs. proposed reserve:")
    print(gap_before_after(problem, selected_ids).to_string(index=False))

    # Irreplaceability flags cells that hold scarce, concentrated features.
    # Here the widespread habitats leave most of the map substitutable
    # (score 0), but the single pikeperch spawning ground is critical at
    # every target level — there is simply nowhere else to protect it.
    print("\nIrreplaceability vs. how ambitious the targets are:")
    for frac in (0.30, 0.60, 0.85):
        tight, _ = build_baltic_problem(target_fraction=frac)
        irr = compute_irreplaceability(tight)
        top = sorted(irr.items(), key=lambda kv: kv[1], reverse=True)[:5]
        max_score = top[0][1] if top else 0.0
        n_critical = sum(1 for s in irr.values() if s > 0)
        leaders = ", ".join(f"{pid}:{s:.2f}" for pid, s in top if s > 0) or "none"
        print(
            f"  target={frac:.0%}: {n_critical:3d} cells critical, "
            f"max score {max_score:.2f}  (top: {leaders})"
        )


if __name__ == "__main__":
    main()
