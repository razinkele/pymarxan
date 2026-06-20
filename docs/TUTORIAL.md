# pymarxan tutorial — Python API

A single end-to-end walk-through of the v0.3.0 ("prioritizr-parity") and
v0.4.0 ("Connectivity + portfolios") surface, plus the v0.7+ river
connectivity / barrier-restoration features (final section). Every
code block below is exercised by `tests/test_tutorial_examples.py` —
if a snippet stops running, that test fails and the doc gets fixed
in the same commit as the code change.

This tutorial assumes you've already installed `pymarxan` via
`pip install pymarxan==0.4.1`. The Shiny UI is a separate surface
covered by `docs/USER_MANUAL.md`; this file is for the Python API.

## Table of Contents

1. [Building a minimal problem](#1-building-a-minimal-problem)
2. [Solving with the default min-set objective](#2-solving-with-the-default-min-set-objective)
3. [Choosing the MIP backend (Phase 21)](#3-choosing-the-mip-backend-phase-21)
4. [Importance scores (Phase 22)](#4-importance-scores-phase-22)
5. [Alternative MIP objectives (Phase 23)](#5-alternative-mip-objectives-phase-23)
6. [Connectivity metrics (Phase 24)](#6-connectivity-metrics-phase-24)
7. [Solution portfolios via no-good cuts (Phase 25)](#7-solution-portfolios-via-no-good-cuts-phase-25)
8. [BLM calibration with Pareto filter (Phase 25)](#8-blm-calibration-with-pareto-filter-phase-25)

---

## 1. Building a minimal problem

`pymarxan` ingests three DataFrames: planning units, features, and the
PU-vs-feature amount table. Targets are per-feature; an SPF (species
penalty factor) controls how much the solver cares about missing the
target.

```python
import pandas as pd
from pymarxan.models.problem import ConservationProblem

planning_units = pd.DataFrame({
    "id": list(range(1, 13)),       # 12 PUs
    "cost": [1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 1.0, 2.0, 1.0],
    "status": [0] * 12,             # 0 = free, 2 = locked-in, 3 = locked-out
})

features = pd.DataFrame({
    "id": [1, 2, 3],
    "name": ["coral", "seagrass", "mangrove"],
    "target": [5.0, 4.0, 3.0],      # minimum amount in the reserve
    "spf": [1.0, 1.0, 1.0],
})

pu_vs_features = pd.DataFrame({
    "species": [1] * 6 + [2] * 4 + [3] * 3,
    "pu":      [1, 2, 3, 4, 5, 6, 3, 4, 7, 8, 9, 10, 11],
    "amount":  [1.0] * 13,
})

problem = ConservationProblem(
    planning_units=planning_units,
    features=features,
    pu_vs_features=pu_vs_features,
)
```

For real workflows the `pymarxan.io.readers.load_project()` helper
ingests a full Marxan project directory (`input.dat` + tabular files);
see the source for details.

## 2. Solving with the default min-set objective

The default `MIPSolver()` minimises `cost + BLM·boundary + penalty`
subject to `Σ amount·x ≥ target·MISSLEVEL` — the classic Marxan
problem.

```python
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver

solver = MIPSolver()
solutions = solver.solve(problem, SolverConfig(num_solutions=1))

best = solutions[0]
assert all(best.targets_met.values())
print(best.metadata["objective"])    # 'min_set'
print(best.metadata["mip_backend"])  # 'cbc' or 'highs', depending on env
print(best.cost)                     # objective cost
```

`Solution.metadata` records every choice the solver made, so a notebook
that runs the same `MIPSolver()` call months later can tell you which
backend and which constraint strategies were active.

## 3. Choosing the MIP backend (Phase 21)

CBC ships with PuLP and is always available. HiGHS is 5-50× faster on
larger problems and is picked automatically when its binary is on
`PATH`. Gurobi requires `pip install gurobipy` separately.

```python
from pymarxan.solvers.mip_solver import _available_backends, MIPSolver

print(_available_backends())
# {'cbc': True, 'highs': False, 'gurobi': False}   (typical CI)

# Explicit backend choice for reproducible scripts:
solver = MIPSolver(mip_backend="cbc")
# Auto-select (default): HiGHS if available, otherwise CBC.
solver = MIPSolver(mip_backend="auto")
```

Once a backend is resolved the choice flows into
`Solution.metadata["mip_backend"]` so downstream tooling can adjust
expectations (e.g. tighter MIP gap on HiGHS).

## 4. Importance scores (Phase 22)

Three complementary metrics for ranking PUs by their conservation
value. None of them re-solves the MIP except `compute_replacement_cost`
— the others are post-hoc on a single solve.

```python
from pymarxan.analysis.ferrier_importance import compute_ferrier_importance
from pymarxan.analysis.rank_importance import compute_rank_importance
from pymarxan.analysis.replacement_cost import compute_replacement_cost

# Closed-form SPF-weighted contribution to each feature's target.
ferrier = compute_ferrier_importance(problem)

# Jung 2021: repeatedly remove the least-damaging PU; rank by removal order.
ranks = compute_rank_importance(problem, best)

# Gold standard: per-PU "how much more does the optimum cost when this
# PU is locked out?". n_selected + 1 MIP solves; pair with HiGHS on
# large problems.
repl_cost = compute_replacement_cost(problem)
```

Each function returns a `dict[int, float]` keyed by PU id. Combine
them for a multi-criteria summary:

```python
pd.DataFrame({
    "PU": list(ferrier.keys()),
    "ferrier": list(ferrier.values()),
    "rank": [ranks[i] for i in ferrier],
    "replacement_cost": [repl_cost[i] for i in ferrier],
})
```

## 5. Alternative MIP objectives (Phase 23)

`MIPSolver(objective=...)` picks from four formulations. `min_set` is
the default Marxan problem; the other three lift you out of pure
minimum-set.

| `objective` | Goal | Required parameters |
| --- | --- | --- |
| `"min_set"` | Minimise `cost + BLM·boundary + penalty` | — (default) |
| `"max_features"` | Maximise count of feature targets met under a cost cap | `COSTBUDGET` |
| `"min_largest_shortfall"` | Minimax: minimise the worst per-feature shortfall | `COSTBUDGET` |
| `"min_penalties"` | Hierarchical: minimise SPF-weighted shortfall first, cost second | — |

```python
problem.parameters["COSTBUDGET"] = 5.0  # tight enough to force trade-offs

max_feat = MIPSolver(objective="max_features").solve(
    problem, SolverConfig(num_solutions=1),
)[0]

min_short = MIPSolver(objective="min_largest_shortfall").solve(
    problem, SolverConfig(num_solutions=1),
)[0]

min_pen = MIPSolver(objective="min_penalties").solve(
    problem, SolverConfig(num_solutions=1),
)[0]
```

Use `max_features` when you have a strict budget and want to know which
targets you can hit; `min_largest_shortfall` when you want to avoid
abandoning any one feature (good for endangered-species portfolios);
`min_penalties` when you'd pay extra to meet every target if at all
feasible (i.e. a softer min-set).

## 6. Connectivity metrics (Phase 24)

Connectivity in pymarxan is an edge list (`id1, id2, value`). Convert
to a matrix once, then compute as many metrics as you like.

```python
from pymarxan.connectivity.io import (
    connectivity_to_boundary,
    connectivity_to_matrix,
)
from pymarxan.connectivity.metrics import (
    compute_donors,
    compute_pagerank_centrality,
    compute_recipients,
)

edges = pd.DataFrame({
    "id1": [1, 2, 3, 2],
    "id2": [2, 3, 4, 5],
    "value": [1.0, 1.0, 1.0, 0.5],
})
# symmetric=False preserves the directed donor → recipient structure.
matrix = connectivity_to_matrix(
    edges, pu_ids=[1, 2, 3, 4, 5], symmetric=False,
)

pagerank = compute_pagerank_centrality(matrix)  # sums to 1
donors = compute_donors(matrix)                 # boolean mask: net source
recipients = compute_recipients(matrix)         # boolean mask: net sink
```

Two further connectivity modules ship in v0.4.0:

- `pymarxan.connectivity.temporal.compute_temporal_connectivity(stack, reduction=...)`
  reduces a `(T, n, n)` stack of per-timestep matrices to a single
  `(n, n)` summary via `"mean"` / `"max"` / `"weighted"`.
- `pymarxan.connectivity.resistance.habitat_resistance_to_matrix(raster, coords)`
  computes least-cost-path connectivity from a 2D habitat-resistance
  raster (networkx Dijkstra under the hood; no scikit-image dep).

You can also feed a connectivity matrix into Marxan's BLM penalty
directly by converting it to `bound.dat` format:

```python
boundary = connectivity_to_boundary(edges, scale=-1.0)
# Negative scale: high connectivity → low boundary penalty (the solver
# treats highly-connected PU pairs as "cheap to keep together").
```

Set `problem.boundary = boundary` and the existing BLM machinery picks
it up.

## 7. Solution portfolios via no-good cuts (Phase 25)

Generate K diverse high-quality MIP solutions without needing Gurobi's
solution-pool feature. Each iteration solves the MIP, then adds a
no-good cut forcing at least one variable to flip on the next pass.

```python
from pymarxan.analysis.portfolio_cuts import generate_portfolio_cuts

portfolio = generate_portfolio_cuts(
    problem,
    solver=MIPSolver(),
    k=3,
    config=SolverConfig(num_solutions=1),
)

for sol in portfolio:
    rank = sol.metadata["portfolio_iteration"]
    print(f"#{rank}: cost={sol.cost:.1f}, n_selected={sol.selected.sum()}")
```

Objectives are weakly non-decreasing across iterations (optimum first,
each cut forces a sub-optimal next). The function returns the partial
list when fewer than `k` distinct feasible solutions exist — no crash.

## 8. BLM calibration with Pareto filter (Phase 25)

The Boundary Length Modifier (BLM) trades cost against reserve
compactness. The classic calibration workflow is to sweep BLM over a
range and look for the "elbow". The Cohon Pareto filter (Phase 25)
drops dominated points so users see only the meaningful trade-offs.

```python
from pymarxan.calibration.blm import calibrate_blm
from pymarxan.calibration.pareto import pareto_frontier

# Boundary edges so BLM actually does something.
problem.boundary = pd.DataFrame([
    {"id1": i, "id2": i + 1, "boundary": 1.0}
    for i in range(1, 12)
])

result = calibrate_blm(
    problem,
    solver=MIPSolver(),
    blm_values=[0.0, 0.5, 1.0, 2.0, 5.0],
    config=SolverConfig(num_solutions=1),
)
pareto = pareto_frontier(result)

print(f"raw sweep: {len(result.blm_values)} points")
print(f"pareto frontier: {len(pareto.blm_values)} points")
for blm, c, b in zip(pareto.blm_values, pareto.costs, pareto.boundaries):
    print(f"  BLM={blm}: cost={c:.1f}, boundary={b:.1f}")
```

A point `i` is dominated by point `j` iff `cost_j ≤ cost_i AND
boundary_j ≤ boundary_i` with strict inequality somewhere. The filter
also deduplicates identical `(cost, boundary)` pairs.

---

## River connectivity and barrier restoration

`pymarxan.rivers` adds river-network connectivity (the Dendritic Connectivity
Index, Côté et al. 2009) and barrier-removal optimization — *which dams / weirs
/ culverts to remove, under a budget, to maximise reconnected habitat.*

```python
import pandas as pd
from pymarxan.rivers import (
    BarrierProblem,
    RiverNetwork,
    budget_dci_frontier,
    dci_diadromous,
    optimize_barriers_greedy,
    optimize_barriers_mip,
)

# A small river: S1 (the outlet) <- S2 <- S3, with an impassable barrier
# at the downstream end of S2 (B1) and of S3 (B2).
net = RiverNetwork(
    segments=pd.DataFrame(
        {"id": [1, 2, 3], "length": [10.0, 10.0, 10.0], "down_id": [-1, 1, 2]}
    ),
    barriers=pd.DataFrame(
        {
            "id": [1, 2],
            "segment": [2, 3],
            "pass_up": [0.0, 0.0],
            "pass_down": [0.0, 0.0],
            "removal_cost": [1.0, 1.0],
            "status": [0, 0],
        }
    ),
)
print(dci_diadromous(net))            # 33.33 — only the outlet reaches the sea

problem = BarrierProblem(net, budget=1.0)
greedy = optimize_barriers_greedy(problem)
exact = optimize_barriers_mip(problem)   # binary diadromous → provably optimal
print(greedy.removed, greedy.dci_after)  # {1} 66.67 — B1 gates B2's reach
print(exact.removed, exact.optimal)      # {1} True

# Efficiency frontier: DCI gained as the budget rises.
print(budget_dci_frontier(net, [0.0, 1.0, 2.0])[["budget", "dci_after"]])
```

Key pieces:

- **`RiverNetwork`** — a rooted river tree using the downstream-pointer
  encoding (`down_id`; the outlet's is `-1`/`0`/NA). A barrier sits at the
  *downstream end* of its `segment`. Build one from a HydroRIVERS / NHDPlus
  GeoDataFrame with `from_hydrorivers(gdf)` and attach barrier points with
  `snap_barriers(net, barriers_gdf)`.
- **DCI** — `dci_diadromous` (sea ↔ segment), `dci_potamodromous` (all
  within-network pairs), and `segment_connectivity` for per-segment values.
- **Optimizers** — `optimize_barriers_greedy` and `optimize_barriers_sa` work
  for general (partial) passability; `optimize_barriers_mip` is exact for the
  binary-passability diadromous case. All honour a `budget` and locked-in /
  locked-out barriers (`status`).
- **Analysis** — `budget_dci_frontier` sweeps budgets into an efficiency
  frontier; `barrier_selection_frequency` ranks barriers by how often they
  appear in good portfolios.

---

## Next steps

For the Shiny app surface, see `docs/USER_MANUAL.md`. For the full
release history, see `CHANGELOG.md`. For the architectural plans behind
each phase, see `docs/plans/2026-05-16-realignment.md` and the
multi-agent review docs under the same directory.
