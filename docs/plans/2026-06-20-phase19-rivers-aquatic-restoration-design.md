# Implementation Plan — `pymarxan.rivers`

**Aquatic (riverine) restoration: dendritic connectivity + barrier-mitigation optimization**

Status: design draft, **revised 2026-06-20 per multi-agent review** · Target: `pymarxan` v0.5 · Owner: maintainer
Filename: `docs/plans/2026-06-20-phase19-rivers-aquatic-restoration-design.md`
Review: [`2026-06-20-phase19-rivers-review.md`](2026-06-20-phase19-rivers-review.md) — four-lens review (architect / grounding / scientific / independent re-design). No CRITICAL findings; HIGH/MEDIUM fixes folded in below (networkx-optional, connectivity wording, reuse-is-pattern, dual-solver note, Y-tree fixture, `dci_before` semantics, backend-factory extraction). Citations all verified real via scite.

---

## 1. Goal and scope

Add a first-class river-network restoration capability to pymarxan, built around the
problem the aquatic literature actually optimizes: **which barriers (dams, weirs,
culverts) to remove or fit with passage, to maximize reconnected habitat per unit
budget.** The connectivity currency is the **Dendritic Connectivity Index (DCI)**
(Cote et al. 2009), in its potamodromous and diadromous forms.

In scope for this plan:
- A directed river-network model (segments + barriers + passabilities).
- Native DCI computation (DCIp, DCId) with an override hook for optimization.
- A barrier-decision problem model (remove / mitigate / keep; cost; budget; locks).
- Three optimizers: greedy, simulated annealing (both for general partial
  passability), and an exact MIP for the binary-passability diadromous case.
- Data ingest from river-network GIS (HydroRIVERS `NEXT_DOWN`-style topology).
- Validation of DCI against the R `dci` package via an `rpy2`-gated test.

Out of scope (handled elsewhere / later): the marine restoration-site-selection
path — it is already served by the existing connectivity facilities (directed
connectivity matrices via `connectivity/io.py::connectivity_to_matrix(symmetric=False)`
plus degree/donor–recipient metrics in `connectivity/metrics.py`, and current-flow /
effective-resistance connectivity in `connectivity/circuit.py` — note the latter is
*symmetric*, not directional) together with the standard selection objectives; it
needs a recipe/tutorial, not new core code.

## 2. Why this fits the existing architecture

The barrier problem maps onto machinery pymarxan already has:

| Need | Existing pymarxan piece | Reuse kind |
|---|---|---|
| Non-removable barriers | `STATUS_LOCKED_OUT` constant (`models/problem.py:16-19`) | import constant |
| Mandatory removals | `STATUS_LOCKED_IN` | import constant |
| Budget cap | existing budget/linear-constraint pattern | pattern |
| Exact solve | `mip_solver.py` PuLP backend factory (`_available_backends`, CBC/HiGHS/Gurobi) | import (after extraction — see §7.3) |
| Cooling schedules | `solvers/cooling.py::CoolingSchedule` | import (clean) |
| Many-run aggregation | `analysis/selection_freq.py` / `analysis/portfolio.py` | **pattern only** — see note |
| Graph traversal | `networkx` | dep — **see note** |
| Config / `seed`/`verbose` convention | `solvers/base.py::SolverConfig` | convention only — `rivers` writes its own config |

New module mirrors the repo's `models/` + `solvers/` split and `from __future__
import annotations` + NumPy-docstring conventions (match the NumPy-style majority;
do **not** copy `connectivity/circuit.py`'s Google-style docstrings).

**Reuse caveats (from design review):**
- **`networkx` is an *optional* dependency** (only under `[project.optional-dependencies]
  shiny` in `pyproject.toml`), not core. Core modules that use it lazy-import inside
  functions (`connectivity/metrics.py`, `connectivity/resistance.py`). `rivers/network.py`
  must do the same **or** networkx must be promoted to a core dependency — decide before
  Phase A.
- **Selection-frequency / portfolio reuse is *pattern* reuse, not API reuse.**
  `analysis/selection_freq.py` and `portfolio.py` are hard-coupled to `Solution.selected`
  (length-`n_pu` bool array) and `Solution.objective`. A `BarrierSolution` carries
  `removed: set[int]`, so barrier-frequency is reimplemented in `rivers/` (a trivial
  membership count), mirroring the pattern — not importing those functions.
- **Barrier optimizers are standalone, not `Solver` subclasses.** `Solver.solve` is typed
  to `ConservationProblem` and `SolverRegistry` takes `type[Solver]`, so barrier
  optimizers cannot register there or appear in the Shiny solver picker. This is
  intentional: barrier removal is a different problem shape (decisions on tree edges,
  nonlinear DCI objective). pymarxan will have two parallel solver universes; that is the
  right call, stated here so it is not mistaken for a gap.

## 3. Module layout

```
src/pymarxan/rivers/
  __init__.py        # public exports
  network.py         # RiverNetwork model + topology helpers
  dci.py             # dci_potamodromous, dci_diadromous, connectivity helpers
  barriers.py        # BarrierProblem (decisions, costs, budget, locks, actions)
  optimize.py        # greedy / anneal / mip barrier optimizers + BarrierSolution
  io.py              # ingest from river GIS; snap barriers to segments (Phase D)
tests/pymarxan/rivers/
  test_network.py
  test_dci.py
  test_optimize.py
  test_dci_validation.py   # rpy2 + R `dci`, gated/skipped when absent
```

## 4. Data model (`network.py`)

Use the **downstream-pointer** encoding — every segment names its one downstream
neighbour. This *is* a rooted tree and maps directly onto hydrological data
(HydroRIVERS `NEXT_DOWN`, NHDPlus `ToNode`), so ingest is trivial.

```python
@dataclass
class RiverNetwork:
    segments: pd.DataFrame   # columns: id, length, down_id, [weight]
    barriers: pd.DataFrame   # columns: id, segment, pass_up, pass_down,
                             #          removal_cost, status, [pass_if_mitigated]
    # outlet = the unique segment whose down_id is NA/-1 (drains to the sea)
```

Conventions:
- A barrier sits at the **downstream end** of its `segment` (i.e. on the link to
  `down_id`). A barrier on the outlet segment models a barrier at the river mouth.
- `length` is the segment habitat length; `weight` defaults to `length` but allows
  habitat-quality weighting (e.g. weight by suitable spawning habitat).
- `pass_up` / `pass_down` ∈ [0, 1]; default symmetric (`pass_up == pass_down`).
- `status` reuses `STATUS_NORMAL / LOCKED_IN / LOCKED_OUT`.
- `pass_if_mitigated` (optional): passability after a fishway is installed — enables
  a three-way action (keep / mitigate / remove) later without a model change.

`__post_init__` validates: unique segment ids, unique barrier ids, every `down_id`
resolves to an existing segment or NA, **exactly one outlet**, acyclic (tree), and
passabilities in range. Build an internal `networkx.DiGraph` (edge segment → down_id)
once and cache it. **Lazy-import `networkx` inside the method** (it is an optional dep —
see §2 caveat), matching `connectivity/`. In practice the rooted-tree topology only needs
the downstream-pointer walk + an LCA table, so a pure-NumPy/dict implementation that drops
the `networkx` dependency entirely is worth considering during Phase A.

Topology helpers (private, cached):
- `_root_products(passabilities)` → for each segment, the product of barrier
  passabilities from that segment **down to the mouth** (= `c_i` for DCId). Single
  downstream walk, O(n).
- `_lca(i, j)` → lowest common ancestor, for `c_ij` (DCIp) via
  `c_ij = root_prod(i) * root_prod(j) / root_prod(lca)**2`.
- `path_barriers(i, j)` / `path_barriers_to_mouth(i)` → barrier ids on a path
  (used by the MIP to build per-segment constraints).

## 5. DCI computation (`dci.py`)

Formulas (Cote et al. 2009), with `L = Σ l_i` and weights `w_i = l_i / L`:

- **Potamodromous:** `DCIp = 100 · Σ_i Σ_j w_i w_j · c_ij`, `c_ij = ∏_{b ∈ path(i,j)} p_b`, `c_ii = 1`.
- **Diadromous:** `DCId = 100 · Σ_i w_i · c_i`, `c_i = ∏_{b ∈ path(i, mouth)} p_b`.

Public API:

```python
def dci_diadromous(network, passabilities=None, *, direction="symmetric") -> float
def dci_potamodromous(network, passabilities=None) -> float
def segment_connectivity(network, passabilities=None, form="diadromous") -> dict[int, float]
```

`passabilities` is an optional `{barrier_id: p}` override so optimizers can score a
candidate solution (removed barrier → p = 1.0) without mutating the network.
Effective passability under a removal decision `y_b ∈ {0,1}`:
`p_b^eff = p_b + (1 − p_b)·y_b`.

**Convention to pin down before coding:** the diadromous form can be single-pass
(product of passabilities segment→mouth) or round-trip (`∏ p_b²`). Default to
**single-pass** to match the `dci` R package; expose `direction="round_trip"` for the
diadromous spawning-migration interpretation. Document this explicitly — it is the
most likely source of a validation mismatch. The directional / round-trip
formalisation (symmetric `∏ p_up·p_down` vs directional single-pass) is made explicit
in Baldan et al. 2022 (`riverconn`); cite that, not just Côté 2009, for this convention.

**Scope recommendation (design review):** ship **symmetric-only** DCI in v0.5 — keep the
`pass_up`/`pass_down`/`pass_if_mitigated` columns in the data model, but defer the
directional *DCI math* (open decision #2) until there is a validation target for it. The
directional combination rule is fiddly and would otherwise be an unvalidated correctness
liability against the (single-pass, symmetric) `dci` package.

Performance notes (implement simple first, optimize later): use the `_root_products`
+ LCA route (O(n) build, O(1) per pair) rather than per-pair path walks; add an
**incremental evaluator** that recomputes only segments whose path-to-mouth contains
a flipped barrier (big speedup inside SA).

### Validated test fixture (hand-computed — use in `test_dci.py`)

Linear chain, outlet below S1, no mouth barrier:
`S1 —[B1]— S2 —[B2]— S3`, lengths `l1 = l2 = l3 = 10` (so `w_i = 1/3`),
`p(B1) = p(B2) = 0.5`.

- `c_1 = 1`, `c_2 = 0.5`, `c_3 = 0.25` → **DCId = 100·(1/3)(1 + 0.5 + 0.25) = 58.333…**
- pair sum `Σ_iΣ_j c_ij = 3 + 2(0.5) + 2(0.25) + 2(0.5) = 5.5` → **DCIp = 100·(1/9)·5.5 = 61.111…**
- all barriers removed → **DCIp = DCId = 100** (sanity).
- single segment → **100** (sanity).
- removing any barrier never decreases DCI (monotonicity property test).

**Confluence (Y-tree) fixture — REQUIRED (design review H5).** The linear chain above
never exercises a non-trivial lowest common ancestor, so a bug in
`c_ij = root_prod(i)·root_prod(j) / root_prod(lca)²` would pass every test above. Add a
Y-shaped network: two headwater segments `S2`, `S3` joining at a confluence segment `S1`
that drains to the mouth, with a barrier on each headwater link (`B2` above `S2`, `B3`
above `S3`) and optionally one on the outlet (`B1` below `S1`). Hand-compute `DCIp` for
this network so the pair `(S2, S3)` — whose LCA is the interior segment `S1`, *not* an
endpoint — is checked explicitly. This is the single most important fixture to add.

## 6. Barrier-decision problem (`barriers.py`)

```python
@dataclass
class BarrierProblem:
    network: RiverNetwork
    budget: float | None = None          # cost cap on the chosen action set
    form: str = "diadromous"             # "diadromous" | "potamodromous"
    objective: str = "max_dci"           # "max_dci" | "max_gain" | "min_cost_for_target"
    target_dci: float | None = None      # required for min_cost_for_target
```

Decision variable: per removable barrier, `y_b ∈ {0, 1}` (1 = remove). Locked-out
barriers are fixed `y_b = 0`; locked-in fixed `y_b = 1`. (Three-way keep/mitigate/
remove is a later extension using `pass_if_mitigated`.)

```python
@dataclass
class BarrierSolution:
    removed: set[int]          # barrier ids selected for removal
    cost: float
    dci_before: float
    dci_after: float
    gain: float
    optimal: bool              # True only from the exact MIP
```

**Baseline semantics (pin before coding — design review M2):** `dci_before` is the DCI of
the network at its **current/as-given passabilities** with **no** decision applied
(locked-in removals are *not* pre-applied in the baseline). `dci_after` applies the chosen
removals (including locked-in), and `gain = dci_after − dci_before`. Every optimizer must
report against this same baseline so `gain` is comparable across greedy / SA / MIP. State
this in the `BarrierSolution` docstring.

## 7. Optimizers (`optimize.py`)

### 7.1 Greedy (always available, no external solver)
Iteratively add the barrier with the best **DCI-gain-per-cost** until the budget is
exhausted or no positive-gain barrier remains. Skips locked-out, pre-includes
locked-in. Uses the incremental DCI evaluator. This is the practitioner default and
the baseline for the others.

### 7.2 Simulated annealing (general partial passability)
Reuse the project's SA philosophy (cooling schedules, flip-and-evaluate) over the
binary barrier vector. **This is the key advantage of the dual-engine design:** DCIp
and partial-passability DCId are *nonlinear*, and SA optimizes them directly with no
linearization. Enforce the budget as a penalty (soft) or rejection (hard). Returns
`optimal=False`.

### 7.3 Exact MIP — binary-passability diadromous (`optimize_barriers_mip`)
For the common modelling assumption "barrier present ⇒ impassable (p=0); removed ⇒
passable (p=1)", segment *i* is connected to the sea **iff every barrier on its path
to the mouth is removed**. Then `c_i = ∏_{b∈path(i)} y_b` (product of binaries),
which linearizes **exactly**:

```
maximize    Σ_i w_i · c_i
subject to  c_i ≤ y_b                 ∀ i, ∀ b ∈ path(i, mouth)
            c_i ≥ Σ_{b∈path(i)} y_b − (|path(i)| − 1)
            Σ_b removal_cost_b · y_b ≤ budget
            y_b = 0   ∀ b locked out
            y_b = 1   ∀ b locked in
            c_i ∈ [0,1],  y_b ∈ {0,1}
```

This is the classic connected-habitat formulation (O'Hanley 2011; Kuby et al. 2005
lineage; the probability-chain technique of King & O'Hanley 2014 and Neeson et al. 2015
is what generalises this binary AND-linearisation to the deferred partial-passability
case). It is linear and solves on the existing CBC/HiGHS/Gurobi backends. Returns
`optimal=True`.

**Prerequisite refactor (design review M1):** `_available_backends()` / `_make_*` are
module-private in `mip_solver.py`. Before this phase, promote the PuLP backend factory to
a shared importable home (e.g. `solvers/_backends.py`) with no behaviour change — keeping
`mip_solver.py`'s tests green — and have both `mip_solver` and `rivers` import it. This
avoids copy-pasting private internals across subpackages. Make it an explicit Phase C
task, not a same-PR afterthought.

The MIP `c_i` are correctly declared **continuous in `[0, 1]`** (not binary): under
maximisation with non-negative weights the `c_i ≤ y_b` constraints pin `c_i` to the AND of
the `y_b`, so only the `y_b` need be integer.

**Potamodromous exact MIP** is deferred: the all-pairs product objective is harder to
linearize compactly. Phase C ships diadromous-exact + heuristics for both forms;
a potamodromous MIP (log-linearization or O'Hanley's network formulation) is logged
as future work so we don't block on it.

## 8. Tests (mirror existing style: plain pytest, `_make_*` builders)

- **test_network.py** — construction; validation errors (duplicate ids, dangling
  `down_id`, two outlets, cycle, out-of-range passability); `path_barriers_*`;
  `_root_products` against hand values.
- **test_dci.py** — the §5 linear-chain fixture (DCIp = 61.111…, DCId = 58.333…);
  **the §5 Y-tree confluence fixture** (hand-computed DCIp exercising an interior-node
  LCA — REQUIRED, H5); all-removed = 100; single-segment = 100; monotonicity;
  weight-vs-length weighting.
- **incremental-delta correctness** — SA incremental DCI delta == full recompute after
  each flip, for **both** forms. The *potamodromous* delta (a flipped barrier changes
  `c_ij` for every pair straddling it) is the error-prone one and needs its own
  delta-vs-from-scratch equality test (design review M3).
- **test_optimize.py** —
  - greedy respects budget and never selects locked-out;
  - MIP optimum on the binary fixture: with `budget = 1` removal it picks **B1**
    (→ DCId 66.67), not B2 (→ 33.33), because B1 gates B2's reach;
  - MIP == brute-force enumeration on every ≤8-barrier random net (exactness check);
  - locked-out B1 ⇒ DCId stuck at 33.33 regardless of budget;
  - SA reaches the MIP optimum on the binary case (within tolerance);
  - all heuristics ≤ MIP optimum (never beat exact) — property test via `hypothesis`.
  - **infeasible / degenerate guards** (per this repo's review history): budget too small
    for any removal → return the baseline solution, not a crash; all barriers locked-out →
    baseline; zero-length segment; single-segment network. Specify whether
    `removal_cost`/`budget` are integers or arbitrary floats (the brute-force and any
    tree-knapsack DP cross-checks must agree on cost semantics).
- **test_dci_validation.py** — `pytest.importorskip("rpy2")` + check the R `dci`
  package is installed; build a random tree, compare native DCIp/DCId to the R
  package within `1e-6`. Skips cleanly (not fails) where R/rpy2 absent, so CI on
  Python-only runners is green.

## 9. Data ingest (`io.py`, Phase D)

- `from_hydrorivers(gdf, *, next_down="NEXT_DOWN", length="LENGTH_KM")` — build a
  `RiverNetwork` straight from a HydroRIVERS/NHDPlus GeoDataFrame using the
  downstream-pointer field. This is why the §4 encoding was chosen.
- `snap_barriers(network, barriers_gdf, tolerance=...)` — assign each barrier point
  to the nearest segment and downstream end; reuse `geopandas`/`shapely` already in
  `spatial/`.
- Optional: derive a network from a DEM (note `pysheds` / WhiteboxTools) for basins
  without a ready vector network — keep as an optional extra, not a core dep.

## 10. Analysis & UI (Phase E)

- **DCI-gain-vs-budget curve**: solve the MIP (or greedy) across a budget sweep →
  the efficiency frontier practitioners ask for. New helper in `analysis/`.
- **Barrier selection frequency**: run many SA solves → reuse
  `analysis/selection_freq.py` to rank barriers by how often they appear in good
  portfolios (robust no-regret picks).
- **Shiny `rivers` panel** (mirror existing modules): network map, barriers coloured
  by selection frequency, the budget–DCI curve, before/after DCI readout.

## 11. External references / validation targets

- **`dci`** (R, open) — primary DCI validation (`test_dci_validation.py`).
- **`riverconn`** (R, Baldan et al.) — secondary cross-check; menu of further indices.
- **Algorithmic citations** for code + JOSS paper: Cote et al. 2009 (DCI);
  O'Hanley 2011, Kuby et al. 2005, King & O'Hanley 2014, Neeson et al. 2015
  (barrier-removal optimization). Reimplement from papers (MIT-clean); do **not** copy
  GPL R source.

## 12. Phasing (smallest valuable unit first)

| Phase | Deliverable | Depends on |
|---|---|---|
| **A** | `network.py` + `dci.py` (read-only DCI metrics) + tests | — |
| **B** | `barriers.py` + greedy + SA (general passability) + tests | A |
| **C** | (C0) extract PuLP backend factory to a shared home (M1); then exact diadromous MIP + tests + brute-force/exactness checks. *Optional:* an O(n·budget) tree-knapsack DP gives a backend-free exact diadromous solver + a clean MIP cross-check | A, B |
| **D** | `io.py` HydroRIVERS ingest + barrier snapping | A |
| **E** | budget–DCI frontier, selection frequency, Shiny panel | B/C |
| **F** | docs (TUTORIAL section), validation harness, JOSS paper update | A–E |

Phase A alone is independently useful (compute DCI for any network) and is the
right first PR.

## 13. JOSS paper update (Phase F)

Reposition the statement of need from "Marxan in Python" to "the Marxan family in
Python **plus native river-connectivity (DCI) and barrier-mitigation optimization**,
spanning marine site-selection and riverine barrier portfolios" — a gap prioritizr
(no dendritic connectivity), restoptr (terrestrial-2D, Java-bound), and the narrow/
closed barrier tools do not jointly fill. Add one figure: a budget–DCId frontier on a
real basin (e.g. Nemunas / a Baltic tributary), heuristic vs exact MIP.

## 14. Open decisions to confirm before coding

1. **DCId convention** — single-pass (match `dci` package) vs round-trip; ship
   single-pass default + `direction` flag. *(blocks §5 + validation)*
2. **Directional passability** in DCIp — how up/down combine on mixed-direction
   paths; default symmetric, document the directional rule.
3. **Three-way action** (keep/mitigate/remove) — ship binary first, design the data
   model (`pass_if_mitigated`) so the extension needs no breaking change.
4. **Potamodromous exact MIP** — accept heuristic-only for v0.5; log linearization
   approach as future work.
5. **Performance target** — basin size to support at interactive speed (drives
   whether the incremental evaluator / LCA optimization lands in Phase A or later).

---

*Once these five decisions are confirmed, Phase A (`network.py` + `dci.py` + tests)
is directly implementable from §4–§5 and the hand-computed fixture.*
