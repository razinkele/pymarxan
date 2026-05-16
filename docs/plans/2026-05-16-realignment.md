# Marxan-Family Realignment Plan

**Date:** 2026-05-16
**Trigger:** Post-v0.1.0 audit of the broader Marxan ecosystem to see where pymarxan stands.
**Method:** Background research agent surveyed Marxan classic, MarZone, MarxanConnect (+ marxanconpy), Vizzuality/marxan-cloud, prioritizr, and MaPP. Findings spot-checked against the actual pymarxan source tree before this plan was written — three overstated gaps corrected in §1.

## TL;DR

pymarxan 0.1.0 is **functionally ahead** of the dormant C++ family (Marxan / MarZone / MarxanConnect, all dormant since 2021) and **at parity** with `prioritizr` on most algorithmic features, but **behind** on a small number of well-cited Marxan-classic features (clumping, separation distance) and on **breadth of MIP backends and importance metrics** that `prioritizr` ships. The cloud-platform features in Vizzuality/marxan-cloud are out of scope for a library and are explicitly **not** targeted.

The recommended path is **eight phases (18-25)** ordered by impact-per-effort. Phase 18-20 close the remaining Marxan-classic gaps; 21-23 reach prioritizr parity on objectives and importance; 24-25 round out the connectivity-metric library and add solution portfolios.

## 1. Corrections to the research report

The research agent overstated three gaps. Items below are **already present in pymarxan** and should NOT be re-implemented:

| Claimed missing | Actually present | Evidence |
|---|---|---|
| Probability features (PROB1D/2D, PTARGET, PROBMODE) | **Partial — PROBMODE 1 (risk premium) and PROBMODE 2 (persistence-adjusted amounts) implemented in MIP + heuristic** | `solvers/utils.py::compute_probability_penalty`, `mip_solver.py:157-220`, `heuristic.py:194-244`, `io/writers.py::write_probability` |
| Eigenvector centrality | **Present** | `connectivity/metrics.py::compute_eigenvector_centrality` |
| In-flow / out-flow | **Present (as in_degree / out_degree)** | `connectivity/metrics.py::compute_in_degree`, `compute_out_degree` |

What's still **genuinely missing** from probability support: PTARGET (probability target as a constraint rather than penalty), PROB2D (per-feature variance), Z-score-based formulation. That residue lives in §3 as the lower half of Phase 18.

## 2. Ecosystem snapshot (May 2026)

| Repo | Status | Last meaningful work |
|---|---|---|
| Marxan classic v4 (`Marxan-source-code/marxan`) | **Dormant** | v4.0.6, 2021-03-19 |
| MarZone (`Marxan-source-code/marzone`) | **Dormant** | v4.0.6, 2021-03-24 |
| MarxanConnect + marxanconpy | **Unmaintained** | 2020-2021 |
| Vizzuality/marxan-cloud | **Active** | 2026-05-14 — Kubernetes web platform |
| prioritizr (R) | **Active** | v8.1.0, 2025-11-10 |
| MaPP (TNC/Microsoft SaaS) | Closed-source service | n/a |

**Implication for pymarxan:** the four C++ family repos won't add new features and won't compete for the Python audience. `prioritizr` is the only living serious alternative, and it's R-only. The remaining gap-to-close is small and well-defined.

## 3. Phased realignment

Phases are ordered by **impact-per-effort**. Effort estimates assume the existing pattern from Phases 1-17 (TDD-per-task with one batch commit). Confidence reflects how cleanly each phase can be scoped from current code.

### Phase 18 — Probability completion (PTARGET + PROB2D + Z-score)

**Effort:** ~1 week. **Confidence:** 80%.

Existing probability infrastructure handles PROBMODE 1 (risk premium) and PROBMODE 2 (persistence-adjusted amounts) but not the Z-score formulation Marxan classic uses for variance-aware targets.

Add:
- Per-feature probability/variance fields (PROB2D — extend `pu_vs_features` with `variance`?).
- Z-score computation in `compute_probability_penalty`.
- PTARGET1D / PTARGET2D — probability target as a constraint (feature must be met with ≥ X probability), not just a penalty.
- Three new `PROBMODE` test cases for the Z-score path.

Touches: `models/problem.py`, `solvers/utils.py`, `solvers/mip_solver.py`, `solvers/simulated_annealing.py`, `io/readers.py`.

**Why first:** completes a feature pymarxan already 60% has — least new surface area for the most marketable headline ("full Marxan classic parity").

### Phase 19 — Clumping (TARGET2, CLUMPTYPE, type-4 species)

**Effort:** ~1.5 weeks. **Confidence:** 85%.

Marxan-classic's `clumping.cpp` lets users specify a minimum patch size per feature. Heavily used in habitat-fragmentation-aware planning. Source in `Marxan-source-code/marxan/clumping.cpp`, `NewPenalty4`, `PartialPen4`.

Add:
- Per-feature `clump_target` (`TARGET2`) and `clump_type` (`CLUMPTYPE` 0-2) fields.
- `compute_clump_penalty()` in `solvers/utils.py` (graph-connected-components on selected PUs filtered by feature presence).
- Integration into SA delta computation via `ProblemCache` (caches PU→adjacency for component recomputation on flip).
- MIP-only `add_min_clump_constraint` constraint for users who want exact clumping (slow but principled).
- Three test cases: no-clump baseline, sparse-target satisfied by single big clump, sparse-target failed by many small clumps.

**Why second:** classic Marxan feature, blocker for some users, doesn't depend on Phase 18.

### Phase 20 — Separation distance (SEPDISTANCE, SEPNUM)

**Effort:** ~3 days. **Confidence:** 90%.

Minimum geographic spacing between occurrences of a feature. MarZone explicitly punted on this ("re-enable when separation is reimplemented") so pymarxan reaches strict parity with v4 here. Reuses `spatial.grid.compute_adjacency` spatial index.

Add:
- `SEPDISTANCE`/`SEPNUM` parameters per feature.
- `CountSeparation()` in `solvers/utils.py`.
- Spatial-index-based pairwise distance check on flip in SA cache.
- MIP-only constraint counterpart.

**Why third:** small, self-contained, completes the Marxan-classic-feature trifecta from Phases 18-20.

### Phase 21 — HiGHS solver backend (and optional Gurobi)

**Effort:** ~3 days for HiGHS, ~2 days for Gurobi gating. **Confidence:** 95%.

PuLP already supports HiGHS via `HiGHS_CMD`. Currently `MIPSolver` hardcodes `pulp.PULP_CBC_CMD`. Refactor to:

- Add `MIPBackend` enum (`CBC`, `HiGHS`, `Gurobi`, `CPLEX`) on `SolverConfig`.
- `_make_pulp_solver(backend, config)` factory in `mip_solver.py`.
- Auto-detect available backends at startup (Gurobi/CPLEX behind optional extras: `pip install pymarxan[gurobi]`).
- Same change for `zones/mip_solver.py`.
- Add HiGHS to one CI matrix entry to keep the path tested.

**Why fourth:** unlocks Phase 22 (portfolios need a fast solver), 5-50× speed-up on large MIPs (HiGHS vs CBC), zero algorithmic changes — pure wiring.

### Phase 22 — Importance scores (replacement-cost, Ferrier, rank)

**Effort:** ~1 week. **Confidence:** 80%.

`prioritizr` ships four importance metrics; pymarxan ships one (`compute_irreplaceability`) plus `compute_selection_frequency`. Replacement cost is the gold standard. Add:

- `analysis/replacement_cost.py::compute_replacement_cost` — per-PU "how much more does the optimum cost when this PU is locked out?". Needs MIP backend (use HiGHS from Phase 21).
- `analysis/ferrier_importance.py::compute_ferrier_importance` — proportion of feature representation provided by each PU, weighted by SPF.
- `analysis/rank_importance.py::compute_rank_importance` — Jung 2021 sequential-removal ranking.
- Three new tests per metric; one new Shiny module under `modules/results/importance.py` to surface scores.

**Why fifth:** prioritizr-parity headline, scientific rigor improvement, single-package scope.

### Phase 23 — Extended objectives (max features, min largest shortfall, min penalties)

**Effort:** ~1 week. **Confidence:** 75%.

`prioritizr` exposes 9 objective formulations to pymarxan's 4 (`min_set`, `max_coverage`, `max_utility`, `min_shortfall`). Add the three most useful:

- `MaxFeaturesObjective` — maximise the *count* of features whose target is met under a cost budget.
- `MinLargestShortfallObjective` — minimax over feature shortfalls; useful when one feature is at risk of being abandoned.
- `MinPenaltiesObjective` — hierarchical: minimise feature penalty first, cost second.

Each is a `solvers/objectives/` subclass; MIP-only initially. Phylogenetic objectives (`MaxPhyloDiv`, `MaxPhyloEnd`) require a tree dependency (`dendropy` or `ete3`) and are deferred to a possible Phase 23.5.

**Why sixth:** broadens the pymarxan use case beyond classic Marxan; benefits from Phase 21's solver speed.

### Phase 24 — Connectivity metric expansion + post-hoc clustering

**Effort:** ~1 week. **Confidence:** 85%.

Close the MarxanConnect feature gap. pymarxan currently has betweenness, eigenvector, in_degree, out_degree. Add:

- `compute_pagerank_centrality` (networkx one-liner).
- `compute_donors`, `compute_recipients` (thresholded in/out-degree variants).
- `connectivity_to_boundary` — convert a connectivity matrix to a `bound.dat`-compatible format so users can feed connectivity into the BLM penalty.
- `connectivity/temporal.py::compute_temporal_connectivity` — multi-timestep connectivity matrices, returns a (n, n) summary.
- `connectivity/resistance.py::habitat_resistance_to_matrix` — least-cost-path connectivity from a habitat-resistance raster (uses scikit-image `route_through_array` or networkx Dijkstra).
- `analysis/posthoc_clusters.py::compute_solution_clusters` — fragmentation diagnostics on a Solution.

One new Shiny module: `modules/connectivity/metrics_dashboard.py` to display all metrics together.

**Why seventh:** medium impact (the MarxanConnect community is small but the GUI is unmaintained), but a clean way to claim the MarxanConnect feature niche before someone else does.

### Phase 25 — Solution portfolios + Cohon BLM calibration

**Effort:** ~1 week. **Confidence:** 70%.

prioritizr's portfolio system (`add_top_portfolio`, `add_gap_portfolio`, `add_extra_portfolio`) lets users generate K high-quality MIP solutions without re-solving. Add:

- `Portfolio` type with `top_k`, `gap`, `extra`, `cuts` strategies.
- Top-k and gap require Gurobi (Phase 21 gates this); cuts is solver-agnostic.
- Wire into `MIPSolver.solve()` via `SolverConfig.portfolio = Portfolio(...)`.
- `calibration/cohon.py::calibrate_cohon_blm` — automatic BLM calibration via Cohon's method; more principled than the BLM grid sweep already in pymarxan.

**Why last:** depends on Phases 21+23, smaller user base, but rounds out feature parity with prioritizr's "professional" tier.

## 4. Explicitly out of scope

Listed so future contributors understand the boundary.

- **Cloud / multi-tenant deployment.** Vizzuality/marxan-cloud is ~14k LoC TS + Kubernetes + dual Postgres + Redis. pymarxan is a Python library; if a team needs a cloud service they can deploy pymarxan behind FastAPI themselves.
- **Hill climbing as a separate solver.** Marxan v4 separates `hill_climbing` from iterative improvement; pymarxan's `ITIMPTYPE` modes 0-3 cover the same conceptual space. Treat as notional.
- **Quantum annealing.** Marxan v4's `quantum_annealing.cpp` source comments admit it's broken. Skip unless user demand emerges.
- **MarxanConnect `.MarCon` import.** MarxanConnect is unmaintained; reverse-engineering its project format for migration support is not worth the effort.
- **BLOCKDEF (species block defaults).** Convenience syntactic-sugar feature; users can express the same thing in code today.
- **Multithreaded SA via OpenMP.** Memory's existing `Pure NumPy vectorization (no Numba/Cython)` decision is intentional. Parallel sweep already covers the multi-run case. Single-run SA speed is dominated by `ProblemCache` delta wins, not thread count.

## 5. What's NOT in any phase but worth tracking

These are real gaps the audit found, but they don't justify a dedicated phase. Track in issues / future minor releases.

- **Marxan v4 output extras:** `writeRichness`, `writeTotalAreas`, `writeWeightedConnectivityFile`, `writeR` (R script for plotting). Add ad-hoc if users request.
- **Marxan-cloud project bundle import/export.** Could be a v0.3 feature if pymarxan gains traction as a desktop alternative.
- **Marxan v4 asymmetric-connection file write.** Internal format detail; add if a real user surfaces with the need.

## 6. Sequencing

Suggested order with optional fast-track:

```
Phase 18 → Phase 19 → Phase 20  ──→  v0.2.0 ("Full Marxan-classic parity")
                            │
Phase 21 → Phase 22 → Phase 23 ──→  v0.3.0 ("prioritizr-parity")
                            │
Phase 24 → Phase 25         ──→  v0.4.0 ("Connectivity + portfolios")
```

Each version bump is a natural release point with a clear marketing line. Phases inside a version target can ship as patch tags (`v0.2.0-rc1`, etc.) if needed.

## 7. Acceptance criteria

A phase is "done" when:

1. All new code has unit tests with the same TDD-per-task pattern as Phases 1-17.
2. `make check` (lint + types + test) stays green — no new ruff/mypy errors and no skipped tests.
3. The Shiny app exposes any user-facing knob the phase introduces.
4. `CHANGELOG.md` gets a new `## [Unreleased]` entry that the release commit promotes to `[0.2.0]` / `[0.3.0]` / `[0.4.0]`.
5. Test count grows; coverage doesn't drop below 90%.

## References

- Research agent transcript: see git note on the commit that created this file.
- Marxan v4 source: https://github.com/Marxan-source-code/marxan (v4.0.6, 2021)
- MarZone source: https://github.com/Marxan-source-code/marzone
- MarxanConnect: https://github.com/remi-daigle/MarxanConnect + https://github.com/remi-daigle/marxanconpy
- marxan-cloud: https://github.com/Vizzuality/marxan-cloud
- prioritizr: https://github.com/prioritizr/prioritizr (v8.1.0, 2025-11)
- Marxan User Manual: https://marxansolutions.org/wp-content/uploads/2021/02/Marxan-User-Manual_2021.pdf
