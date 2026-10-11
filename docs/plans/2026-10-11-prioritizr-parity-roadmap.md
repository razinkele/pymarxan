# prioritizr-parity roadmap (v0.39 → v0.42)

**Date:** 2026-10-11 · **Baseline:** pymarxan v0.38.0 (`e5a066c`) · **Reference:** prioritizr
v9.0.1 reference index (<https://prioritizr.net/reference/>), checked 2026-10-11.

## Why

The 2026-06-12 ecosystem survey found broad parity with prioritizr. Since then prioritizr 9.0
shipped its multi-objective framework (survey item C5, "watch, don't build yet") and 8.1/9 added
about twenty published target-setting methods. A gap check of pymarxan's source against the v9
index left four groups of work worth building:

| Stage | Release | Theme | Size | Depends on |
|---|---|---|---|---|
| 1 | v0.39.0 | Zone-MIP parity + solver tooling | Small–medium | v0.38 zone rules |
| 2 | v0.40.0 | Published target-setting methods | Medium (many small parts) | — |
| 3 | v0.41.0 | Multi-objective optimisation | Medium–large | Stage 1 (problem export helps debugging) |
| 4 | v0.42.0 | Proportion / semi-continuous decisions | Large | Stage 3 (objective builders refactored) |

Every stage follows the process that works here: brainstorm → spec → 4-lens design review
(grounding lens executes code) → plan → dry-run plan review loop → subagent-driven TDD →
whole-branch review → gate → release. Every solver-touching stage keeps the MIP anchor
(`tests/data/simple` = 35.0) and v0.37/v0.38 golden tests green.

### Already at parity (don't rebuild)

- **Objectives:** min set, max features met (`max_features`), weighted features met, min shortfall, min largest shortfall, min penalties, phylogenetic diversity.
- **Constraints:** contiguity, feature contiguity, min-neighbour, linear, budget, locked in/out, zone pulock/puzone.
- **Penalties:** boundary, symmetric and asymmetric connectivity, Cohon BLM calibration with a Pareto frontier.
- **Portfolios:** cuts, gap filter, selection frequency.
- **Importance:** replacement cost, Ferrier, incremental rank.
- **Targets:** relative, log-linear, group.

---

## Stage 1 — v0.39.0: Zone-MIP parity + solver tooling

**Goal:** zone problems get the same exact objectives as single-zone problems, and users can see
and check the optimisation model before solving.

**Scope**
1. **Zone-MIP objectives.** `ZoneMIPSolver(objective=...)` with `"min_set"` (default, today's
   model), `"max_features"`, `"min_shortfall"`, `"min_largest_shortfall"` (budgeted, as in
   `MIPSolver`). Zone targets (overall + per-zone, `zone_target_lookup`) become the met/shortfall
   terms.
2. **Mandatory allocation.** `ZoneMIPSolver(mandatory_allocation=True)` (and an equivalent
   ZonalProblem parameter honoured by SA/heuristic/II) forces every non-locked-out PU into some
   zone — implemented as `may_be_unassigned=False` in `pu_zone_rules()`, so all four solvers
   inherit it from the v0.38 rule machinery.
3. **Problem export.** `write_problem(problem, path, solver=...)` for `MIPSolver` and
   `ZoneMIPSolver`; `.lp` or `.mps` chosen by extension, via PuLP `writeLP` / `writeMPS`.
4. **Presolve check.** `presolve_check(problem) -> list[str]` (core and zone problems) warns before
   solving about:
   - cost, amount or boundary ranges wider than about 1e6, which risk numerical trouble;
   - features with zero total amount;
   - targets above the total amount;
   - every PU locked;
   - negative or non-finite costs;
   - a BLM-times-boundary scale far from the cost scale.

   It reuses `diagnose_zone_infeasibility` for zones. Solvers call it only when asked
   (`check=True`), so default behaviour does not change.

**Acceptance**
- Zone anchor problems solved by enumeration match each new zone objective.
- Mandatory allocation leaves no PU in zone 0 in any of the four solvers, checked with
  `rule_violations` on their outputs.
- An exported `.lp` file re-solved by CBC gives the same objective as the solver did.
- `presolve_check` has one test per warning, and gives no warnings on `tests/data/simple`.

**Risks:** budgeted zone objectives need a zone cost definition (sum of zone costs is the obvious
one — confirm against MarZone and prioritizr zones semantics in the design review).

---

## Stage 2 — v0.40.0: Published target-setting methods

**Goal:** users can set targets with the methods conservation reports cite, not just relative or
log-linear rules.

**Scope** — functions in `pymarxan.targets` that return per-feature targets and work with
`apply_targets` / `group_targets`:

| Method | Source | Needed inputs (beyond feature totals) |
|---|---|---|
| Rodrigues et al. 2004 | range-size interpolation | feature range size |
| Watson et al. 2010 | range-size rule | range size |
| Wilson et al. 2010 | (verify) | (verify) |
| Polak et al. 2015 | (verify) | (verify) |
| Durán et al. 2020 | (verify) | (verify) |
| Jung et al. 2021 | (verify) | (verify) |
| Ward et al. 2025 | (verify) | (verify) |
| Sreekar & Watson 2026 | (verify) | (verify) |
| IUCN Red List of Threatened Species | threat-category rules | category per feature |
| IUCN Red List of Ecosystems | criterion rules | category per feature |
| Population size | minimum viable population | density / population per unit |
| Area-based, interpolated absolute/area | threshold interpolation | units (km²) |
| Rule-based, min/max caps | composition helpers | — |

"(verify)" marks entries where the method details have not yet been read. Each method's formula
must be checked against its paper (scite: the paper exists, is not retracted, and says what is
claimed) and against prioritizr's implementation before it is coded. This is the science lens
of the design review, and it is the critical step in this stage.

**Acceptance**
- Each method has a worked example taken from its paper, or from prioritizr's documentation
  where the paper gives none, pinned as a test.
- Unit handling (km² vs native units) is explicit, with no silent unit assumptions.
- The CHANGELOG cites each source.

**Risks:** some methods need inputs pymarxan does not model yet (range size, IUCN category,
population density). Take them as plain per-feature columns or dicts; do not build new data
models.

---

## Stage 3 — v0.41.0: Multi-objective optimisation

**Goal:** trade off several objectives (e.g. cost vs representation vs connectivity) exactly,
the way prioritizr 9's `multi_problem()` does.

**Scope**
1. **Objective builders.** Refactor the MIP objective blocks in `mip_solver.py` into reusable
   builders that each add their variables and constraints and return an expression. Today they
   are `if/elif` blocks, and this refactor is the prerequisite for the rest.
2. **Approaches:**
   - **Hierarchical (lexicographic):** solve objective 1, constrain it to within a relative
     tolerance of its optimum, solve objective 2, and so on.
   - **Weighted sum:** combine objectives with weights. A weights-matrix sweep produces a
     frontier.
   - **Reference point:** an achievement scalarising function (Chebyshev distance to an
     aspiration point). The design must confirm normalisation (ideal/nadir) against prioritizr.
3. **API:** `MultiObjectiveProblem(problem, objectives=[...])` plus an approach object, returning
   a list of `Solution`s. It sits alongside `calibration.pareto` and reuses its frontier filter.

MIP only; CBC/HiGHS. Zones in a follow-up unless Stage 1's builders make it free.

**Acceptance**
- The hierarchical approach on a 2-objective toy problem matches a hand-enumerated
  lexicographic optimum.
- The weighted-sum sweep reproduces the convex hull of an enumerated frontier.
- The reference-point approach returns the closest feasible plan on an enumerated example.
- Single-objective results do not change: `tests/data/simple` still gives 35.0.

**Risks:**
- The objective-builder refactor touches the most-used solver; golden tests for every
  objective must be recorded first, as v0.38 did for the zone solvers.
- Scaling between objectives with very different magnitudes; normalisation is a design decision.

---

## Stage 4 — v0.42.0: Proportion and semi-continuous decisions

**Goal:** a PU can be partly selected (0–1), or selected above a floor if selected at all. This
covers partial protection, restoration areas and land-use shares, and pairs with
`pymarxan.restoration`.

**Scope**
1. `MIPSolver(decisions="binary" | "proportion" | "semicontinuous", upper=..., lower=...)`.
   Continuous `x_i ∈ [0, u]`; semi-continuous `x_i = 0 or x_i ∈ [l, u]`. In the MIP this is
   mostly a variable-type change.
2. **Solution model:** `Solution.selected` may hold fractions. Representation, gap, cost and
   targets met become fraction-weighted. Boundary and connectivity with fractional decisions
   need a stated linearisation; check what prioritizr does.
3. **Analyses:** each `pymarxan.analysis` function either supports fractions or raises a clear
   error. No silent rounding.
4. SA, heuristic, II and Zonation stay binary and raise a clear error on fractional decisions.

**Acceptance**
- The proportion LP optimum is never above the binary optimum, and equals it on problems with an
  integral LP.
- The semi-continuous result respects the floor.
- Representation of a fractional solution matches a hand calculation.
- Binary results don't change: the anchor still gives 35.0.

**Risks:** the widest blast radius of the four stages (`Solution` consumers, Shiny maps,
writers). Run the design review's blast-radius lens over every reader of `Solution.selected`.

---

## Backlog (small parity items, not scheduled)

Fold these into a stage when one touches the same code:
- `max_cover` and `max_wtd_sum` objectives: maximise the amount held rather than targets met.
- Phylogenetic endemism objective, reusing `pymarxan.phylo`.
- Rarity-weighted-richness importance.
- Cost penalties and linear penalties.
- Neighbour penalties.
- Shuffle portfolio. Top and extra portfolios need a solver solution pool, which CBC and HiGHS
  lack.
- `simulate_data` / `simulate_species` / `simulate_cost` synthetic project generator, useful for
  tutorials and the JOSS paper.
