# Changelog

All notable changes to **pymarxan** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Phase 23 — Extended MIP objectives.** ``MIPSolver`` gains an
  ``objective`` kwarg with four choices:
  - ``"min_set"`` (default, pre-Phase-23 behaviour) — minimise
    ``cost + BLM·boundary + connectivity + probability + penalty``
    subject to ``amount ≥ target·MISSLEVEL``.
  - ``"max_features"`` — maximise the count of feature targets met
    under a cost cap. Adds binary ``feat_met_j`` per feature; relaxes
    ``amount ≥ target`` to ``amount ≥ target·feat_met_j``; requires
    ``problem.parameters["COSTBUDGET"]``.
  - ``"min_largest_shortfall"`` — minimax over per-feature shortfalls.
    Auxiliary ``slack_j ≥ 0`` and ``t ≥ slack_j`` for every feature;
    objective is ``min t``. Useful when one feature is at risk of being
    abandoned by other formulations; requires ``COSTBUDGET``.
  - ``"min_penalties"`` — hierarchical: minimise SPF-weighted shortfall
    first, cost second. Implemented as weighted scalarisation
    ``M · Σ SPF·slack + cost`` with ``M = sum_of_costs + 1`` so any
    non-zero penalty term dominates the worst-case cost.
  - Unknown objective names rejected at ``__init__`` time (fail-fast,
    matching the Phase 20/21 strategy/backend validation pattern).
  - ``Solution.metadata["objective"]`` records the formulation used.

- **Phase 22 — Importance scores (Ferrier, Jung rank, replacement-cost).**
  Three new ``pymarxan.analysis`` modules complement the existing
  ``compute_irreplaceability`` to give pymarxan prioritizr-parity on
  per-PU prioritisation metrics.
  - ``analysis.ferrier_importance.compute_ferrier_importance(problem)`` —
    closed-form SPF-weighted contribution-to-target. No MIP re-solves.
    Reference: Ferrier, Pressey & Barrett (2000) *Biological Conservation*
    93(3): 303-325. https://doi.org/10.1016/S0006-3207(99)00149-4
  - ``analysis.rank_importance.compute_rank_importance(problem, solution)``
    — Jung 2021 sequential-removal ranking. Repeatedly removes the
    least-damaging PU and records the order; later-removed PUs rank
    higher (more important). Reference: Jung et al. (2021) *Methods in
    Ecology and Evolution* 12(5): 869-877.
    https://doi.org/10.1111/2041-210X.13578
  - ``analysis.replacement_cost.compute_replacement_cost(problem)`` —
    for each PU in the MIP optimum, lock it out, re-solve, report the
    objective gap. ``n_selected + 1`` MIP solves total; pair with
    ``MIPSolver(mip_backend="highs")`` from Phase 21 on large problems.
    Returns ``+inf`` when lock-out is infeasible. Reference: Ferrier et
    al. (2000); Cabeza & Moilanen (2006) *Operations Research* 53(1):
    174-191. https://doi.org/10.1287/opre.1040.0167

- **Phase 21 — HiGHS / Gurobi MIP backends.** ``MIPSolver`` and
  ``ZoneMIPSolver`` now dispatch through a shared backend factory.
  - New ``mip_backend`` kwarg (default ``"auto"``). Values: ``"auto"``,
    ``"cbc"``, ``"highs"``, ``"gurobi"``. ``"auto"`` prefers HiGHS when
    available (5-50× faster than CBC on large MIPs), falls back to CBC.
  - CBC remains shipped with PuLP and is always available; HiGHS uses
    the system ``highs`` binary (or PyPI-distributed wheels); Gurobi
    requires the user to install ``gurobipy`` separately.
  - ``_available_backends()`` exposes ``{name: bool}`` for callers
    wanting to surface which backends are usable on the current machine.
  - Solution ``metadata["mip_backend"]`` records the resolved backend
    (``"cbc"`` / ``"highs"`` / ``"gurobi"``) so users can confirm which
    solver actually ran.
  - Unknown backend names rejected at ``__init__`` time, matching the
    fail-fast pattern of the Phase 20 strategy validators.
  - Phase 21 is "pure wiring" — no algorithmic changes; existing CBC
    behaviour is unchanged.

## [0.2.0] — 2026-05-16

The v0.2.0 milestone closes "full Marxan-classic parity" for the
single-zone solver suite. Builds on v0.2.0a1 (PROBMODE 3) and v0.2.0a2
(TARGET2 / CLUMPTYPE) by adding Phase 20 (SEPDISTANCE / SEPNUM).

### Added

- **Phase 20 — SEPDISTANCE / SEPNUM (separation distance).** Per-feature
  geographic-spread constraints, validated line-by-line against Marxan v4
  ``computation.hpp::computeSepPenalty`` and ``clumping.cpp::CountSeparation2``
  via three rounds of multi-agent design review.
  - Optional ``sepdistance`` (float ≥ 0, default 0) and ``sepnum`` (int ≥ 1,
    default 1) columns on ``spec.dat``. A feature is separation-active iff
    ``sepdistance > 0 AND sepnum > 1``. Writers omit both when all-default,
    so legacy projects round-trip byte-identical.
  - PU coordinates resolve via three-tier fallback: GeoDataFrame
    ``.geometry.centroid`` → ``pu.dat`` ``xloc``/``yloc`` columns →
    ``PUCoordinatesUnavailableError`` at ``ProblemCache.from_problem``
    (only when separation-active). NaN guard prevents silent
    under-counting from empty geometries.
  - New ``pymarxan.solvers.separation`` module exposes ``compute_sep_penalty``
    (Marxan hyperbolic curve ``1/(7·fval + 0.2) − 1/7.2`` with the count==0
    bump), ``count_separation`` (greedy admission in ascending PU-id order,
    capped at ``sepnum``), ``compute_sep_penalty_from_scratch``,
    ``evaluate_solution_separation``, ``get_pu_coordinates``, the mutable
    ``SepState`` companion to ``ProblemCache``, ``is_separation_active`` /
    ``raise_if_separation_active`` guard helpers, and the
    ``PUCoordinatesUnavailableError`` exception class.
  - ``MIPSolver`` and ``ZoneMIPSolver`` gain ``mip_sep_strategy`` kwarg
    (default ``"drop"``). ``"socp"`` rejected at ``__init__`` (separation
    is combinatorial, not conic); ``"big_m"`` raises
    ``NotImplementedError`` at solve time. All three MIP strategy kwargs
    (``mip_chance_strategy``, ``mip_clump_strategy``, ``mip_sep_strategy``)
    now route through a shared ``_validate_mip_strategy`` helper.
  - New ``Solver.supports_separation()`` capability method (default
    ``True``). Zone solvers (``ZoneSASolver``, ``ZoneIterativeImprovementSolver``,
    ``ZoneHeuristicSolver``, ``ZoneMIPSolver``) override to ``False`` and
    raise ``NotImplementedError`` on separation-active problems
    (previously would silently no-op). Per-zone SEPDISTANCE deferred to
    v0.3.
  - ``Solution`` gains ``sep_shortfalls: dict[int, int] | None`` and
    ``sep_penalty: float | None`` attributes (purely additive — all
    existing keyword-only construction patterns unchanged).
  - ``ProblemCache`` precomputes a ``pu_to_sep_feats`` inverse PU→feature
    index so ``SepState.delta_penalty`` is O(features-at-PU) rather than
    O(n_feat) per flip. The compound deterministic-penalty mask
    (``feat_target2 <= 0`` × ``feat_sepnum <= 1``) is centralised as a
    cached ``_det_spf`` field — single source of truth across
    ``compute_full_objective`` and ``compute_delta_objective``.
  - ``write_mvbest`` now emits ``Separation_Count`` / ``Separation_Met``
    columns when separation-active, plus ``Clump_Short`` (Phase 19
    backport) and ``Prob_Gap`` (Phase 18 backport) when those constraint
    paths are active.
  - Shiny UI: ``sepdistance`` / ``sepnum`` editable in ``feature_table``
    (split int-validator — ``sepnum >= 0``, distinct from ``clumptype ∈
    {0,1,2}``); ``target_met`` shows ``sep_short`` column when active;
    ``help_content`` documents the hyperbolic penalty curve with
    citations.
  - ``ScenarioSet`` ``_OVERRIDABLE_FIELDS`` extended to include
    ``sepdistance``, ``sepnum`` (plus ``target2``, ``clumptype``,
    ``ptarget`` backports — Phase 18 + 19 coverage gap).
  - References: Watts et al. (2009). *Environmental Modelling & Software*
    24(12): 1513–1521. https://doi.org/10.1016/j.envsoft.2009.06.005.
    Watts, Stewart & Martin (2017). *Learning landscape ecology*, 211–227.
    https://doi.org/10.1007/978-1-4939-6374-4_13

### Changed

- ``ProblemCache.from_problem`` emits ``UserWarning`` for two no-op
  separation configurations (``sepdistance > 0`` on a geographic CRS;
  ``sepnum > 1`` with ``sepdistance == 0``) so users editing in the
  Shiny grid actually see them (the previous ``validate()`` warnings
  only fired on Shiny upload).
- ``read_spec`` emits ``UserWarning`` for unrecognised columns in
  ``spec.dat`` — catches typos like ``sepnnum``, ``targt2``, ``ptraget``,
  ``clumptpe`` across the Phase 18 + 19 + 20 column whitelist in one
  shot.
- When any feature has a non-default ``sepdistance`` or ``sepnum``, the
  column is written for *all* features in ``spec.dat`` (same behaviour
  as Phase 18 ``ptarget`` and Phase 19 ``target2`` / ``clumptype``).

### Notes

- ``Solution`` and ``ProblemCache`` serialized state produced under
  v0.2.0 cannot be deserialized by v0.2.0a2 or earlier (one-way forward
  compatibility). Same shape as Phase 18 / 19 alpha bumps.
- Users running with ``python -W error`` or pytest ``filterwarnings =
  error::UserWarning`` should filter pymarxan warnings explicitly:
  ``warnings.filterwarnings("always", category=UserWarning,
  module="pymarxan")``.
- ``Solution.all_targets_met`` remains amount-only (consistent with
  Phases 18 + 19). Per-constraint completeness is exposed on
  ``Solution.prob_shortfalls`` / ``clump_shortfalls`` / ``sep_shortfalls``
  individually; the ``SolutionMetrics`` named-tuple refactor that
  collapses these is on the v0.3 backlog.

### Quality and infrastructure

- 1303 tests (+91 vs v0.2.0a2). Coverage stays ≥91 %.
- 0 lint, 0 mypy errors across 128 source files.
- Plan / review documents: three rounds of multi-agent design review
  caught two parity-critical Marxan-source bugs (hyperbolic penalty
  curve, PU-id greedy ordering), two performance blockers (per-flip
  O(n²) memory allocation, validate() warnings fire in the wrong place),
  one zone-solver silent-no-op risk, and the cross-phase typo-warning
  observability gap. See ``docs/plans/2026-05-16-phase20-{design,
  implementation,review,review-round2,review-round3}.md``.

## [0.2.0a2] — 2026-05-16

Second alpha of the v0.2.0 release line. Adds Phase 19 (TARGET2 /
CLUMPTYPE / clumping) on top of v0.2.0a1's Phase 18 (PROBMODE 3).
Phase 20 (separation distance) ships in a later alpha / final v0.2.0.

### Added

- **Phase 19 — TARGET2 / CLUMPTYPE / clumping ("type-4 species").** Marxan
  v4 minimum-patch-size constraints, validated line-by-line against
  ``clumping.cpp::PartialPen4`` and ``score_change.cpp::computeChangePenalty``
  via a multi-agent design review.
  - Optional ``target2`` column on ``spec.dat``: minimum amount per
    contiguous patch for the feature to count toward its target.
    ``target2 <= 0`` (default) disables clumping for the feature.
  - Optional ``clumptype`` column: 0 = binary (sub-target contributes 0),
    1 = half (``occ / 2``, "nicer step"), 2 = quadratic
    (``occ² / target2``, NOT linear despite the User Manual phrasing).
  - Writers omit both columns when at default, so legacy non-clumping
    projects round-trip byte-identical.
  - New ``pymarxan.solvers.clumping`` module exposes the pure-functional
    Marxan-faithful math (``partial_pen4``, ``compute_feature_components``
    via ``scipy.sparse.csgraph``, ``compute_baseline_penalty``,
    ``compute_clump_penalty_from_scratch``, ``evaluate_solution_clumping``)
    plus the mutable ``ClumpState`` companion to ``ProblemCache`` for
    the SA / iterative-improvement inner loops.
  - SA and iterative-improvement (removal + addition passes) honour
    clumping natively: ``ClumpState.delta_penalty`` supplies the
    type-4 penalty delta alongside the cache's deterministic delta
    (cache excludes type-4 features from its raw-amount penalty path).
  - ``MIPSolver`` gains ``mip_clump_strategy`` kwarg (default ``"drop"``):
    deterministic relaxation solved; chance-constraint-style clump
    shortfall reported post-hoc on ``Solution.clump_shortfalls`` /
    ``Solution.clump_penalty``. ``"big_m"`` raises ``NotImplementedError``
    pointing at a future phase.
  - ``ZoneMIPSolver`` gains an ``__init__`` for API symmetry; per-zone
    TARGET2 is explicitly out of scope.
  - The heuristic stays clumping-blind during scoring; ``build_solution``
    reports the gap post-hoc through the same Solution attrs.
  - New ``Solver.supports_clumping()`` capability method (default True).
  - Shiny UI: ``target2`` and ``clumptype`` columns editable in
    ``feature_table``; ``target_met`` table shows ``target2`` /
    ``clumptype`` / ``clump_short`` columns when active; help content
    documents the Marxan source-of-truth and cites Ball-Possingham-Watts
    (2009) and Metcalfe et al. (2015).
  - References: Ball, Possingham, & Watts (2009). *Spatial Conservation
    Prioritization*, Oxford University Press.
    https://doi.org/10.1093/oso/9780199547760.003.0014. Metcalfe et al.
    (2015). *Conservation Biology* 29(6): 1615–1625.
    https://doi.org/10.1111/cobi.12571

## [0.2.0a1] — 2026-05-16

First alpha of the v0.2.0 "full Marxan-classic parity" release line.
Ships Phase 18 (Z-score chance constraints) ahead of the full v0.2.0
so the citable feature is dated; Phases 19 + 20 ship in v0.2.0b1 and
v0.2.0 final.

### Added

- **PROBMODE 3 — Z-score chance constraints (Phase 18).** Marxan v4
  PROB2D + PTARGET2D support across all solvers, with the formulation
  validated against the reference C++ source (`probability.cpp::computeProbMeasures`)
  via a multi-agent design review.
  - Optional `prob` column on `puvspr.dat`: per-cell Bernoulli probability.
    Variance is derived internally as `amount² · p · (1-p)` (Marxan-faithful).
  - Optional `ptarget` column on `spec.dat`: per-feature probability target.
    Default `-1` matches Marxan's "disabled" sentinel.
  - Writers omit both columns when all values are at default, so legacy
    deterministic projects round-trip byte-identical.
  - SA, heuristic, and iterative-improvement solvers handle PROBMODE 3
    natively via the existing `ProblemCache` infrastructure (new
    `expected_matrix`, `var_matrix`, `feat_ptarget` fields).
  - `MIPSolver` gains a `mip_chance_strategy` kwarg (default `"drop"`):
    the deterministic relaxation is solved and the chance-constraint gap
    is reported post-hoc on `Solution.prob_shortfalls` /
    `Solution.prob_penalty`. Strategies `"piecewise"` (Phase 18.5) and
    `"socp"` (Phase 21) raise `NotImplementedError` with phase pointers.
  - New module `pymarxan.solvers.probability` exposes
    `compute_zscore_per_feature`, `compute_zscore_penalty`, and
    `evaluate_solution_chance` for direct use.
  - New `Solver.supports_probmode3()` capability method (default `True`).
  - Shiny UI: PROBMODE radio in `probability_config` extended with mode 3;
    `target_met` table shows `ptarget` / `P(met)` / `prob_gap` columns
    when active; `run_panel` shows a banner when MIP + PROBMODE 3 are
    combined explaining the drop-and-report-gap behaviour.
  - References: Game et al. 2008 (`10.1890/07-1027.1`), Tulloch et al.
    2013 (`10.1016/j.biocon.2013.01.003`), Carvalho et al. 2011
    (`10.1016/j.biocon.2011.04.024`).

## [0.1.0] — 2026-05-16

First public release. A modular Python toolkit for systematic conservation
planning that ships full algorithmic parity with classic Marxan, Marxan with
Zones, and Marxan Connectivity Edition, plus a Shiny-for-Python UI and a
spatial workflow built on `geopandas` / `rasterio`.

### Added

#### Core algorithm

- **Solver suite** with eight registered solvers, dispatched via a registry:
  - Native-Python simulated annealing (`SimulatedAnnealingSolver`) backed by
    a precomputed `ProblemCache` for O(degree + features_per_pu) delta
    computation per iteration.
  - MIP solver (`MIPSolver`) via PuLP + CBC, with configurable
    `MIP_TIME_LIMIT`, `MIP_GAP`, and verbose controls.
  - Heuristic greedy solver (`HeuristicSolver`) supporting all eight Marxan
    `HEURTYPE` scoring modes.
  - Iterative improvement solver (`IterativeImprovementSolver`) supporting
    `ITIMPTYPE` modes 0–3, including the two-step removal/addition pass
    that loops to convergence.
  - Run-mode pipeline (`RunModePipeline`) implementing `RUNMODE` 0–6.
  - C++ binary wrapper (`MarxanBinarySolver`) for users who want to invoke
    the reference C++ implementation.
- **Marxan with Zones**: `ZonalProblem`, `ZoneProblemCache`, `ZoneSASolver`,
  `ZoneHeuristicSolver`, `ZoneIterativeImprovementSolver`, `ZoneMIPSolver`,
  zone-aware objective and boundary computation, MISSLEVEL applied per
  (zone, feature), `COOLING` parameter support.
- **Connectivity edition**: graph-theoretic metrics (betweenness, eigenvector,
  bridge edges), edge-list and matrix I/O, four distance-decay kernels,
  connectivity terms in both SA and MIP objectives.
- **Parameter support**: BLM, SPF, MISSLEVEL, COSTTHRESH, THRESHPEN1/2,
  NUMITNS, NUMTEMP, STARTTEMP, COOLING (geometric / exponential / linear /
  Lundy-Mees with O(1) lookup), HEURTYPE, ITIMPTYPE, RUNMODE.
- **Output files** matching the reference Marxan format: `mvbest`, `ssoln`,
  `sum`, plus the per-run scenario file.

#### Models and I/O

- `ConservationProblem` dataclass with planning units, features, puvspr,
  boundary, optional zones / connectivity / probability fields.
- `load_project()` / `save_project()` round-trip for full Marxan project
  directories (`input.dat` + tabular files).
- Defensive defaults when readers find missing `status`, `spf`, or `name`
  columns; `prop` column auto-resolved to effective targets.
- `ConservationProblem.clone()`, `apply_feature_overrides()`, `copy_with()`
  for scenario branching without mutating the source.
- `ScenarioSet` with `clone_scenario()` and `run_with_overrides()` for
  bulk scenario comparison.

#### Calibration and analysis

- BLM, SPF, parameter-sweep (sequential and parallel), and sensitivity
  calibration — all guarded against infeasible solver returns.
- `compute_selection_frequency`, `compute_irreplaceability` (MISSLEVEL-aware,
  excludes locked-out PUs), `compute_gap_analysis`.

#### Spatial workflow (Phases 14–17)

- `pymarxan.spatial.grid`: planning-unit grid generation (square / hex /
  flat-top hex), polygon clipping, adjacency via spatial index.
- `pymarxan.spatial.gadm`: country-boundary fetching via the geoBoundaries
  API with name-filter support.
- `pymarxan.spatial.wdpa`: WDPA protected-area fetching with pagination
  warnings, plus `apply_wdpa_status` for status overlay (handles invalid
  geometries via `buffer(0)`).
- `pymarxan.spatial.importers`: shapefile / GeoJSON / GeoPackage planning-unit
  import and vector feature overlay. `.zip` archives auto-extract so Shiny
  uploads can deliver shapefile bundles in a single upload slot.
- `pymarxan.spatial.cost_surface`: raster zonal cost surfaces with multiple
  aggregation modes, plus vector overlay cost layers.

#### Shiny UI

- Eight-tab `pymarxan_app` covering Data, Spatial, Solver, Results, Maps,
  Calibration, Connectivity, and Scenarios.
- Twenty-two modular Shiny components under `pymarxan_shiny.modules/`:
  - Mapping: solution / spatial-grid / frequency / comparison / network
    views — all real `ipyleaflet` `@render_widget` maps with automatic
    EPSG:4326 reprojection.
  - Solver UX: run panel with thread-safe progress polling, convergence
    plot, scenario comparison, run controls.
  - Data input: project upload, directory browser, GIS import wizard,
    cost surface upload, GADM picker, WDPA overlay.
  - Calibration UI: BLM / SPF / sweep / sensitivity dashboards.
  - Connectivity: matrix CSV upload.
  - Results: summary table, target achievement, export (CSV / GeoPackage /
    Shapefile) with `session.on_ended` tempfile cleanup.

### Quality and infrastructure

- 1094 tests covering every solver, every Marxan parameter, the spatial
  subpackage, and Shiny module behaviour. 91.93 % statement coverage.
- `make check` (ruff + mypy + pytest) is green with zero errors across
  125 source files, including the Shiny layer.
- GitHub Actions CI for lint, types, and the full test suite on every
  push.
- `pdoc`-generated API documentation via `make docs`.
- Six full codebase reviews fixed 110 issues prior to release, plus two
  maintenance sweeps that cleared 191 baseline ruff errors and 34 mypy
  errors. See `docs/plans/2026-02-23-codebase-review{2..6}-*.md` for the
  audit history.

### Known issues

- One occasionally flaky stochastic test (`test_solutions_are_different`)
  passes on rerun.
- `shinywidgets` hooks `ipywidgets` globally; unit tests that construct
  `ipyleaflet.Map` outside a Shiny session need the
  `_allow_widget_outside_session` fixture.

[Unreleased]: https://github.com/razinkele/pymarxan/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/razinkele/pymarxan/releases/tag/v0.2.0
[0.2.0a2]: https://github.com/razinkele/pymarxan/releases/tag/v0.2.0a2
[0.2.0a1]: https://github.com/razinkele/pymarxan/releases/tag/v0.2.0a1
[0.1.0]: https://github.com/razinkele/pymarxan/releases/tag/v0.1.0
