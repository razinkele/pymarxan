# Changelog

All notable changes to **pymarxan** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Target: v0.2.0 — "full Marxan-classic parity". Remaining work: Phase 20
(separation distance / SEPDISTANCE / SEPNUM).

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

[Unreleased]: https://github.com/razinkele/pymarxan/compare/v0.2.0a2...HEAD
[0.2.0a2]: https://github.com/razinkele/pymarxan/releases/tag/v0.2.0a2
[0.2.0a1]: https://github.com/razinkele/pymarxan/releases/tag/v0.2.0a1
[0.1.0]: https://github.com/razinkele/pymarxan/releases/tag/v0.1.0
