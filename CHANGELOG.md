# Changelog

All notable changes to **pymarxan** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Vectorized `GridGeometry.build_boundary`.** The analytic rook-adjacency boundary is now
  built with O(n) numpy array ops (shifted-mask edges + exposed-side self-boundary) instead of
  a per-cell Python loop — so `include_boundary` scales to million-cell raster grids (identical
  output; the shapely-parity anchor is unchanged; ~9 ms for a 200×200 grid).

## [0.20.0] — 2026-07-15

### Changed

- **Sparse solver cache (S3a).** ``ProblemCache`` now stores the feature amounts as a
  ``scipy.sparse`` CSR (``ConservationProblem.build_pu_feature_csr``) instead of a dense
  ``(n_pu×n_feat)`` matrix, and builds the PROBMODE-3 ``expected``/``var`` matrices only when
  needed — cutting SA / iterative-improvement cache memory by ~10–40× on large sparse (raster)
  problems. ``cache.pu_feat_matrix`` is preserved as a lazily-densified property for clumping /
  separation / analysis; the delta is now O(nnz-per-PU). Solver results are unchanged on
  integer-amount problems (MIP still 35.0 on the reference problem; SA/greedy ≥ 35.0); the delta
  / ``compute_held`` differ only by float summation order (≤ a few ULP) on arbitrary-float
  problems. Scope: plain SA / iterative-improvement — clumping / separation / probmode-3 /
  analysis / zone paths still densify (future work).

## [0.19.0] — 2026-07-15

### Added

- **Windowed raster ingestion (`from_rasters(window_size=...)`, S3c).** Large rasters ingest
  in tiles without loading full ``(H×W)`` arrays: a two-pass windowed builder (bool validity
  mask → ``flat_valid`` index → sparse ``pu_vs_features`` via row-major ``searchsorted``),
  bit-identical to the full-array path. ``window_size`` (``int | "auto" | None``, default
  ``"auto"``) auto-switches on the estimated dense-stack size; on the windowed path
  ``include_boundary`` defaults off (the analytic ``build_boundary`` is a per-cell Python
  loop — a scale bottleneck to vectorize later) and an auto-skipped boundary warns. +13 tests.

## [0.18.0] — 2026-07-15

### Added

- **Raster-grid ingestion (`from_rasters` / `from_arrays`, S2).** ``spatial/raster.py``
  builds a ``ConservationProblem`` (carrying a S1 ``GridGeometry``) directly from aligned
  rasters — a pure-NumPy ``from_arrays`` core plus a thin rasterio ``from_rasters`` wrapper.
  One feature container (``dict[int, path | (path, band)]``) covers separate files and a
  multi-band stack; validity precedence mask → cost → feature-union; sparse
  ``pu_vs_features``; the analytic boundary is wired in by default. Rotated / non-north-up /
  misaligned rasters raise (reprojection deferred). +25 tests.

## [0.17.0] — 2026-07-15

### Added

- **Grid-geometry model for raster-grid planning units (`GridGeometry`, S1).**
  ``models/grid.py`` — a pure-NumPy grid descriptor (origin + cell size + validity
  mask + CRS) whose valid cells are planning units, with an analytic rook-adjacency
  ``build_boundary()`` (matches the shapely ``compute_boundary``, without materializing
  a polygon per cell) — carried as a ``kw_only`` ``grid`` field on
  ``ConservationProblem`` (validated for cell-count/PU-count agreement). First step
  toward raster-grid PUs; ingestion (S2) follows. +16 tests.

## [0.16.0] — 2026-07-15

### Added

- **Zonation Shiny tab + solver-picker entry (Phase D).** A "Zonation" tab shows
  the priority-rank choropleth + per-feature performance curves for the loaded
  problem (CAZ/ABF), and "Zonation (rank-removal)" is selectable in the solver
  picker — completing Zonation end-to-end across the stack (core → solver →
  smoothing → UI). +6 tests.

## [0.15.0] — 2026-07-14

### Added

- **Zonation distribution smoothing (`SmoothingSpec`, Phase C).** An optional
  ``smoothing=SmoothingSpec(alpha, coords=...)`` on ``rank_removal`` /
  ``ZonationSolver`` spreads each feature's amount to nearby planning units via a
  mass-conserving dispersal kernel before ranking (Zonation's distribution
  smoothing), reusing ``connectivity.smoothing``. +10 tests.

## [0.14.0] — 2026-07-14

### Added

- **`ZonationSolver` (Zonation Phase B).** A `Solver`-ABC adapter over the Phase A
  rank-removal engine: ``ZonationSolver(rule=..., top_fraction=0.3).solve(problem)``
  thresholds the priority ranking into one deterministic reserve (rank map +
  performance curves in ``Solution.metadata``), enforcing PU locks as hard
  constraints, registered as ``"zonation"`` in the default solver registry.
  +10 tests.

## [0.13.0] — 2026-07-14

### Added

- **Zonation rank-removal prioritization (`pymarxan.zonation`, Phase A).** A
  Zonation-style engine (Moilanen et al. 2005; Moilanen 2007) that ranks every
  planning unit by iterative backward removal — ``rank_removal(problem, rule=...)``
  with core-area (CAZ, ``max`` over features → favors rarity) and additive-benefit
  (ABF, ``sum`` over features → favors richness) rules, cost- and status-aware,
  with a warp factor. Returns a ``ZonationResult`` with a continuous 0-1 priority
  map and per-feature performance curves (by cell count and by cost) — the
  priority-rank paradigm Marxan's min-set cannot express. +14 tests (hand-computed
  CAZ order; CAZ-vs-ABF divergence).

## [0.12.0] — 2026-07-14

### Added

- **Phylogenetic-diversity objectives (`pymarxan.phylo`).** Faith (1992) PD via a
  branch-as-feature decomposition (Rodrigues & Gaston 2002): a pure-Python
  ``PhylogeneticTree`` (edge table or Newick, zero new deps),
  ``compute_phylogenetic_diversity`` scoring (rooted PD, total & representable
  fractions, unresolved-tip warning), and ``phylogenetic_branch_problem`` — each
  branch becomes a synthetic feature weighted by length so the existing solvers
  maximize PD (``min_set`` for the cheapest full-PD reserve; the new additive
  ``max_weighted_features`` MIP objective for max PD under a ``COSTBUDGET``).
  ``max_features`` is unchanged. +29 tests (Newick parse, hand-computed PD,
  MIP == brute-force).

## [0.11.0] — 2026-07-14

### Added

- **Omniscape omnidirectional connectivity (`pymarxan.connectivity.omniscape`).**
  ``omniscape(resistance, radius, source_strength=None)`` maps omnidirectional
  current density across a landscape with a moving window over the existing
  grounded grid-Laplacian solve (McRae 2016; Landau et al. 2021), returning
  ``cumulative_current``, the flat-resistance ``flow_potential`` null, and
  ``normalized_current`` (>1 marks channelling "pinch points"). +6 tests
  (uniform→normalized≈1 invariant; corridor pinch-point).
- **Climate-refugia scoring (`pymarxan.connectivity.refugia`).**
  ``refugia_score(velocity, connectivity=None, ...)`` composes climate stability
  (inverse velocity) and connectivity into a [0, 1] "resilient-and-connected"
  priority surface (Keppel et al. 2015; Anderson et al. 2023), weighted or
  geometric, treating flat-climate (infinite-velocity) cells as the worst.
  +8 tests.
- **Dynamic / multi-period reserve design (`pymarxan.temporal`).**
  ``dynamic_reserve_greedy`` — the Costello–Polasky (2004) informed-myopic
  schedule (protect by value × loss-risk over a per-period budget; beats the
  naive value-only baseline) — and ``two_stage_reserve_mip`` — the
  Snyder–Haight–ReVelle (2005) two-stage stochastic maximal-coverage MIP
  (act-now vs. recourse under future scenarios) on the shared CBC/HiGHS/Gurobi
  backend. +13 tests (informed > naive; MIP == brute-force; scenario-loss
  recourse). Climate-adaptive and temporal features from the 2026-06-20
  frontier-methods survey.

## [0.10.0] — 2026-06-20

### Added

- **Climate velocity (`pymarxan.connectivity.velocity`).** New
  ``spatial_gradient`` (Horn 1981 3×3 finite-difference gradient of a climate
  raster, as used by Burrows et al. 2014 / the VoCC package) and
  ``climate_velocity`` (local temporal climate trend ÷ spatial gradient — the
  speed a species must move to track its niche, e.g. km/yr). Accepts a scalar
  or per-cell temporal trend, returns a non-negative velocity-magnitude raster
  (``inf`` on flat climate, with an optional ``max_velocity`` cap). A
  climate-adaptation layer that feeds Marxan as a cost / feature / boundary
  (e.g. prioritise low-velocity refugia). Pure NumPy; georeferencing to
  planning units is handled by the existing spatial raster pipeline. First
  climate-adaptive feature from the 2026-06-20 frontier-methods survey. +12 tests.

## [0.9.0] — 2026-06-20

### Added

- **Project prioritization (`pymarxan.projects`).** A spatially-implicit
  complement to site selection (the "Project Prioritization Protocol", Joseph
  et al. 2009; exact form Hanson et al. 2019): fund a budget-constrained set of
  management **actions** to maximise weighted expected feature persistence,
  where **projects** (bundles of possibly-shared actions) secure features. New
  ``ProjectProblem`` / ``ProjectSolution`` model, an ``evaluate_projects``
  scorer, an exact ``prioritize_projects_mip`` (assignment linearisation on the
  shared CBC/HiGHS/Gurobi backend, ``optimal=True``), and a greedy
  cost-effectiveness ``prioritize_projects_greedy`` baseline. Shared actions
  make ranking suboptimal — the exact MIP provably beats greedy (guarded by a
  brute-force exactness test). First slice from the 2026-06-20 AI/frontier
  methods survey. +6 tests.

## [0.8.4] — 2026-06-20

### Added

- **Real-world Marxan format fixtures + end-to-end test.** Two synthetic
  projects under ``tests/data/formats/`` reproduce the file-format variations
  of real public datasets — ``double_tab/`` (columns padded with repeated tabs,
  boundary column named ``bound``; MarOpt / AdrienBrunel style) and
  ``quoted_csv/`` (quoted ``puvspr`` header with comma rows; MarxanConnect
  style) — without bundling any license-encumbered third-party data. Each is a
  deterministic 5×5-grid project generated by ``tests/data/formats/_generate.py``
  and exercised by ``tests/test_e2e_marxan_formats.py``, which loads every
  variant end-to-end and solves it with both simulated annealing and the exact
  MIP. Locks in the v0.8.3 loader-robustness fixes against regression on full
  projects. +4 tests.

## [0.8.3] — 2026-06-20

### Fixed

- **Load real-world Marxan projects robustly.** Validating against several
  public datasets (prioritizr's bundled example, a MarxanConnect Great Barrier
  Reef tutorial, and a published reserve-site-selection study) surfaced two
  format variations the reader mishandled: columns padded with repeated tabs or
  aligned spaces (`id\t\tcost`) became spurious empty `Unnamed` columns, and
  some exports name the boundary column `bound` rather than `boundary`. The
  reader now collapses whitespace runs for non-comma files and accepts either
  boundary-column name.
- **Multiple solver runs in one session.** The run panel's result-transfer
  effect stopped polling once the first run finished, so a *second* run in the
  same session (e.g. loading another project and re-running) never had its
  result transferred to the UI. It now polls robustly across runs.

## [0.8.2] — 2026-06-20

### Fixed

- **Maps no longer crash on a non-spatial (classic Marxan-format) project.**
  ``create_geo_map`` accessed ``gdf.crs`` on its input, raising
  ``AttributeError: 'DataFrame' object has no attribute 'crs'`` whenever a
  geometry-less project (the standard Marxan ``input.dat`` format) reached it.
  The mapping modules already gate on ``has_geometry()``, but
  ``grid_builder``'s map preview did not — so loading any classic project
  flooded the server log and broke the grid preview. ``create_geo_map`` now
  raises a clear ``ValueError`` when handed a plain ``DataFrame``, and
  ``grid_builder`` gates on ``has_geometry()`` with a synthetic-grid fallback
  like the other map modules. +1 test.

## [0.8.1] — 2026-06-20

### Fixed

- **Shiny plots now render reliably (server-side matplotlib).** Three panels
  built their charts with ``plotly`` via
  ``ui.HTML(fig.to_html(include_plotlyjs="cdn", ...))`` injected through
  ``render.ui`` — which left the chart **blank** ("Plotly is not defined": the
  inline ``Plotly.newPlot`` runs before/without the CDN script under Shiny's
  dynamic HTML injection). Converted the **rivers budget–DCI frontier**, the
  **SA convergence** plot, and the **target-sensitivity** chart to
  ``@render.plot`` with matplotlib (already a dependency; same pattern as the
  BLM explorer). Verified live via Playwright for the rivers panel.

## [0.8.0] — 2026-06-20

### Added

- **River barrier analysis + Shiny panel (`pymarxan.rivers`, Phase E).** New
  ``budget_dci_frontier`` (solve the barrier problem across a budget sweep →
  the DCI-gain-vs-budget efficiency frontier as a DataFrame; greedy / SA / exact
  MIP) and ``barrier_selection_frequency`` (run many SA solves and rank barriers
  by how often they appear in good portfolios — robust no-regret picks,
  deterministic per ``base_seed``). New Shiny ``rivers_panel`` module
  (`pymarxan_shiny.modules.rivers`) showing the budget–DCI frontier, a
  before/after DCI readout, and the barrier selection-frequency table. +8 tests.
  (Wiring the panel into the assembled app — a river-network upload flow — is
  deferred to Phase F.)
- **Rivers app wiring, tutorial & JOSS (`pymarxan.rivers`, Phase F).** The
  assembled Shiny app gains a **Rivers** tab (a demo-network loader + the
  ``rivers_panel``), so barrier-removal optimization is explorable end-to-end in
  the GUI. ``docs/TUTORIAL.md`` gains a "River connectivity and barrier
  restoration" section (pinned by ``tests/test_tutorial_examples.py``), and the
  JOSS paper's Summary / Statement of need are repositioned to cover the
  marine-site-selection **plus** riverine-barrier breadth, citing Côté et al.
  (2009) and O'Hanley (2011). +3 tests. Completes the rivers feature (Phases A–F).

## [0.7.0] — 2026-06-20

### Added

- **River connectivity — DCI metrics (`pymarxan.rivers`, Phase A).** New
  ``RiverNetwork`` model (rooted downstream-pointer tree of segments + barriers,
  validated for a single outlet / acyclicity / passability range, with cached
  pure-NumPy topology — **no networkx dependency**) and the Dendritic
  Connectivity Index (Côté et al. 2009): ``dci_diadromous`` (sea↔segment,
  ``direction="single_pass"`` default; ``round_trip`` deferred),
  ``dci_potamodromous`` (all within-network pairs), and ``segment_connectivity``
  (per-segment ``c_i`` or potamodromous marginals). Pairwise ``c_ij`` is a
  direct path-product, avoiding the ``0/0→NaN`` of the closed-form LCA division
  when a sub-confluence barrier is impassable. First slice of the river
  barrier-restoration feature; barrier-decision optimizers (greedy/SA/MIP) land
  in later phases. +27 tests (hand-computed chain & Y-tree confluence fixtures,
  incl. the p=0 NaN-gate).
- **River barrier optimization — greedy + SA (`pymarxan.rivers`, Phase B).** New
  ``BarrierProblem`` (budget-constrained DCI maximisation; barrier ``status``
  reuses the Marxan locked-in/locked-out convention; only the wired ``max_dci``
  objective is exposed) and ``BarrierSolution`` (``removed`` set, cost,
  ``dci_before``/``dci_after``/``gain`` with the baseline excluding locked-in,
  ``optimal``). Two optimizers: ``optimize_barriers_greedy`` (best
  DCI-gain-per-cost until the budget is spent) and ``optimize_barriers_sa``
  (simulated annealing over the free barriers, reusing
  ``solvers.cooling.CoolingSchedule``, budget by hard rejection, deterministic
  per ``seed``). Both honour locks and never beat the exact optimum. +9 tests
  (brute-force oracle, gating-barrier pick, budget/lock guards, SA determinism).
- **River barrier optimization — exact MIP (`pymarxan.rivers`, Phase C).**
  ``optimize_barriers_mip`` solves the binary-passability **diadromous** case
  exactly: a segment reaches the sea iff every blocking barrier on its path is
  removed, linearised as ``c_i = ∏ y_b`` (``c_i ≤ y_b``; ``c_i ≥ Σy_b −
  (|path|−1)``; continuous ``c_i``), maximising ``Σ w_i c_i`` under the budget
  and locks (O'Hanley 2011 / Kuby 2005 lineage). Returns ``optimal=True``;
  refuses partial passability (``p ∉ {0,1}`` → use SA) and the potamodromous
  form. The PuLP backend factory (``_available_backends`` / ``_make_pulp_solver``)
  was extracted from ``mip_solver.py`` to ``solvers/_backends.py`` so the core,
  zone, and river MIP solvers share one importable home (no behaviour change).
  +8 tests, incl. MIP == brute-force on random binary trees.
- **River-network ingest (`pymarxan.rivers`, Phase D).** ``from_hydrorivers``
  builds a ``RiverNetwork`` directly from a HydroRIVERS / NHDPlus-style
  GeoDataFrame via the downstream-pointer field (configurable ``id_col`` /
  ``next_down`` / ``length``; ``NEXT_DOWN`` 0/NA → outlet), retaining segment
  geometry. ``snap_barriers`` assigns barrier points to their nearest segment
  (``geopandas.sjoin_nearest``) within an optional ``tolerance`` — dropping
  too-far barriers with a warning, reprojecting on CRS mismatch, and carrying
  over ``pass_up``/``pass_down``/``removal_cost``/``status``. +10 tests.

## [0.6.0] — 2026-06-20

### Fixed

- **PROBMODE 3 (Z-score chance constraints) brought to true Marxan PROB2D
  parity.** An audit against the authoritative Marxan C++ source
  (`probability.cpp`, `computation.hpp::computeProbMeasures`) found two
  divergences in the Phase 18 implementation:
  - **Expected-amount polarity.** The expected reserve amount was computed as
    `Σ amount·(1 − p)` (Marxan's *1D* loss convention) instead of
    `Σ amount·p` (Marxan's *2D* presence convention,
    `ComputeP_AllPUsSelected_2D`). The per-(PU, feature) `prob` column is now
    interpreted as the **probability of presence**, matching Marxan PROB2D.
  - **SPF weighting.** The probability shortfall was multiplied by each
    feature's SPF; Marxan's `ComputeProbability2D` scales it by
    `PROBABILITYWEIGHTING` only (SPF weights the separate representation
    penalty). The SPF factor has been removed from the probability penalty.

  The Z-score, upper-tail probability (`norm.sf` ≡ Marxan `probZUT`),
  ptarget normalisation, and zero-variance sentinel were already correct and
  are unchanged.

### Changed

- **BREAKING (PROBMODE 3 only): the `puvspr` `prob` column now means
  probability of presence, not loss.** Existing PROBMODE 3 inputs that
  supplied loss probabilities must be complemented (`p → 1 − p`) to keep the
  same behaviour. A missing `prob` now defaults to `1.0` (certain presence ⇒
  deterministic). PROBMODE 1 and 2 (per-PU `prob`, Marxan 1D loss convention)
  are unaffected. Deterministic problems (no `prob` column) are unaffected.

## [0.5.1] — 2026-06-20

### Changed

- **Release-script toolchain pre-flight (``scripts/release.sh``).** The
  release now verifies its full toolchain up front, before any
  irreversible step: ``ruff``, ``mypy`` and ``pytest`` must resolve on
  ``PATH`` and the ``build`` module must be importable by the build
  interpreter (a new, overridable ``PYTHON`` variable, default
  ``python3``). Previously a missing ``build`` failed the wheel/sdist
  step *after* the version-bump commit had already landed, leaving a
  partial release; the check now aborts cleanly with nothing committed.
  ``--dry-run`` only warns, mirroring the existing twine pre-flight, so
  the script stays runnable in CI without the full toolchain. +4 tests.

## [0.5.0] — 2026-06-20

### Added

- **Circuit-theory connectivity (``pymarxan.connectivity.circuit``).** New
  ``current_flow_to_matrix(raster, coords)`` computes the pairwise
  effective-resistance (resistance-distance) connectivity between planning
  units from a habitat-resistance raster, modelling the landscape as a
  resistor network and integrating all paths (the Circuitscape measure)
  rather than only the cheapest one like the existing least-cost-path
  matrix. Solved with a single reused ``scipy.sparse`` LU factorisation of
  the grounded graph Laplacian. First Tier-B feature from the 2026-06-12
  ecosystem survey; McRae et al. (2008). +10 tests.
- **Automatic target-setting rules (``pymarxan.targets``).** ``relative_targets``
  (fraction of each feature's total), ``loglinear_targets`` (IUCN-style
  range-size, log-linear interpolation between two area thresholds),
  ``group_targets`` (per-group relative target), and ``apply_targets`` to
  write a ``{feature_id: target}`` mapping onto a problem. Tier-A feature
  from the 2026-06-12 ecosystem survey (prioritizr ``add_auto_targets``
  parity). +10 tests.
- **Distribution smoothing (``pymarxan.connectivity.smoothing``).**
  ``smooth_distribution`` spreads a feature's per-unit amounts to nearby
  units via a negative-exponential dispersal kernel (column-normalised and
  mass-conserving by default), plus ``distance_matrix_from_points``. The
  planning-unit analogue of Zonation's distribution smoothing. +5 tests.
- **Multi-scenario robustness (``pymarxan.analysis.robustness``).**
  ``minimax_regret`` selects the most robust plan from a plans-by-scenarios
  cost matrix (minimax regret and minimax worst-case cost), and
  ``evaluate_plans_across_scenarios`` builds that matrix by re-evaluating
  each plan's objective under every scenario problem. Complements the
  existing no-regrets overlap (``compute_selection_frequency``). +9 tests.
- **Shiny "Representation (30x30)" panel.** New results module exposing
  ``compute_representation`` with a threshold slider and per-feature table.
- **Area-based representation reporting (``pymarxan.analysis.representation``).**
  New ``compute_representation(problem, solution, threshold=0.30)`` reports,
  per feature, the total amount, the amount represented by a solution, the
  percentage represented, and whether it clears a uniform policy threshold
  (default 30% for the Kunming-Montreal "30x30" / GBF Target 3), plus a
  summary of how many features are met. Distinct from gap analysis (which
  scores ``status==2`` protection against each feature's optimisation
  target). Tier-A feature from the 2026-06-12 ecosystem survey. +9 tests.
- **Distributional-equity analysis (``pymarxan.analysis.equity``).** New
  ``compute_equity(problem, solution, groups, value=...)`` reports how a
  reserve's value (cost burden, unit count, or a custom per-PU benefit)
  distributes across social/spatial groups, returning per-group totals,
  shares, and the Gini coefficient of the group totals (0 = perfectly
  even). First of the Tier-A "modern conservation planning" features from
  the 2026-06-12 ecosystem survey; motivated by Gopalakrishna et al.
  (2024), PNAS. +9 tests.
- **JOSS paper draft (``paper/paper.md`` + ``paper/paper.bib``).** A Journal
  of Open Source Software submission presenting pymarxan, positioned
  against dormant Marxan (C++) and R-only prioritizr. All six references
  verified against real literature; author ORCID left as a flagged TODO
  for the maintainer to fill before submission.
- **Validation harness + ``docs/VALIDATION.md``.**
  ``examples/validate_marxan_parity.py`` loads the bundled six-unit project
  in native Marxan format, solves it with exact MIP + simulated annealing +
  greedy, and checks the heuristics meet every target without beating the
  hand-verified exact optimum (cost 35.0, reserve ``{2, 4, 6}``), then
  confirms a Marxan-format round-trip. ``docs/VALIDATION.md`` documents the
  methodology and scopes the C++-binary comparison as opt-in via
  ``MarxanBinarySolver``. +2 tests.
- **PyPI publishing in ``scripts/release.sh``.** The release script now
  publishes to PyPI as its final step (after the GitHub Release), via
  ``twine``. Opt out with ``--no-pypi`` (GitHub-only, the previous
  behaviour) or retarget to TestPyPI with ``--test-pypi``. Credentials
  (a ``TWINE_PASSWORD`` API token or ``~/.pypirc``) and the ``twine``
  tool are verified in **pre-flight**, so a missing token fails *before*
  any irreversible commit/tag/push; artifacts are validated with
  ``twine check`` before the push too. Five new ``--dry-run`` tests
  cover the publish steps and the new flags.
- **Worked example: south-eastern Baltic marine spatial planning**
  (``examples/baltic_marine_planning.py`` + ``examples/README.md``).
  A self-contained, deterministic end-to-end scenario (Curonian Lagoon /
  Klaipėda shelf) — grid generation, a cost surface, four conservation
  features, locked-in existing reserves, an exact (MIP) vs. heuristic
  (SA) reserve comparison, before/after gap analysis, and
  irreplaceability ranking. Runs with no network access and is pinned by
  ``tests/test_examples.py`` so it can't rot. ``make lint`` now also
  covers ``examples/``.
- **Release automation script (``scripts/release.sh``).** Captures the
  manual workflow run for v0.2.0 / v0.3.0 / v0.4.0 / v0.4.1:
  pre-flight safety checks → ``make check`` → bump pyproject.toml →
  promote CHANGELOG ``[Unreleased]`` to ``[VERSION] — DATE`` → commit
  → build wheel + sdist → tag → push → ``gh release`` with the
  just-promoted CHANGELOG section as the release notes (no separate
  copy to maintain). Supports ``--dry-run`` for verifying every step
  before any side effect lands. Pre-flight refuses: dirty tree,
  non-main branch, out-of-sync origin, existing tag, empty
  ``[Unreleased]`` section. Smoke-tested via
  ``tests/test_release_script.py`` (14 tests, ~3s with shared
  dry-run fixture).

- **End-to-end Python API tutorial.** New ``docs/TUTORIAL.md`` walks
  through the v0.3 / v0.4 surface in a single coherent workflow:
  build a problem, solve with the default min-set objective, choose
  the MIP backend, compute importance scores, try alternative
  objectives, work with connectivity metrics, generate a no-good-cut
  portfolio, and view a Pareto-filtered BLM sweep. Every code block
  has a parallel runnable test in
  ``tests/test_tutorial_examples.py`` so the doc can't silently rot
  when the API changes — if a snippet breaks, that test fails and
  forces a same-commit doc fix.

- **Phase 18 + 19 benchmark coverage.** Two new
  ``tests/benchmarks/`` files mirror the Phase 20 ``bench_sep.py``
  perf gate:
  - ``bench_prob.py`` — per-flip ``compute_delta_objective`` under
    PROBMODE 3 must run median < 2.5 ms on a 500-PU × 10-feature
    problem. Catches accidental O(n_pu²) scans in the
    ``_compute_zscore_penalty`` path.
  - ``bench_clump.py`` — per-flip ``ClumpState.delta_penalty``
    median < 5 ms on a 400-PU × 5-feature grid problem with
    TARGET2 active. Catches regressions in the affected-features
    short-circuit and the per-feature ``connected_components``
    recompute.
  - Budgets sized at ~2.5× the measured median on the development
    machine so genuine regressions trip the gate without normal
    cross-machine variance breaking CI. Run via ``make bench``.

### Changed

- **CI / Makefile cleanup.** Perf-budget benchmarks
  (``tests/benchmarks/bench_*.py``) are now skipped by default in
  ``make test`` and CI to avoid spurious failures on slower CI
  runners. Run them manually via the new ``make bench`` target. All
  11 bench tests (Phase 18 / 19 / 20 inner-loop costs + the
  pre-existing SA / Zone SA full-run gates) carry the new
  ``pytest.mark.bench`` marker. CI also extends ``mypy`` to cover
  ``src/pymarxan_shiny/`` so the Shiny layer can't drift
  type-unsafe between releases (the Makefile already did this; CI
  is now in sync).

## [0.4.1] — 2026-05-17

Pure UI patch — closes the last item deferred from the Phase 24 plan.

### Added

- **Shiny connectivity metrics dashboard.** New
  ``pymarxan_shiny.modules.connectivity.metrics_dashboard`` surfaces all
  seven Phase 24 connectivity metrics (in/out-degree, betweenness,
  eigenvector centrality, PageRank, donor/recipient flags) in a single
  sortable table — one row per PU. Was originally listed in the
  Phase 24 plan but deferred from the v0.4.0 release because it's a
  pure UI addition; lands cleanly in a patch.

## [0.4.0] — 2026-05-17

The v0.4.0 milestone closes "Connectivity + portfolios". Combines Phase
24 (MarxanConnect feature gap — PageRank, donors, recipients, temporal
connectivity, habitat-resistance LCP, post-hoc clustering) and Phase 25
(solver-agnostic no-good-cut portfolios + Cohon Pareto filter on BLM
sweeps), plus a round of polish items deferred from the Phase 20
multi-agent review process.

### Added

- **Phase 25 — Solution portfolios + Cohon Pareto filter.** Two
  pieces, both solver-agnostic (work on any PuLP backend).
  - ``analysis.portfolio_cuts.generate_portfolio_cuts(problem, *,
    solver=None, k, config=None)`` — generates up to ``k`` distinct
    high-quality MIP solutions by iteratively solving and adding
    no-good cuts of the form ``Σ_{i: s_i=1} (1 - x_i) + Σ_{i: s_i=0}
    x_i ≥ 1`` (at least one variable must flip from the previous
    solution). Returns the partial list when fewer than ``k``
    distinct feasible solutions exist. Each solution's metadata
    records its ``"portfolio_iteration"``.
  - ``calibration.pareto.pareto_frontier(BLMResult)`` — drops
    dominated points from a BLM calibration sweep so users see only
    the Pareto-optimal cost–boundary trade-offs. Named after Cohon
    (1978) *Multiobjective Programming and Planning*.
  - ``MIPSolver`` learns the ``forbidden_selections`` key in
    ``config.metadata`` so external code can add no-good cuts without
    rebuilding the model — the mechanism powering ``portfolio_cuts``.
  - Gurobi-dependent strategies (``top_k``, ``gap``, ``extra``)
    deferred — they need solution-pool features absent in CBC/HiGHS.

- **Phase 24 — Connectivity metric expansion + post-hoc clustering.**
  Closes the MarxanConnect feature gap.
  - ``connectivity.metrics.compute_pagerank_centrality`` — networkx
    PageRank with configurable damping factor; sum normalised to 1.
  - ``connectivity.metrics.compute_donors`` /
    ``compute_recipients`` — boolean masks for nodes whose
    out/in-degree exceeds the opposite by ``threshold`` (thresholded
    in/out-degree variants).
  - ``connectivity.io.connectivity_to_boundary`` — convert a
    connectivity edge list to a ``bound.dat``-compatible DataFrame
    with optional ``scale`` factor. Lets users feed connectivity into
    the BLM penalty without a manual rename. Zero-value rows dropped.
  - ``connectivity.temporal.compute_temporal_connectivity`` — reduce
    a ``(T, n, n)`` stack of per-timestep connectivity matrices to an
    ``(n, n)`` summary via ``"mean"`` / ``"max"`` / ``"weighted"``.
  - ``connectivity.resistance.habitat_resistance_to_matrix`` —
    least-cost-path connectivity from a 2D habitat-resistance raster.
    Uses networkx Dijkstra on a 4-neighbour grid graph (avoids the
    scikit-image dependency that ``route_through_array`` would
    introduce).
  - ``analysis.posthoc_clusters.compute_solution_clusters`` —
    partitions a Solution's selected PUs into connected components
    using the problem's boundary graph; returns ``n_clusters``,
    sorted ``cluster_sizes``, ``max_cluster_fraction``, and a
    ``pu_to_cluster`` dict for map colour-coding.

### Changed

- **Phase 20 round-3 backlog cleanup.** Three deferred items from the
  Phase 20 review process now land:
  - **CR3 (Shiny run_panel crash traceback visibility):** when the
    solver thread raises, ``progress.error`` now carries the full
    ``traceback.format_exc()`` instead of a bare ``str(e)``, and
    ``progress.message`` is updated so the progress card visually
    leaves the running state. Users see file + line of the failure
    instead of a stalled progress bar.
  - **CR4 (warning visibility for non-Shiny users):** README now
    documents that pymarxan emits ``UserWarning`` from
    ``ProblemCache.from_problem`` and ``read_spec``, and that strict-mode
    users (``python -W error`` or pytest ``filterwarnings =
    error::UserWarning``) should filter via
    ``warnings.filterwarnings("always", category=UserWarning,
    module="pymarxan")``. All four ``warnings.warn`` call sites already
    use ``stacklevel=2`` so source locations point to user code, not
    pymarxan internals.
  - **bench_sep.py (perf gate):** new ``tests/benchmarks/bench_sep.py``
    pins the round-2 CR1 ``SepState`` memory-shape claim with an actual
    measurement — per-flip ``delta_penalty`` median under 500 µs on a
    500-PU × 5-sep-features problem; ``SepState.from_selection`` under
    50 ms. Run via ``make bench``, not ``make check``.

## [0.3.0] — 2026-05-17

The v0.3.0 milestone closes "prioritizr-parity" for pymarxan's MIP path.
Combines Phase 21 (HiGHS / Gurobi backends), Phase 22 (Ferrier + Jung
rank + replacement-cost importance scores), and Phase 23 (extended MIP
objectives) into a single release.

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

[Unreleased]: https://github.com/razinkele/pymarxan/compare/v0.20.0...HEAD
[0.20.0]: https://github.com/razinkele/pymarxan/releases/tag/v0.20.0
[0.19.0]: https://github.com/razinkele/pymarxan/releases/tag/v0.19.0
[0.18.0]: https://github.com/razinkele/pymarxan/releases/tag/v0.18.0
[0.17.0]: https://github.com/razinkele/pymarxan/releases/tag/v0.17.0
[0.16.0]: https://github.com/razinkele/pymarxan/releases/tag/v0.16.0
[0.15.0]: https://github.com/razinkele/pymarxan/releases/tag/v0.15.0
[0.14.0]: https://github.com/razinkele/pymarxan/releases/tag/v0.14.0
[0.13.0]: https://github.com/razinkele/pymarxan/releases/tag/v0.13.0
[0.12.0]: https://github.com/razinkele/pymarxan/releases/tag/v0.12.0
[0.11.0]: https://github.com/razinkele/pymarxan/releases/tag/v0.11.0
[0.10.0]: https://github.com/razinkele/pymarxan/releases/tag/v0.10.0
[0.9.0]: https://github.com/razinkele/pymarxan/releases/tag/v0.9.0
[0.8.4]: https://github.com/razinkele/pymarxan/releases/tag/v0.8.4
[0.8.3]: https://github.com/razinkele/pymarxan/releases/tag/v0.8.3
[0.8.2]: https://github.com/razinkele/pymarxan/releases/tag/v0.8.2
[0.8.1]: https://github.com/razinkele/pymarxan/releases/tag/v0.8.1
[0.8.0]: https://github.com/razinkele/pymarxan/releases/tag/v0.8.0
[0.7.0]: https://github.com/razinkele/pymarxan/releases/tag/v0.7.0
[0.6.0]: https://github.com/razinkele/pymarxan/releases/tag/v0.6.0
[0.5.1]: https://github.com/razinkele/pymarxan/releases/tag/v0.5.1
[0.5.0]: https://github.com/razinkele/pymarxan/releases/tag/v0.5.0
[0.4.1]: https://github.com/razinkele/pymarxan/releases/tag/v0.4.1
[0.4.0]: https://github.com/razinkele/pymarxan/releases/tag/v0.4.0
[0.3.0]: https://github.com/razinkele/pymarxan/releases/tag/v0.3.0
[0.2.0]: https://github.com/razinkele/pymarxan/releases/tag/v0.2.0
[0.2.0a2]: https://github.com/razinkele/pymarxan/releases/tag/v0.2.0a2
[0.2.0a1]: https://github.com/razinkele/pymarxan/releases/tag/v0.2.0a1
[0.1.0]: https://github.com/razinkele/pymarxan/releases/tag/v0.1.0
