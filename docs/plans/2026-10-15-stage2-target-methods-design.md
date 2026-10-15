# Stage 2 (v0.40.0): published target-setting methods — design

**Date:** 2026-10-15 · **Baseline:** v0.39.0 (`aff3465`) + `5c57fcb` · **Roadmap:**
`docs/plans/2026-10-11-prioritizr-parity-roadmap.md` Stage 2 · **Reference:** prioritizr 9.0.1
source (`R/spec_*_targets.R`, `R/internal_*`), read 2026-10-14.

## 1. Intent

Users of real projects (Baltic / ROWER) can set feature targets with the methods that
conservation reports cite, not only relative or log-linear rules. The methods follow
prioritizr 9's **code** (not its docs, where the two disagree), and fix prioritizr's bugs.

**User decisions (brainstorm, 2026-10-14/15):**

| Question | Decision |
|---|---|
| Purpose | Real projects (Baltic / ROWER) |
| Scope | All four families: interpolation, IUCN Red List, simple + composition, population. Follow prioritizr's code, fix its bugs. |
| Units | Per-call `km2_per_unit=` argument (float or `{fid: factor}`), required, no default; no data-model change |
| `loglinear_targets` | Deprecate, keep exact v0.39 behaviour, `DeprecationWarning` pointing to `interp_absolute_targets` |
| API shape | Plain functions returning `{fid: target}` dicts (approach A); not spec objects, not a dispatcher |

**Success criteria (roadmap acceptance):** each method has a worked example pinned as a test;
unit handling is explicit; the CHANGELOG cites each source; no new data models.

## 2. Package layout

`src/pymarxan/targets.py` becomes the package `src/pymarxan/targets/`. `__init__.py`
re-exports every public name flat (`from pymarxan.targets import rodrigues_targets`) and keeps
the module docstring. Existing imports keep working.

| Module | Public contents |
|---|---|
| `_basic.py` | `relative_targets`, `group_targets`, `apply_targets` (moved unchanged); `loglinear_targets` (deprecated) |
| `_units.py` | private: km² totals, `km2_per_unit` resolution, per-feature dict validation, cap/clip |
| `_interp.py` | `interp_absolute_targets`, `interp_area_targets`, `rodrigues_targets`, `sreekar_targets`, `polak_targets`, `watson_targets` |
| `_redlist.py` | `rl_species_targets`, `rl_ecosystem_targets`, `jung_targets`, `ward_targets` |
| `_population.py` | `pop_size_targets`, `wilson_targets`, `duran_targets` |
| `_compose.py` | `absolute_targets`, `area_targets`, `rule_targets`, `min_targets`, `max_targets` |

No top-level `pymarxan` export, no Shiny wiring (see §9).

## 3. Shared contract

Notation: for feature `j`, `F_j` = native total = `problem.feature_amounts()[j]` (sum over all
PUs, including locked-out ones, as prioritizr does); `k_j` = km² per native unit; `f_j = F_j·k_j`
(km²); `t_j` = target in km²; the function returns `t_j / k_j` (native units).

1. **Units.** Every area-based method takes keyword-only `km2_per_unit: float | Mapping[int, float]`
   with no default. Each factor must be finite and > 0. A dict must contain every feature being
   computed (extra keys allowed); a missing one raises `ValueError` naming the id(s).
2. **Cap.** `cap_km2: float | None = 1_000_000.0` (prioritizr default; Butchart et al. 2015).
   `None` disables. Applied as `min(t, cap)` (prioritizr v9, not v8's flat replacement).
   Exception: `duran_targets` defaults to `cap_km2=None` (prioritizr applies no cap there).
   A non-`None` cap must be finite and ≥ 0.
3. **Clip.** Every area-based result is `min(t, f)`: never infeasible by construction.
4. **Zero totals.** `F_j == 0` → target `0.0` (prioritizr returns NaN).
5. **`features=`.** `Iterable[int] | None`. `None` = every feature in `problem.features["id"]`.
   An id not in the problem raises `ValueError`. The output contains exactly the selected ids,
   in `problem.features` order. With `apply_targets` keeping unlisted features' targets, this is
   how methods are mixed across feature groups.
6. **Per-feature inputs** (`status`, `pop_size`, `density_per_km2`, …) accept a scalar (applied
   to every selected feature) or a `Mapping[int, ...]`. Mappings may have extra keys; a missing
   key for a selected feature, or a non-finite / out-of-range value, raises `ValueError`
   naming the feature id(s). Values are looked up **by feature id**, never by position.
7. **Unit-free exceptions.** `relative_targets` and `group_targets` (proportions) and
   `absolute_targets`, `interp_absolute_targets` (native units) take no `km2_per_unit`.
8. **Errors.** All validation failures raise `ValueError` with the offending argument and/or
   feature ids. No warn-and-continue, except `loglinear_targets`' `DeprecationWarning`.
9. **Return type.** `dict[int, float]`, plain Python floats.

## 4. Interpolation family (`_interp.py`)

### 4.1 `interp_area_targets`

```python
interp_area_targets(problem, *, km2_per_unit, rare_area_threshold, rare_relative_target,
                    common_area_threshold, common_relative_target, interp,
                    rare_area_target=None, rare_method=None,
                    common_area_target=None, common_method=None,
                    cap_km2=1_000_000.0, features=None) -> dict[int, float]
```

`a` = rare threshold, `b` = rare proportion, `c` = rare area target, `e` = common threshold,
`g` = common proportion, `h` = common area target (all km²). `interp ∈ {"log10", "linear"}`;
`rare_method`/`common_method ∈ {"min", "max"}`.

- `f < a` (or `f == a` when `a == e`, see below): `t = b·f`; if `c` given, `t = method(c, b·f)`.
- `f > e`: `t = g·f`; if `h` given, `t = method(h, g·f)`.
- `a ≤ f ≤ e` (with `a < e`): `t = f · (b + (g − b)·(φ(f) − φ(a)) / (φ(e) − φ(a)))`,
  `φ = log10` or identity.
- then `t = min(t, cap, f)`.

Required (no default): thresholds, proportions, `interp` — no hidden default rule.
Validation: `0 ≤ a ≤ e`, finite; `interp == "log10"` requires `a > 0`; `b, g ∈ [0, 1]`;
`c, h ≥ 0` finite; a target and its method must be given together (one without the other →
`ValueError`); unknown `interp`/method strings → `ValueError`.

**Equal thresholds (prioritizr crash fix).** When `a == e`, `f == a` takes the rare branch.
prioritizr routes it into interpolation with two identical points and errors. Everywhere
else the branch boundaries match prioritizr (`f < a` rare, `f > e` common, inclusive middle).

### 4.2 `interp_absolute_targets`

Same rule and argument names with `absolute` for `area` (`rare_absolute_threshold`,
`rare_absolute_target`, …), native units, no `km2_per_unit`; `cap: float | None = None`.
Equivalence pinned by test: `loglinear_targets(lower_area=L, lower_target=p, upper_area=U,
upper_target=q)` == `interp_absolute_targets(rare_absolute_threshold=L, rare_relative_target=p,
common_absolute_threshold=U, common_relative_target=q, interp="log10")` for every feature
(including `F ≤ L`, `F ≥ U`, `F == 0`). Note `loglinear_targets` requires `0 < L < U` and
`interp_absolute_targets` also accepts `L == U` and (linear) `L == 0`; the equivalence holds
on `loglinear_targets`' domain.

### 4.3 Presets (thin wrappers; every default overridable by keyword)

| Function | Fixed | Defaults | Pinned examples (km², `k = 1`) |
|---|---|---|---|
| `rodrigues_targets` | log10, no area targets | a=1,000, b=1, e=250,000, g=0.1, cap 1e6 | f=10,000 → 6,246.778; f=100,000 → 24,935.570; f=2e7 → 1e6 |
| `sreekar_targets` | log10, no area targets | a=200, b=1, e=250,000, g=0.3, cap 1e6 | f=1,000 → 842.011; f=100,000 → 38,994.708; f=5e6 → 1e6 |
| `polak_targets` | linear, no area targets | a=1,000, b=1, e=10,000, g=0.1, cap 1e6 | f=5,500 → 3,025; f=9,000 → 1,800; f=10,000 → 1,000 |
| `watson_targets` | e := a, rare_method "min" | a=10,000, b=1, c=1,000, g=0.1, cap 1e6 | f=500 → 500; f=2,000 → 1,000; f=10,000 → 1,000; f=50,000 → 5,000 |

Provenance (docstrings): Rodrigues et al. 2004 (10.1641/0006-3568(2004)054[1092:GGAPRF]2.0.CO;2),
rule and 1e6 cap confirmed via Butchart et al. 2015 (10.1111/conl.12158) methods text. Polak
et al. 2015 (10.1098/rsos.150107), Watson et al. 2010 (10.1111/j.1523-1739.2010.01587.x),
Sreekar & Watson 2026 (10.1038/s44358-026-00169-7): numeric breakpoints follow prioritizr's
implementation; not confirmable from accessible paper text — docstrings say so. Polak's
docstring notes its absolute target is non-monotone in `f` (inherent to the rule).

## 5. IUCN Red List family (`_redlist.py`)

```python
rl_species_targets(problem, *, km2_per_unit, status, criterion_a, criterion_b,
                   prop_uplift=0.0, method="max", cap_km2=1_000_000.0, features=None)
rl_ecosystem_targets(...same signature...)
```

`t = min( m( B·(1+p), f·(1−A)·(1+p) ), cap, f )`, `m ∈ {max, min}`, `A` = reduction threshold
for (`status`, `criterion_a`), `B` = range threshold (km²) for (`status`, `criterion_b`),
`p = prop_uplift ≥ 0`.

Species table:

| | CR | EN | VU |
|---|---|---|---|
| A1 | 0.9 | 0.7 | 0.5 |
| A2 = A3 = A4 | 0.8 | 0.5 | 0.3 |
| B1 (EOO km²) | 100 | 5,000 | 20,000 |
| B2 (AOO km²) | 10 | 500 | 2,000 |

Ecosystem table:

| | CR | EN | VU |
|---|---|---|---|
| A1 = A2a = A2b | 0.8 | 0.5 | 0.3 |
| A3 | 0.9 | 0.7 | 0.5 |
| B1 (km²) | 2,000 | 20,000 | 50,000 |
| B2 (km²) | 200 | 2,000 | 5,000 |

- `status`: `str | Mapping[int, str]` (per-feature, contract §3.6). Values case-insensitive in
  {CR, EN, VU}; anything else (LC, NT, DD, typos) → `ValueError` naming the feature. `criterion_a`,
  `criterion_b`: single strings, case-insensitive (`"a2b"`, `"A2B"`, `"A2b"` all accepted).
- **Deviations from prioritizr (each in docstring + CHANGELOG):**
  1. RLE VU-A3 = **0.5** (IUCN RLE guidelines v2.0 thresholds 90/70/50%); prioritizr has 0.3.
  2. Case-insensitive criteria (prioritizr can never match `"A2b"`).
  3. Per-feature `status` (prioritizr takes one status per method object).
  Where prioritizr's docs disagree with its code we follow the code, which agrees with IUCN:
  species EN-B2 = 500 km² (docs: 50).

Presets (all keywords overridable):

| Function | Fixed | Defaults | Effective rule | Pinned examples |
|---|---|---|---|---|
| `jung_targets` | A2, B2, max | status "VU", p=0.1 | `min(max(2,200, 0.77f), cap, f)` | f=2,000 → 2,000; f=2,500 → 2,200; f=5,000 → 3,850; f=50,000 → 38,500; f=2e7 → 1e6 |
| `ward_targets` | A2, B1, max | status "CR", p=0 | `min(max(100, 0.2f), cap, f)` | f=50 → 50; f=500 → 100; f=50,000 → 10,000; f=2e7 → 1e6 |

prioritizr docs say Jung = 80% and Ward = 100,000 km²; code and its tests give 77% and 100 km².
Jung et al. 2021 (10.1038/s41559-021-01528-7) has a correction (10.1038/s41559-021-01569-y),
cited in the docstring. Ward et al. 2025 (10.1073/pnas.2414985122). Exact preset parameters
follow prioritizr; not confirmable from accessible paper text — docstrings say so. IUCN (2025)
Red List Categories and Criteria / IUCN (2024) RLE Guidelines v2.0 cited for the tables.

## 6. Population family (`_population.py`)

```python
pop_size_targets(problem, *, km2_per_unit, pop_size, density_per_km2,
                 cap_km2=1_000_000.0, features=None)
wilson_targets(problem, *, km2_per_unit, mean_growth_rate, var_growth_rate, density_per_km2,
               cap_km2=1_000_000.0, features=None)
duran_targets(problem, *, km2_per_unit, probability_target, historical_area_km2,
              cap_km2=None, features=None)
```

- **pop_size:** `t = min(n / d, cap, f)`; `n > 0`, `d > 0`. Density is per km² by name; no
  unit strings (users convert, e.g. per-ha × 100). prioritizr's docs say `n·d`; its code and
  tests divide. Example: n=500, d=2 → 250.
- **wilson:** `β = 2r/σ² − 1`; `k = (1e5·σ²·β²/2)^(1/β)`; `t = min(k / d, cap, f)`. `r > 0`,
  `σ² > 0`, `d > 0`; `β ≤ 0` → `ValueError` naming the feature (prioritizr returns NaN/Inf).
  Examples: r=0.1, σ²=0.04 → β=4, k=13.3748; r=0.3, σ²=1e-4 → k=1.00317. Wilson et al. 2010
  (10.1890/09-1051.1); formula checked against prioritizr only — the design review's science
  lens must try to confirm it against the paper, else the docstring says it follows prioritizr.
- **duran:** `t = min(h·p⁴, f)` (+ cap if given); `p ∈ [0, 1]`, `h > 0` (km²). Persistence
  `P = (A/A_hist)^0.25` (z = 0.25, supported by Durán et al. 2020 text, 10.1111/2041-210X.13427).
  Examples: h=1,000, p=0.9 → 656.1; p=0.95 → 814.506. Docstring lists Davis 2018 and Gumbs 2023
  per-category probabilities as guidance only (no built-in constants).

Each per-feature input is scalar or `{fid: value}` (§3.6).

## 7. Simple methods and composition (`_compose.py`)

- `absolute_targets(problem, amount, *, features=None)`: `t = a` native units, scalar or dict,
  `a ≥ 0` finite. **Not clipped** to `F` (explicit user target; matches prioritizr;
  `presolve_check` flags targets above totals).
- `area_targets(problem, *, km2_per_unit, area_km2, cap_km2=1_000_000.0, features=None)`:
  `t = min(a, cap, f)`.
- `rule_targets(problem, *, km2_per_unit, baseline, rules, traits, cap_km2=1_000_000.0,
  features=None)`: `rules: Mapping[str, float]` (each in [−1, 1]); `traits: Mapping[int,
  Collection[str]]` (every selected feature present; empty set allowed); a trait not in `rules`
  → `ValueError`. `prop = clamp(baseline + Σ rules[τ] for τ in traits[j], 0, 1)`,
  `t = min(prop·f, cap, f)`. `baseline ∈ [0, 1]`. Harris & Holness 2023
  (10.1016/j.biocon.2023.110218). Joined by feature id (fixes prioritizr's inverse-permutation
  bug).
- `min_targets(*results)`, `max_targets(*results)`: ≥ 2 result dicts, feature-wise min/max.
  Inputs must have identical key sets, else `ValueError` listing the difference (prioritizr
  silently drops NaN). Fewer than 2 inputs → `ValueError`.

## 8. `loglinear_targets` deprecation

Body unchanged (v0.39 behaviour exactly). Emits `DeprecationWarning` (`stacklevel=2`):
"loglinear_targets is deprecated and will be removed in a future release; use
interp_absolute_targets(..., interp='log10')". Removal version not fixed here.

## 9. Testing

Under `tests/pymarxan/targets/` (no `__init__.py` there → repo-unique basenames:
`test_targets_units.py`, `test_targets_interp.py`, `test_targets_redlist.py`,
`test_targets_population.py`, `test_targets_compose.py`).

Fixture: a small synthetic `ConservationProblem` whose feature totals are chosen to hit the
pinned `f` values with `km2_per_unit=1` (and a rescaled variant).

1. **Worked examples:** every pinned value in §4–7 (`pytest.approx`, rel 1e-6).
2. **Regime boundaries:** Jung 2,200 / 2,857.14; Ward 100 / 500; Watson f = 10,000;
   Rodrigues cap onset 1e7; Wilson β just above 0 (finite, large); f = 0 → 0 for every method.
3. **Invariants** (parametrised over all area-based methods): `0 ≤ t ≤ F`; `t·k ≤ cap`.
   **Rescaled units:** amounts ×100 with `km2_per_unit` ÷100 → native targets ×100 exactly.
4. **prioritizr-bug regressions:** subset + out-of-order dict keys; `"A2b"`; RLE VU-A3 = 0.5;
   Watson at threshold; f = 0; Wilson β ≤ 0 raises; `rule_targets` with shuffled `traits`.
5. **Validation:** one test per `ValueError` path (missing dict key, unknown feature id, bad
   status, half-given target/method pair, bad thresholds, log10 with a = 0, min/max key mismatch).
6. **Backwards compatibility:** `tests/pymarxan/test_targets.py` passes unchanged apart from
   expecting `DeprecationWarning` on `loglinear_targets`; loglinear ≡ interp_absolute test.
7. **End to end:** `rodrigues_targets` → `apply_targets` on `tests/data/simple` →
   `MIPSolver` solves with all targets met (anchor 35.0 untouched — it uses the file's targets).
8. **Docs:** a `docs/TUTORIAL.md` section (mixed methods: `jung_targets` on threatened ids,
   `rodrigues_targets` on the rest, `apply_targets`) pinned by `tests/test_tutorial_examples.py`.

## 10. Docs and release notes

- CHANGELOG `[Unreleased]` → `### Added` per method with DOI; `### Deprecated`
  `loglinear_targets`; a "Deviations from prioritizr" list (RLE VU-A3 0.5, Watson threshold,
  f = 0, Wilson β ≤ 0, id-keyed inputs, case-insensitive criteria, min/max key check, Jung 77% /
  Ward 100 km² / EN-B2 500 / pop_size division following code over docs).
- README "Targets" bullets and architecture line updated; roadmap Stage 2 marked shipped at
  release.

## 11. Out of scope (follow-ons)

Shiny UI for target methods; zone / per-zone targets; string shortcuts (`"jung"`); deriving
`km2_per_unit` from raster cell size or PU geometry; any `ConservationProblem` field;
`group_targets` dispatching arbitrary methods per group (use `features=` + `apply_targets`).

## 12. Risks

- **Paper verification:** Polak, Watson, Jung, Ward, Sreekar and Wilson parameters are from
  prioritizr's code, not confirmed in paper text. Mitigation: honest docstrings; science lens
  retries confirmation.
- **Locked-out PUs** count toward `F`, as in prioritizr; large locked-out shares can make a
  clipped target unattainable. Documented; `presolve_check` / infeasibility diagnostics apply.
- **Package conversion** must not break `from pymarxan.targets import ...` or pdoc output;
  covered by the unchanged backwards-compat tests and `make docs`.
