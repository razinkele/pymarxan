# Phase 18 — Implementation Plan (revised post-review)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Each task follows TDD: failing test → minimal impl → passing test. Commit after each batch.

**Goal:** Ship `PROBMODE = 3` (Z-score chance constraints, Marxan-faithful) for all four solvers, plus the optional `prob` column on `puvspr.dat` and `ptarget` column on `spec.dat` (Marxan-compatible wire format). MIP's default strategy is "drop" (probability term excluded from the MIP objective; reported post-hoc on `Solution`).

**Design doc:** `2026-05-16-phase18-design.md` (also revised post-review)
**Review doc:** `2026-05-16-phase18-review.md` — explains why the v1 design changed.
**Tech stack:** Python, NumPy, SciPy (norm.sf — upper-tail probability), pandas, pytest, Shiny.
**New dependency:** `scipy` is already in core deps (`pyproject.toml` line 26).

---

## Batch 1 — Schema + I/O (3 tasks)

Lays the data foundation so subsequent tasks can read/write PROB2D + PTARGET without breaking existing fixtures.

### Task 1: `pu_vs_features` accepts optional `prob` column (Marxan PROB2D wire format)

**Files:**
- `src/pymarxan/io/readers.py::read_puvspr` — handle absent column; preserve existing pass-through.
- `src/pymarxan/io/writers.py::write_puvspr` — emit column only when non-zero values exist.
- `tests/pymarxan/io/test_io_probability.py` (new file).

(Note: `validate()` — not `_validate_columns()` as the v1 plan said — already permits extra columns by virtue of set-membership checks on required columns only; no model change needed.)

**Failing test:**
```python
def test_puvspr_round_trip_with_prob(tmp_path):
    df = pd.DataFrame({
        "species": [1, 1, 2],
        "pu": [1, 2, 3],
        "amount": [10.0, 8.0, 5.0],
        "prob": [0.10, 0.05, 0.0],
    })
    write_puvspr(df, tmp_path / "puvspr.dat")
    read = read_puvspr(tmp_path / "puvspr.dat")
    assert "prob" in read.columns
    np.testing.assert_array_almost_equal(read["prob"].values, [0.10, 0.05, 0.0])

def test_puvspr_no_prob_when_all_zero(tmp_path):
    df = pd.DataFrame({
        "species": [1], "pu": [1], "amount": [10.0], "prob": [0.0],
    })
    write_puvspr(df, tmp_path / "puvspr.dat")
    content = (tmp_path / "puvspr.dat").read_text()
    assert "prob" not in content, "writer must omit all-zero prob column"
```

**Implementation:** in `write_puvspr`, conditionally include the `prob` column. In `read_puvspr`, the existing pass-through already accepts it; just confirm tests pin it. Document the column semantics in the function docstring (per-cell Bernoulli probability; variance derived internally as `amount² · prob · (1 − prob)`).

### Task 2: `features` accepts optional `ptarget` column (Marxan PTARGET2D)

**Files:**
- `src/pymarxan/io/readers.py::read_spec`
- `src/pymarxan/io/writers.py::write_spec`
- `tests/pymarxan/io/test_io_probability.py`

**Failing test:**
```python
def test_spec_round_trip_with_ptarget(tmp_path):
    df = pd.DataFrame({
        "id": [1, 2], "name": ["a", "b"],
        "target": [10.0, 20.0], "spf": [1.0, 2.0],
        "ptarget": [0.95, -1.0],
    })
    write_spec(df, tmp_path / "spec.dat")
    read = read_spec(tmp_path / "spec.dat")
    assert "ptarget" in read.columns
    np.testing.assert_array_almost_equal(read["ptarget"].values, [0.95, -1.0])

def test_spec_omits_ptarget_when_all_disabled(tmp_path):
    df = pd.DataFrame({
        "id": [1], "name": ["a"], "target": [10.0], "spf": [1.0],
        "ptarget": [-1.0],   # the disabled-sentinel default
    })
    write_spec(df, tmp_path / "spec.dat")
    content = (tmp_path / "spec.dat").read_text()
    assert "ptarget" not in content

def test_spec_default_ptarget_when_column_absent(tmp_path):
    """Reading a legacy spec.dat without ptarget gives -1 (disabled)."""
    legacy_csv = "id,name,target,spf\n1,a,10.0,1.0\n"
    path = tmp_path / "spec.dat"
    path.write_text(legacy_csv)
    df = read_spec(path)
    # Either the column is absent (caller checks .get) or filled with -1.
    # Pick one — design says reader fills with -1 to ease downstream code.
    assert "ptarget" in df.columns
    assert df["ptarget"].iloc[0] == -1.0
```

**Implementation:** symmetric to Task 1. Default for absent `ptarget` is `-1.0` (Marxan's disabled sentinel). Writer omits the column when every value is `-1.0`.

### Task 3: `ConservationProblem.validate()` accepts new columns

**Files:**
- `src/pymarxan/models/problem.py::validate()`  (the method is named `validate`, NOT `_validate_columns`)
- `tests/pymarxan/models/test_problem.py`

The current `validate()` checks for required columns by set membership and never rejects extra columns, so this Task is mostly a smoke-test guard against future regressions. Add one test that builds a problem with `prob` and `ptarget` columns and asserts `validate()` returns the empty-list (no errors).

**Commit message for Batch 1:**
```
Phase 18 Batch 1: PROB2D + PTARGET schema and I/O (Marxan-compatible)

Adds optional `prob` column to pu_vs_features (Marxan v4 wire format)
and `ptarget` column to features (Marxan PTARGET2D, default -1 = disabled).
Round-trip via read/write_puvspr and read/write_spec. Writers omit the
columns when all values are at default (0.0 / -1.0). Existing projects
round-trip byte-identical.
```

---

## Batch 2 — Z-score math (3 tasks, Marxan-faithful)

Pure-functional layer. No solver integration yet. New module: `src/pymarxan/solvers/probability.py`.

### Task 4: `compute_zscore_per_feature` (Marxan sign convention: `(T − E)/√V`)

**File:** `src/pymarxan/solvers/probability.py` (new).

**Failing test (in `tests/pymarxan/solvers/test_zscore.py`):**
```python
def test_zscore_zero_variance_returns_marxan_sentinel():
    # Marxan classic computation.hpp sets rZ = 4 when variance == 0
    # (corresponds to P ≈ 1, i.e. target trivially met probabilistically)
    z = compute_zscore_per_feature(
        achieved_mean={1: 20.0}, achieved_variance={1: 0.0}, targets={1: 10.0}
    )
    assert z[1] == 4.0  # Marxan's sentinel for "no uncertainty"

def test_zscore_normal_case_marxan_sign():
    # μ = 12, σ² = 4, T = 10 -> Marxan Z = (10-12)/2 = -1.0 (negative = exceedance)
    z = compute_zscore_per_feature(
        achieved_mean={1: 12.0}, achieved_variance={1: 4.0}, targets={1: 10.0}
    )
    assert z[1] == pytest.approx(-1.0)

def test_zscore_shortfall_case():
    # μ = 8, σ² = 4, T = 10 -> Z = (10-8)/2 = 1.0 (positive = shortfall side)
    z = compute_zscore_per_feature(
        achieved_mean={1: 8.0}, achieved_variance={1: 4.0}, targets={1: 10.0}
    )
    assert z[1] == pytest.approx(1.0)
```

**Implementation:**
```python
def compute_zscore_per_feature(
    achieved_mean: dict[int, float],
    achieved_variance: dict[int, float],
    targets: dict[int, float],
) -> dict[int, float]:
    """Z = (T − E[T]) / √Var[T] (Marxan sign — positive = shortfall side).

    Returns Marxan's sentinel value 4.0 (≈ P=1.0 via probZUT) when variance
    is zero. The deterministic case is handled separately in
    compute_zscore_penalty (zero variance means deterministic shortfall
    handled by the standard penalty path).
    """
    z: dict[int, float] = {}
    for fid, mean in achieved_mean.items():
        var = achieved_variance.get(fid, 0.0)
        target = targets.get(fid, 0.0)
        if var <= 0:
            z[fid] = 4.0  # Marxan sentinel
        else:
            z[fid] = (target - mean) / math.sqrt(var)
    return z
```

### Task 5: `compute_zscore_penalty` (normalised, Marxan-faithful)

**Failing test:**
```python
from scipy.stats import norm

def test_zscore_penalty_zero_when_prob_target_met():
    # Z = -2.0 (exceedance side) -> P = norm.sf(-2.0) ≈ 0.977 > ptarget 0.5
    p = compute_zscore_penalty(
        zscore_per_feature={1: -2.0},
        prob_targets={1: 0.5},
        spf={1: 1.0},
        weight=1.0,
    )
    assert p == pytest.approx(0.0)

def test_zscore_penalty_normalised_by_ptarget():
    # Z = 1.0 (shortfall side) -> P = norm.sf(1.0) ≈ 0.159
    # ptarget = 0.95, SPF = 2.0
    # penalty = γ · SPF · (ptarget − P) / ptarget = 1.0 · 2.0 · (0.95 − 0.159) / 0.95
    p = compute_zscore_penalty(
        zscore_per_feature={1: 1.0},
        prob_targets={1: 0.95},
        spf={1: 2.0},
        weight=1.0,
    )
    expected = 2.0 * (0.95 - norm.sf(1.0)) / 0.95
    assert p == pytest.approx(expected, abs=1e-9)

def test_zscore_penalty_disabled_when_ptarget_negative():
    # ptarget = -1 (disabled sentinel) -> feature contributes 0 regardless of Z
    p = compute_zscore_penalty(
        zscore_per_feature={1: 5.0},  # huge shortfall
        prob_targets={1: -1.0},
        spf={1: 1.0},
        weight=1.0,
    )
    assert p == 0.0

def test_zscore_penalty_weight_scaling():
    base = compute_zscore_penalty(
        zscore_per_feature={1: 1.0},
        prob_targets={1: 0.95},
        spf={1: 1.0},
        weight=1.0,
    )
    scaled = compute_zscore_penalty(
        zscore_per_feature={1: 1.0},
        prob_targets={1: 0.95},
        spf={1: 1.0},
        weight=3.0,
    )
    assert scaled == pytest.approx(3.0 * base)
```

**Implementation:**
```python
from scipy.stats import norm

def compute_zscore_penalty(
    zscore_per_feature: dict[int, float],
    prob_targets: dict[int, float],
    spf: dict[int, float],
    weight: float = 1.0,
) -> float:
    """γ · Σ_j SPF_j · max(0, (ptarget_j − P_j) / ptarget_j).

    P_j = norm.sf(Z_j) (upper tail; Marxan's probZUT). Features with
    ptarget ≤ 0 contribute 0 (disabled sentinel).
    """
    total = 0.0
    for fid, z in zscore_per_feature.items():
        ptarget = prob_targets.get(fid, -1.0)
        if ptarget <= 0:
            continue
        prob = norm.sf(z)  # 1 - Φ(z) = probZUT
        spf_j = spf.get(fid, 1.0)
        shortfall = max(0.0, (ptarget - prob) / ptarget)
        total += spf_j * shortfall
    return float(weight * total)
```

### Task 6: `compute_probability_penalty` dispatches on `PROBMODE`

**Files:** `src/pymarxan/solvers/utils.py`, `tests/pymarxan/solvers/test_probability_dispatch.py`.

**Failing tests** verify all four code paths (PROBMODE 0/1/2/3) return what they should given a constructed problem.

**Implementation:** dispatch in `compute_probability_penalty`. Mode 3 builds `achieved_mean` / `achieved_variance` via vectorised selection mask, computes Z, returns penalty. Mode 2 returns 0 (handled upstream as amount substitution). Mode 1 keeps existing risk-premium logic.

**Commit message for Batch 2:**
```
Phase 18 Batch 2: Z-score / PROBMODE 3 math

Pure-functional Z-score and penalty computation. Routed via
compute_probability_penalty dispatching on the PROBMODE parameter.
Edge cases covered: zero variance (deterministic), inf/-inf Z handling,
multi-feature SPF-weighted sum. No solver integration yet.
```

---

## Batch 3 — Solver integration (4 tasks)

Wire the Z-score penalty into SA, heuristic, iterative improvement, and the MIP guard.

### Task 7: `ProblemCache` PROB2D bookkeeping (with sparse per-PU feature index)

**Files:**
- `src/pymarxan/solvers/cache.py` — add `prob_matrix`, `var_matrix`, `feat_ptarget`, `pu_feat_idx` (sparse index), `prob_weight` to the dataclass + `from_problem`. Maintain `held_variance` alongside `held` in the solver.
- `tests/pymarxan/solvers/test_cache.py` — extend `TestComputeDeltaObjective` (the class exists at `test_cache.py:214`) to cover PROBMODE 3.

**Failing test:**
```python
def test_problem_cache_delta_under_probmode3(problem_probmode3):
    cache = ProblemCache.from_problem(problem_probmode3)
    rng = np.random.default_rng(7)
    selected = rng.random(cache.n_pu) > 0.5
    held = cache.compute_held(selected)
    held_var = cache.compute_held_variance(selected)  # new helper
    total_cost = float(np.sum(cache.costs[selected]))
    blm = 1.0
    for _ in range(10):
        idx = rng.integers(cache.n_pu)
        full_before = cache.compute_full_objective(selected, held, blm,
                                                    held_variance=held_var)
        delta = cache.compute_delta_objective(idx, selected, held, total_cost,
                                              blm, held_variance=held_var)
        # Flip and update held/held_var via the cache helpers
        sign = -1.0 if selected[idx] else 1.0
        held += sign * cache.pu_feat_matrix[idx]
        held_var += sign * cache.var_matrix[idx]
        np.clip(held_var, 0.0, None, out=held_var)  # underflow guard
        selected[idx] = not selected[idx]
        total_cost += sign * cache.costs[idx]
        full_after = cache.compute_full_objective(selected, held, blm,
                                                   held_variance=held_var)
        assert delta == pytest.approx(full_after - full_before, abs=1e-9)
```

**Implementation:**
- Precompute `prob_matrix[pu_idx, feat_idx]` from `pu_vs_features.prob` (default 0 when absent).
- Precompute `var_matrix[pu_idx, feat_idx] = amount² · prob · (1 − prob)`.
- Precompute `feat_ptarget[feat_idx]` from `features.ptarget` (default -1).
- Precompute `pu_feat_idx[pu_idx] = np.where(amount[pu_idx, :] != 0 | var_matrix[pu_idx, :] != 0)` — sparse list of features touched by this PU. Used in `compute_delta_objective` to avoid scanning all features.
- Cache `prob_weight = parameters.get("PROBABILITYWEIGHTING", 1.0)`.
- `compute_full_objective` accepts an optional `held_variance` kwarg; when PROBMODE==3 it computes E[T_j] from `held` (which is mean amount) by adjusting for `prob` — actually `E[T_j] = Σ amount · (1-prob)` so the cache should also maintain a separate `held_expected = Σ x_i · μ_ij · (1 − p_ij)`. Track this as `held_expected` distinct from `held` to avoid conflating the deterministic-target path.

### Task 8: `SimulatedAnnealingSolver` honours PROBMODE 3

**Failing test:** an end-to-end SA solve on a small PROB2D problem reaches a solution whose `targets_met` accounts for probability. Use a constructed scenario where deterministic optimum (PROBMODE 0) and probabilistic optimum (PROBMODE 3) differ — usually by adding a "safe but expensive" PU that's only attractive under uncertainty.

**Implementation:** `SimulatedAnnealingSolver.solve` passes the problem through unchanged; the heavy lifting is in `ProblemCache` from Task 7. Verify the solve runs and `pymarxan.__version__`-recognised output looks reasonable.

### Task 9: `HeuristicSolver` PROBMODE 3 branch (inline only — NOT IterativeImprovementSolver)

**File:** `src/pymarxan/solvers/heuristic.py`.

**IMPORTANT — correction from v1 plan:** `IterativeImprovementSolver` DOES use `ProblemCache` (`iterative_improvement.py:114`). It inherits PROBMODE 3 support automatically from Task 7's cache changes. Only `HeuristicSolver` (which builds its own `contributions` dict instead of using the cache) needs inline logic.

**Implementation:** in the heuristic's scoring loop, when `PROBMODE == 3`, compute the contributory variance for each candidate add and call `compute_zscore_penalty` against the running held mean and held variance. Add the penalty to the candidate's score so it factors into selection.

**Failing test:** parameterised over `[SimulatedAnnealingSolver, IterativeImprovementSolver, HeuristicSolver]`; each must run end-to-end under PROBMODE 3 and (a) populate `Solution.prob_penalty`, (b) produce a different solution under PROBMODE 3 vs PROBMODE 0 on a constructed problem where variance correlates with cheap PUs.

### Task 10: `MIPSolver` strategy="drop" (default) + post-hoc evaluation

**Files:** `src/pymarxan/solvers/mip_solver.py`, `src/pymarxan/zones/mip_solver.py`.

**Failing test:**
```python
def test_mip_drop_strategy_returns_solution(problem_probmode3):
    """MIP with default 'drop' strategy returns a deterministic solution
    and reports the chance-constraint gap post-hoc."""
    sol = MIPSolver().solve(problem_probmode3)[0]
    # Solver returns OK — no exception
    assert sol is not None
    # Solution is deterministic-feasible
    assert sol.objective >= 0
    # PROBMODE 3 evaluation populated post-hoc
    assert sol.prob_penalty is not None
    assert sol.prob_shortfalls is not None

def test_mip_piecewise_strategy_not_implemented():
    with pytest.raises(NotImplementedError, match="piecewise"):
        MIPSolver(mip_chance_strategy="piecewise").solve(problem_probmode3)

def test_mip_socp_strategy_not_implemented():
    with pytest.raises(NotImplementedError, match="Phase 21"):
        MIPSolver(mip_chance_strategy="socp").solve(problem_probmode3)
```

**Implementation:**
1. Add `mip_chance_strategy: Literal["drop", "piecewise", "socp"] = "drop"` kwarg on `MIPSolver.__init__` and `ZoneMIPSolver.__init__`.
2. In `solve()`: if `PROBMODE != 3`, behaviour is unchanged. If `PROBMODE == 3` and strategy is `"drop"`: solve the deterministic problem (skip any probability term in the MIP objective), then call `evaluate_solution_chance(problem, solution)` to populate `Solution.prob_shortfalls` and `Solution.prob_penalty`.
3. `"piecewise"` and `"socp"` raise `NotImplementedError` with phase pointers (18.5 and 21 respectively).
4. Add a `_warned` flag so the "MIP is dropping the chance constraint" warning fires once per session, not once per solve call.

### Task 10b (NEW): `Solver.supports_probmode3()` capability method

**Files:** `src/pymarxan/solvers/base.py`, all solver subclasses.

**Implementation:** add `def supports_probmode3(self) -> bool: return True` on the `Solver` ABC. All current solvers inherit `True` (MIP supports mode 3 with the "drop" fallback). No override needed unless a future solver genuinely can't fall back. The Shiny run-panel uses this to decide whether to enable mode 3 as an option.

**Commit message for Batch 3:**
```
Phase 18 Batch 3: PROBMODE 3 in all solvers; MIP drop+evaluate strategy

ProblemCache tracks per-cell prob → expected/variance with sparse
per-PU feature index, keeping delta computation O(features_per_pu).
SA and IterativeImprovementSolver inherit PROBMODE 3 from the cache
automatically. HeuristicSolver gets an inline PROBMODE 3 branch.
MIPSolver and ZoneMIPSolver gain mip_chance_strategy kwarg: "drop"
(default) solves the deterministic problem and reports the
chance-constraint gap post-hoc on Solution.prob_{shortfalls,penalty};
"piecewise" and "socp" raise NotImplementedError pointing at Phase
18.5 and 21. New Solver.supports_probmode3() capability method
returns True everywhere (MIP falls back via drop).
```

---

## Batch 4 — Shiny + smoke test (2 tasks)

### Task 11: Shiny PROBMODE 3 toggle + Z-score display

**Files (corrected from v1 plan — `probability_config.py` is the existing PROBMODE control):**
- `src/pymarxan_shiny/modules/probability/probability_config.py` — add `"3"` to the existing PROBMODE radio (which currently only has `"1"` and `"2"`). When `"3"` is selected, conditionally show a small panel referencing the `prob` column on puvspr and `ptarget` on spec.
- `src/pymarxan_shiny/modules/results/target_met.py` — show Z-score and P_j columns when the active problem's PROBMODE == 3.
- `src/pymarxan_shiny/modules/run_control/run_panel.py` — add a notice banner "MIP solver drops chance constraint; gap reported post-hoc" when MIP + PROBMODE 3 are both selected. Do NOT add a duplicate PROBMODE control here.
- `src/pymarxan_shiny/modules/help/help_content.py` — user-facing explainer for PROBMODE 3 (with the foundational paper citations).

**Failing test (file-based, like the Review 6 H4 cleanup tests):**
```python
def test_probability_config_offers_probmode_3():
    src = Path(probability_config_module.__file__).read_text()
    assert '"3"' in src and "PROBMODE" in src

def test_target_met_module_references_zscore():
    src = Path(target_met_module.__file__).read_text()
    assert "Z-score" in src or "zscore" in src.lower()

def test_run_panel_warns_about_mip_drop():
    src = Path(run_panel_module.__file__).read_text()
    assert "post-hoc" in src.lower() or "chance constraint" in src.lower()
```

**Implementation:** standard Shiny additions. Read existing patterns from `probability_config.py` lines 53-65.

### Task 12: End-to-end smoke test on a probabilistic problem

**Files:**
- `tests/integration/__init__.py` (new — the `tests/integration/` directory does not exist; create both)
- `tests/integration/test_phase18_smoke.py` (new)

A small but meaningful synthetic problem with 6 PUs, 2 features, non-zero `prob` for half the cells, `ptarget=0.95` on one feature and `ptarget=-1` on the other. Verifies:

1. Each of SA / heuristic / iterative-improvement runs end-to-end and produces a `Solution` with non-None `prob_penalty`.
2. MIP runs end-to-end (strategy="drop") and reports the gap post-hoc.
3. The feature with `ptarget=-1` contributes 0 to the probability penalty regardless of selection.
4. Increasing the SPF on the probability-targeted feature pushes the SA solution toward higher-variance-tolerant PUs.

**Commit message for Batch 4:**
```
Phase 18 Batch 4: Shiny UI integration + integration smoke test

Adds PROBMODE "3" to the existing probability_config radio (not a
duplicate widget in run_panel). target_met module shows Z-score and
P_j when active problem is PROBMODE 3. run_panel shows a notice when
MIP+PROBMODE 3 are both selected. New tests/integration/ directory
holds the end-to-end smoke test exercising all four solvers.
```

---

## Verification

After all 4 batches:

```bash
make check                  # lint + types + tests
gh release view v0.1.0      # ensure no release drift
/opt/micromamba/envs/shiny/bin/pytest tests/integration/ -v  # the new smoke tests
```

Targets:
- Tests: 1094 → ~1110
- Coverage: 91.93 % → ≥91 %
- `make check` green
- Benchmark check: SA iteration count under PROBMODE 3 within 2× of PROBMODE 0 wall-clock (caught by `tests/benchmarks/`).
- `CHANGELOG.md` has new `## [Unreleased]` entry summarising Phase 18

## Memory update

After Phase 18 lands, append to MEMORY.md `Phase Status Summary` table:

```
| 18 | COMPLETE | ~1110 | PROBMODE 3 / Z-score / PROB2D `prob` column / PTARGET. Marxan-faithful Z=(T−E)/√V, ptarget=-1 disables, MIP drops chance constraint and reports gap post-hoc on Solution.prob_{shortfalls,penalty}. |
```

## Commits

Four batch commits, then one release-bump commit when v0.2.0 ships with Phases 18-20 bundled.

## Changelog ride-along

Add to `CHANGELOG.md` under `[Unreleased]`:

```markdown
### Added
- **PROBMODE 3 (chance-constrained reserve design)** — Marxan v4 PROB2D + PTARGET2D support across all solvers. Per-cell Bernoulli probability via optional `prob` column on `puvspr.dat`; per-feature probability target via optional `ptarget` column on `spec.dat` (default `-1` = disabled). SA / heuristic / iterative-improvement solve chance-constrained problems natively. MIP solves the deterministic relaxation and reports the chance-constraint gap post-hoc on `Solution.prob_shortfalls` and `Solution.prob_penalty`. References: Game et al. 2008, Tulloch et al. 2013, Carvalho et al. 2011.
- New module `pymarxan.solvers.probability` with `compute_zscore_per_feature`, `compute_zscore_penalty`, `evaluate_solution_chance`.
- `Solver.supports_probmode3()` capability method.
- `MIPSolver(mip_chance_strategy=...)` kwarg with `"drop"` (default), `"piecewise"` (Phase 18.5), `"socp"` (Phase 21).
```
