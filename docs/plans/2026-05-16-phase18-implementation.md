# Phase 18 — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Each task follows TDD: failing test → minimal impl → passing test. Commit after each batch.

**Goal:** Ship `PROBMODE = 3` (Z-score chance constraints) for SA / heuristic / iterative improvement, plus PROB2D variance + PTARGET probability target column support. MIP raises `NotImplementedError` (deferred to Phase 21).

**Design doc:** `2026-05-16-phase18-design.md`
**Tech stack:** Python, NumPy, SciPy (norm.cdf), pandas, pytest, Shiny.
**New dependency:** `scipy` is already in core deps (`pyproject.toml` line 26).

---

## Batch 1 — Schema + I/O (3 tasks)

Lays the data foundation so subsequent tasks can read/write PROB2D + PTARGET without breaking existing fixtures.

### Task 1: `pu_vs_features` accepts optional `variance` column

**Files:**
- `src/pymarxan/models/problem.py` — `_validate_columns()` and any column-presence checks.
- `src/pymarxan/io/readers.py::read_puvspr` — handle absent column.
- `src/pymarxan/io/writers.py::write_puvspr` — emit column only when non-zero values exist.
- `tests/pymarxan/io/test_io_probability.py` (new file).

**Failing test:**
```python
def test_puvspr_round_trip_with_variance(tmp_path):
    df = pd.DataFrame({
        "species": [1, 1, 2],
        "pu": [1, 2, 3],
        "amount": [10.0, 8.0, 5.0],
        "variance": [0.5, 0.2, 0.0],
    })
    write_puvspr(df, tmp_path / "puvspr.dat")
    read = read_puvspr(tmp_path / "puvspr.dat")
    assert "variance" in read.columns
    np.testing.assert_array_almost_equal(read["variance"].values, [0.5, 0.2, 0.0])

def test_puvspr_no_variance_when_all_zero(tmp_path):
    df = pd.DataFrame({
        "species": [1], "pu": [1], "amount": [10.0], "variance": [0.0],
    })
    write_puvspr(df, tmp_path / "puvspr.dat")
    content = (tmp_path / "puvspr.dat").read_text()
    assert "variance" not in content, "writer must omit all-zero variance column"
```

**Implementation:** in `write_puvspr`, conditionally include the `variance` column. In `read_puvspr`, accept it when present, drop nothing otherwise.

### Task 2: `features` accepts optional `prob_target` column

**Files:**
- `src/pymarxan/io/readers.py::read_spec`
- `src/pymarxan/io/writers.py::write_spec`
- `tests/pymarxan/io/test_io_probability.py`

**Failing test:**
```python
def test_spec_round_trip_with_prob_target(tmp_path):
    df = pd.DataFrame({
        "id": [1, 2], "name": ["a", "b"],
        "target": [10.0, 20.0], "spf": [1.0, 2.0],
        "prob_target": [0.95, 0.8],
    })
    write_spec(df, tmp_path / "spec.dat")
    read = read_spec(tmp_path / "spec.dat")
    assert "prob_target" in read.columns
    np.testing.assert_array_almost_equal(read["prob_target"].values, [0.95, 0.8])

def test_spec_omits_prob_target_when_all_default(tmp_path):
    df = pd.DataFrame({
        "id": [1], "name": ["a"], "target": [10.0], "spf": [1.0],
        "prob_target": [0.5],   # the default
    })
    write_spec(df, tmp_path / "spec.dat")
    content = (tmp_path / "spec.dat").read_text()
    assert "prob_target" not in content
```

**Implementation:** symmetric to Task 1. Default value for absent `prob_target` is `0.5`.

### Task 3: `ConservationProblem.validate()` accepts new columns

**Files:**
- `src/pymarxan/models/problem.py::validate()`
- `tests/pymarxan/models/test_problem.py`

Just confirm `validate()` doesn't raise on a problem with both new columns. Smoke test.

**Commit message for Batch 1:**
```
Phase 18 Batch 1: PROB2D + PTARGET schema and I/O

Adds optional `variance` column to pu_vs_features and `prob_target`
column to features. Round-trip via read/write_puvspr and read/write_spec.
Writers omit the columns when values are all-default (zero / 0.5) so
existing projects continue to round-trip byte-identical.
```

---

## Batch 2 — Z-score math (3 tasks)

Pure-functional layer. No solver integration yet.

### Task 4: `compute_zscore_per_feature` pure function

**File:** `src/pymarxan/solvers/utils.py` (extend).

**Failing test (in `tests/pymarxan/solvers/test_zscore.py`):**
```python
def test_zscore_zero_variance_returns_inf_when_target_met():
    # μ = 20, σ² = 0, T = 10 -> deterministic exceedance
    z = compute_zscore_per_feature(
        achieved_mean={1: 20.0}, achieved_variance={1: 0.0}, targets={1: 10.0}
    )
    assert z[1] == np.inf

def test_zscore_zero_variance_returns_neg_inf_when_target_missed():
    z = compute_zscore_per_feature(
        achieved_mean={1: 5.0}, achieved_variance={1: 0.0}, targets={1: 10.0}
    )
    assert z[1] == -np.inf

def test_zscore_normal_case():
    # μ = 12, σ² = 4, T = 10 -> Z = (12-10)/2 = 1.0
    z = compute_zscore_per_feature(
        achieved_mean={1: 12.0}, achieved_variance={1: 4.0}, targets={1: 10.0}
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
    """Z = (Σ μ_i x_i − T) / √(Σ σ²_i x_i).

    Returns +inf when variance is 0 and target met,
    −inf when variance is 0 and target missed.
    """
    z: dict[int, float] = {}
    for fid, mean in achieved_mean.items():
        var = achieved_variance.get(fid, 0.0)
        target = targets.get(fid, 0.0)
        if var <= 0:
            z[fid] = float("inf") if mean >= target else float("-inf")
        else:
            z[fid] = (mean - target) / math.sqrt(var)
    return z
```

### Task 5: `compute_zscore_penalty` pure function

**Failing test:**
```python
def test_zscore_penalty_zero_when_prob_target_met():
    # Z = 2.0 -> Φ(Z) ≈ 0.977 > prob_target 0.5
    p = compute_zscore_penalty(
        zscore_per_feature={1: 2.0},
        prob_targets={1: 0.5},
        spf={1: 1.0},
    )
    assert p == pytest.approx(0.0)

def test_zscore_penalty_when_prob_target_missed():
    # Z = -1.0 -> Φ(Z) ≈ 0.159; prob_target 0.95; SPF 2.0
    # penalty = SPF · max(0, p_target − Φ(Z)) = 2.0 · (0.95 − 0.159) ≈ 1.582
    p = compute_zscore_penalty(
        zscore_per_feature={1: -1.0},
        prob_targets={1: 0.95},
        spf={1: 2.0},
    )
    assert p == pytest.approx(2.0 * (0.95 - norm.cdf(-1.0)), abs=1e-9)

def test_zscore_penalty_sums_over_features():
    p = compute_zscore_penalty(
        zscore_per_feature={1: -1.0, 2: -2.0},
        prob_targets={1: 0.95, 2: 0.95},
        spf={1: 1.0, 2: 3.0},
    )
    expected = (0.95 - norm.cdf(-1.0)) + 3.0 * (0.95 - norm.cdf(-2.0))
    assert p == pytest.approx(expected, abs=1e-9)
```

**Implementation:**
```python
from scipy.stats import norm

def compute_zscore_penalty(
    zscore_per_feature: dict[int, float],
    prob_targets: dict[int, float],
    spf: dict[int, float],
) -> float:
    total = 0.0
    for fid, z in zscore_per_feature.items():
        target_prob = prob_targets.get(fid, 0.5)
        cdf = norm.cdf(z)
        spf_j = spf.get(fid, 1.0)
        total += spf_j * max(0.0, target_prob - cdf)
    return float(total)
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

### Task 7: `ProblemCache` variance bookkeeping

**Files:**
- `src/pymarxan/solvers/cache.py` — add `var_matrix`, `held_variance` to precomputed state and the flip logic.
- `tests/pymarxan/solvers/test_cache.py` — extend `TestComputeDeltaObjective` to cover PROBMODE 3.

**Failing test:**
```python
def test_problem_cache_delta_under_probmode3(problem_probmode3):
    cache = ProblemCache.from_problem(problem_probmode3)
    rng = np.random.default_rng(7)
    selected = rng.random(cache.n_pu) > 0.5
    for _ in range(10):
        flip_idx = rng.integers(cache.n_pu)
        full_before = cache.compute_full_objective(selected, ...)
        delta = cache.compute_delta_objective(flip_idx, selected, ...)
        selected[flip_idx] = not selected[flip_idx]
        full_after = cache.compute_full_objective(selected, ...)
        assert delta == pytest.approx(full_after - full_before, abs=1e-9)
```

**Implementation:** precompute `var_matrix[pu_idx, feat_idx]` exactly like `feat_matrix`; maintain `held_variance` analog to `held` and update both on flip. `compute_full_objective` and `compute_delta_objective` add a `probability` branch that constructs `achieved_variance` from `held_variance` and calls `compute_zscore_penalty`.

### Task 8: `SimulatedAnnealingSolver` honours PROBMODE 3

**Failing test:** an end-to-end SA solve on a small PROB2D problem reaches a solution whose `targets_met` accounts for probability. Use a constructed scenario where deterministic optimum (PROBMODE 0) and probabilistic optimum (PROBMODE 3) differ — usually by adding a "safe but expensive" PU that's only attractive under uncertainty.

**Implementation:** `SimulatedAnnealingSolver.solve` passes the problem through unchanged; the heavy lifting is in `ProblemCache` from Task 7. Verify the solve runs and `pymarxan.__version__`-recognised output looks reasonable.

### Task 9: `HeuristicSolver` and `IterativeImprovementSolver` paths

**Files:** `src/pymarxan/solvers/heuristic.py`, `src/pymarxan/solvers/iterative_improvement.py`.

**Implementation:** both solvers compute `achieved` via the existing helper; add a PROBMODE-3 branch that computes Z-score penalty inline and adds it to the objective. No ProblemCache changes — these solvers don't use one.

**Failing test:** parameterised over the three solvers; each must run end-to-end under PROBMODE 3 and produce a solution whose objective decreases when a high-variance PU is added (everything else equal).

### Task 10: `MIPSolver` raises `NotImplementedError` for PROBMODE 3

**Files:** `src/pymarxan/solvers/mip_solver.py`, `src/pymarxan/zones/mip_solver.py`.

**Failing test:**
```python
def test_mip_raises_on_probmode3(problem_probmode3):
    with pytest.raises(NotImplementedError, match="PROBMODE.*3.*SA"):
        MIPSolver().solve(problem_probmode3)
```

**Implementation:** early-return guard in `solve()`:
```python
if int(problem.parameters.get("PROBMODE", 0)) == 3:
    raise NotImplementedError(
        "PROBMODE=3 (Z-score chance constraints) is nonlinear and not "
        "supported by the CBC MIP backend. Use the SA, heuristic, or "
        "iterative improvement solvers instead. (See Phase 21 plan for "
        "Gurobi/SOCP support.)"
    )
```

**Commit message for Batch 3:**
```
Phase 18 Batch 3: PROBMODE 3 in SA, heuristic, iterative-improvement; MIP guard

ProblemCache tracks variance per feature with O(1) delta on flip.
SA / heuristic / iterative-improvement compute Z-score penalty inline.
MIP raises NotImplementedError with a clear pointer at SA and Phase 21.
```

---

## Batch 4 — Shiny + smoke test (2 tasks)

### Task 11: Shiny PROBMODE 3 toggle + Z-score display

**Files:**
- `src/pymarxan_shiny/modules/run_control/run_panel.py` — extend the solver-config form with a `PROBMODE` radio (0/1/2/3) when probability data is present.
- `src/pymarxan_shiny/modules/results/target_met.py` — show Z-score column when PROBMODE == 3.
- `src/pymarxan_shiny/modules/help/help_content.py` — short user-facing explainer for PROBMODE 3.

**Failing test:** `inspect.getsource` on the modules contains the strings `"PROBMODE"` and `"Z-score"`. Lightweight, like the H4 cleanup tests from Review 6.

**Implementation:** standard Shiny additions following the existing patterns. Read tests' code patterns from `tests/pymarxan_shiny/test_run_panel.py`.

### Task 12: End-to-end smoke test on a probabilistic problem

**File:** `tests/integration/test_phase18_smoke.py`.

A small but meaningful synthetic problem with 6 PUs, 2 features, variance present, PROBMODE 3, and a known-correct (computed via brute force) optimal solution. Verifies SA converges to within `objective * 1.05` of the brute-force optimum.

**Commit message for Batch 4:**
```
Phase 18 Batch 4: Shiny UI + end-to-end smoke test
```

---

## Verification

After all 4 batches:

```bash
make check                  # lint + types + tests
gh release view v0.1.0      # ensure no release drift
```

Targets:
- Tests: 1094 → ~1110
- Coverage: 91.93 % → ≥91 %
- `make check` green
- `CHANGELOG.md` has new `## [Unreleased]` entry summarising Phase 18

## Memory update

After Phase 18 lands, append to MEMORY.md `Phase Status Summary` table:

```
| 18 | COMPLETE | ~1110 | PROBMODE 3 / Z-score / PROB2D / PTARGET. MIP gated NotImplementedError (Phase 21). |
```

## Commits

Four batch commits, then one release-bump commit when v0.2.0 ships with Phases 18-20 bundled.
