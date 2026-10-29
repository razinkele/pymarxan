# Chance-Term Core, 2D (v0.47.0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** PROBMODE 3 (2D chance constraints) follows Marxan 3's zero-variance rule (Z = 4) by default with a `PROBZEROVAR 1` opt-out, every path scores features through one shared core, and simulated annealing / iterative improvement update the chance term incrementally (about 5× mode 0 per flip instead of about 300×), so release v0.48.0 can add Probability 1D on the same machinery.

**Architecture:** One pure scoring function (`score_feature` in `solvers/probability.py`) decides Z, P, shortfall and penalty from a feature's E, V and count of nonzero-variance cells. A new `ChanceState` sidecar (`solvers/chance_state.py`, the `ClumpState` / `SepState` pattern) keeps per-feature running E / V / count with float error bounds and an error-bounded recompute; `ProblemCache` stays stateless and only holds immutable per-cell contribution arrays (CSR and CSC). Two helpers, `_chance_terms` (requested) and `_active_chance_terms` (requested and with data), replace every `PROBMODE == 3` branch.

**Tech Stack:** Python 3.11+, numpy, scipy (sparse, `norm.sf`), PuLP 3.3.2 + CBC, pytest.

**Spec:** `docs/plans/2026-10-18-probability-1d-design.md` (reviewed 2026-10-29: four loop rounds and a fresh-eyes final review; synthesis `docs/plans/2026-10-29-probability-1d-review.md`). This plan is release **A** of spec §6 (groups A0–A4); release B (Probability 1D, modes 4 / 5, diagnostics, Shiny) is out of scope. Executors read the spec; it is the authority, and this plan argues from it.

**Provenance:** every code block below is extracted (`git diff` / `git show`) from a prototype built task by task in a scratch worktree, one commit per task, with the full gate green at every commit (branch `proto/v0.47-a`). Prototype commits: Task 1 `5d1ac60`, Task 2 `492d5f0`, Task 3 `b82cf34`, Task 4 `0e329e0`, Task 5 `ffb7d6c`, Task 6 `4d4b434`, Task 7 `fe6bd76` (base `a4248b4`). Modified files are diffs against the previous task's tree and apply with `git apply --whitespace=nowarn`; new files are given in full. Every block was replayed that way on a fresh worktree from the base, and each task's tree matched its prototype commit exactly (2026-10-29). Editors that strip trailing whitespace from pasted context lines break a plain `git apply`. Each task's RED and GREEN counts were measured by running the task's test files on the previous task's tree and then on the task's own tree.

## Global Constraints

- Python 3.11+ floor; `from __future__ import annotations` first in every new file; full type hints. mypy runs as `mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports`.
- ruff rules E, F, I, UP; line length 99. **Run `ruff check` on the test files too at every task handoff.**
- Tests run ONLY under `/opt/micromamba/envs/shiny/bin/pytest`, never `.venv`. In a worktree without `.venv`, use `PYTHONPATH=src:.` with the shiny pytest, `~/.local/bin/ruff`, and the main checkout's `.venv/bin/mypy`.
- **Marxan 3 score, per feature** (spec §1), over the selected PUs: `E = Σ a·q`, `V = Σ a²·q(1 − q)`; `Z = (target − E)/√V` if V > 0, else `Z = 4`; `P = norm.sf(Z)`; `penalty = [ptarget > P]·(ptarget − P)/ptarget` if `ptarget > 0`, else 0; total `PROBABILITYWEIGHTING · Σ penalty`. No SPF, no MISSLEVEL; `target` is the resolved deterministic target.
- **Zero variance** = count of selected cells whose variance term (as computed in float) is > 0 equals 0 — never a float threshold. Rule `"marxan"` (default, `PROBZEROVAR 0`): Z = 4 through the normal path; rule `"met"` (`PROBZEROVAR 1`): penalty 0, shortfall 0.
- **Not reproduced:** Marxan 3's delta bugs (`spec[i]` vs `spec[isp]`; no E/V update for clumping species; running-sum zero test). pymarxan's SA delta equals the difference of its own full recompute.
- **Error-bounded recompute** (spec §3.5): `BV += eps·(|V_old| + |ΔV|)`, `BE += eps·(|E_old| + |ΔE|)`; recompute a feature over the selection *after* the previewed flip when `V ≤ 0` with count > 0, `BV > τ·V`, or `BE > max(τ·√V, 64·eps·|E|)`, τ = 1e-12, V post-update; reset bounds to `4·eps·|value|`; `from_selection` counts as a recompute.
- **Commit == preview:** `apply_flip` commits the cached preview when `(idx, adding)` matches, else runs the same code path; every `apply_flip` clears the preview; `delta_penalty` never mutates.
- **Chance term counted once:** `compute_full_objective` / `compute_delta_objective` carry no chance term; SA adds `ChanceState.penalty_total()`; the swap pass adds a from-scratch term at both `compute_full_objective` calls; the evaluator / `build_solution` carry the full term.
- **Release A scope:** only PROBMODE 3 requests a chance term (`{"2d"}`); every other mode requests none. Mode 3 without a puvspr `prob` column: term off, penalty 0, silent (the diagnostic is release B).
- **Performance model:** no per-flip allocation beyond the preview cache. Benches comparative only (`bench` marker): modes 3 per flip (delta, plus commit on accept) ≤ 6× mode 0 at 2000 × 20 and 10 000 × 50, dense and sparse-risk layouts.
- **Goldens:** no golden JSON is regenerated (no golden record uses PROBMODE 3). The `tests/data/simple` anchor (MIP 35.0; SA / greedy at or above) holds.
- Commit trailer on every commit:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01G9s5x1nmcqSTDeRUXcRPVW
  ```

## Review Focus

Inputs a user will plausibly hit that the spec's acceptance list does not name directly. Each is pinned by a test in its owning task.
1. **A puvspr `prob` column that is all 1** (presence recorded as certain everywhere): every feature has zero variance, so under the new default each one with `ptarget > 0` pays ≈ PROBABILITYWEIGHTING, and saving / reloading must keep the column (dropping it would switch the term off). → `test_puvspr_all_one_prob_round_trips` (Task 1), `test_zero_variance_default_is_marxans_rule[all q = 1]` (Task 7).
2. **A typo in `PROBZEROVAR`** (`2`, `0.5`, `"yes"`): `validate()` reports it and the solvers fall back to Marxan's rule rather than raising mid-run. → `test_validate_reports_bad_probzerovar`, `test_rule_from_parameter` (Task 7).
3. **A feature none of whose cells is selected** (an early SA state, or a deliberately empty plan): zero variance, so Marxan's penalty applies by default and 0 under the opt-out. → `test_zero_variance_default_is_marxans_rule[nothing selected]`, `test_zero_variance_opt_out_is_met` (Task 7).
4. **An older PROBMODE 3 project with no `prob` column in puvspr**: the chance term is off (penalty 0, `Solution.prob_shortfalls` None) and SA gives the mode-0 result instead of a constant ≈ PROBABILITYWEIGHTING per feature. → `test_2d_term_without_puvspr_prob_is_requested_but_not_active`, `test_solution_fields_follow_active_terms` (Task 3), `test_term_off_without_data_matches_mode_0` (Task 6).
5. **Mostly certain presence with a few risky cells** (the usual real-data layout): the E-bound floor keeps recomputes rare and the delta exact. → `test_sparse_risk_recompute_floor` (Task 5) and the sparse-risk bench cases (Task 6).

## Rulings made while prototyping

Where the spec was silent, or wrong about the code, the prototype made these choices; each names its task.
- **Task 1:** the all-one round-trip test is a guard, not RED (the old writer dropped only an all-zero column); RED = the two all-zero tests.
- **Task 2:** the RED test's sweep is seeded (`seed=42`): the greedy heuristic breaks ties with random noise, so unseeded `run_sweep` and `run_sweep_parallel` differ by tie-breaking alone.
- **Task 3:** spec §3.1 lists `probability_config.py`, `help_content.py:865` and `base.py:129` as `PROBMODE == 3` sites; none has such a branch (radio values, prose, a docstring), so nothing is routed there in release A.
- **Task 3:** `build_solution` gates `Solution.prob_shortfalls` / `prob_penalty` on the *active* terms, so mode 3 without a puvspr `prob` column now reports None / None (was `{fid: 0.0}` / 0.0); the objective is unchanged. `compute_probability_penalty` keeps two gates: requested → reject fractional selections; active → compute.
- **Task 3:** the MIP's `_BaseModel` / `_BuiltModel` gain `chance_terms` (requested) beside `probmode` (still read for mode 1); `_strategy_metadata` takes `chance_terms` (callers: `MIPSolver.solve`, `MultiObjectiveMIP.solve`). The dense matrices are still built on *requested* terms until Task 6 deletes them.
- **Task 4:** `score_feature` is a thin wrapper over a public vectorised `score_features(E, V, nonzero, target, ptarget, zero_var_rule) -> (z, p, shortfall, penalty)`; a test pins scalar == vectorised. Under `"met"` a zero-variance feature reports `p = 1.0`. `nonzero > 0` with `V ≤ 0` is not special-cased (it yields inf / NaN; `ChanceState` recomputes before it can happen). `evaluate_solution_chance(term=...)` raises `ValueError` for any term but `"2d"`.
- **Task 4:** `compute_zscore_penalty` no longer infers zero variance from `Z == 4`, so `test_sentinel_variance_means_deterministic_no_penalty` is rewritten already here (zero variance named via `zero_variance`, rule "met") and again in Task 7 for the new default; it keeps its name so spec §3.6 still resolves. `_compute_probmode3_zscore_penalty` delegates to `evaluate_solution_chance` (the second E / V loop is gone). The rule lives in `utils._zero_var_rule(problem)` (returns "met" until Task 7) and a new `ProblemCache.zero_var_rule` field.
- **Task 5:** per-cell arrays are a frozen `ChanceCells` in `cache.py` (`cache.chance_cells[term]`, built for active terms only); E and V terms are stacked (`ev`, 2 × nnz) with an `abs_ev` copy and a CSC copy `c_ev`, so one fancy index serves both on the hot path (−15 % per flip). Duplicate (pu, species) rows: amounts summed, the last row's `prob` wins, as in the dense matrix it replaces.
- **Task 5:** the snap applies on the transition to `nonzero == 0` (E recomputed exactly from the column, V = 0, BV = 0); while the count stays 0 the E bound keeps growing, so a later refill is caught by the E trigger. The recompute is a float matvec over the CSC column with the previewed PU's entry toggled (no full-selection copy). `n_recompute` (a diagnostic counter) is the only thing `delta_penalty` mutates. `probability.py` gains `_chance_penalty` / `_ptarget_den` for the hot path; a test pins `_chance_penalty == score_features(...)[3]` bit for bit. `chance_state.py` joins `INTERNAL_OR_COUNT` in `test_fractional_readers.py`.
- **Task 5:** the round-2 "p = 1e-17 beside a = 1e4" layout breaks running sums only when the feature's other cells are near-certain too; the test uses big (a = 1e4, q = 0.5) and tiny (q = 1e-17) cells only (recompute disabled → |delta − full| = 4.0). `sparse_risk` runs 2500 flips (its reference is a dense matvec), plus 3000 flips at 2000 PUs in `test_sparse_risk_recompute_floor`.
- **Task 6:** RED beyond the new API: the dense-matrix removal (`test_dense_chance_matrices_are_gone`, `test_chance_arrays_gated_on_chance_terms`) and the cache's full objective dropping the chance term (`test_full_objective_plus_chance_state_is_the_evaluator_objective`). The SA / II behaviour tests in `test_sa_chance.py` pass on the dense path too: they are regression guards, mutation-checked (removing any one wiring site fails 2–4 of them).
- **Task 6:** II has no observable tracked objective, so "reported == tracked" for II is: after each ITIMPTYPE every move the pass rejected also fails under the evaluator (a local optimum of the reported objective), plus a removal-pass case and a swap-pass case only the chance term decides. The swap pass's from-scratch term is `ChanceState.from_selection(...).penalty_total()` per candidate.
- **Task 6, bench:** layout ≈ 5 feature cells per PU, amounts U(0.1, 3), dense risk q = 1 − U(0, 0.3), sparse risk 1 % of PUs at q = 0.95; ratio = best of 3 × 6000 flips, 60 % accepted. Measured 4.50× / 4.72× (2000 × 20, dense / sparse), 4.54× / 4.61× (10 000 × 50); the dense-matrix delta it replaces was 17× and 103×. **Spec correction:** on a dense layout (60 % of cells present, ~30 features per PU) at 10 000 × 50 the E trigger fires about once per 33 committed updates per feature and the ratio is 5.6×, close to the 6× bound; spec §3.5's 4.9–5.1× holds for the sparse layout only. The two Phase 18 absolute smoke budgets (500 PUs) are kept, re-pointed at `ChanceState`.
- **Task 7:** `PROBZEROVAR` parses numerically (0 / 0.0 → "marxan"; 1 / 1.0 / "1" → "met"); anything else falls back to "marxan" in the pure `_zero_var_rule` and is an error in `validate()`, which imports the shared parser lazily (`problem.py` must not import solvers at load). `probability.DEFAULT_ZERO_VAR_RULE` is now "marxan". Tests written in Tasks 4 / 6 that relied on the "met" default name `PROBZEROVAR=1`.
- **Task 7:** the Marxan 3 transcription test ports `ReturnProbabilityAmounts2D` + `ComputeProbability2D` + `probZUT` and compares P at 1e-6, totals at weight·1e-6 / min(ptarget); all |z| < 5. The manual drops "zero variance is treated as met" and the zero-variance caveat on the "prob = 1 − p" 1D workaround, and gains a `PROBZEROVAR` paragraph; the rest of the 1D text is release B. The CHANGELOG also lists the API changes direct `ProblemCache` / `probability.py` users see (no `expected_matrix` / `var_matrix`, full objective without the chance term, new keyword-only arguments, `Solution` fields None without data). `base.py`'s `Solution` comment loses its wrong "γ · Σ SPF ·" formula.
- **Gate:** `pyproject` collects `bench_*.py`; the gate (as the Makefile and CI do) passes `-m "not bench"`.

---

### Task 1: `write_puvspr` always writes a present `prob` column

**Files:**
- Modify: `src/pymarxan/io/writers.py`
- Modify: `tests/pymarxan/io/test_io_probability.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `write_puvspr` writes the `prob` column whenever `pu_vs_features` has one (spec §3.2, final-review I-7), so presence 0 no longer reloads as presence 1 and an all-1 column survives (under Task 7's rule it is not a no-op).

- [ ] **Step 1: Write the failing tests**

Modify `tests/pymarxan/io/test_io_probability.py` (apply this diff):

```diff
diff --git a/tests/pymarxan/io/test_io_probability.py b/tests/pymarxan/io/test_io_probability.py
index 2ee3dea..924c7e8 100644
--- a/tests/pymarxan/io/test_io_probability.py
+++ b/tests/pymarxan/io/test_io_probability.py
@@ -31,10 +31,12 @@ def test_puvspr_round_trip_with_prob(tmp_path):
     np.testing.assert_array_almost_equal(read["prob"].values, [0.10, 0.05, 0.0])
 
 
-def test_puvspr_no_prob_when_all_zero(tmp_path):
-    """The writer omits the `prob` column when every row is 0.
+def test_puvspr_all_zero_prob_is_written(tmp_path):
+    """An all-zero `prob` column is written, not dropped.
 
-    Keeps legacy projects byte-identical on round-trip.
+    Rewritten (v0.47.0): the old test pinned a bug. The writer dropped an
+    all-zero column while the readers treat a missing `prob` as presence 1.0,
+    so presence 0 round-tripped as presence 1 (spec §3.2, round 1 M3).
     """
     df = pd.DataFrame({
         "species": [1], "pu": [1], "amount": [10.0], "prob": [0.0],
@@ -42,8 +44,37 @@ def test_puvspr_no_prob_when_all_zero(tmp_path):
     out = tmp_path / "puvspr.dat"
     write_puvspr(df, out)
 
-    content = out.read_text()
-    assert "prob" not in content, "writer must omit all-zero prob column"
+    assert "prob" in out.read_text().splitlines()[0]
+
+
+def test_puvspr_all_zero_prob_round_trips(tmp_path):
+    """Presence 0 stays presence 0 through write → read."""
+    df = pd.DataFrame({
+        "species": [1, 1], "pu": [1, 2], "amount": [10.0, 8.0], "prob": [0.0, 0.0],
+    })
+    out = tmp_path / "puvspr.dat"
+    write_puvspr(df, out)
+
+    read = read_puvspr(out)
+    assert "prob" in read.columns
+    np.testing.assert_array_equal(read["prob"].values, [0.0, 0.0])
+
+
+def test_puvspr_all_one_prob_round_trips(tmp_path):
+    """An all-1 `prob` column is kept: its header switches Marxan's 2D term on.
+
+    Under the Z = 4 zero-variance rule an all-1 column is not a no-op
+    (spec §3.2, final review I-7), so dropping it would change the problem.
+    """
+    df = pd.DataFrame({
+        "species": [1, 2], "pu": [1, 1], "amount": [10.0, 8.0], "prob": [1.0, 1.0],
+    })
+    out = tmp_path / "puvspr.dat"
+    write_puvspr(df, out)
+
+    read = read_puvspr(out)
+    assert "prob" in read.columns
+    np.testing.assert_array_equal(read["prob"].values, [1.0, 1.0])
 
 
 def test_puvspr_legacy_file_loads_without_prob(tmp_path):
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/io/test_io_probability.py -q -m 'not bench'`
Expected: FAIL — 2 failed, 6 passed: `test_puvspr_all_zero_prob_is_written` and `test_puvspr_all_zero_prob_round_trips` (the column is absent after writing). `test_puvspr_no_prob_when_all_zero` asserted the bug and is replaced (reason in its successor's docstring); `test_puvspr_all_one_prob_round_trips` already passes (a guard).

- [ ] **Step 3: Implement**

Modify `src/pymarxan/io/writers.py` (apply this diff):

```diff
diff --git a/src/pymarxan/io/writers.py b/src/pymarxan/io/writers.py
index 564e49a..1cf3b8b 100644
--- a/src/pymarxan/io/writers.py
+++ b/src/pymarxan/io/writers.py
@@ -62,9 +62,11 @@ def write_spec(df: pd.DataFrame, path: str | Path) -> None:
 def write_puvspr(df: pd.DataFrame, path: str | Path) -> None:
     """Write a planning-unit-vs-species DataFrame to a CSV file.
 
-    The optional ``prob`` column (Marxan PROB2D — per-cell Bernoulli
-    probability) is omitted from the output when every value is zero,
-    so legacy deterministic projects round-trip byte-identical.
+    The optional ``prob`` column (Marxan PROB2D — per-cell probability of
+    presence) is always written when present, whatever its values: the readers
+    treat a missing column as presence 1.0, so dropping an all-zero column
+    turned presence 0 into presence 1 on reload, and Marxan 3 switches its 2D
+    term on from a ``prob`` header, so even an all-1 column is not a no-op.
 
     Parameters
     ----------
@@ -73,8 +75,6 @@ def write_puvspr(df: pd.DataFrame, path: str | Path) -> None:
     path : str | Path
         Output file path.
     """
-    if "prob" in df.columns and (df["prob"] == 0).all():
-        df = df.drop(columns=["prob"])
     df.to_csv(path, index=False)
 
 
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/io/test_io_probability.py -q -m 'not bench'`
Expected: 8 passed

- [ ] **Step 5: Gate**

Run: `~/.local/bin/ruff check src tests examples && /home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/ -q -m 'not bench' -p no:cacheprovider && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/python examples/validate_marxan_parity.py`
Expected: ruff and mypy clean; the full suite passes (prototype: 3780 passed, 101 skipped HiGHS cases which run in CI); `examples/validate_marxan_parity.py` prints MIP 35.0, SA 43.0, greedy 45.0; no golden JSON regenerated.

- [ ] **Step 6: Commit**

```bash
git add -A src/pymarxan/io/writers.py tests/pymarxan/io/test_io_probability.py
git commit -F - <<'MSG'
fix(io): write_puvspr always writes a present prob column (A0a)

write_puvspr dropped an all-zero `prob` column while the readers default
a missing column to presence 1.0, so presence 0 round-tripped as presence
1. It now writes the column whenever it is present: under the Z = 4
zero-variance rule an all-1 column is not a no-op either, and Marxan 3
switches its 2D term on from a `prob` header (spec §3.2, final review I-7).

test_puvspr_no_prob_when_all_zero asserted the bug and is replaced by
all-zero and all-one round-trip tests.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01G9s5x1nmcqSTDeRUXcRPVW
MSG
```

---

### Task 2: `run_sweep_parallel` passes the whole problem

**Files:**
- Modify: `src/pymarxan/calibration/parallel.py`
- Modify: `tests/pymarxan/calibration/test_parallel.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `calibration/parallel.py` sends each worker the pickled problem (subclass type, `probability`, `connectivity` kept) with the point's parameters applied via `copy_with`, so `run_sweep` and `run_sweep_parallel` agree (spec §3.1 "Fix while here", round 4 I-2).

- [ ] **Step 1: Write the failing tests**

Modify `tests/pymarxan/calibration/test_parallel.py` (apply this diff):

```diff
diff --git a/tests/pymarxan/calibration/test_parallel.py b/tests/pymarxan/calibration/test_parallel.py
index 3ee702d..5c1f2d0 100644
--- a/tests/pymarxan/calibration/test_parallel.py
+++ b/tests/pymarxan/calibration/test_parallel.py
@@ -120,3 +120,108 @@ def test_parallel_sweep_skips_infeasible(small_problem: ConservationProblem):
     # Surviving points should be BLM=1.0 and BLM=5.0
     blms = sorted(p["BLM"] for p in result.param_dicts)
     assert blms == [1.0, 5.0]
+
+
+# --- v0.47.0: workers receive the whole problem (spec §3.1, round 4 I-2) ----
+
+
+class _TaggedProblem(ConservationProblem):
+    """A subclass, so the test sees whether the workers keep the problem's type."""
+
+
+class _RecordingSolver(Solver):
+    """Records what the worker received in the solution metadata."""
+
+    def solve(
+        self, problem: ConservationProblem, config: SolverConfig | None = None
+    ) -> list[Solution]:
+        n = problem.n_planning_units
+        return [
+            Solution(
+                selected=np.ones(n, dtype=bool),
+                cost=1.0,
+                boundary=0.0,
+                objective=1.0,
+                targets_met={1: True},
+                metadata={
+                    "type": type(problem).__name__,
+                    "has_probability": problem.probability is not None,
+                    "has_connectivity": problem.connectivity is not None,
+                },
+            )
+        ]
+
+    def name(self) -> str:
+        return "recording"
+
+    def supports_zones(self) -> bool:
+        return False
+
+
+def _mode1_problem() -> ConservationProblem:
+    """PROBMODE 1 (risk premium) from a prob.dat table: the premium is in the objective."""
+    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [10.0, 12.0, 8.0], "status": [0, 0, 0]})
+    feat = pd.DataFrame({"id": [1], "name": ["f1"], "target": [5.0], "spf": [100.0]})
+    puvspr = pd.DataFrame({"species": [1, 1, 1], "pu": [1, 2, 3], "amount": [3.0, 4.0, 5.0]})
+    prob = pd.DataFrame({"pu": [1, 2, 3], "probability": [0.5, 0.1, 0.9]})
+    return ConservationProblem(
+        planning_units=pu, features=feat, pu_vs_features=puvspr, probability=prob,
+        parameters={"PROBMODE": 1, "PROBABILITYWEIGHTING": 2.0},
+    )
+
+
+def test_parallel_sweep_keeps_probability_table():
+    """A prob.dat-driven objective agrees between run_sweep and run_sweep_parallel.
+
+    Before v0.47.0 the workers rebuilt the problem from five fields and dropped
+    ``problem.probability``, so the PROBMODE 1 premium vanished in the workers.
+    """
+    from pymarxan.calibration.sweep import run_sweep
+    from pymarxan.solvers.heuristic import HeuristicSolver
+
+    problem = _mode1_problem()
+    config = SweepConfig(
+        param_dicts=[{"BLM": 0.0}, {"BLM": 1.0}],
+        # Seeded: the greedy heuristic breaks ties with random noise.
+        solver_config=SolverConfig(num_solutions=1, seed=42),
+    )
+    seq = run_sweep(problem, HeuristicSolver(), config)
+    par = run_sweep_parallel(problem, HeuristicSolver(), config, max_workers=2)
+    assert par.objectives == pytest.approx(seq.objectives, rel=1e-12)
+    assert [s.objective for s in par.solutions] == pytest.approx(
+        [s.objective for s in seq.solutions], rel=1e-12,
+    )
+
+
+def test_parallel_sweep_keeps_subclass_and_optional_tables():
+    """The workers see the problem's subclass, probability and connectivity tables."""
+    base = _mode1_problem()
+    problem = _TaggedProblem(
+        planning_units=base.planning_units,
+        features=base.features,
+        pu_vs_features=base.pu_vs_features,
+        probability=base.probability,
+        connectivity=pd.DataFrame({"id1": [1], "id2": [2], "value": [1.0]}),
+        parameters=dict(base.parameters),
+    )
+    config = SweepConfig(param_dicts=[{"BLM": 0.0}, {"BLM": 1.0}])
+    result = run_sweep_parallel(problem, _RecordingSolver(), config, max_workers=2)
+    assert len(result.solutions) == 2
+    for sol in result.solutions:
+        assert sol.metadata == {
+            "type": "_TaggedProblem",
+            "has_probability": True,
+            "has_connectivity": True,
+        }
+
+
+def test_parallel_sweep_applies_point_parameters_on_whole_problem(
+    small_problem: ConservationProblem,
+):
+    """Each point's parameters still override the problem's, and the caller's
+    problem is left untouched."""
+    small_problem.parameters["BLM"] = 99.0
+    config = SweepConfig(param_dicts=[{"BLM": 0.0}, {"BLM": 5.0}])
+    result = run_sweep_parallel(small_problem, _StubSolver(), config, max_workers=2)
+    assert result.costs == [10.0, 15.0]
+    assert small_problem.parameters["BLM"] == 99.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/calibration/test_parallel.py -q -m 'not bench'`
Expected: FAIL — 2 failed, 5 passed: `test_parallel_sweep_keeps_probability_table` (parallel 8.0 vs sequential 34.4: the PROBMODE 1 premium vanished in the workers) and `test_parallel_sweep_keeps_subclass_and_optional_tables` (workers received a plain `ConservationProblem`).

- [ ] **Step 3: Implement**

Modify `src/pymarxan/calibration/parallel.py` (apply this diff):

```diff
diff --git a/src/pymarxan/calibration/parallel.py b/src/pymarxan/calibration/parallel.py
index 16a03a7..362ebb7 100644
--- a/src/pymarxan/calibration/parallel.py
+++ b/src/pymarxan/calibration/parallel.py
@@ -12,28 +12,19 @@ from pymarxan.solvers.base import Solution, Solver, SolverConfig
 
 
 def _solve_single(
-    problem_data: dict,
+    problem: ConservationProblem,
     solver: Solver,
     params: dict,
     solver_config: SolverConfig,
     index: int,
 ) -> tuple[int, Solution]:
-    """Solve a single sweep point. Returns (index, best_solution)."""
-    import pandas as pd
-
-    from pymarxan.models.problem import ConservationProblem
+    """Solve a single sweep point. Returns (index, best_solution).
 
-    modified = ConservationProblem(
-        planning_units=pd.DataFrame(problem_data["planning_units"]),
-        features=pd.DataFrame(problem_data["features"]),
-        pu_vs_features=pd.DataFrame(problem_data["pu_vs_features"]),
-        boundary=(
-            pd.DataFrame(problem_data["boundary"])
-            if problem_data["boundary"] is not None
-            else None
-        ),
-        parameters={**problem_data["parameters"], **params},
-    )
+    The whole problem is pickled to the worker and derived with ``copy_with``, exactly
+    as ``run_sweep`` does, so its subclass and every optional table (probability,
+    connectivity, ...) survive the trip.
+    """
+    modified = problem.copy_with(parameters={**problem.parameters, **params})
     sols = solver.solve(modified, solver_config)
     if not sols:
         raise ValueError("Solver returned no solutions (infeasible)")
@@ -57,26 +48,13 @@ def run_sweep_parallel(
     solver_config = config.solver_config or SolverConfig(num_solutions=1)
     param_dicts = config.expand()
 
-    # Serialise problem data for pickling across processes
-    problem_data = {
-        "planning_units": problem.planning_units.to_dict(orient="list"),
-        "features": problem.features.to_dict(orient="list"),
-        "pu_vs_features": problem.pu_vs_features.to_dict(orient="list"),
-        "boundary": (
-            problem.boundary.to_dict(orient="list")
-            if problem.boundary is not None
-            else None
-        ),
-        "parameters": dict(problem.parameters),
-    }
-
     # Submit all jobs; collect successful results and drop infeasible points
     # (mirrors sequential run_sweep, which skips with `if not sols: continue`).
     indexed_results: dict[int, Solution] = {}
     with ProcessPoolExecutor(max_workers=max_workers) as executor:
         futures = {
             executor.submit(
-                _solve_single, problem_data, solver, params, solver_config, i
+                _solve_single, problem, solver, params, solver_config, i
             ): i
             for i, params in enumerate(param_dicts)
         }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/calibration/test_parallel.py -q -m 'not bench'`
Expected: 7 passed

- [ ] **Step 5: Gate**

Run: `~/.local/bin/ruff check src tests examples && /home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/ -q -m 'not bench' -p no:cacheprovider && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/python examples/validate_marxan_parity.py`
Expected: ruff and mypy clean; the full suite passes (prototype: 3783 passed, 101 skipped HiGHS cases which run in CI); `examples/validate_marxan_parity.py` prints MIP 35.0, SA 43.0, greedy 45.0; no golden JSON regenerated.

- [ ] **Step 6: Commit**

```bash
git add -A src/pymarxan/calibration/parallel.py tests/pymarxan/calibration/test_parallel.py
git commit -F - <<'MSG'
fix(calibration): run_sweep_parallel passes the whole problem (A0b)

_solve_single rebuilt ConservationProblem from five fields in each worker,
dropping problem.probability, problem.connectivity and the subclass type,
so a PROBMODE 1 sweep lost its risk premium in the workers and
run_sweep_parallel disagreed with run_sweep (spec §3.1, round 4 I-2).
The problem is now pickled whole and derived with copy_with, as run_sweep
does.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01G9s5x1nmcqSTDeRUXcRPVW
MSG
```

---

### Task 3: Requested vs active chance terms (`_chance_terms` / `_active_chance_terms`)

**Files:**
- Modify: `src/pymarxan/multiobjective.py`
- Modify: `src/pymarxan/solvers/cache.py`
- Modify: `src/pymarxan/solvers/mip_solver.py`
- Modify: `src/pymarxan/solvers/utils.py`
- Modify: `src/pymarxan_shiny/modules/results/target_met.py`
- Modify: `src/pymarxan_shiny/modules/run_control/run_panel.py`
- Create: `tests/pymarxan/solvers/test_chance_terms.py`

**Interfaces:**
- Consumes: nothing new.
- Produces (`solvers/utils.py`): `_chance_terms(problem) -> frozenset[str]` (PROBMODE 3 → `frozenset({"2d"})`, every other mode → `frozenset()` in release A) and `_active_chance_terms(problem) -> frozenset[str]` (requested and `pu_vs_features` has a `prob` column). `ProblemCache.chance_terms: frozenset[str]` (active). MIP `_BaseModel` / `_BuiltModel.chance_terms` (requested); `_strategy_metadata(..., chance_terms)`. Requested gates: non-binary rejection, MIP strategy gate / metadata, `run_panel` notice. Active gates: the evaluator penalty, the cache, `Solution.prob_shortfalls` / `prob_penalty`, target-met columns.

A pure refactor apart from one reporting detail (Rulings, Task 3): the existing suite is the regression guard.

- [ ] **Step 1: Write the failing tests**

Create `tests/pymarxan/solvers/test_chance_terms.py`:

```python
"""``_chance_terms`` / ``_active_chance_terms`` (spec §3.1, final review I-4).

Release A (v0.47.0) knows one chance term, ``"2d"`` (PROBMODE 3, from the puvspr
``prob`` column). *Requested* terms follow the mode alone; *active* terms are the
requested ones whose data exists, so a term without its data is off (penalty 0).
"""
from __future__ import annotations

import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.utils import _active_chance_terms, _chance_terms


def _problem(params: dict, *, prob_column: bool = True, table: bool = False):
    pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 2.0], "status": [0, 0]})
    feat = pd.DataFrame({
        "id": [1], "name": ["a"], "target": [5.0], "spf": [1.0], "ptarget": [0.9],
    })
    pv = {"species": [1, 1], "pu": [1, 2], "amount": [3.0, 4.0]}
    if prob_column:
        pv["prob"] = [0.5, 0.8]
    probability = (
        pd.DataFrame({"pu": [1, 2], "probability": [0.1, 0.2]}) if table else None
    )
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pd.DataFrame(pv),
        probability=probability, parameters=params,
    )


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_modes_without_a_chance_term_request_none(mode: int) -> None:
    p = _problem({"PROBMODE": mode})
    assert _chance_terms(p) == frozenset()
    assert _active_chance_terms(p) == frozenset()


def test_probmode_3_requests_the_2d_term() -> None:
    p = _problem({"PROBMODE": 3})
    assert _chance_terms(p) == frozenset({"2d"})
    assert _active_chance_terms(p) == frozenset({"2d"})


def test_2d_term_without_puvspr_prob_is_requested_but_not_active() -> None:
    p = _problem({"PROBMODE": 3}, prob_column=False)
    assert _chance_terms(p) == frozenset({"2d"})
    assert _active_chance_terms(p) == frozenset()


@pytest.mark.parametrize("mode", [4, 5, 7, -1])
def test_other_modes_request_nothing_in_release_a(mode: int) -> None:
    """Modes 4 / 5 (Probability 1D) arrive in release B; until then they request nothing."""
    p = _problem({"PROBMODE": mode})
    assert _chance_terms(p) == frozenset()
    assert _active_chance_terms(p) == frozenset()


def test_default_mode_follows_probmode_rule() -> None:
    """No PROBMODE: the v0.45 default (1 with a table, else 0) — no chance term."""
    assert _chance_terms(_problem({})) == frozenset()
    assert _chance_terms(_problem({}, table=True)) == frozenset()


def test_helpers_return_frozensets() -> None:
    p = _problem({"PROBMODE": 3})
    assert isinstance(_chance_terms(p), frozenset)
    assert isinstance(_active_chance_terms(p), frozenset)


def test_cache_exposes_active_terms() -> None:
    from pymarxan.solvers.cache import ProblemCache

    assert ProblemCache.from_problem(_problem({"PROBMODE": 3})).chance_terms == {"2d"}
    no_data = _problem({"PROBMODE": 3}, prob_column=False)
    assert ProblemCache.from_problem(no_data).chance_terms == frozenset()
    assert ProblemCache.from_problem(_problem({"PROBMODE": 0})).chance_terms == frozenset()


def test_solution_fields_follow_active_terms() -> None:
    """``Solution.prob_shortfalls`` / ``prob_penalty`` are None unless 2D is active."""
    import numpy as np

    from pymarxan.solvers.utils import build_solution

    sel = np.array([True, True])
    on = build_solution(_problem({"PROBMODE": 3}), sel, 0.0)
    assert on.prob_shortfalls is not None and on.prob_penalty is not None
    off = build_solution(_problem({"PROBMODE": 3}, prob_column=False), sel, 0.0)
    assert off.prob_shortfalls is None and off.prob_penalty is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_chance_terms.py -q -m 'not bench'`
Expected: ERROR — 1 error collecting `test_chance_terms.py` (ImportError: cannot import `_chance_terms`).

- [ ] **Step 3: Implement**

Modify `src/pymarxan/multiobjective.py` (apply this diff):

```diff
diff --git a/src/pymarxan/multiobjective.py b/src/pymarxan/multiobjective.py
index 98438eb..dc44d20 100644
--- a/src/pymarxan/multiobjective.py
+++ b/src/pymarxan/multiobjective.py
@@ -646,7 +646,9 @@ class MultiObjectiveMIP:
                 objective_senses=dict(senses), mip_backend=backend,
             )
             meta.update(
-                mip._strategy_metadata(base.probmode, base.has_target2, base.has_sep_active)
+                mip._strategy_metadata(
+                    base.chance_terms, base.has_target2, base.has_sep_active,
+                )
             )
             solutions.append(build_solution(problem, selected, base.blm, metadata=meta))
         return solutions
```

Modify `src/pymarxan/solvers/cache.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/cache.py b/src/pymarxan/solvers/cache.py
index f856229..c005203 100644
--- a/src/pymarxan/solvers/cache.py
+++ b/src/pymarxan/solvers/cache.py
@@ -57,6 +57,8 @@ from scipy.sparse import coo_matrix, csr_matrix
 
 from pymarxan.models.problem import ConservationProblem
 from pymarxan.solvers.utils import (
+    _active_chance_terms,
+    _chance_terms,
     _probability_vector,
     _probmode,
     compute_cost_threshold_penalty,
@@ -134,6 +136,9 @@ class ProblemCache:
     # but only used when probmode == 3. See solvers/probability.py for the math.
     probmode: int = 0
     prob_weight: float = 1.0
+    # Active chance terms (``_active_chance_terms``: requested by PROBMODE and with data);
+    # ``"2d"`` = the Z-score term from the puvspr ``prob`` column.
+    chance_terms: frozenset[str] = frozenset()
     # expected_matrix[i, j] = amount_ij · p_ij (presence); collapses to amount
     # when there is no `prob` column or when prob == 1.
     expected_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
@@ -316,6 +321,7 @@ class ProblemCache:
         # Mode 3 state below is zero-cost when probmode != 3.
         probmode = _probmode(problem)
         prob_weight = float(params.get("PROBABILITYWEIGHTING", 1.0))
+        chance_terms = _active_chance_terms(problem)
 
         # Per-cell probability of PRESENCE — read once into a dense matrix.
         # Marxan's PROB2D `prob` is the probability the feature is present at
@@ -328,7 +334,7 @@ class ProblemCache:
         # amount is `amount * (1 - prob)`.
         # expected/var are read ONLY under probmode == 3 — build them (densely) only then;
         # a plain problem pays zero memory for these two (n_pu, n_feat) matrices.
-        if probmode == 3:
+        if "2d" in _chance_terms(problem):
             dense = np.asarray(pu_feat_csr.toarray())
             prob_matrix = np.ones_like(dense)
             if (
@@ -545,6 +551,7 @@ class ProblemCache:
             thresh_pen2=thresh_pen2,
             probmode=probmode,
             prob_weight=prob_weight,
+            chance_terms=chance_terms,
             expected_matrix=expected_matrix,
             var_matrix=var_matrix,
             feat_ptarget=feat_ptarget,
@@ -616,12 +623,13 @@ class ProblemCache:
     ) -> float:
         """Marxan-faithful Z-score chance-constraint penalty.
 
-        Active only when ``self.probmode == 3``; returns 0 otherwise (the
+        Active only when the 2D chance term is (``"2d" in self.chance_terms``); returns 0
+        otherwise (the
         caller's hot path can skip this branch entirely via the probmode
         check).  Features with ptarget ≤ 0 are skipped.  Features with
         zero variance contribute 0 (Marxan's "degenerate → P=1").
         """
-        if self.probmode != 3 or self.n_feat == 0:
+        if "2d" not in self.chance_terms or self.n_feat == 0:
             return 0.0
         from scipy.stats import norm  # local import keeps cold path cheap
         total = 0.0
@@ -695,8 +703,8 @@ class ProblemCache:
                 total_cost, self.cost_thresh, self.thresh_pen1, self.thresh_pen2
             )
 
-        # PROBMODE 3 chance-constraint penalty (additive; no-op when probmode!=3)
-        if self.probmode == 3:
+        # PROBMODE 3 chance-constraint penalty (additive; no-op unless the 2D term is active)
+        if "2d" in self.chance_terms:
             held_expected = self.expected_matrix[selected].sum(axis=0)
             held_variance = self.var_matrix[selected].sum(axis=0)
             obj += self._compute_zscore_penalty(held_expected, held_variance)
@@ -831,7 +839,7 @@ class ProblemCache:
         # the deterministic delta but cheap relative to the SA outer loop.
         # The future optimisation (sparse per-PU feature index) lands in a
         # later patch if benchmarks demand it.
-        if self.probmode == 3:
+        if "2d" in self.chance_terms:
             held_exp_before = self.expected_matrix[selected].sum(axis=0)
             held_var_before = self.var_matrix[selected].sum(axis=0)
             held_exp_after = held_exp_before + sign * self.expected_matrix[idx]
```

Modify `src/pymarxan/solvers/mip_solver.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/mip_solver.py b/src/pymarxan/solvers/mip_solver.py
index cb75929..2fd0e6c 100644
--- a/src/pymarxan/solvers/mip_solver.py
+++ b/src/pymarxan/solvers/mip_solver.py
@@ -58,6 +58,7 @@ from pymarxan.solvers.presolve import (
     _validate_presolve_flag,
 )
 from pymarxan.solvers.utils import (
+    _chance_terms,
     _nonbinary_conflicts,
     _probability_map,
     _probmode,
@@ -73,7 +74,7 @@ def _reject_nonbinary(
     if decisions == "binary":
         return
     causes: list[str] = []
-    if _probmode(problem) == 3:
+    if _chance_terms(problem):
         causes.append("PROBMODE 3 (Z-score chance constraints)")
     causes += _nonbinary_conflicts(problem)
     if config.metadata and config.metadata.get("forbidden_selections"):
@@ -293,6 +294,7 @@ class _BaseModel:
     terms: list[TargetTerm]
     implied: list[ImpliedRule]
     probmode: int
+    chance_terms: frozenset[str]  # requested (``_chance_terms``)
     has_target2: bool
     has_sep_active: bool
     s: dict[int, pulp.LpVariable]  # semicontinuous on/off binaries; empty otherwise
@@ -324,6 +326,7 @@ class _BuiltModel:
     blm: float
     objective_expr: pulp.LpAffineExpression  # natural units
     probmode: int
+    chance_terms: frozenset[str]  # requested (``_chance_terms``)
     has_target2: bool
     has_sep_active: bool
     s: dict[int, pulp.LpVariable]
@@ -476,7 +479,8 @@ class MIPSolver(Solver):
 
         # PROBMODE 3 gating — see class docstring for rationale.
         probmode = _probmode(problem)
-        if probmode == 3:
+        chance_terms = _chance_terms(problem)
+        if chance_terms:
             if self.mip_chance_strategy == "piecewise":
                 raise NotImplementedError(
                     "mip_chance_strategy='piecewise' (piecewise-linear "
@@ -719,6 +723,7 @@ class MIPSolver(Solver):
             model=model, x=x, pu_ids=pu_ids, blm=blm, cost_expr=cost_expr,
             boundary_expr=boundary_expr, conn_expr=conn_expr, prob_expr=prob_expr,
             amount_exprs=amount_exprs, terms=terms, implied=implied, probmode=probmode,
+            chance_terms=chance_terms,
             has_target2=bool(has_target2), has_sep_active=bool(has_sep_active),
             s=s, decisions=self.decisions, upper=self.upper, lower=self.lower,
         )
@@ -767,18 +772,19 @@ class MIPSolver(Solver):
             model += base.cost_expr <= float(cost_budget), "cost_budget"
         return _BuiltModel(
             model=model, x=base.x, pu_ids=base.pu_ids, blm=base.blm, objective_expr=term.expr,
-            probmode=base.probmode, has_target2=base.has_target2,
+            probmode=base.probmode, chance_terms=base.chance_terms,
+            has_target2=base.has_target2,
             has_sep_active=base.has_sep_active, s=base.s, decisions=base.decisions,
             upper=base.upper, lower=base.lower, lex=lex,
         )
 
     def _strategy_metadata(
-        self, probmode: int, has_target2: bool, has_sep_active: bool,
+        self, chance_terms: frozenset[str], has_target2: bool, has_sep_active: bool,
     ) -> dict[str, str]:
         """The ``mip_*_strategy`` keys for the gaps this problem has (shared with
         ``MultiObjectiveMIP``)."""
         meta: dict[str, str] = {}
-        if probmode == 3:
+        if chance_terms:
             meta["mip_chance_strategy"] = self.mip_chance_strategy
         if has_target2:
             meta["mip_clump_strategy"] = self.mip_clump_strategy
@@ -818,7 +824,7 @@ class MIPSolver(Solver):
         built = self._build(problem, config)
         _emit_presolve(self.presolve_check, data, built.model, self.name(), stacklevel=3)
         model, x, pu_ids = built.model, built.x, built.pu_ids
-        probmode, has_target2 = built.probmode, built.has_target2
+        has_target2 = built.has_target2
         has_sep_active, blm = built.has_sep_active, built.blm
 
         solver, resolved_backend = _mip_solver_for(problem, self.mip_backend, config.verbose)
@@ -860,6 +866,8 @@ class MIPSolver(Solver):
             "mip_objective_value": obj_value,
             "shortfall": self.shortfall,
         }
-        meta.update(self._strategy_metadata(probmode, has_target2, has_sep_active))
+        meta.update(
+            self._strategy_metadata(built.chance_terms, has_target2, has_sep_active)
+        )
         sol = build_solution(problem, selected, blm, metadata=meta)
         return [copy.deepcopy(sol) for _ in range(config.num_solutions)]
```

Modify `src/pymarxan/solvers/utils.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/utils.py b/src/pymarxan/solvers/utils.py
index 7931373..9dd3a17 100644
--- a/src/pymarxan/solvers/utils.py
+++ b/src/pymarxan/solvers/utils.py
@@ -98,6 +98,32 @@ def _probmode(problem: ConservationProblem) -> int:
     return int(problem.parameters.get("PROBMODE", default))
 
 
+def _chance_terms(problem: ConservationProblem) -> frozenset[str]:
+    """Chance terms **requested** by ``PROBMODE`` (spec §3.1): 3 → ``{"2d"}``, else none.
+
+    ``"2d"`` is the Z-score chance term from the puvspr ``prob`` column (Marxan
+    Probability 2D). Requested terms gate what depends on the mode alone: non-binary
+    rejection, the MIP chance-strategy gate and metadata, the Shiny MIP notice. Pure; never
+    warns.
+    """
+    if _probmode(problem) == 3:
+        return frozenset({"2d"})
+    return frozenset()
+
+
+def _active_chance_terms(problem: ConservationProblem) -> frozenset[str]:
+    """Requested chance terms whose data exists (spec §3.1 "term without data").
+
+    ``"2d"`` needs a ``prob`` column on ``pu_vs_features``; without one the term is off
+    (penalty 0). Active terms gate the evaluator, ``ProblemCache``, the ``Solution``
+    fields and the reports. Pure; never warns.
+    """
+    terms = _chance_terms(problem)
+    if "2d" in terms and "prob" not in problem.pu_vs_features.columns:
+        terms = terms - {"2d"}
+    return terms
+
+
 def _probmode2_active(problem: ConservationProblem) -> bool:
     """True iff PROBMODE 2 applies: a probability table and ``PROBMODE == 2``."""
     return problem.probability is not None and _probmode(problem) == 2
@@ -457,14 +483,17 @@ def compute_probability_penalty(
     prob_mode = _probmode(problem)
     if prob_mode == 1:
         return _compute_probmode1_risk_premium(problem, selected)
-    if prob_mode == 3:
+    # Two gates: a *requested* chance term rejects proportion decisions whether or not its
+    # data exists; only an *active* one (requested, with data) adds a penalty.
+    if _chance_terms(problem):
         if _is_fractional(selected):
             raise ValueError(
                 "PROBMODE 3 (Z-score chance constraints) is only defined for binary "
                 "selections; this selection holds proportion decisions. Solve with "
                 "decisions='binary' or convert with binarize_solution()."
             )
-        return _compute_probmode3_zscore_penalty(problem, selected)
+        if "2d" in _active_chance_terms(problem):
+            return _compute_probmode3_zscore_penalty(problem, selected)
     return 0.0
 
 
@@ -713,7 +742,7 @@ def build_solution(
     # via compute_probability_penalty.
     prob_shortfalls: dict[int, float] | None = None
     prob_penalty_val: float | None = None
-    if _probmode(problem) == 3:
+    if "2d" in _active_chance_terms(problem):
         from pymarxan.solvers.probability import evaluate_solution_chance
         prob_shortfalls, prob_penalty_val = evaluate_solution_chance(
             problem, selected,
```

Modify `src/pymarxan_shiny/modules/results/target_met.py` (apply this diff):

```diff
diff --git a/src/pymarxan_shiny/modules/results/target_met.py b/src/pymarxan_shiny/modules/results/target_met.py
index 8479103..cb0343b 100644
--- a/src/pymarxan_shiny/modules/results/target_met.py
+++ b/src/pymarxan_shiny/modules/results/target_met.py
@@ -6,7 +6,7 @@ from __future__ import annotations
 
 from shiny import Inputs, Outputs, Session, module, reactive, render, ui
 
-from pymarxan.solvers.utils import _probmode
+from pymarxan.solvers.utils import _active_chance_terms
 from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup
 
 
@@ -43,13 +43,13 @@ def target_met_server(
         if p is None or s is None:
             return None
 
-        # Show probability-target columns only when PROBMODE 3 is active
+        # Show probability-target columns only when the PROBMODE 3 (2D) chance term is active
         # and the solution actually carries a Z-score evaluation. Show
         # clumping columns only when any feature has target2 > 0 and the
         # solution carries a clump evaluation. Same for separation when
         # any feature is sep-active. Keeps the table tidy for legacy runs.
         is_probmode3 = (
-            _probmode(p) == 3
+            "2d" in _active_chance_terms(p)
             and s.prob_shortfalls is not None
         )
         has_clumping = (
```

Modify `src/pymarxan_shiny/modules/run_control/run_panel.py` (apply this diff):

```diff
diff --git a/src/pymarxan_shiny/modules/run_control/run_panel.py b/src/pymarxan_shiny/modules/run_control/run_panel.py
index e700dbc..802a12e 100644
--- a/src/pymarxan_shiny/modules/run_control/run_panel.py
+++ b/src/pymarxan_shiny/modules/run_control/run_panel.py
@@ -7,7 +7,7 @@ import threading
 from shiny import Inputs, Outputs, Session, module, reactive, render, ui
 
 from pymarxan.solvers.base import SolverConfig
-from pymarxan.solvers.utils import _probmode
+from pymarxan.solvers.utils import _chance_terms
 from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup
 from pymarxan_shiny.modules.run_control.progress import SolverProgress
 
@@ -247,7 +247,7 @@ def run_panel_server(
         p = problem()
         if p is None:
             return ui.tags.div()
-        if _probmode(p) != 3:
+        if not _chance_terms(p):
             return ui.tags.div()
         try:
             active = solver()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_chance_terms.py -q -m 'not bench'`
Expected: 13 passed

- [ ] **Step 5: Gate**

Run: `~/.local/bin/ruff check src tests examples && /home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/ -q -m 'not bench' -p no:cacheprovider && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/python examples/validate_marxan_parity.py`
Expected: ruff and mypy clean; the full suite passes (prototype: 3796 passed, 101 skipped HiGHS cases which run in CI); `examples/validate_marxan_parity.py` prints MIP 35.0, SA 43.0, greedy 45.0; no golden JSON regenerated.

- [ ] **Step 6: Commit**

```bash
git add -A src/pymarxan/multiobjective.py src/pymarxan/solvers/cache.py src/pymarxan/solvers/mip_solver.py src/pymarxan/solvers/utils.py src/pymarxan_shiny/modules/results/target_met.py src/pymarxan_shiny/modules/run_control/run_panel.py tests/pymarxan/solvers/test_chance_terms.py
git commit -F - <<'MSG'
refactor(probability): route PROBMODE 3 through _chance_terms helpers (A1)

Two pure helpers in solvers/utils.py replace every `PROBMODE == 3` branch
(spec §3.1, final review I-4): `_chance_terms` (terms requested by the
mode; release A: 3 -> {"2d"}, everything else -> {}) and
`_active_chance_terms` (requested and with data: "2d" needs a puvspr
`prob` column).

Requested: non-binary rejection (evaluator and MIP), the MIP chance
strategy gate and `_strategy_metadata` (MIPSolver, MultiObjectiveMIP),
the Shiny MIP notice. Active: the evaluator's penalty, ProblemCache
(new `chance_terms` field), the Solution fields and the target-met table.

No objective changes. The one observable change: PROBMODE 3 without a
puvspr `prob` column now leaves Solution.prob_shortfalls / prob_penalty
None (they were all-zero), as the spec assigns.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01G9s5x1nmcqSTDeRUXcRPVW
MSG
```

---

### Task 4: `score_feature` core and the new public signatures

**Files:**
- Modify: `src/pymarxan/solvers/cache.py`
- Modify: `src/pymarxan/solvers/probability.py`
- Modify: `src/pymarxan/solvers/utils.py`
- Create: `tests/pymarxan/solvers/test_score_feature.py`
- Modify: `tests/pymarxan/solvers/test_zscore.py`

**Interfaces:**
- Consumes: Task 3's helpers.
- Produces (`solvers/probability.py`): `@dataclass(frozen=True) FeatureChance(z: float, p: float, shortfall: float, penalty: float)`; `score_feature(E, V, nonzero, target, ptarget, zero_var_rule) -> FeatureChance`; `score_features(...)` (arrays, same arithmetic); `compute_zscore_penalty(zscore_per_feature, prob_targets, weight=1.0, *, zero_variance=frozenset(), zero_var_rule=DEFAULT_ZERO_VAR_RULE) -> float`; `evaluate_solution_chance(problem, selected, term="2d") -> tuple[dict[int, float], float]`; `_term_moments(...)`. `solvers/utils.py`: `_zero_var_rule(problem) -> str` ("met" until Task 7). `ProblemCache.zero_var_rule: str`. All three zero-variance skips and `_compute_probmode3_zscore_penalty` go through the core. Effective rule stays "met": no solver result changes.

- [ ] **Step 1: Write the failing tests**

Create `tests/pymarxan/solvers/test_score_feature.py`:

```python
"""The chance-term core ``score_feature`` (spec §3.4, final review I-5).

One function scores a feature from ``(E, V, nonzero, target, ptarget, zero_var_rule)``.
Zero variance is ``nonzero == 0`` (the number of selected cells whose variance term is
> 0), never a float threshold on V. ``zero_var_rule`` is ``"marxan"`` (Marxan 3:
Z = 4, then the normal path) or ``"met"`` (penalty and shortfall 0).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.probability import (
    FeatureChance,
    compute_zscore_penalty,
    evaluate_solution_chance,
    score_feature,
    score_features,
)

SF4 = float(norm.sf(4.0))


class TestScoreFeature:
    def test_is_frozen_dataclass(self) -> None:
        fc = score_feature(8.0, 4.0, 2, 10.0, 0.95, "marxan")
        assert isinstance(fc, FeatureChance)
        assert [f.name for f in dataclasses.fields(fc)] == [
            "z", "p", "shortfall", "penalty",
        ]
        with pytest.raises(dataclasses.FrozenInstanceError):
            fc.z = 0.0  # type: ignore[misc]

    @pytest.mark.parametrize("rule", ["marxan", "met"])
    def test_normal_path(self, rule: str) -> None:
        """E = 8, V = 4, target 10 -> Z = 1; both rules agree away from zero variance."""
        fc = score_feature(8.0, 4.0, 3, 10.0, 0.95, rule)
        p = float(norm.sf(1.0))
        assert fc.z == pytest.approx(1.0)
        assert fc.p == pytest.approx(p, abs=1e-15)
        assert fc.shortfall == pytest.approx(0.95 - p, abs=1e-15)
        assert fc.penalty == pytest.approx((0.95 - p) / 0.95, abs=1e-15)

    def test_exceedance_has_no_penalty(self) -> None:
        fc = score_feature(14.0, 4.0, 1, 10.0, 0.9, "marxan")
        assert fc.z == pytest.approx(-2.0)
        assert fc.penalty == 0.0 and fc.shortfall == 0.0

    def test_zero_variance_marxan_rule(self) -> None:
        """Marxan 3: zero variance -> Z = 4, P = sf(4) ~ 3.17e-5, the step penalty follows."""
        fc = score_feature(20.0, 0.0, 0, 10.0, 0.95, "marxan")
        assert fc.z == 4.0
        assert fc.p == pytest.approx(SF4, abs=1e-18)
        assert fc.shortfall == pytest.approx(0.95 - SF4)
        assert fc.penalty == pytest.approx((0.95 - SF4) / 0.95)

    def test_zero_variance_met_rule(self) -> None:
        fc = score_feature(5.0, 0.0, 0, 10.0, 0.95, "met")
        assert fc.penalty == 0.0 and fc.shortfall == 0.0

    def test_zero_variance_marxan_rule_with_tiny_ptarget(self) -> None:
        """ptarget below sf(4): even the Z = 4 rule is met."""
        fc = score_feature(0.0, 0.0, 0, 10.0, 1e-5, "marxan")
        assert fc.penalty == 0.0 and fc.shortfall == 0.0

    def test_classification_uses_the_count_not_the_value(self) -> None:
        """A tiny positive variance with nonzero > 0 is NOT zero variance."""
        fc = score_feature(10.0, 1e-30, 1, 10.0, 0.95, "met")
        assert fc.z == 0.0
        assert fc.p == pytest.approx(0.5)
        assert fc.penalty == pytest.approx((0.95 - 0.5) / 0.95)

    @pytest.mark.parametrize("rule", ["marxan", "met"])
    @pytest.mark.parametrize("ptarget", [-1.0, 0.0])
    def test_disabled_ptarget(self, rule: str, ptarget: float) -> None:
        for args in ((5.0, 4.0, 1), (0.0, 0.0, 0)):
            fc = score_feature(*args, 10.0, ptarget, rule)
            assert fc.penalty == 0.0 and fc.shortfall == 0.0

    def test_unknown_rule_raises(self) -> None:
        with pytest.raises(ValueError, match="zero_var_rule"):
            score_feature(1.0, 1.0, 1, 1.0, 0.5, "trivially_met")


class TestVectorisedAgreesWithScalar:
    @pytest.mark.parametrize("rule", ["marxan", "met"])
    def test_batch(self, rule: str) -> None:
        rng = np.random.default_rng(3)
        n = 400
        E = rng.uniform(0, 20, n)
        V = rng.uniform(0, 9, n)
        N = rng.integers(0, 4, n)
        V[N == 0] = 0.0
        V[:5] = 1e-300  # tiny but counted
        N[:5] = 1
        t = rng.uniform(0, 20, n)
        pt = rng.choice([-1.0, 0.0, 0.2, 0.5, 0.95, 1.0, 1e-6], n)
        z, p, sf, pen = score_features(E, V, N, t, pt, rule)
        for k in range(n):
            fc = score_feature(float(E[k]), float(V[k]), int(N[k]), float(t[k]),
                               float(pt[k]), rule)
            assert (fc.z, fc.p, fc.shortfall, fc.penalty) == (
                float(z[k]), float(p[k]), float(sf[k]), float(pen[k]),
            )


class TestComputeZscorePenaltyZeroVariance:
    def test_real_z_of_four_is_scored(self) -> None:
        """A real Z = 4 (not a zero-variance feature) goes through the normal path."""
        p = compute_zscore_penalty({1: 4.0}, {1: 0.95}, weight=2.0)
        assert p == pytest.approx(2.0 * (0.95 - SF4) / 0.95)

    def test_zero_variance_feature_marxan_rule(self) -> None:
        p = compute_zscore_penalty(
            {1: 4.0, 2: 1.0}, {1: 0.95, 2: 0.95}, weight=1.0,
            zero_variance=frozenset({1}), zero_var_rule="marxan",
        )
        expected = (0.95 - SF4) / 0.95 + (0.95 - float(norm.sf(1.0))) / 0.95
        assert p == pytest.approx(expected)

    def test_zero_variance_feature_met_rule(self) -> None:
        p = compute_zscore_penalty(
            {1: 4.0, 2: 1.0}, {1: 0.95, 2: 0.95}, weight=1.0,
            zero_variance=frozenset({1}), zero_var_rule="met",
        )
        assert p == pytest.approx((0.95 - float(norm.sf(1.0))) / 0.95)

    def test_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            compute_zscore_penalty({1: 1.0}, {1: 0.5}, 1.0, frozenset())  # type: ignore[misc]


def _two_feature_problem() -> ConservationProblem:
    """Feature 1 uncertain (q = 0.7), feature 2 certain (q = 1: zero variance)."""
    pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]})
    feat = pd.DataFrame({
        "id": [1, 2], "name": ["a", "b"], "target": [5.0, 5.0], "spf": [1.0, 1.0],
        "ptarget": [0.9, 0.8],
    })
    pv = pd.DataFrame({
        "species": [1, 1, 2], "pu": [1, 2, 1], "amount": [4.0, 4.0, 10.0],
        "prob": [0.7, 0.7, 1.0],
    })
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pv,
        parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 3.0},
    )


class TestEvaluateSolutionChance:
    def test_signature_has_term(self) -> None:
        sel = np.array([True, True])
        a = evaluate_solution_chance(_two_feature_problem(), sel)
        b = evaluate_solution_chance(_two_feature_problem(), sel, term="2d")
        assert a == b

    def test_unknown_term_raises(self) -> None:
        with pytest.raises(ValueError, match="term"):
            evaluate_solution_chance(_two_feature_problem(), np.array([True, True]), term="3d")

    def test_matches_score_feature(self) -> None:
        sel = np.array([True, True])
        shortfalls, penalty = evaluate_solution_chance(_two_feature_problem(), sel)
        e, v = 8.0 * 0.7, 2 * 16.0 * 0.7 * 0.3
        f1 = score_feature(e, v, 2, 5.0, 0.9, "met")
        assert shortfalls[1] == pytest.approx(f1.shortfall, abs=1e-15)
        assert shortfalls[2] == 0.0  # zero variance under today's rule
        assert penalty == pytest.approx(3.0 * f1.penalty, abs=1e-15)


class TestRoutedThroughScoreFeature:
    """The evaluator and the cache both score through the core and its rule."""

    def test_evaluator_honours_the_rule(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pymarxan.solvers.utils as utils

        sel = np.array([True, True])
        _, met = evaluate_solution_chance(_two_feature_problem(), sel)
        monkeypatch.setattr(utils, "_zero_var_rule", lambda problem: "marxan")
        shortfalls, marxan = evaluate_solution_chance(_two_feature_problem(), sel)
        assert shortfalls[2] == pytest.approx(0.8 - SF4)
        assert marxan - met == pytest.approx(3.0 * (0.8 - SF4) / 0.8, abs=1e-12)
        # build_solution's objective goes through the same evaluator
        from pymarxan.solvers.utils import build_solution

        sol = build_solution(_two_feature_problem(), sel, 0.0)
        assert sol.prob_penalty == pytest.approx(marxan, abs=1e-15)

    def test_cache_honours_the_rule(self) -> None:
        from pymarxan.solvers.cache import ProblemCache

        problem = _two_feature_problem()
        cache = ProblemCache.from_problem(problem)
        marxan = dataclasses.replace(cache, zero_var_rule="marxan")
        sel = np.array([True, True])
        held = cache.compute_held(sel)
        diff = (
            marxan.compute_full_objective(sel, held, 0.0)
            - cache.compute_full_objective(sel, held, 0.0)
        )
        assert diff == pytest.approx(3.0 * (0.8 - SF4) / 0.8, abs=1e-12)
        _, pen = evaluate_solution_chance(problem, sel)
        assert cache.compute_full_objective(sel, held, 0.0) == pytest.approx(
            2.0 + pen, abs=1e-12,
        )
```

Modify `tests/pymarxan/solvers/test_zscore.py` (apply this diff):

```diff
diff --git a/tests/pymarxan/solvers/test_zscore.py b/tests/pymarxan/solvers/test_zscore.py
index 0e1f75d..7c5dad7 100644
--- a/tests/pymarxan/solvers/test_zscore.py
+++ b/tests/pymarxan/solvers/test_zscore.py
@@ -179,18 +179,18 @@ class TestZscorePenalty:
         assert p == pytest.approx(expected, abs=1e-9)
 
     def test_sentinel_variance_means_deterministic_no_penalty(self):
-        """Z=4 (Marxan zero-variance sentinel) -> no chance-constraint
-        penalty. The deterministic-target shortfall path handles any
-        deterministic miss; the probability penalty is vacuously
-        satisfied (P=1) when there's no uncertainty.
-
-        Matches Marxan's intent (probability.cpp comment: "degenerate
-        case → P=1") even though norm.sf(4) ≈ 3e-5 would give a
-        non-trivial penalty if we passed it through naïvely.
+        """A zero-variance feature has no chance penalty under the "met" rule.
+
+        Rewritten (v0.47.0, A2): the zero-variance feature is now named in
+        ``zero_variance`` instead of being inferred from ``Z == 4``, so a real
+        Z of 4 is told apart from Marxan's zero-variance 4 (spec §3.4, final
+        review I-5); ``zero_var_rule="met"`` keeps this release's default.
         """
         p = compute_zscore_penalty(
             zscore_per_feature={1: 4.0},
             prob_targets={1: 0.95},
             weight=1.0,
+            zero_variance=frozenset({1}),
+            zero_var_rule="met",
         )
         assert p == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_score_feature.py tests/pymarxan/solvers/test_zscore.py -q -m 'not bench'`
Expected: ERROR — 1 error collecting `test_score_feature.py` (ImportError: cannot import `FeatureChance`).

- [ ] **Step 3: Implement**

Modify `src/pymarxan/solvers/cache.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/cache.py b/src/pymarxan/solvers/cache.py
index c005203..d290e57 100644
--- a/src/pymarxan/solvers/cache.py
+++ b/src/pymarxan/solvers/cache.py
@@ -61,6 +61,7 @@ from pymarxan.solvers.utils import (
     _chance_terms,
     _probability_vector,
     _probmode,
+    _zero_var_rule,
     compute_cost_threshold_penalty,
 )
 
@@ -139,6 +140,8 @@ class ProblemCache:
     # Active chance terms (``_active_chance_terms``: requested by PROBMODE and with data);
     # ``"2d"`` = the Z-score term from the puvspr ``prob`` column.
     chance_terms: frozenset[str] = frozenset()
+    # Zero-variance rule of the chance terms (``_zero_var_rule``): "marxan" or "met".
+    zero_var_rule: str = "met"
     # expected_matrix[i, j] = amount_ij · p_ij (presence); collapses to amount
     # when there is no `prob` column or when prob == 1.
     expected_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
@@ -322,6 +325,7 @@ class ProblemCache:
         probmode = _probmode(problem)
         prob_weight = float(params.get("PROBABILITYWEIGHTING", 1.0))
         chance_terms = _active_chance_terms(problem)
+        zero_var_rule = _zero_var_rule(problem)
 
         # Per-cell probability of PRESENCE — read once into a dense matrix.
         # Marxan's PROB2D `prob` is the probability the feature is present at
@@ -552,6 +556,7 @@ class ProblemCache:
             probmode=probmode,
             prob_weight=prob_weight,
             chance_terms=chance_terms,
+            zero_var_rule=zero_var_rule,
             expected_matrix=expected_matrix,
             var_matrix=var_matrix,
             feat_ptarget=feat_ptarget,
@@ -619,38 +624,29 @@ class ProblemCache:
         held[cols] += sign * csr.data[s:e]
 
     def _compute_zscore_penalty(
-        self, held_expected: np.ndarray, held_variance: np.ndarray,
+        self,
+        held_expected: np.ndarray,
+        held_variance: np.ndarray,
+        held_nonzero: np.ndarray,
     ) -> float:
         """Marxan-faithful Z-score chance-constraint penalty.
 
         Active only when the 2D chance term is (``"2d" in self.chance_terms``); returns 0
-        otherwise (the
-        caller's hot path can skip this branch entirely via the probmode
-        check).  Features with ptarget ≤ 0 are skipped.  Features with
-        zero variance contribute 0 (Marxan's "degenerate → P=1").
+        otherwise. Scored by ``probability.score_features``: zero variance is
+        ``held_nonzero == 0`` (no selected cell with a variance term > 0) and follows
+        ``self.zero_var_rule``; features with ptarget ≤ 0 are disabled.
         """
         if "2d" not in self.chance_terms or self.n_feat == 0:
             return 0.0
-        from scipy.stats import norm  # local import keeps cold path cheap
-        total = 0.0
-        for j in range(self.n_feat):
-            ptarget = float(self.feat_ptarget[j])
-            if ptarget <= 0:
-                continue
-            var = float(held_variance[j])
-            if var <= 0:
-                # Degenerate / no uncertainty — deterministic path handles miss
-                continue
-            mean = float(held_expected[j])
-            target = float(self.feat_targets[j])
-            z = (target - mean) / np.sqrt(var)
-            prob_met = float(norm.sf(z))
-            shortfall = max(0.0, (ptarget - prob_met) / ptarget)
-            # Marxan's ComputeProbability2D weights the probability shortfall
-            # only by PROBABILITYWEIGHTING — NOT by SPF (unlike the
-            # representation penalty). See score_change.cpp:611.
-            total += shortfall
-        return self.prob_weight * total
+        from pymarxan.solvers.probability import score_features  # keeps cold path cheap
+
+        # Marxan's ComputeProbability2D weights the probability shortfall only by
+        # PROBABILITYWEIGHTING — NOT by SPF (unlike the representation penalty).
+        penalty = score_features(
+            held_expected, held_variance, held_nonzero, self.feat_targets,
+            self.feat_ptarget, self.zero_var_rule,
+        )[3]
+        return self.prob_weight * float(np.sum(penalty))
 
     def compute_full_objective(
         self,
@@ -707,7 +703,8 @@ class ProblemCache:
         if "2d" in self.chance_terms:
             held_expected = self.expected_matrix[selected].sum(axis=0)
             held_variance = self.var_matrix[selected].sum(axis=0)
-            obj += self._compute_zscore_penalty(held_expected, held_variance)
+            held_nonzero = (self.var_matrix[selected] > 0).sum(axis=0)
+            obj += self._compute_zscore_penalty(held_expected, held_variance, held_nonzero)
 
         # Connectivity term (no-op when conn_mode == 0)
         if self.conn_mode:
@@ -846,11 +843,13 @@ class ProblemCache:
             held_var_after = held_var_before + sign * self.var_matrix[idx]
             # Underflow guard — held_var must be non-negative
             np.clip(held_var_after, 0.0, None, out=held_var_after)
+            nz_before = (self.var_matrix[selected] > 0).sum(axis=0)
+            nz_after = nz_before + int(sign) * (self.var_matrix[idx] > 0)
             prob_before = self._compute_zscore_penalty(
-                held_exp_before, held_var_before,
+                held_exp_before, held_var_before, nz_before,
             )
             prob_after = self._compute_zscore_penalty(
-                held_exp_after, held_var_after,
+                held_exp_after, held_var_after, nz_after,
             )
             delta += prob_after - prob_before
 
```

Modify `src/pymarxan/solvers/probability.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/probability.py b/src/pymarxan/solvers/probability.py
index 70fc995..fefde9f 100644
--- a/src/pymarxan/solvers/probability.py
+++ b/src/pymarxan/solvers/probability.py
@@ -40,13 +40,131 @@ References
 from __future__ import annotations
 
 import math
+from dataclasses import dataclass
+from typing import Literal
 
-from scipy.stats import norm
+import numpy as np
+from scipy.special import ndtr
 
-# Marxan v4 short-circuit value when variance == 0: rZ = 4 gives
-# probZUT(4) ≈ 3.17e-5, i.e. effectively "target is trivially met".
+# Marxan's Z for a zero-variance feature (``probability.c``: ``rZ = 4``);
+# probZUT(4) = norm.sf(4) ≈ 3.17e-5.
 _MARXAN_ZERO_VARIANCE_SENTINEL = 4.0
 
+ZeroVarRule = Literal["marxan", "met"]
+_ZERO_VAR_RULES = ("marxan", "met")
+# The rule used when a caller names none. "met" keeps the pre-v0.47 behaviour.
+DEFAULT_ZERO_VAR_RULE: ZeroVarRule = "met"
+
+_TERMS = ("2d",)
+
+
+@dataclass(frozen=True)
+class FeatureChance:
+    """One feature's chance-term score (spec §3.4).
+
+    ``z`` and ``p`` (= ``norm.sf(z)``) are the Marxan Z-score and probability of meeting
+    the target; ``shortfall`` = ``max(0, ptarget − p)`` (absolute, for reports);
+    ``penalty`` = ``[ptarget > p]·(ptarget − p)/ptarget``, unweighted. Both are 0 for a
+    disabled feature (``ptarget ≤ 0``). Under the ``"met"`` rule a zero-variance feature
+    reports ``p = 1`` and no shortfall or penalty.
+    """
+
+    z: float
+    p: float
+    shortfall: float
+    penalty: float
+
+
+def _check_rule(zero_var_rule: str) -> None:
+    if zero_var_rule not in _ZERO_VAR_RULES:
+        raise ValueError(
+            f"zero_var_rule must be one of {_ZERO_VAR_RULES}, got {zero_var_rule!r}"
+        )
+
+
+def _chance_z(
+    E: np.ndarray, V: np.ndarray, zero: np.ndarray, target: np.ndarray,
+) -> np.ndarray:
+    """Z = (target − E)/√V; Marxan's 4 where ``zero`` (no selected cell with variance)."""
+    with np.errstate(divide="ignore", invalid="ignore"):
+        z = (target - E) / np.sqrt(np.where(zero, 1.0, V))
+    return np.where(zero, _MARXAN_ZERO_VARIANCE_SENTINEL, z)
+
+
+def _chance_from_z(
+    z: np.ndarray, zero: np.ndarray, ptarget: np.ndarray, zero_var_rule: str,
+) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
+    """``(p, shortfall, penalty)`` from Z: the one copy of the probability arithmetic."""
+    p = ndtr(-z)  # = norm.sf(z), Marxan's probZUT up to ~1e-9
+    enabled = ptarget > 0
+    gap = np.where(enabled & (ptarget > p), ptarget - p, 0.0)
+    penalty = gap / np.where(enabled, ptarget, 1.0)
+    if zero_var_rule == "met":
+        p = np.where(zero, 1.0, p)
+        gap = np.where(zero, 0.0, gap)
+        penalty = np.where(zero, 0.0, penalty)
+    return p, gap, penalty
+
+
+def score_features(
+    E: np.ndarray,
+    V: np.ndarray,
+    nonzero: np.ndarray,
+    target: np.ndarray,
+    ptarget: np.ndarray,
+    zero_var_rule: str,
+) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
+    """Vectorised :func:`score_feature`: arrays ``(z, p, shortfall, penalty)``.
+
+    The solvers' incremental path (``ChanceState``) and the evaluator use this; it is
+    the same arithmetic as :func:`score_feature`, element for element.
+    """
+    _check_rule(zero_var_rule)
+    E = np.asarray(E, dtype=np.float64)
+    V = np.asarray(V, dtype=np.float64)
+    target = np.asarray(target, dtype=np.float64)
+    ptarget = np.asarray(ptarget, dtype=np.float64)
+    zero = np.asarray(nonzero) == 0
+    z = _chance_z(E, V, zero, target)
+    p, shortfall, penalty = _chance_from_z(z, zero, ptarget, zero_var_rule)
+    return z, p, shortfall, penalty
+
+
+def score_feature(
+    E: float,
+    V: float,
+    nonzero: int,
+    target: float,
+    ptarget: float,
+    zero_var_rule: str,
+) -> FeatureChance:
+    """Score one feature's chance term (spec §3.4).
+
+    Parameters
+    ----------
+    E, V
+        Expected amount and variance of the feature over the selected cells.
+    nonzero
+        Number of selected cells whose variance term, as computed in float, is > 0.
+        **Zero variance is ``nonzero == 0``** (exact; no float threshold on ``V``).
+    target
+        Deterministic target (the resolved target, without MISSLEVEL).
+    ptarget
+        Probability target; ``≤ 0`` disables the feature (Marxan's −1).
+    zero_var_rule
+        ``"marxan"``: zero variance gives Marxan 3's ``Z = 4``, then the normal path
+        (so ``P ≈ 3.17e-5`` fails any ``ptarget`` above it). ``"met"``: zero variance
+        counts as met (penalty and shortfall 0).
+    """
+    z, p, shortfall, penalty = score_features(
+        np.array([E]), np.array([V]), np.array([nonzero]), np.array([target]),
+        np.array([ptarget]), zero_var_rule,
+    )
+    return FeatureChance(
+        z=float(z[0]), p=float(p[0]), shortfall=float(shortfall[0]),
+        penalty=float(penalty[0]),
+    )
+
 
 def compute_zscore_per_feature(
     achieved_mean: dict[int, float],
@@ -87,13 +205,16 @@ def compute_zscore_penalty(
     zscore_per_feature: dict[int, float],
     prob_targets: dict[int, float],
     weight: float = 1.0,
+    *,
+    zero_variance: frozenset[int] = frozenset(),
+    zero_var_rule: str = DEFAULT_ZERO_VAR_RULE,
 ) -> float:
     """Marxan-faithful normalised probability shortfall penalty.
 
     For each feature *j*::
 
         P_j = norm.sf(Z_j)
-        penalty_j = max(0, (ptarget_j − P_j) / ptarget_j)
+        penalty_j = [ptarget_j > P_j] · (ptarget_j − P_j) / ptarget_j
 
     Total: ``weight · Σ penalty_j``. Following Marxan's ``ComputeProbability2D``
     the probability shortfall is scaled **only** by ``PROBABILITYWEIGHTING``,
@@ -109,38 +230,77 @@ def compute_zscore_penalty(
         Per-feature probability target. ``-1`` (or any ≤ 0) means disabled.
     weight
         Global ``PROBABILITYWEIGHTING`` scaling factor.
+    zero_variance
+        Ids of the zero-variance features, so a real ``Z = 4`` is told apart from
+        Marxan's zero-variance 4; ``zero_var_rule`` applies to these only.
+    zero_var_rule
+        ``"marxan"`` or ``"met"`` (see :func:`score_feature`).
 
     Returns
     -------
     float
         The total weighted probability penalty.
     """
-    total = 0.0
-    for fid, z in zscore_per_feature.items():
-        ptarget = prob_targets.get(fid, -1.0)
-        if ptarget <= 0:
+    _check_rule(zero_var_rule)
+    fids = list(zscore_per_feature)
+    if not fids:
+        return 0.0
+    z = np.array([zscore_per_feature[f] for f in fids], dtype=np.float64)
+    pt = np.array([prob_targets.get(f, -1.0) for f in fids], dtype=np.float64)
+    zero = np.array([f in zero_variance for f in fids], dtype=bool)
+    _, _, penalty = _chance_from_z(z, zero, pt, zero_var_rule)
+    return float(weight * float(np.sum(penalty)))
+
+
+def _term_moments(
+    problem,  # ConservationProblem
+    selected,  # np.ndarray
+    term: str,
+    feat_ids: np.ndarray,
+) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
+    """Per-feature ``(E, V, nonzero)`` of one chance term over the selected PUs."""
+    if term not in _TERMS:
+        raise ValueError(f"unknown chance term {term!r}; release A knows {_TERMS}")
+    col = {int(f): j for j, f in enumerate(feat_ids)}
+    n = len(feat_ids)
+    E = np.zeros(n)
+    V = np.zeros(n)
+    N = np.zeros(n, dtype=np.int64)
+    pu_ids = problem.planning_units["id"].values
+    selected_pu_ids = {int(pu_ids[i]) for i in range(len(pu_ids)) if selected[i]}
+    puvspr = problem.pu_vs_features
+    pv_pu = puvspr["pu"].values
+    pv_sp = puvspr["species"].values
+    pv_am = puvspr["amount"].values
+    pv_pr = puvspr["prob"].values if "prob" in puvspr.columns else None
+    for k in range(len(pv_pu)):
+        if int(pv_pu[k]) not in selected_pu_ids:
             continue
-        # Marxan zero-variance sentinel means "no uncertainty"; the
-        # chance constraint is satisfied vacuously (P = 1) and the
-        # deterministic shortfall penalty handles any deterministic miss.
-        if z == _MARXAN_ZERO_VARIANCE_SENTINEL:
+        j = col.get(int(pv_sp[k]))
+        if j is None:
             continue
-        prob = float(norm.sf(z))  # upper tail; matches Marxan's probZUT
-        shortfall = max(0.0, (ptarget - prob) / ptarget)
-        total += shortfall
-    return float(weight * total)
+        amt = float(pv_am[k])
+        # PROB2D presence convention: q = P(present); default 1.0 (certain presence)
+        # when there is no `prob` column. Marxan 3 ComputeProbability2D:
+        # E += amount·q, V += amount²·q·(1 − q).
+        q = float(pv_pr[k]) if pv_pr is not None else 1.0
+        v = amt * amt * q * (1.0 - q)
+        E[j] += amt * q
+        V[j] += v
+        N[j] += int(v > 0)
+    return E, V, N
 
 
 def evaluate_solution_chance(
     problem,  # ConservationProblem — typed loosely to avoid circular import
     selected,  # np.ndarray
+    term: str = "2d",
 ) -> tuple[dict[int, float], float]:
-    """Post-hoc Z-score evaluation of a fully-formed solution.
+    """Post-hoc evaluation of one chance term for a fully-formed solution.
 
-    Used by the MIP "drop" strategy: the MIP returns a solution against
-    the deterministic problem; this evaluator computes the
-    chance-constraint gap so the Solution can carry
-    ``prob_shortfalls`` and ``prob_penalty``.
+    Used by ``build_solution`` (every solver's reported objective and the MIP "drop"
+    strategy's post-hoc report): computes the per-feature E / V / nonzero over the
+    selected PUs and scores them with :func:`score_features`.
 
     Returns
     -------
@@ -148,10 +308,13 @@ def evaluate_solution_chance(
         ``shortfalls`` is a feature_id → ``max(0, ptarget − P)`` dict
         (per-feature absolute gap in probability space, NOT normalised
         by ptarget — easier to interpret in the UI).
-        ``penalty`` is the weighted normalised total that would have been
-        added to the objective under PROBMODE 3.
+        ``penalty`` is the weighted normalised total that the term adds to the
+        objective.
     """
+    from pymarxan.solvers.utils import _zero_var_rule
 
+    if term not in _TERMS:
+        raise ValueError(f"unknown chance term {term!r}; release A knows {_TERMS}")
     if "ptarget" not in problem.features.columns:
         return {}, 0.0
     feat_ids = problem.features["id"].astype(int).values
@@ -161,51 +324,10 @@ def evaluate_solution_chance(
     if not any(pt > 0 for pt in feat_ptarget):
         return {fid: 0.0 for fid in feat_ids.tolist()}, 0.0
 
-    pu_ids = problem.planning_units["id"].values
-    selected_pu_ids = {
-        int(pu_ids[i]) for i in range(len(pu_ids)) if selected[i]
-    }
-
-    puvspr = problem.pu_vs_features
-    has_prob = "prob" in puvspr.columns
-    achieved_mean: dict[int, float] = {}
-    achieved_var: dict[int, float] = {}
-    pv_pu = puvspr["pu"].values
-    pv_sp = puvspr["species"].values
-    pv_am = puvspr["amount"].values
-    pv_pr = puvspr["prob"].values if has_prob else None
-    for k in range(len(pv_pu)):
-        pid = int(pv_pu[k])
-        if pid not in selected_pu_ids:
-            continue
-        fid = int(pv_sp[k])
-        amt = float(pv_am[k])
-        # PROB2D presence convention: p = P(present); default 1.0 (certain
-        # presence) when there is no `prob` column. Matches Marxan
-        # ComputeP_AllPUsSelected_2D (`ExpectedAmount2D += amount * prob`).
-        p = float(pv_pr[k]) if pv_pr is not None else 1.0
-        achieved_mean[fid] = achieved_mean.get(fid, 0.0) + amt * p
-        achieved_var[fid] = achieved_var.get(fid, 0.0) + amt * amt * p * (1.0 - p)
-
-    targets = {int(feat_ids[j]): float(feat_target[j]) for j in range(len(feat_ids))}
-    ptargets = {int(feat_ids[j]): float(feat_ptarget[j]) for j in range(len(feat_ids))}
-
-    # Ensure every feature is keyed (E=Var=0 → sentinel Z)
-    for fid in targets:
-        achieved_mean.setdefault(fid, 0.0)
-        achieved_var.setdefault(fid, 0.0)
-
-    z_per = compute_zscore_per_feature(achieved_mean, achieved_var, targets)
+    E, V, N = _term_moments(problem, selected, term, feat_ids)
+    _, _, shortfall, penalty = score_features(
+        E, V, N, feat_target, feat_ptarget, _zero_var_rule(problem),
+    )
     weight = float(problem.parameters.get("PROBABILITYWEIGHTING", 1.0))
-    penalty = compute_zscore_penalty(z_per, ptargets, weight=weight)
-
-    shortfalls: dict[int, float] = {}
-    for fid, z in z_per.items():
-        pt = ptargets.get(fid, -1.0)
-        if pt <= 0 or z == _MARXAN_ZERO_VARIANCE_SENTINEL:
-            shortfalls[fid] = 0.0
-        else:
-            prob_met = float(norm.sf(z))
-            shortfalls[fid] = max(0.0, pt - prob_met)
-
-    return shortfalls, penalty
+    shortfalls = {int(f): float(sf) for f, sf in zip(feat_ids, shortfall)}
+    return shortfalls, float(weight * float(np.sum(penalty)))
```

Modify `src/pymarxan/solvers/utils.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/utils.py b/src/pymarxan/solvers/utils.py
index 9dd3a17..12a8edf 100644
--- a/src/pymarxan/solvers/utils.py
+++ b/src/pymarxan/solvers/utils.py
@@ -124,6 +124,15 @@ def _active_chance_terms(problem: ConservationProblem) -> frozenset[str]:
     return terms
 
 
+def _zero_var_rule(problem: ConservationProblem) -> str:
+    """The zero-variance rule for the chance terms (spec §3.4): ``"met"`` in this release.
+
+    ``"met"``: a feature with no selected cell carrying variance has no chance penalty.
+    Pure; never warns.
+    """
+    return "met"
+
+
 def _probmode2_active(problem: ConservationProblem) -> bool:
     """True iff PROBMODE 2 applies: a probability table and ``PROBMODE == 2``."""
     return problem.probability is not None and _probmode(problem) == 2
@@ -531,89 +540,21 @@ def _compute_probmode3_zscore_penalty(
     problem: ConservationProblem,
     selected: np.ndarray,
 ) -> float:
-    """PROBMODE 3: Marxan-faithful Z-score chance-constraint penalty.
+    """PROBMODE 3: the weighted Z-score chance penalty of the 2D term.
 
-    Computes E[T_j] and Var[T_j] from the per-cell ``prob`` column on
-    ``pu_vs_features`` (Marxan PROB2D wire format):
+    Per feature, over the selected PUs, with the per-cell presence probability ``q`` from
+    the ``prob`` column on ``pu_vs_features`` (Marxan PROB2D wire format)::
 
-        E[T_j]   = Σ_i x_i · amount_ij · (1 − p_ij)
-        Var[T_j] = Σ_i x_i · amount_ij² · p_ij · (1 − p_ij)
+        E[T_j]   = Σ_i x_i · amount_ij · q_ij
+        Var[T_j] = Σ_i x_i · amount_ij² · q_ij · (1 − q_ij)
 
-    Then routes to ``compute_zscore_per_feature`` and
-    ``compute_zscore_penalty`` which implement the Marxan sign convention
-    ``Z = (T − E)/sqrt(V)`` and ptarget-normalised shortfall.
+    scored by ``probability.score_features`` (one copy of the arithmetic, shared with
+    ``evaluate_solution_chance``, which this delegates to).
     """
     # Lazy import: keep solvers.utils free of scipy unless mode 3 is used.
-    from pymarxan.solvers.probability import (
-        compute_zscore_penalty,
-        compute_zscore_per_feature,
-    )
-
-    prob_weight = float(problem.parameters.get("PROBABILITYWEIGHTING", 1.0))
-    if prob_weight == 0.0:
-        return 0.0
-
-    # Build per-feature expected and variance from selected PUs.
-    pu_ids = problem.planning_units["id"].values
-    selected_pu_ids = {int(pu_ids[i]) for i in range(len(pu_ids)) if selected[i]}
-    if not selected_pu_ids:
-        # Empty reserve: E=0, Var=0 -> sentinel Z; penalty zero anyway
-        # for features whose ptarget≤0, otherwise norm.sf(4) ≈ 0 < ptarget
-        # which produces a maximal shortfall. That's the right behaviour
-        # for an empty reserve under chance constraints.
-        pass
-
-    puvspr = problem.pu_vs_features
-    has_prob = "prob" in puvspr.columns
-
-    achieved_mean: dict[int, float] = {}
-    achieved_variance: dict[int, float] = {}
-    if len(puvspr) > 0:
-        pv_pu = puvspr["pu"].values
-        pv_sp = puvspr["species"].values
-        pv_am = puvspr["amount"].values
-        pv_pr = puvspr["prob"].values if has_prob else None
-        for k in range(len(pv_pu)):
-            pid = int(pv_pu[k])
-            if pid not in selected_pu_ids:
-                continue
-            fid = int(pv_sp[k])
-            amount = float(pv_am[k])
-            # PROB2D presence convention: p = P(present); default 1.0
-            # (deterministic) when no `prob` column. E[T] += amount * p,
-            # matching Marxan ComputeP_AllPUsSelected_2D.
-            p = float(pv_pr[k]) if pv_pr is not None else 1.0
-            achieved_mean[fid] = achieved_mean.get(fid, 0.0) + amount * p
-            achieved_variance[fid] = (
-                achieved_variance.get(fid, 0.0) + (amount ** 2) * p * (1.0 - p)
-            )
-
-    # Feature-side data
-    feat_ids = problem.features["id"].astype(int).values
-    feat_target = problem.features["target"].astype(float).values
-    if "ptarget" in problem.features.columns:
-        feat_ptarget = problem.features["ptarget"].astype(float).values
-    else:
-        # Legacy spec without ptarget -> all disabled
-        return 0.0
-
-    # Skip work entirely if every feature is disabled
-    if not any(pt > 0 for pt in feat_ptarget):
-        return 0.0
-
-    targets = {int(feat_ids[j]): float(feat_target[j]) for j in range(len(feat_ids))}
-    prob_targets = {
-        int(feat_ids[j]): float(feat_ptarget[j]) for j in range(len(feat_ids))
-    }
-
-    # Features absent from achieved_* (no selected PU contributed) get
-    # E=0, Var=0 -> Marxan sentinel Z=4 -> P≈0 -> max shortfall.
-    for fid in targets:
-        achieved_mean.setdefault(fid, 0.0)
-        achieved_variance.setdefault(fid, 0.0)
+    from pymarxan.solvers.probability import evaluate_solution_chance
 
-    z_per_feat = compute_zscore_per_feature(achieved_mean, achieved_variance, targets)
-    return compute_zscore_penalty(z_per_feat, prob_targets, weight=prob_weight)
+    return evaluate_solution_chance(problem, selected, "2d")[1]
 
 
 def compute_objective_terms(
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_score_feature.py tests/pymarxan/solvers/test_zscore.py -q -m 'not bench'`
Expected: 38 passed

- [ ] **Step 5: Gate**

Run: `~/.local/bin/ruff check src tests examples && /home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/ -q -m 'not bench' -p no:cacheprovider && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/python examples/validate_marxan_parity.py`
Expected: ruff and mypy clean; the full suite passes (prototype: 3820 passed, 101 skipped HiGHS cases which run in CI); `examples/validate_marxan_parity.py` prints MIP 35.0, SA 43.0, greedy 45.0; no golden JSON regenerated.

- [ ] **Step 6: Commit**

```bash
git add -A src/pymarxan/solvers/cache.py src/pymarxan/solvers/probability.py src/pymarxan/solvers/utils.py tests/pymarxan/solvers/test_score_feature.py tests/pymarxan/solvers/test_zscore.py
git commit -F - <<'MSG'
feat(probability): score_feature core with zero_var_rule (A2)

One function scores a feature's chance term from (E, V, nonzero, target,
ptarget, zero_var_rule) -> FeatureChance(z, p, shortfall, penalty)
(spec §3.4, final review I-5). Zero variance is `nonzero == 0`, the count
of selected cells whose variance term is > 0, not a float test on V.
`zero_var_rule` is "marxan" (Z = 4, then the normal path) or "met"
(penalty and shortfall 0). `score_features` is the vectorised twin with
the same arithmetic.

New public signatures: compute_zscore_penalty(..., *, zero_variance,
zero_var_rule), evaluate_solution_chance(problem, selected, term="2d").
All three zero-variance skips (compute_zscore_penalty, the evaluator's
shortfall loop, ProblemCache._compute_zscore_penalty) and
utils._compute_probmode3_zscore_penalty now go through the core.

The effective rule stays "met" everywhere (utils._zero_var_rule,
ProblemCache.zero_var_rule), so solver results do not change; tests pin
both rules at the function level.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01G9s5x1nmcqSTDeRUXcRPVW
MSG
```

---

### Task 5: `ChanceState` sidecar and per-cell chance arrays (not yet wired)

**Files:**
- Modify: `src/pymarxan/solvers/cache.py`
- Create: `src/pymarxan/solvers/chance_state.py`
- Modify: `src/pymarxan/solvers/probability.py`
- Create: `tests/pymarxan/solvers/test_chance_state.py`
- Modify: `tests/pymarxan/test_fractional_readers.py`

**Interfaces:**
- Consumes: Task 4's `score_features` / `_chance_penalty`; Task 3's `cache.chance_terms`.
- Produces: `ProblemCache.chance_cells: dict[str, ChanceCells]` (frozen; CSR `indptr` / `indices` shared with `held_csr`, stacked `ev` (2 × nnz; rows `e`, `v`), `abs_ev`, `pos`; CSC `c_indptr` / `c_rows` / `c_ev`), built for active terms only. New `src/pymarxan/solvers/chance_state.py`: `ChanceState.from_selection(cache, selected) -> ChanceState`, `.penalty_total() -> float`, `.delta_penalty(cache, idx: int, adding: bool) -> float` (never mutates state; caches the preview), `.apply_flip(cache, idx: int, adding: bool) -> None` (commits the matching preview, else the same path; clears the preview), `.n_recompute: int`; constants `TAU = 1e-12`, `_E_FLOOR = 64·eps`, `_RESET = 4·eps`. The dense matrices and the cache's dense delta still exist (removed in Task 6).

- [ ] **Step 1: Write the failing tests**

Create `tests/pymarxan/solvers/test_chance_state.py`:

```python
"""``ChanceState`` sidecar and the cache's per-cell chance arrays (spec §3.5).

The contract SA relies on: ``delta_penalty`` equals the difference of the from-scratch
chance penalty across a flip (tolerance 1e-9), never mutates, and ``apply_flip`` commits
exactly the previewed state. The adversarial layouts are the ones the spec review used
to break running sums: p forced to 0 / 1, features whose variance empties and refills,
amounts spread over 1e6 with the dominant-variance cell removed, p = 1e-17 cells beside
large-variance cells, and the sparse-risk layout (1 % risky PUs).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.cache import ProblemCache
from pymarxan.solvers.chance_state import ChanceState
from pymarxan.solvers.probability import evaluate_solution_chance, score_features

TOL = 1e-9


# ----------------------------------------------------------------------------- fixtures


def _problem(pv: pd.DataFrame, n_pu: int, n_feat: int, rng, *, weight: float = 2.0,
             pu_ids=None, target_frac=(0.2, 0.6)) -> ConservationProblem:
    pu_ids = np.arange(1, n_pu + 1) if pu_ids is None else pu_ids
    pu = pd.DataFrame({"id": pu_ids, "cost": rng.uniform(1, 5, n_pu), "status": 0})
    tot = (pv.amount * pv.prob).groupby(pv.species).sum().reindex(
        range(1, n_feat + 1), fill_value=0.0,
    )
    feat = pd.DataFrame({
        "id": range(1, n_feat + 1), "name": [f"f{j}" for j in range(n_feat)],
        "target": tot.values * rng.uniform(*target_frac, n_feat), "spf": 1.0,
        # >= 2 features with different ptarget, one disabled
        "ptarget": np.r_[rng.uniform(0.3, 0.95, n_feat - 1), -1.0],
    })
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pv,
        parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": weight},
    )


def mixed(seed: int, n_pu: int = 120, n_feat: int = 6) -> ConservationProblem:
    """Shuffled PU ids; q forced to 0 or 1 on ~40 % of cells."""
    rng = np.random.default_rng(seed)
    ids = rng.permutation(np.arange(1, n_pu + 1) * 3)
    k = n_pu * 4
    pv = pd.DataFrame({
        "species": rng.integers(1, n_feat + 1, k), "pu": rng.choice(ids, k),
        "amount": rng.uniform(0.1, 3, k),
    }).drop_duplicates(["species", "pu"]).reset_index(drop=True)
    q = rng.uniform(0, 1, len(pv))
    r = rng.random(len(pv))
    q[r < 0.25] = 0.0
    q[(r >= 0.25) & (r < 0.4)] = 1.0
    pv["prob"] = q
    return _problem(pv, n_pu, n_feat, rng, pu_ids=ids)


def few_risky(seed: int, n_pu: int = 40, n_feat: int = 3) -> ConservationProblem:
    """Each feature has only 2-3 uncertain cells: its variance empties and refills."""
    rng = np.random.default_rng(seed)
    rows = [(f, i) for f in range(1, n_feat + 1) for i in range(1, n_pu + 1)
            if rng.random() < 0.5]
    pv = pd.DataFrame(rows, columns=["species", "pu"])
    pv["amount"] = rng.uniform(0.5, 4, len(pv))
    pv["prob"] = 1.0
    for f in range(1, n_feat + 1):
        idx = pv.index[pv.species == f][:3]
        pv.loc[idx, "prob"] = rng.uniform(0.2, 0.8, len(idx))
    return _problem(pv, n_pu, n_feat, rng)


def wide_amounts(seed: int, n_pu: int = 60, n_feat: int = 4) -> ConservationProblem:
    """Amounts spanning 1e0 .. 1e6, q in (0.01, 0.99)."""
    rng = np.random.default_rng(seed)
    rows = [(f, i) for f in range(1, n_feat + 1) for i in range(1, n_pu + 1)
            if rng.random() < 0.6]
    pv = pd.DataFrame(rows, columns=["species", "pu"])
    pv["amount"] = 10 ** rng.uniform(0, 6, len(pv))
    pv["prob"] = rng.uniform(0.01, 0.99, len(pv))
    return _problem(pv, n_pu, n_feat, rng)


def tiny_beside_big(seed: int, n_pu: int = 60, n_feat: int = 4) -> ConservationProblem:
    """q = 1e-17 cells sharing features with a = 1e4, q = 0.5 cells.

    Once the large cells leave, the running V is float residue of order 1e-8 while the
    true V is ~1e-17: without the error-bounded recompute the delta is off by up to the
    weight·ptarget scale (measured 4.0) or NaN.
    """
    rng = np.random.default_rng(seed)
    rows = [(f, i) for f in range(1, n_feat + 1) for i in range(1, n_pu + 1)
            if rng.random() < 0.6]
    pv = pd.DataFrame(rows, columns=["species", "pu"])
    pv["amount"] = rng.uniform(0.5, 3, len(pv))
    q = np.full(len(pv), 1e-17)
    big = rng.random(len(pv)) < 0.15
    pv.loc[big, "amount"] = 1e4
    q[big] = 0.5
    pv["prob"] = q
    return _problem(pv, n_pu, n_feat, rng)


def sparse_risk(seed: int, n_pu: int = 400, n_feat: int = 5) -> ConservationProblem:
    """1 % of PUs risky (presence q = 0.95, i.e. loss p = 0.05); every other cell q = 1."""
    rng = np.random.default_rng(seed)
    rows = [(f, i) for f in range(1, n_feat + 1) for i in range(1, n_pu + 1)
            if rng.random() < 0.6]
    pv = pd.DataFrame(rows, columns=["species", "pu"])
    pv["amount"] = rng.uniform(1, 10, len(pv))
    risky = rng.choice(np.arange(1, n_pu + 1), max(1, n_pu // 100), replace=False)
    pv["prob"] = np.where(pv.pu.isin(risky), 0.95, 1.0)
    return _problem(pv, n_pu, n_feat, rng, target_frac=(0.3, 0.5))


LAYOUTS = {
    "mixed": mixed, "few_risky": few_risky, "wide_amounts": wide_amounts,
    "tiny_beside_big": tiny_beside_big, "sparse_risk": sparse_risk,
}


def _cache(problem: ConservationProblem, rule: str) -> ProblemCache:
    return dataclasses.replace(ProblemCache.from_problem(problem), zero_var_rule=rule)


def _dense_cells(cache: ProblemCache) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = cache.chance_cells["2d"]
    E = np.zeros((cache.n_pu, cache.n_feat))
    V = np.zeros_like(E)
    rows = np.repeat(np.arange(cache.n_pu), np.diff(c.indptr))
    E[rows, c.indices] = c.e
    V[rows, c.indices] = c.v
    return E, V, (V > 0).astype(float)


def _reference(cache: ProblemCache, dense, sel: np.ndarray) -> float:
    """From-scratch chance penalty, independent of ChanceState's arithmetic."""
    Ec, Vc, Pc = dense
    x = sel.astype(float)
    pen = score_features(
        x @ Ec, x @ Vc, np.rint(x @ Pc).astype(np.int64), cache.feat_targets,
        cache.feat_ptarget, cache.zero_var_rule,
    )[3]
    return cache.prob_weight * float(np.sum(pen))


# ----------------------------------------------------------------------- per-cell arrays


class TestChanceCells:
    def test_cells_are_aligned_with_held_csr(self) -> None:
        cache = ProblemCache.from_problem(mixed(0))
        c = cache.chance_cells["2d"]
        np.testing.assert_array_equal(c.indptr, cache.held_csr.indptr)
        np.testing.assert_array_equal(c.indices, cache.held_csr.indices)
        assert c.e.shape == c.v.shape == c.pos.shape == cache.held_csr.data.shape

    def test_cell_values_match_the_puvspr_rows(self) -> None:
        problem = mixed(1)
        cache = ProblemCache.from_problem(problem)
        E, V, P = _dense_cells(cache)
        for row in problem.pu_vs_features.itertuples():
            i = cache.pu_id_to_idx[int(row.pu)]
            j = cache.feat_id_to_col[int(row.species)]
            assert E[i, j] == pytest.approx(row.amount * row.prob, rel=1e-15)
            assert V[i, j] == pytest.approx(
                row.amount ** 2 * row.prob * (1.0 - row.prob), rel=1e-15,
            )
            assert P[i, j] == float(V[i, j] > 0)

    def test_csc_is_the_same_cells_feature_major(self) -> None:
        cache = ProblemCache.from_problem(mixed(2))
        c = cache.chance_cells["2d"]
        E, V, _ = _dense_cells(cache)
        for j in range(cache.n_feat):
            a, b = c.c_indptr[j], c.c_indptr[j + 1]
            rows = c.c_rows[a:b]
            assert np.all(np.diff(rows) > 0)
            np.testing.assert_array_equal(c.c_e[a:b], E[rows, j])
            np.testing.assert_array_equal(c.c_v[a:b], V[rows, j])
        assert c.c_indptr[-1] == len(c.e)

    def test_duplicate_rows_sum_amounts_and_last_prob_wins(self) -> None:
        pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]})
        feat = pd.DataFrame({"id": [1], "name": ["a"], "target": [1.0], "spf": [1.0],
                             "ptarget": [0.5]})
        pv = pd.DataFrame({"species": [1, 1, 1, 9], "pu": [1, 1, 2, 1],
                           "amount": [2.0, 3.0, 4.0, 1.0], "prob": [0.1, 0.4, 0.5, 0.9]})
        problem = ConservationProblem(planning_units=pu, features=feat, pu_vs_features=pv,
                                      parameters={"PROBMODE": 3})
        cache = ProblemCache.from_problem(problem)
        c = cache.chance_cells["2d"]
        assert c.e.tolist() == [5.0 * 0.4, 4.0 * 0.5]
        assert c.v.tolist() == [25.0 * 0.4 * 0.6, 16.0 * 0.5 * 0.5]

    def test_built_only_for_active_terms(self) -> None:
        problem = mixed(3)
        assert set(ProblemCache.from_problem(problem).chance_cells) == {"2d"}
        off = problem.copy_with(parameters={"PROBMODE": 0})
        assert ProblemCache.from_problem(off).chance_cells == {}
        no_data = problem.copy_with(
            pu_vs_features=problem.pu_vs_features.drop(columns=["prob"]),
        )
        assert ProblemCache.from_problem(no_data).chance_cells == {}


# ------------------------------------------------------------------------- ChanceState


@pytest.mark.parametrize("rule", ["marxan", "met"])
def test_from_selection_equals_the_evaluator(rule: str, monkeypatch) -> None:
    import pymarxan.solvers.cache as cache_mod
    import pymarxan.solvers.utils as utils

    monkeypatch.setattr(utils, "_zero_var_rule", lambda problem: rule)
    monkeypatch.setattr(cache_mod, "_zero_var_rule", lambda problem: rule)
    rng = np.random.default_rng(5)
    for build in LAYOUTS.values():
        problem = build(11)
        cache = ProblemCache.from_problem(problem)
        assert cache.zero_var_rule == rule
        for _ in range(5):
            sel = rng.random(cache.n_pu) < rng.uniform(0, 1)
            cs = ChanceState.from_selection(cache, sel)
            assert cs.penalty_total() == pytest.approx(
                evaluate_solution_chance(problem, sel)[1], rel=1e-12, abs=1e-12,
            )


def _run(problem, rule: str, seed: int, flips: int, accept: float,
         sel=None) -> tuple[ChanceState, np.ndarray, float, float]:
    cache = _cache(problem, rule)
    dense = _dense_cells(cache)
    rng = np.random.default_rng(seed)
    sel = rng.random(cache.n_pu) < 0.5 if sel is None else sel.copy()
    cs = ChanceState.from_selection(cache, sel)
    full = _reference(cache, dense, sel)
    tracked = cs.penalty_total()
    assert tracked == pytest.approx(full, abs=TOL)
    worst = 0.0
    for _ in range(flips):
        idx = int(rng.integers(cache.n_pu))
        add = not sel[idx]
        d = cs.delta_penalty(cache, idx, add)
        assert np.isfinite(d)
        s2 = sel.copy()
        s2[idx] = add
        f2 = _reference(cache, dense, s2)
        worst = max(worst, abs(d - (f2 - full)))
        if rng.random() < accept:  # mixes accepted and rejected moves
            cs.apply_flip(cache, idx, add)
            sel, full = s2, f2
            tracked += d
    np.testing.assert_array_equal(cs.selected, sel)
    return cs, sel, worst, abs(tracked - full)


@pytest.mark.parametrize("rule", ["marxan", "met"])
@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_delta_equals_full_difference(layout: str, rule: str) -> None:
    flips = 5000 if layout != "sparse_risk" else 2500
    cs, _, worst, drift = _run(LAYOUTS[layout](7), rule, seed=17, flips=flips, accept=0.6)
    assert worst <= TOL
    assert drift <= 1e-8


def test_removing_the_dominant_variance_cells_largest_first() -> None:
    """A largest-first removal chain at amounts spanning 1e6 (round 3's drift case)."""
    n = 40
    rng = np.random.default_rng(4)
    a = 10 ** (6 * np.arange(n) / (n - 1)) * rng.uniform(0.9, 1.1, n)
    pu = pd.DataFrame({"id": np.arange(1, n + 1), "cost": 1.0, "status": 0})
    pv = pd.DataFrame({"species": 1, "pu": np.arange(1, n + 1), "amount": a, "prob": 0.5})
    keep = 3
    e_k = (a[:keep] * 0.5).sum()
    sd = np.sqrt((a[:keep] ** 2 * 0.25).sum())
    feat = pd.DataFrame({"id": [1], "name": ["f"], "target": [e_k + 0.3 * sd], "spf": 1.0,
                         "ptarget": [0.9]})
    problem = ConservationProblem(planning_units=pu, features=feat, pu_vs_features=pv,
                                  parameters={"PROBMODE": 3})
    cache = ProblemCache.from_problem(problem)
    dense = _dense_cells(cache)
    sel = np.ones(n, dtype=bool)
    cs = ChanceState.from_selection(cache, sel)
    full = _reference(cache, dense, sel)
    for idx in range(n - 1, keep - 1, -1):
        d = cs.delta_penalty(cache, idx, False)
        s2 = sel.copy()
        s2[idx] = False
        f2 = _reference(cache, dense, s2)
        assert abs(d - (f2 - full)) <= TOL
        cs.apply_flip(cache, idx, False)
        sel, full = s2, f2
    assert cs.penalty_total() == pytest.approx(full, abs=TOL)


@pytest.mark.parametrize("rule", ["marxan", "met"])
def test_delta_never_mutates(rule: str) -> None:
    problem = tiny_beside_big(3)
    cache = _cache(problem, rule)
    rng = np.random.default_rng(1)
    cs = ChanceState.from_selection(cache, rng.random(cache.n_pu) < 0.5)
    t = cs.terms["2d"]
    snap = [x.copy() for x in (t.E, t.V, t.N, t.BE, t.BV, t.pen, cs.selected)]
    for _ in range(500):
        idx = int(rng.integers(cache.n_pu))
        cs.delta_penalty(cache, idx, not cs.selected[idx])
    for before, after in zip(snap, (t.E, t.V, t.N, t.BE, t.BV, t.pen, cs.selected)):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("rule", ["marxan", "met"])
def test_commit_equals_preview_and_fresh_state(rule: str) -> None:
    problem = mixed(9)
    cache = _cache(problem, rule)
    rng = np.random.default_rng(2)
    cs = ChanceState.from_selection(cache, rng.random(cache.n_pu) < 0.5)
    t = cs.terms["2d"]
    for _ in range(300):
        idx = int(rng.integers(cache.n_pu))
        add = not cs.selected[idx]
        cs.delta_penalty(cache, idx, add)
        assert cs._preview is not None
        cols, EV2, _, N2, pen2 = cs._preview[3][0]
        E2, V2 = EV2
        cs.apply_flip(cache, idx, add)
        assert cs._preview is None
        # equals the previewed scratch state exactly
        np.testing.assert_array_equal(t.E[cols], E2)
        np.testing.assert_array_equal(t.V[cols], V2)
        np.testing.assert_array_equal(t.N[cols], N2)
        np.testing.assert_array_equal(t.pen[cols], pen2)
        # equals a fresh from_selection to 1e-12 relative
        fresh = ChanceState.from_selection(cache, cs.selected).terms["2d"]
        np.testing.assert_array_equal(t.N, fresh.N)
        np.testing.assert_allclose(t.E, fresh.E, rtol=1e-12, atol=1e-300)
        np.testing.assert_allclose(t.V, fresh.V, rtol=1e-12, atol=1e-300)
        np.testing.assert_allclose(t.pen, fresh.pen, rtol=0, atol=1e-12)


def test_stale_preview_is_never_committed() -> None:
    """delta(i), apply_flip(j), apply_flip(i) commits i against the post-j state."""
    problem = mixed(4)
    cache = ProblemCache.from_problem(problem)
    dense = _dense_cells(cache)
    sel = np.zeros(cache.n_pu, dtype=bool)
    sel[::2] = True
    cs = ChanceState.from_selection(cache, sel)
    i, j = 1, 3
    cs.delta_penalty(cache, i, True)
    cs.apply_flip(cache, j, True)
    cs.apply_flip(cache, i, True)
    sel[[i, j]] = True
    assert cs.penalty_total() == pytest.approx(_reference(cache, dense, sel), abs=TOL)
    np.testing.assert_allclose(
        cs.terms["2d"].E, ChanceState.from_selection(cache, sel).terms["2d"].E, rtol=1e-12,
    )


def test_a_pu_without_cells_has_zero_delta() -> None:
    problem = mixed(5)
    pu = problem.planning_units
    extra = pd.DataFrame({"id": [10_000], "cost": [1.0], "status": [0]})
    problem = problem.copy_with(planning_units=pd.concat([pu, extra], ignore_index=True))
    cache = ProblemCache.from_problem(problem)
    cs = ChanceState.from_selection(cache, np.zeros(cache.n_pu, dtype=bool))
    before = cs.penalty_total()
    assert cs.delta_penalty(cache, cache.n_pu - 1, True) == 0.0
    cs.apply_flip(cache, cache.n_pu - 1, True)
    assert cs.penalty_total() == before
    assert cs.selected[-1]


def test_sparse_risk_recompute_floor() -> None:
    """The 64·eps·|E| floor: mostly-certain features do not recompute on every touch."""
    cs, *_ = _run(sparse_risk(0, n_pu=2000, n_feat=5), "marxan", seed=3, flips=3000,
                  accept=0.6)
    assert cs.n_recompute < 300  # without the floor: ~one per touched feature per flip


def test_tiny_probability_has_variance() -> None:
    """q = 1e-17 gives a positive variance term: nonzero > 0, not zero variance."""
    pu = pd.DataFrame({"id": [1], "cost": [1.0], "status": [0]})
    feat = pd.DataFrame({"id": [1], "name": ["a"], "target": [1.0], "spf": [1.0],
                         "ptarget": [0.5]})
    pv = pd.DataFrame({"species": [1], "pu": [1], "amount": [5.0], "prob": [1e-17]})
    problem = ConservationProblem(planning_units=pu, features=feat, pu_vs_features=pv,
                                  parameters={"PROBMODE": 3})
    cache = ProblemCache.from_problem(problem)
    cs = ChanceState.from_selection(cache, np.array([True]))
    assert cs.terms["2d"].N[0] == 1
    assert cs.terms["2d"].V[0] > 0


def test_no_active_term_is_inert() -> None:
    problem = mixed(6).copy_with(parameters={"PROBMODE": 0})
    cache = ProblemCache.from_problem(problem)
    cs = ChanceState.from_selection(cache, np.ones(cache.n_pu, dtype=bool))
    assert cs.penalty_total() == 0.0
    assert cs.delta_penalty(cache, 0, False) == 0.0
    cs.apply_flip(cache, 0, False)
    assert not cs.selected[0]


def test_variance_snaps_to_zero_when_nonzero_reaches_zero() -> None:
    """Removing a feature's last uncertain cell leaves V exactly 0 and E the exact sum."""
    problem = few_risky(2)
    cache = ProblemCache.from_problem(problem)
    c = cache.chance_cells["2d"]
    sel = np.ones(cache.n_pu, dtype=bool)
    cs = ChanceState.from_selection(cache, sel)
    t = cs.terms["2d"]
    risky_rows = np.unique(np.repeat(np.arange(cache.n_pu), np.diff(c.indptr))[c.pos > 0])
    for i in risky_rows:
        cs.delta_penalty(cache, int(i), False)
        cs.apply_flip(cache, int(i), False)
        sel[i] = False
    assert (t.N == 0).all()
    assert (t.V == 0.0).all()
    fresh = ChanceState.from_selection(cache, sel).terms["2d"]
    np.testing.assert_allclose(t.E, fresh.E, rtol=1e-15)


@pytest.mark.parametrize("rule", ["marxan", "met"])
def test_hot_path_penalty_is_the_core_arithmetic(rule: str) -> None:
    """``ChanceState``'s lean scorer equals ``score_features``' penalty bit for bit."""
    from pymarxan.solvers.probability import _chance_penalty, _ptarget_den

    rng = np.random.default_rng(8)
    n = 500
    N = rng.integers(0, 3, n)
    V = np.where(N > 0, rng.uniform(1e-300, 9, n), 0.0)
    E = rng.uniform(0, 20, n)
    t = rng.uniform(0, 20, n)
    pt = rng.choice([-1.0, 0.0, 1e-6, 0.3, 0.95, 1.0], n)
    ref = score_features(E, V, N, t, pt, rule)[3]
    got = _chance_penalty(E, V, N == 0, t, pt, _ptarget_den(pt), rule == "met")
    np.testing.assert_array_equal(got, ref)
```

Modify `tests/pymarxan/test_fractional_readers.py` (apply this diff):

```diff
diff --git a/tests/pymarxan/test_fractional_readers.py b/tests/pymarxan/test_fractional_readers.py
index 48cbf5c..4d1d871 100644
--- a/tests/pymarxan/test_fractional_readers.py
+++ b/tests/pymarxan/test_fractional_readers.py
@@ -44,6 +44,7 @@ INTERNAL_OR_COUNT = {
     "pymarxan/calibration/sensitivity.py",
     "pymarxan/calibration/sweep.py",
     "pymarxan/solvers/base.py",
+    "pymarxan/solvers/chance_state.py",
     "pymarxan/solvers/clumping.py",
     "pymarxan/solvers/separation.py",
     "pymarxan/solvers/space_state.py",
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_chance_state.py tests/pymarxan/test_fractional_readers.py -q -m 'not bench'`
Expected: ERROR — 1 error collecting `test_chance_state.py` (ModuleNotFoundError: `pymarxan.solvers.chance_state`). The behavioural RED of this task is the new API: its tests were mutation-checked instead (disabled recompute, disabled E floor, pre-flip recompute, mutating preview, missing snap each fail them).

- [ ] **Step 3: Implement**

Modify `src/pymarxan/solvers/cache.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/cache.py b/src/pymarxan/solvers/cache.py
index d290e57..0690b59 100644
--- a/src/pymarxan/solvers/cache.py
+++ b/src/pymarxan/solvers/cache.py
@@ -51,6 +51,7 @@ from __future__ import annotations
 import warnings
 from dataclasses import dataclass, field
 from functools import cached_property
+from typing import cast
 
 import numpy as np
 from scipy.sparse import coo_matrix, csr_matrix
@@ -66,6 +67,51 @@ from pymarxan.solvers.utils import (
 )
 
 
+@dataclass(frozen=True)
+class ChanceCells:
+    """Immutable per-cell contributions of one chance term (spec §3.5).
+
+    PU-major (CSR) arrays share ``held_csr``'s ``indptr`` / ``indices``, so row ``i``'s
+    cells are ``indptr[i]:indptr[i + 1]``. ``ev[0]`` (= ``e``) is each cell's
+    expected-amount term, ``ev[1]`` (= ``v``) its variance term — stacked so one fancy
+    index serves both on the hot path — ``abs_ev`` their absolute values (the error
+    bounds' increments), and ``pos`` is 1 where ``v > 0`` as computed in float, else 0.
+    The feature-major (CSC) copy — ``c_indptr`` / ``c_rows`` / ``c_ev`` (rows ``c_e`` /
+    ``c_v``), rows ascending within a feature — serves ``ChanceState``'s per-feature
+    recompute. For the 2D term (puvspr ``prob`` = ``q``, the probability of presence): ``e = a·q``,
+    ``v = a²·q·(1 − q)``.
+    """
+
+    indptr: np.ndarray
+    indices: np.ndarray
+    ev: np.ndarray
+    abs_ev: np.ndarray
+    pos: np.ndarray
+    c_indptr: np.ndarray
+    c_rows: np.ndarray
+    c_ev: np.ndarray
+
+    @property
+    def c_e(self) -> np.ndarray:
+        """Per-cell expected-amount terms (CSC order)."""
+        return cast(np.ndarray, self.c_ev[0])
+
+    @property
+    def c_v(self) -> np.ndarray:
+        """Per-cell variance terms (CSC order)."""
+        return cast(np.ndarray, self.c_ev[1])
+
+    @property
+    def e(self) -> np.ndarray:
+        """Per-cell expected-amount terms (CSR order)."""
+        return cast(np.ndarray, self.ev[0])
+
+    @property
+    def v(self) -> np.ndarray:
+        """Per-cell variance terms (CSR order)."""
+        return cast(np.ndarray, self.ev[1])
+
+
 @dataclass(frozen=True)
 class ProblemCache:
     """Precomputed arrays derived from a ConservationProblem.
@@ -142,6 +188,11 @@ class ProblemCache:
     chance_terms: frozenset[str] = frozenset()
     # Zero-variance rule of the chance terms (``_zero_var_rule``): "marxan" or "met".
     zero_var_rule: str = "met"
+    # Per-cell contributions of each ACTIVE chance term (``ChanceCells``), keyed by term;
+    # empty when no term is active. Read by ``ChanceState`` (solvers/chance_state.py).
+    chance_cells: dict[str, ChanceCells] = field(
+        default_factory=dict, compare=False, repr=False,
+    )
     # expected_matrix[i, j] = amount_ij · p_ij (presence); collapses to amount
     # when there is no `prob` column or when prob == 1.
     expected_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
@@ -362,6 +413,12 @@ class ProblemCache:
             expected_matrix = np.zeros((0, 0))
             var_matrix = np.zeros((0, 0))
 
+        chance_cells: dict[str, ChanceCells] = {}
+        if "2d" in chance_terms:
+            chance_cells["2d"] = _build_chance_cells_2d(
+                problem, pu_id_to_idx, feat_id_to_col, pu_feat_csr,
+            )
+
         # Per-feature probability target; default -1 (disabled sentinel) for
         # legacy specs without a ptarget column (read_spec already fills this).
         if "ptarget" in feat_df.columns:
@@ -557,6 +614,7 @@ class ProblemCache:
             prob_weight=prob_weight,
             chance_terms=chance_terms,
             zero_var_rule=zero_var_rule,
+            chance_cells=chance_cells,
             expected_matrix=expected_matrix,
             var_matrix=var_matrix,
             feat_ptarget=feat_ptarget,
@@ -958,6 +1016,57 @@ class ProblemCache:
         return total
 
 
+def _build_chance_cells_2d(
+    problem: ConservationProblem,
+    pu_id_to_idx: dict[int, int],
+    feat_id_to_col: dict[int, int],
+    pu_feat_csr: csr_matrix,
+) -> ChanceCells:
+    """``ChanceCells`` of the 2D term from the puvspr ``prob`` column (presence ``q``).
+
+    Each CSR cell (duplicate ``(pu, species)`` rows already summed by
+    ``build_pu_feature_csr``) takes the ``prob`` of the last puvspr row naming it; rows
+    with an unknown id are skipped; a cell no row names keeps ``q = 1`` (certain).
+    """
+    csr = pu_feat_csr
+    n_feat = csr.shape[1]
+    indptr = np.asarray(csr.indptr, dtype=np.int64)
+    indices = np.asarray(csr.indices, dtype=np.int64)
+    amt = np.asarray(csr.data, dtype=np.float64)
+    cell_row = np.repeat(np.arange(csr.shape[0], dtype=np.int64), np.diff(indptr))
+    cell_key = cell_row * n_feat + indices  # ascending: canonical CSR (sum_duplicates)
+
+    pv = problem.pu_vs_features
+    q = np.ones(len(amt), dtype=np.float64)
+    if len(pv) and len(amt):
+        rows = np.fromiter((pu_id_to_idx.get(int(u), -1) for u in pv["pu"].values),
+                           np.int64, len(pv))
+        cols = np.fromiter((feat_id_to_col.get(int(f), -1) for f in pv["species"].values),
+                           np.int64, len(pv))
+        vals = np.asarray(pv["prob"].values, dtype=np.float64)
+        ok = (rows >= 0) & (cols >= 0)
+        keys = rows[ok] * n_feat + cols[ok]
+        vals = vals[ok]
+        # Last row wins: keep each key's last occurrence.
+        last = len(keys) - 1 - np.unique(keys[::-1], return_index=True)[1]
+        keys, vals = keys[last], vals[last]
+        pos = np.searchsorted(cell_key, keys)
+        hit = (pos < len(cell_key)) & (cell_key[np.minimum(pos, len(cell_key) - 1)] == keys)
+        q[pos[hit]] = vals[hit]
+
+    e = amt * q
+    v = (amt ** 2) * q * (1.0 - q)
+    ev = np.vstack([e, v])
+    flag = (v > 0).astype(np.int64)
+    order = np.lexsort((cell_row, indices))  # feature-major, rows ascending
+    c_indptr = np.zeros(n_feat + 1, dtype=np.int64)
+    np.cumsum(np.bincount(indices, minlength=n_feat), out=c_indptr[1:])
+    return ChanceCells(
+        indptr=indptr, indices=indices, ev=ev, abs_ev=np.abs(ev), pos=flag,
+        c_indptr=c_indptr, c_rows=cell_row[order], c_ev=ev[:, order],
+    )
+
+
 def _build_probability(
     problem: ConservationProblem,
     pu_id_to_idx: dict[int, int],
```

Create `src/pymarxan/solvers/chance_state.py`:

```python
"""``ChanceState`` — incremental chance-term sidecar for SA / iterative improvement.

The ``ClumpState`` / ``SepState`` pattern (spec §3.5): ``ProblemCache`` stays stateless
with respect to the selection and holds the immutable per-cell contributions of each
active chance term (``cache.chance_cells``); this sidecar keeps, per term and per feature,
the running expected amount ``E``, variance ``V`` and ``nonzero`` (selected cells whose
variance term is > 0), the float error bounds of ``E`` and ``V`` since the feature's last
recompute, its unweighted penalty, and its own copy of the selection.

- ``delta_penalty(cache, idx, adding)`` re-scores only the flipped PU's features and
  **never mutates** the state; it caches its result (the preview).
- ``apply_flip(cache, idx, adding)`` commits the cached preview when ``(idx, adding)``
  matches, otherwise computes it through the same code path, so the committed state is
  bit for bit the previewed one. Every ``apply_flip`` clears the preview.

Error-bounded recompute. Running sums lose precision (cancellation when a dominant
cell leaves, drift over long chains). Each update adds ``eps·(|old| + |Δ|)`` to the
feature's bound (``BE`` for ``E``, ``BV`` for ``V``). A touched feature is recomputed from
its cells (the CSC column) over the selection *after* the previewed flip when
``V ≤ 0`` while ``nonzero > 0``, or ``BV > τ·V``, or ``BE > max(τ·√V, 64·eps·|E|)``
(τ = 1e-12); the ``64·eps·|E|`` floor keeps the E trigger from firing on every touch when
the coefficient of variation is tiny (mostly-certain cells beside a few risky ones). After
a recompute both bounds reset to ``4·eps·|value|``; ``from_selection`` counts as one. When
``nonzero`` reaches 0, ``E`` and ``V`` snap to the exact sums of the remaining
zero-variance cells (``V = 0``). Zero variance itself is the exact count, never a float
test on ``V``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from pymarxan.solvers.cache import ChanceCells
from pymarxan.solvers.probability import _chance_penalty, _ptarget_den, score_features

EPS = float(np.finfo(np.float64).eps)
TAU = 1e-12
_E_FLOOR = 64.0 * EPS
_RESET = 4.0 * EPS


@dataclass
class _TermState:
    """Running per-feature state of one chance term.

    ``EV`` (2, n_feat): running ``E`` (row 0) and ``V`` (row 1); ``B`` (2, n_feat): their
    float error bounds since the feature's last recompute; ``N``: nonzero counts; ``pen``:
    unweighted penalties.
    """

    cells: ChanceCells
    EV: np.ndarray
    B: np.ndarray
    N: np.ndarray
    pen: np.ndarray

    @property
    def E(self) -> np.ndarray:
        return cast(np.ndarray, self.EV[0])

    @property
    def V(self) -> np.ndarray:
        return cast(np.ndarray, self.EV[1])

    @property
    def BE(self) -> np.ndarray:
        return cast(np.ndarray, self.B[0])

    @property
    def BV(self) -> np.ndarray:
        return cast(np.ndarray, self.B[1])


# One term's preview for the flipped PU's features: (cols, EV2, B2, N2, pen2).
_Payload = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class ChanceState:
    """Mutable chance-term companion to ``ProblemCache`` (see the module docstring)."""

    def __init__(
        self,
        selected: np.ndarray,
        terms: dict[str, _TermState],
        weight: float,
        zero_var_rule: str,
        targets: np.ndarray,
        ptargets: np.ndarray,
    ) -> None:
        self.selected = selected
        self.terms = terms
        self._term_list = list(terms.values())
        self.weight = weight
        self.zero_var_rule = zero_var_rule
        self._met = zero_var_rule == "met"
        self.targets = targets
        self.ptargets = ptargets
        self._ptarget_den = _ptarget_den(ptargets)
        self.n_recompute = 0
        self._preview: tuple[int, bool, float, list[_Payload]] | None = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_selection(cls, cache, selected: np.ndarray) -> ChanceState:
        """Fresh state for ``selected`` (one per SA run / II pass); a recompute of all."""
        sel = np.asarray(selected, dtype=bool).copy()
        n_feat = cache.n_feat
        terms: dict[str, _TermState] = {}
        for name in sorted(cache.chance_terms):
            cells = cache.chance_cells[name]
            mask = sel[cells.c_rows]
            col = np.repeat(np.arange(n_feat), np.diff(cells.c_indptr))[mask]
            EV = np.vstack([
                np.bincount(col, weights=cells.c_e[mask], minlength=n_feat),
                np.bincount(col, weights=cells.c_v[mask], minlength=n_feat),
            ])
            N = np.bincount(col, weights=(cells.c_v[mask] > 0), minlength=n_feat)
            N = N.astype(np.int64)
            pen = score_features(
                EV[0], EV[1], N, cache.feat_targets, cache.feat_ptarget,
                cache.zero_var_rule,
            )[3]
            terms[name] = _TermState(cells=cells, EV=EV, B=_RESET * np.abs(EV), N=N, pen=pen)
        return cls(
            selected=sel, terms=terms, weight=float(cache.prob_weight),
            zero_var_rule=cache.zero_var_rule,
            targets=np.asarray(cache.feat_targets, dtype=np.float64),
            ptargets=np.asarray(cache.feat_ptarget, dtype=np.float64),
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def penalty_total(self) -> float:
        """Weighted chance penalty of the current selection (all active terms)."""
        return self.weight * float(sum(float(np.sum(t.pen)) for t in self._term_list))

    def delta_penalty(self, cache, idx: int, adding: bool) -> float:
        """Δ(chance penalty) for flipping PU ``idx``. Never mutates; caches the preview."""
        delta = 0.0
        payloads: list[_Payload] = []
        for t in self._term_list:
            pay = self._preview_term(t, idx, adding)
            payloads.append(pay)
            if len(pay[0]):
                delta += float(np.sum(pay[4] - t.pen[pay[0]]))
        delta *= self.weight
        self._preview = (idx, adding, delta, payloads)
        return delta

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def apply_flip(self, cache, idx: int, adding: bool) -> None:
        """Commit flipping PU ``idx``: the cached preview if it matches, else recomputed."""
        prev = self._preview
        self._preview = None
        if prev is not None and prev[0] == idx and prev[1] == adding:
            payloads = prev[3]
        else:
            payloads = [self._preview_term(t, idx, adding) for t in self._term_list]
        for t, (cols, EV2, B2, N2, pen2) in zip(self._term_list, payloads):
            if len(cols):
                t.EV[:, cols] = EV2
                t.B[:, cols] = B2
                t.N[cols] = N2
                t.pen[cols] = pen2
        self.selected[idx] = adding

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _preview_term(self, t: _TermState, idx: int, adding: bool) -> _Payload:
        """The touched features' post-flip state of one term (fresh arrays).

        Mutates nothing but the diagnostic ``n_recompute`` counter.
        """
        c = t.cells
        s, e = int(c.indptr[idx]), int(c.indptr[idx + 1])
        cols = c.indices[s:e]
        if s == e:
            return cols, np.zeros((2, 0)), np.zeros((2, 0)), np.zeros(0, np.int64), np.zeros(0)
        d = c.ev[:, s:e]
        N0 = t.N[cols]
        if adding:
            N2 = N0 + c.pos[s:e]
        else:
            d = -d
            N2 = N0 - c.pos[s:e]
        EV0 = t.EV[:, cols]
        EV2 = EV0 + d
        B2 = t.B[:, cols] + EPS * (np.abs(EV0) + c.abs_ev[:, s:e])
        E2 = EV2[0]
        V2 = EV2[1]
        live = N2 > 0
        redo = live & (
            (V2 <= 0)
            | (B2[1] > TAU * V2)
            | ((B2[0] > TAU * np.sqrt(np.maximum(V2, 0.0))) & (B2[0] > _E_FLOOR * np.abs(E2)))
        )
        # nonzero reached 0: snap to the exact sums of the remaining zero-variance cells.
        snap = ~live & (N0 > 0)
        fix = redo | snap
        if fix.any():
            for k in np.flatnonzero(fix):
                ee, vv = self._column_sums(c, int(cols[k]), idx, adding)
                if snap[k]:
                    vv = 0.0
                EV2[0, k] = ee
                EV2[1, k] = vv
                B2[0, k] = _RESET * abs(ee)
                B2[1, k] = _RESET * abs(vv)
                self.n_recompute += 1
        pen2 = _chance_penalty(
            E2, V2, ~live, self.targets[cols], self.ptargets[cols],
            self._ptarget_den[cols], self._met,
        )
        return cols, EV2, B2, N2, pen2

    def _column_sums(
        self, c: ChanceCells, j: int, idx: int, adding: bool,
    ) -> tuple[float, float]:
        """Exact (fresh) ``E``, ``V`` of feature ``j`` over the selection after the flip."""
        a, b = int(c.c_indptr[j]), int(c.c_indptr[j + 1])
        rows = c.c_rows[a:b]
        m = self.selected[rows].astype(np.float64)  # fancy index: a copy
        k = int(np.searchsorted(rows, idx))
        if k < len(rows) and rows[k] == idx:
            m[k] = 1.0 if adding else 0.0
        ee, vv = c.c_ev[:, a:b] @ m
        return float(ee), float(vv)
```

Modify `src/pymarxan/solvers/probability.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/probability.py b/src/pymarxan/solvers/probability.py
index fefde9f..639319a 100644
--- a/src/pymarxan/solvers/probability.py
+++ b/src/pymarxan/solvers/probability.py
@@ -41,7 +41,7 @@ from __future__ import annotations
 
 import math
 from dataclasses import dataclass
-from typing import Literal
+from typing import Literal, cast
 
 import numpy as np
 from scipy.special import ndtr
@@ -85,27 +85,51 @@ def _check_rule(zero_var_rule: str) -> None:
 def _chance_z(
     E: np.ndarray, V: np.ndarray, zero: np.ndarray, target: np.ndarray,
 ) -> np.ndarray:
-    """Z = (target − E)/√V; Marxan's 4 where ``zero`` (no selected cell with variance)."""
-    with np.errstate(divide="ignore", invalid="ignore"):
-        z = (target - E) / np.sqrt(np.where(zero, 1.0, V))
+    """Z = (target − E)/√V; Marxan's 4 where ``zero`` (no selected cell with variance).
+
+    No ``np.errstate`` here (it costs more than the arithmetic on the hot path): the
+    caller guarantees ``V > 0`` wherever ``zero`` is False, or owns the float state.
+    """
+    z = (target - E) / np.sqrt(np.where(zero, 1.0, V))
     return np.where(zero, _MARXAN_ZERO_VARIANCE_SENTINEL, z)
 
 
+def _ptarget_den(ptarget: np.ndarray) -> np.ndarray:
+    """Penalty denominator: ``ptarget`` where enabled (> 0), else 1 (its gap is 0)."""
+    return np.where(ptarget > 0, ptarget, 1.0)
+
+
 def _chance_from_z(
-    z: np.ndarray, zero: np.ndarray, ptarget: np.ndarray, zero_var_rule: str,
+    z: np.ndarray, zero: np.ndarray, ptarget: np.ndarray, ptarget_den: np.ndarray,
+    met: bool,
 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
-    """``(p, shortfall, penalty)`` from Z: the one copy of the probability arithmetic."""
+    """``(p, shortfall, penalty)`` from Z: the one copy of the probability arithmetic.
+
+    ``p ≥ 0``, so ``ptarget > p`` is False for a disabled feature (``ptarget ≤ 0``).
+    """
     p = ndtr(-z)  # = norm.sf(z), Marxan's probZUT up to ~1e-9
-    enabled = ptarget > 0
-    gap = np.where(enabled & (ptarget > p), ptarget - p, 0.0)
-    penalty = gap / np.where(enabled, ptarget, 1.0)
-    if zero_var_rule == "met":
+    gap = np.where(ptarget > p, ptarget - p, 0.0)
+    penalty = gap / ptarget_den
+    if met:
         p = np.where(zero, 1.0, p)
         gap = np.where(zero, 0.0, gap)
         penalty = np.where(zero, 0.0, penalty)
     return p, gap, penalty
 
 
+def _chance_penalty(
+    E: np.ndarray, V: np.ndarray, zero: np.ndarray, target: np.ndarray,
+    ptarget: np.ndarray, ptarget_den: np.ndarray, met: bool,
+) -> np.ndarray:
+    """Unweighted penalties only (``ChanceState``'s hot path): the same operations, in
+    the same order, as the ``penalty`` of :func:`_chance_z` + :func:`_chance_from_z`."""
+    p = ndtr(-_chance_z(E, V, zero, target))
+    penalty = np.where(ptarget > p, ptarget - p, 0.0) / ptarget_den
+    if met:
+        penalty = np.where(zero, 0.0, penalty)
+    return cast(np.ndarray, penalty)
+
+
 def score_features(
     E: np.ndarray,
     V: np.ndarray,
@@ -116,8 +140,8 @@ def score_features(
 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
     """Vectorised :func:`score_feature`: arrays ``(z, p, shortfall, penalty)``.
 
-    The solvers' incremental path (``ChanceState``) and the evaluator use this; it is
-    the same arithmetic as :func:`score_feature`, element for element.
+    The evaluator and ``ChanceState`` use this arithmetic; it is the same as
+    :func:`score_feature`, element for element.
     """
     _check_rule(zero_var_rule)
     E = np.asarray(E, dtype=np.float64)
@@ -125,8 +149,11 @@ def score_features(
     target = np.asarray(target, dtype=np.float64)
     ptarget = np.asarray(ptarget, dtype=np.float64)
     zero = np.asarray(nonzero) == 0
-    z = _chance_z(E, V, zero, target)
-    p, shortfall, penalty = _chance_from_z(z, zero, ptarget, zero_var_rule)
+    with np.errstate(divide="ignore", invalid="ignore"):
+        z = _chance_z(E, V, zero, target)
+    p, shortfall, penalty = _chance_from_z(
+        z, zero, ptarget, _ptarget_den(ptarget), zero_var_rule == "met",
+    )
     return z, p, shortfall, penalty
 
 
@@ -248,7 +275,7 @@ def compute_zscore_penalty(
     z = np.array([zscore_per_feature[f] for f in fids], dtype=np.float64)
     pt = np.array([prob_targets.get(f, -1.0) for f in fids], dtype=np.float64)
     zero = np.array([f in zero_variance for f in fids], dtype=bool)
-    _, _, penalty = _chance_from_z(z, zero, pt, zero_var_rule)
+    _, _, penalty = _chance_from_z(z, zero, pt, _ptarget_den(pt), zero_var_rule == "met")
     return float(weight * float(np.sum(penalty)))
 
 
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_chance_state.py tests/pymarxan/test_fractional_readers.py -q -m 'not bench'`
Expected: 34 passed

- [ ] **Step 5: Gate**

Run: `~/.local/bin/ruff check src tests examples && /home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/ -q -m 'not bench' -p no:cacheprovider && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/python examples/validate_marxan_parity.py`
Expected: ruff and mypy clean; the full suite passes (prototype: 3850 passed, 101 skipped HiGHS cases which run in CI); `examples/validate_marxan_parity.py` prints MIP 35.0, SA 43.0, greedy 45.0; no golden JSON regenerated.

- [ ] **Step 6: Commit**

```bash
git add -A src/pymarxan/solvers/cache.py src/pymarxan/solvers/chance_state.py src/pymarxan/solvers/probability.py tests/pymarxan/solvers/test_chance_state.py tests/pymarxan/test_fractional_readers.py
git commit -F - <<'MSG'
feat(solvers): ChanceState sidecar and per-cell chance arrays (A3a)

New solvers/chance_state.py: ChanceState, in the ClumpState / SepState
pattern (spec §3.5). Per active chance term and per feature it keeps the
running E, V, nonzero count, float error bounds and penalty, and its own
copy of the selection:

- from_selection(cache, selected) builds it (counts as a recompute);
- delta_penalty(cache, idx, adding) re-scores only the flipped PU's
  features, never mutates, and caches the preview;
- apply_flip(cache, idx, adding) commits that preview when (idx, adding)
  match, else computes it through the same code path; it always clears
  the preview;
- penalty_total() is the weighted chance penalty.

Error-bounded recompute exactly as §3.5: BV/BE grow by eps·(|old|+|Δ|);
a feature is recomputed from its CSC column over the selection after the
previewed flip when V <= 0 with nonzero > 0, BV > τ·V or
BE > max(τ·√V, 64·eps·|E|) (τ = 1e-12); bounds reset to 4·eps·|value|;
E and V snap to exact sums (V = 0) when nonzero reaches 0.

ProblemCache gains `chance_cells` (ChanceCells: per-cell E and V terms
and the V > 0 flag, CSR-aligned with held_csr plus a CSC copy), built
for active terms only. Nothing is wired yet: the dense expected/var
matrices and the cache's own chance delta are untouched.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01G9s5x1nmcqSTDeRUXcRPVW
MSG
```

---

### Task 6: Wire `ChanceState` into SA and II; remove the dense chance path

**Files:**
- Modify: `src/pymarxan/solvers/cache.py`
- Modify: `src/pymarxan/solvers/iterative_improvement.py`
- Modify: `src/pymarxan/solvers/simulated_annealing.py`
- Modify: `tests/benchmarks/bench_prob.py`
- Modify: `tests/pymarxan/solvers/test_cache.py`
- Modify: `tests/pymarxan/solvers/test_cache_probmode3.py`
- Create: `tests/pymarxan/solvers/test_sa_chance.py`
- Modify: `tests/pymarxan/solvers/test_score_feature.py`

**Interfaces:**
- Consumes: Task 5's `ChanceState` and `chance_cells`.
- Produces: `compute_full_objective` / `compute_delta_objective` carry no chance term; `expected_matrix` / `var_matrix` and the dense delta are gone. SA builds a `ChanceState` per run (initial objective + `penalty_total()`; `delta_penalty` in the temperature-sampling loop and per move; `apply_flip` on accept). II removal / addition passes build one per pass; the swap pass adds `ChanceState.from_selection(...).penalty_total()` at both `compute_full_objective` calls. `tests/benchmarks/bench_prob.py` measures `ChanceState` (≤ 6× mode 0, dense and sparse risk, 2000 × 20 and 10 000 × 50).

- [ ] **Step 1: Write the failing tests**

Modify `tests/benchmarks/bench_prob.py` (apply this diff):

```diff
diff --git a/tests/benchmarks/bench_prob.py b/tests/benchmarks/bench_prob.py
index ae47bc9..ebe134a 100644
--- a/tests/benchmarks/bench_prob.py
+++ b/tests/benchmarks/bench_prob.py
@@ -1,12 +1,15 @@
-"""Phase 18 PROBMODE 3 SA inner-loop performance benchmark.
+"""PROBMODE 3 (2D chance term) SA inner-loop performance benchmarks.
 
-Pins per-flip cost for the Z-score chance-constraint penalty path —
-the most expensive constraint type in the system because it runs a
-full ``_compute_zscore_penalty`` evaluation per flip (O(n_feat) with
-``scipy.stats.norm.sf`` per active feature).
+v0.47.0 (spec §3.5): the chance term moved out of ``ProblemCache`` into the incremental
+``ChanceState`` sidecar, which re-scores only the flipped PU's features. Before, every
+flip re-summed dense (n_pu, n_feat) expected / variance matrices (~300× mode 0).
 
-Run via ``make bench`` / ``pytest tests/benchmarks/bench_prob.py``; not
-part of ``make check``.
+"Per flip" means the delta, plus the commit on an accepted flip, for mode 0 and mode 3
+alike. Budgets are comparative (mode 3 ≤ 6× mode 0 on the same machine), except the
+two small absolute smoke budgets kept from Phase 18.
+
+Run via ``make bench`` / ``pytest tests/benchmarks/bench_prob.py -m bench``; not part
+of ``make check``.
 """
 from __future__ import annotations
 
@@ -18,98 +21,143 @@ import pytest
 
 from pymarxan.models.problem import ConservationProblem
 from pymarxan.solvers.cache import ProblemCache
+from pymarxan.solvers.chance_state import ChanceState
 
 pytestmark = pytest.mark.bench
 
+RATIO_BOUND = 6.0
+
 
 def _make_probmode3_problem(
-    n_pu: int = 500, n_features: int = 10, seed: int = 0,
+    n_pu: int = 500, n_features: int = 10, seed: int = 0, risk: str = "dense",
+    cells_per_pu: int = 5,
 ) -> ConservationProblem:
-    """Synthetic PROBMODE 3 problem with per-cell Bernoulli probabilities."""
+    """Synthetic PROBMODE 3 problem (the spec review's layout: ~``cells_per_pu`` feature
+    cells per PU).
+
+    ``risk="dense"``: presence q = 1 − p with loss p ~ U(0, 0.3) on every cell.
+    ``risk="sparse"``: 1 % of PUs risky (p = 0.05, q = 0.95); every other cell q = 1.
+    """
     rng = np.random.default_rng(seed)
     pu_ids = np.arange(1, n_pu + 1)
     pu = pd.DataFrame({
-        "id": pu_ids,
-        "cost": rng.uniform(1.0, 10.0, size=n_pu),
+        "id": pu_ids, "cost": rng.uniform(1.0, 10.0, size=n_pu),
         "status": np.zeros(n_pu, dtype=int),
     })
-
-    feat_ids = np.arange(1, n_features + 1)
+    k = n_pu * cells_per_pu
+    pv = pd.DataFrame({
+        "species": rng.integers(1, n_features + 1, k), "pu": rng.integers(1, n_pu + 1, k),
+        "amount": rng.uniform(0.1, 3.0, k),
+    }).drop_duplicates(["species", "pu"]).reset_index(drop=True)
+    if risk == "dense":
+        pv["prob"] = 1.0 - rng.uniform(0.0, 0.3, len(pv))
+    else:
+        risky = rng.choice(pu_ids, max(1, n_pu // 100), replace=False)
+        pv["prob"] = np.where(pv["pu"].isin(risky), 0.95, 1.0)
+    total = (pv["amount"] * pv["prob"]).groupby(pv["species"]).sum()
+    total = total.reindex(range(1, n_features + 1), fill_value=0.0).to_numpy()
     features = pd.DataFrame({
-        "id": feat_ids,
-        "name": [f"p_f{i}" for i in feat_ids],
-        "target": [10.0] * n_features,
-        "spf": [1.0] * n_features,
-        "ptarget": [0.9] * n_features,  # every feature is PROBMODE 3 active
+        "id": np.arange(1, n_features + 1),
+        "name": [f"p_f{i}" for i in range(1, n_features + 1)],
+        "target": 0.4 * total, "spf": 1.0,
+        "ptarget": rng.uniform(0.3, 0.95, n_features),
     })
-
-    rows = []
-    for fid in feat_ids:
-        present = rng.random(n_pu) < 0.6
-        for k in range(n_pu):
-            if present[k]:
-                rows.append({
-                    "species": int(fid),
-                    "pu": int(pu_ids[k]),
-                    "amount": float(rng.uniform(0.5, 2.0)),
-                    "prob": float(rng.uniform(0.05, 0.3)),
-                })
-    puvspr = pd.DataFrame(rows)
-
     return ConservationProblem(
-        planning_units=pu, features=features, pu_vs_features=puvspr,
+        planning_units=pu, features=features, pu_vs_features=pv,
         parameters={"PROBMODE": 3, "BLM": 0.0},
     )
 
 
+def _per_flip_seconds(
+    cache: ProblemCache, chance: bool, sel0: np.ndarray, idxs: np.ndarray,
+    accept: np.ndarray,
+) -> float:
+    """Mean per-flip time: the cache delta (+ ChanceState delta), and on an accepted flip
+    the held update (+ ChanceState commit). Best of 3 repeats."""
+    best = float("inf")
+    for _ in range(3):
+        sel = sel0.copy()
+        held = cache.compute_held(sel)
+        total_cost = float(cache.costs[sel].sum())
+        state = ChanceState.from_selection(cache, sel) if chance else None
+        t0 = time.perf_counter()
+        for i, acc in zip(idxs.tolist(), accept.tolist()):
+            adding = not sel[i]
+            cache.compute_delta_objective(i, sel, held, total_cost, 0.0)
+            if state is not None:
+                state.delta_penalty(cache, i, adding)
+            if acc:
+                sign = 1.0 if adding else -1.0
+                sel[i] = adding
+                cache.apply_flip_to_held(held, i, sign)
+                total_cost += sign * cache.costs[i]
+                if state is not None:
+                    state.apply_flip(cache, i, adding)
+        best = min(best, (time.perf_counter() - t0) / len(idxs))
+    return best
+
+
+@pytest.mark.parametrize("risk", ["dense", "sparse"])
+@pytest.mark.parametrize("n_pu, n_feat", [(2000, 20), (10_000, 50)])
+def test_probmode3_per_flip_within_6x_mode0(n_pu: int, n_feat: int, risk: str) -> None:
+    """Per-flip cost of mode 3 (cache + ChanceState) ≤ 6× mode 0 (cache only)."""
+    problem = _make_probmode3_problem(n_pu, n_feat, seed=0, risk=risk)
+    cache3 = ProblemCache.from_problem(problem)
+    assert cache3.chance_terms == {"2d"}
+    cache0 = ProblemCache.from_problem(problem.copy_with(parameters={"PROBMODE": 0}))
+    rng = np.random.default_rng(1)
+    sel0 = rng.random(n_pu) < 0.5
+    idxs = rng.integers(0, n_pu, 6000)
+    accept = rng.random(6000) < 0.6
+    t0 = _per_flip_seconds(cache0, False, sel0, idxs, accept)
+    t3 = _per_flip_seconds(cache3, True, sel0, idxs, accept)
+    ratio = t3 / t0
+    assert ratio <= RATIO_BOUND, (
+        f"PROBMODE 3 per flip {t3 * 1e6:.1f} µs = {ratio:.2f}× mode 0 "
+        f"({t0 * 1e6:.1f} µs) at {n_pu}×{n_feat}, {risk} risk (bound {RATIO_BOUND}×)."
+    )
+
+
 def test_probmode3_delta_under_2500us_at_500pu():
-    """Per-flip ``compute_delta_objective`` under PROBMODE 3 must run
-    in median < 2.5 ms on a 500-PU × 10-feature problem.
-
-    The current implementation does a full Z-score recomputation
-    pre/post flip (cache.py:515-528). The expected-matrix sum is
-    O(n_selected · n_feat); the norm.sf loop is O(n_feat). Measured
-    median on first install was ~1 ms; the 2.5 ms budget gives ~2.5×
-    headroom for slower CI machines while still catching genuine
-    regressions (e.g. accidental O(n_pu²) scans)."""
+    """Per-flip ``ChanceState`` cost (delta + commit) under PROBMODE 3 on a 500-PU ×
+    10-feature problem: median < 2.5 ms. Re-pointed (v0.47.0) from the cache's dense
+    delta, which no longer carries the chance term (it would pass measuring nothing)."""
     problem = _make_probmode3_problem(n_pu=500, n_features=10, seed=0)
     cache = ProblemCache.from_problem(problem)
-    assert cache.probmode == 3, "fixture is meant to be PROBMODE 3"
+    assert cache.chance_terms == {"2d"}, "fixture is meant to be PROBMODE 3"
 
     rng = np.random.default_rng(0)
     selected = rng.random(cache.n_pu) > 0.5
-    held = cache.compute_held(selected)
-    total_cost = float(np.sum(cache.costs[selected]))
+    state = ChanceState.from_selection(cache, selected)
 
     samples: list[float] = []
     for _ in range(200):
         idx = int(rng.integers(0, cache.n_pu))
+        adding = not state.selected[idx]
         t0 = time.perf_counter()
-        cache.compute_delta_objective(idx, selected, held, total_cost, blm=0.0)
+        state.delta_penalty(cache, idx, adding)
+        state.apply_flip(cache, idx, adding)
         samples.append(time.perf_counter() - t0)
 
     median_us = float(np.median(samples)) * 1e6
     assert median_us < 2500.0, (
-        f"PROBMODE 3 delta median per-flip cost: {median_us:.1f} µs "
-        "(budget 2500 µs). compute_delta_objective likely regressed — "
-        "verify _compute_zscore_penalty still O(n_feat) per flip."
+        f"PROBMODE 3 ChanceState per-flip median: {median_us:.1f} µs (budget 2500 µs)."
     )
 
 
 def test_probmode3_full_objective_under_5ms_at_500pu():
-    """Full objective recomputation (no incremental state) — budget 5 ms
-    at 500 PU × 10 features. Used as a reference for delta speedup."""
+    """From-scratch chance term (``ChanceState.from_selection``) — budget 5 ms at 500 PU ×
+    10 features. Re-pointed (v0.47.0) from ``compute_full_objective``, which no longer
+    includes the chance term."""
     problem = _make_probmode3_problem(n_pu=500, n_features=10, seed=0)
     cache = ProblemCache.from_problem(problem)
     rng = np.random.default_rng(0)
     selected = rng.random(cache.n_pu) > 0.5
-    held = cache.compute_held(selected)
 
     t0 = time.perf_counter()
-    cache.compute_full_objective(selected, held, blm=0.0)
+    ChanceState.from_selection(cache, selected).penalty_total()
     elapsed = time.perf_counter() - t0
 
     assert elapsed < 0.005, (
-        f"PROBMODE 3 full objective took {elapsed*1000:.1f} ms "
-        "(budget 5 ms)."
+        f"PROBMODE 3 from-scratch chance term took {elapsed * 1000:.1f} ms (budget 5 ms)."
     )
```

Modify `tests/pymarxan/solvers/test_cache.py` (apply this diff):

```diff
diff --git a/tests/pymarxan/solvers/test_cache.py b/tests/pymarxan/solvers/test_cache.py
index 1c997ec..8e69bee 100644
--- a/tests/pymarxan/solvers/test_cache.py
+++ b/tests/pymarxan/solvers/test_cache.py
@@ -610,9 +610,13 @@ def test_delta_matches_dense_penalty(n_pu, n_feat):
         assert abs(got - expected) <= 1e-9 * (1.0 + abs(expected))
 
 
-def test_expected_var_gated_on_probmode():
+def test_chance_arrays_gated_on_chance_terms():
+    """Rewritten (v0.47.0): the dense expected / variance matrices are gone; the per-cell
+    chance arrays are built only for active chance terms (``cache.chance_terms``)."""
     cache = ProblemCache.from_problem(_random_problem())  # probmode 0
-    assert cache.expected_matrix.size == 0 and cache.var_matrix.size == 0
+    assert cache.chance_terms == frozenset()
+    assert cache.chance_cells == {}
+    assert not hasattr(cache, "expected_matrix") and not hasattr(cache, "var_matrix")
 
 
 def test_plain_solve_does_not_densify():
```

Modify `tests/pymarxan/solvers/test_cache_probmode3.py` (apply this diff):

```diff
diff --git a/tests/pymarxan/solvers/test_cache_probmode3.py b/tests/pymarxan/solvers/test_cache_probmode3.py
index 47239af..1a6141c 100644
--- a/tests/pymarxan/solvers/test_cache_probmode3.py
+++ b/tests/pymarxan/solvers/test_cache_probmode3.py
@@ -1,13 +1,14 @@
 """ProblemCache PROB2D / PROBMODE 3 bookkeeping.
 
-Phase 18 Batch 3 Task 7. Verifies the cache:
-
-- Precomputes expected_matrix, var_matrix, feat_ptarget from the new
-  optional columns (with sensible defaults when columns absent).
-- compute_full_objective adds the PROBMODE 3 chance-constraint penalty.
-- compute_delta_objective tracks variance and produces deltas that match
-  the full-recompute difference (delta correctness — the bedrock SA
-  contract that has to hold for the inner loop to be valid).
+Phase 18 Batch 3 Task 7, reworked in v0.47.0 (spec §3.5). Verifies the cache:
+
+- Precomputes the per-cell 2D chance arrays (``chance_cells["2d"]``) and
+  feat_ptarget from the optional columns; no term (and no arrays) without
+  a puvspr ``prob`` column.
+- compute_full_objective / compute_delta_objective no longer include the
+  chance term: it lives in the ``ChanceState`` sidecar, whose delta plus the
+  cache's delta matches the difference of the full objectives (the bedrock
+  SA contract).
 """
 from __future__ import annotations
 
@@ -19,6 +20,8 @@ import pytest
 
 from pymarxan.models.problem import ConservationProblem
 from pymarxan.solvers.cache import ProblemCache
+from pymarxan.solvers.chance_state import ChanceState
+from pymarxan.solvers.utils import build_solution
 
 
 def _make_probmode3_problem(num_pus: int = 6, seed: int = 0) -> ConservationProblem:
@@ -65,34 +68,44 @@ class TestProbmode3CacheFields:
         cache = ProblemCache.from_problem(problem)
         assert cache.prob_weight == 2.5
 
-    def test_expected_matrix_uses_amount_times_prob(self):
+    def test_expected_cells_use_amount_times_prob(self):
         """Marxan PROB2D presence convention: expected = amount * prob
-        (probability.cpp ComputeP_AllPUsSelected_2D), NOT amount * (1 - prob)
-        (that's the PROBMODE 2 / Marxan 1D loss convention)."""
+        (probability.c ComputeProbability2D), NOT amount * (1 - prob)
+        (that's the PROBMODE 2 / Marxan 1D loss convention).
+
+        Moved (v0.47.0) from the removed dense ``expected_matrix`` to the
+        per-cell arrays ``chance_cells["2d"].e``.
+        """
         problem = _make_probmode3_problem(num_pus=2, seed=42)
         cache = ProblemCache.from_problem(problem)
-        puvspr = problem.pu_vs_features
-        for _, row in puvspr.iterrows():
-            pu_idx = cache.pu_id_to_idx[int(row["pu"])]
-            feat_idx = cache.feat_id_to_col[int(row["species"])]
-            amount = float(row["amount"])
-            prob = float(row["prob"])
-            assert cache.expected_matrix[pu_idx, feat_idx] == pytest.approx(
-                amount * prob
-            )
-
-    def test_var_matrix_is_bernoulli_form(self):
+        cells = cache.chance_cells["2d"]
+        for _, row in problem.pu_vs_features.iterrows():
+            i = cache.pu_id_to_idx[int(row["pu"])]
+            j = cache.feat_id_to_col[int(row["species"])]
+            s, e = cells.indptr[i], cells.indptr[i + 1]
+            k = s + int(np.flatnonzero(cells.indices[s:e] == j)[0])
+            assert cells.e[k] == pytest.approx(float(row["amount"]) * float(row["prob"]))
+
+    def test_variance_cells_are_bernoulli_form(self):
+        """var = amount² · prob · (1 − prob); moved (v0.47.0) from ``var_matrix``."""
         problem = _make_probmode3_problem(num_pus=2, seed=42)
         cache = ProblemCache.from_problem(problem)
-        # var = amount^2 * prob * (1 - prob)
-        puvspr = problem.pu_vs_features
-        for _, row in puvspr.iterrows():
-            pu_idx = cache.pu_id_to_idx[int(row["pu"])]
-            feat_idx = cache.feat_id_to_col[int(row["species"])]
-            amount = float(row["amount"])
-            prob = float(row["prob"])
-            expected = amount ** 2 * prob * (1.0 - prob)
-            assert cache.var_matrix[pu_idx, feat_idx] == pytest.approx(expected)
+        cells = cache.chance_cells["2d"]
+        for _, row in problem.pu_vs_features.iterrows():
+            i = cache.pu_id_to_idx[int(row["pu"])]
+            j = cache.feat_id_to_col[int(row["species"])]
+            s, e = cells.indptr[i], cells.indptr[i + 1]
+            k = s + int(np.flatnonzero(cells.indices[s:e] == j)[0])
+            amount, prob = float(row["amount"]), float(row["prob"])
+            assert cells.v[k] == pytest.approx(amount ** 2 * prob * (1.0 - prob))
+            assert cells.pos[k] == int(cells.v[k] > 0)
+
+    def test_dense_chance_matrices_are_gone(self):
+        """The dense (n_pu, n_feat) expected / variance matrices were removed (v0.47.0)."""
+        cache = ProblemCache.from_problem(_make_probmode3_problem())
+        assert not hasattr(cache, "expected_matrix")
+        assert not hasattr(cache, "var_matrix")
+        assert "pu_feat_matrix" not in cache.__dict__  # mode 3 no longer densifies
 
     def test_feat_ptarget_stored(self):
         problem = _make_probmode3_problem()
@@ -101,47 +114,55 @@ class TestProbmode3CacheFields:
             cache.feat_ptarget, np.array([0.95, 0.80]),
         )
 
-    def test_missing_prob_column_is_deterministic(self):
-        """No `prob` column → presence defaults to 1.0 (certain): variance is
-        0 throughout and expected collapses to the plain amount."""
+    def test_missing_prob_column_turns_the_term_off(self):
+        """No `prob` column → the 2D chance term is off ("term without data", spec §3.1).
+
+        Rewritten (v0.47.0): this used to assert that the dense matrices collapsed to the
+        deterministic case (variance 0). The term is now inactive instead: no per-cell
+        arrays, no ChanceState, chance penalty 0 and no chance fields on the Solution.
+        """
         problem = _make_probmode3_problem()
         problem = problem.copy_with(
             pu_vs_features=problem.pu_vs_features.drop(columns=["prob"]),
         )
         cache = ProblemCache.from_problem(problem)
-        np.testing.assert_array_equal(
-            cache.var_matrix, np.zeros_like(cache.var_matrix),
-        )
-        # expected_matrix collapses to amount (since prob defaults to 1.0)
-        np.testing.assert_array_almost_equal(
-            cache.expected_matrix, cache.pu_feat_matrix,
-        )
+        assert cache.chance_terms == frozenset()
+        assert cache.chance_cells == {}
+        sel = np.ones(cache.n_pu, dtype=bool)
+        assert ChanceState.from_selection(cache, sel).penalty_total() == 0.0
+        sol = build_solution(problem, sel, 0.0)
+        assert sol.prob_penalty is None and sol.prob_shortfalls is None
+        det = build_solution(problem.copy_with(parameters={"PROBMODE": 0}), sel, 0.0)
+        assert sol.objective == det.objective
 
 
 class TestProbmode3FullObjective:
     """compute_full_objective adds the PROBMODE 3 penalty on top of
     the deterministic objective."""
 
-    def test_full_objective_includes_probability_penalty(self):
+    def test_full_objective_plus_chance_state_is_the_evaluator_objective(self):
+        """The cache's full objective excludes the chance term; adding
+        ``ChanceState.penalty_total()`` gives the evaluator's (reported) objective.
+
+        Rewritten (v0.47.0): the old test only asserted ``obj_3 >= obj_0``, which
+        stays true (vacuously) once the term moves out of the cache (spec §3.6).
+        """
         problem = _make_probmode3_problem(num_pus=4, seed=7)
         cache = ProblemCache.from_problem(problem)
-        rng = np.random.default_rng(7)
-        selected = rng.random(cache.n_pu) > 0.5
-        # Ensure something is selected (otherwise penalty path is degenerate)
-        if not selected.any():
-            selected[0] = True
-        held = cache.compute_held(selected)
-
-        # Same problem under PROBMODE 0 (no probability penalty)
         problem_det = copy.deepcopy(problem)
         problem_det.parameters["PROBMODE"] = 0
         cache_det = ProblemCache.from_problem(problem_det)
-        held_det = cache_det.compute_held(selected)
-
-        obj_3 = cache.compute_full_objective(selected, held, blm=0.0)
-        obj_0 = cache_det.compute_full_objective(selected, held_det, blm=0.0)
-        # PROBMODE 3 must add a non-negative term
-        assert obj_3 >= obj_0 - 1e-9
+        rng = np.random.default_rng(7)
+        for _ in range(8):
+            selected = rng.random(cache.n_pu) > 0.5
+            held = cache.compute_held(selected)
+            obj_3 = cache.compute_full_objective(selected, held, blm=0.0)
+            assert obj_3 == cache_det.compute_full_objective(selected, held, blm=0.0)
+            chance = ChanceState.from_selection(cache, selected).penalty_total()
+            reported = build_solution(problem, selected, 0.0)
+            assert reported.prob_penalty is not None
+            assert chance == pytest.approx(reported.prob_penalty, abs=1e-12)
+            assert obj_3 + chance == pytest.approx(reported.objective, abs=1e-9)
 
     def test_full_objective_disabled_ptarget_no_penalty(self):
         """When every feature has ptarget=-1, PROBMODE 3 contributes 0."""
@@ -163,8 +184,8 @@ class TestProbmode3FullObjective:
 
 
 class TestProbmode3Delta:
-    """compute_delta_objective produces deltas that match the full-recompute
-    difference — this is the bedrock SA inner-loop contract."""
+    """The cache's delta plus ``ChanceState.delta_penalty`` matches the full-recompute
+    difference — the bedrock SA inner-loop contract (moved to ChanceState in v0.47.0)."""
 
     def test_delta_matches_full_under_probmode3(self):
         problem = _make_probmode3_problem(num_pus=8, seed=42)
@@ -176,18 +197,27 @@ class TestProbmode3Delta:
         held = cache.compute_held(selected)
         total_cost = float(np.sum(cache.costs[selected]))
         blm = 1.5
+        chance = ChanceState.from_selection(cache, selected)
 
-        for _ in range(10):
+        for _ in range(50):
             flip_idx = int(rng.integers(cache.n_pu))
-            full_before = cache.compute_full_objective(selected, held, blm)
+            adding = not selected[flip_idx]
+            full_before = (
+                cache.compute_full_objective(selected, held, blm)
+                + ChanceState.from_selection(cache, selected).penalty_total()
+            )
             delta = cache.compute_delta_objective(
                 flip_idx, selected, held, total_cost, blm,
-            )
+            ) + chance.delta_penalty(cache, flip_idx, adding)
             # Apply the flip
-            sign = -1.0 if selected[flip_idx] else 1.0
-            held = held + sign * cache.pu_feat_matrix[flip_idx]
+            sign = 1.0 if adding else -1.0
+            cache.apply_flip_to_held(held, flip_idx, sign)
             total_cost += sign * cache.costs[flip_idx]
             selected = selected.copy()
-            selected[flip_idx] = not selected[flip_idx]
-            full_after = cache.compute_full_objective(selected, held, blm)
-            assert delta == pytest.approx(full_after - full_before, abs=1e-7)
+            selected[flip_idx] = adding
+            chance.apply_flip(cache, flip_idx, adding)
+            full_after = (
+                cache.compute_full_objective(selected, held, blm)
+                + ChanceState.from_selection(cache, selected).penalty_total()
+            )
+            assert delta == pytest.approx(full_after - full_before, abs=1e-9)
```

Create `tests/pymarxan/solvers/test_sa_chance.py`:

```python
"""SA and iterative improvement optimise the 2D chance term through ``ChanceState``.

v0.47.0 (spec §3.5): the chance term left ``ProblemCache``'s full objective and delta; SA
builds a ``ChanceState`` per run, II per removal / addition pass, and the swap pass adds a
from-scratch chance term at both of its full-objective calls. The reported objective (the
evaluator's, via ``build_solution``) must equal what the solvers tracked.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.cache import ProblemCache
from pymarxan.solvers.iterative_improvement import IterativeImprovementSolver
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
from pymarxan.solvers.utils import build_solution


def _random_problem(seed: int, n_pu: int = 14, n_feat: int = 3) -> ConservationProblem:
    rng = np.random.default_rng(seed)
    pu = pd.DataFrame({
        "id": rng.permutation(np.arange(1, n_pu + 1) * 7),
        "cost": rng.uniform(1, 5, n_pu), "status": 0,
    })
    rows = [(f, int(pid)) for f in range(1, n_feat + 1) for pid in pu.id if rng.random() < 0.6]
    pv = pd.DataFrame(rows, columns=["species", "pu"])
    pv["amount"] = rng.uniform(1, 6, len(pv))
    q = rng.uniform(0.3, 0.95, len(pv))
    q[rng.random(len(pv)) < 0.2] = 1.0
    pv["prob"] = q
    tot = (pv.amount * pv.prob).groupby(pv.species).sum().values
    feat = pd.DataFrame({
        "id": range(1, n_feat + 1), "name": [f"f{j}" for j in range(n_feat)],
        "target": tot * 0.35, "spf": 3.0, "ptarget": [0.95, 0.7, 0.85][:n_feat],
    })
    b = [(int(a), int(c), 1.0) for a, c in zip(pu.id[:-1], pu.id[1:])]
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pv,
        boundary=pd.DataFrame(b, columns=["id1", "id2", "boundary"]),
        parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 20.0, "BLM": 0.3,
                    "NUMITNS": 20_000, "NUMTEMP": 100},
    )


def _keeps_needed_pu_problem() -> ConservationProblem:
    """4 PUs x 1 feature, each PU alone meets the deterministic target (amount 10, target 10)
    but the chance target needs all four (q = 0.6, ptarget 0.9): k = 4 PUs cost 4 with no
    chance penalty; k = 3 cost 3 + 50·(0.9 − P3)/0.9 ≈ 7.06 (SPF 100 rules out k = 0)."""
    pu = pd.DataFrame({"id": [1, 2, 3, 4], "cost": [1.0] * 4, "status": [0] * 4})
    feat = pd.DataFrame({"id": [1], "name": ["f"], "target": [10.0], "spf": [100.0],
                         "ptarget": [0.9]})
    pv = pd.DataFrame({"species": [1] * 4, "pu": [1, 2, 3, 4], "amount": [10.0] * 4,
                       "prob": [0.6] * 4})
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pv,
        parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 50.0, "BLM": 0.0},
    )


def _obj(problem: ConservationProblem, sel: np.ndarray) -> float:
    return build_solution(problem, sel, float(problem.parameters.get("BLM", 0.0))).objective


class TestSimulatedAnnealing:
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_reported_objective_equals_tracked(self, seed: int) -> None:
        problem = _random_problem(seed)
        (sol,) = SimulatedAnnealingSolver().solve(
            problem, SolverConfig(num_solutions=1, seed=seed),
        )
        assert sol.prob_penalty is not None
        assert sol.objective == pytest.approx(
            sol.metadata["history"]["best_objective"][-1], abs=1e-9,
        )

    def test_chance_term_is_optimised(self) -> None:
        problem = _keeps_needed_pu_problem().copy_with(
            parameters={**_keeps_needed_pu_problem().parameters,
                        "NUMITNS": 5000, "NUMTEMP": 50},
        )
        sols = SimulatedAnnealingSolver().solve(problem, SolverConfig(num_solutions=3, seed=4))
        for sol in sols:
            assert sol.selected.all()
            assert sol.objective == pytest.approx(4.0)

    def test_term_off_without_data_matches_mode_0(self) -> None:
        """Mode 3 with no puvspr `prob`: the term is off (silently, in this release) and SA
        runs exactly as mode 0."""
        p3 = _random_problem(5)
        p3 = p3.copy_with(pu_vs_features=p3.pu_vs_features.drop(columns=["prob"]))
        p0 = p3.copy_with(parameters={**p3.parameters, "PROBMODE": 0})
        assert ProblemCache.from_problem(p3).chance_terms == frozenset()
        cfg = SolverConfig(num_solutions=2, seed=9)
        s3 = SimulatedAnnealingSolver().solve(p3, cfg)
        s0 = SimulatedAnnealingSolver().solve(p0, cfg)
        for a, b in zip(s3, s0):
            assert a.selected.tolist() == b.selected.tolist()
            assert a.objective == b.objective
            assert a.prob_penalty is None


def _single_moves_do_not_improve(problem, sol, *, removals: bool, additions: bool) -> None:
    base = sol.objective
    for i in range(len(sol.selected)):
        if (sol.selected[i] and removals) or (not sol.selected[i] and additions):
            s2 = sol.selected.copy()
            s2[i] = not s2[i]
            assert _obj(problem, s2) >= base - 1e-9


class TestIterativeImprovement:
    @pytest.mark.parametrize("itimptype", [1, 2])
    def test_removal_pass_keeps_a_pu_the_chance_term_needs(self, itimptype: int) -> None:
        problem = _keeps_needed_pu_problem()
        (sol,) = IterativeImprovementSolver(itimptype=itimptype).solve(
            problem, SolverConfig(num_solutions=1, seed=1),
        )
        assert sol.selected.all()
        assert sol.objective == pytest.approx(4.0)

    @pytest.mark.parametrize("seed", range(4))
    @pytest.mark.parametrize("itimptype", [1, 2])
    def test_passes_stop_at_an_evaluator_local_optimum(self, itimptype: int, seed: int) -> None:
        """Reported == tracked: every move the pass rejected also fails under the
        evaluator's objective (the passes saw the evaluator's chance term)."""
        problem = _random_problem(10 + seed)
        (sol,) = IterativeImprovementSolver(itimptype=itimptype).solve(
            problem, SolverConfig(num_solutions=1),
        )
        assert sol.objective == pytest.approx(_obj(problem, sol.selected), abs=1e-12)
        _single_moves_do_not_improve(
            problem, sol, removals=True, additions=itimptype == 2,
        )

    @pytest.mark.parametrize("seed", range(3))
    def test_swap_pass_stops_at_an_evaluator_local_optimum(self, seed: int) -> None:
        problem = _random_problem(20 + seed, n_pu=10)
        rng = np.random.default_rng(seed)
        start = build_solution(problem, rng.random(10) < 0.5, problem.parameters["BLM"])
        sol = IterativeImprovementSolver(itimptype=3).improve(problem, start)
        assert sol.prob_penalty is not None
        base = sol.objective
        assert base <= start.objective + 1e-12
        sel = sol.selected
        for r, a in itertools.product(np.flatnonzero(sel), np.flatnonzero(~sel)):
            s2 = sel.copy()
            s2[r], s2[a] = False, True
            assert _obj(problem, s2) >= base - 1e-9

    def test_swap_pass_optimises_the_chance_term(self) -> None:
        """Swap a risky PU for a safe one that costs more: worth it only with the term.

        PU 1 (cost 1, q = 0.5) vs PU 2 (cost 2, q = 1.0); amount 10, target 5, ptarget 0.9,
        weight 50. Holding PU 1: P = sf((5 − 5)/5) = 0.5 → penalty 50·0.4/0.9 ≈ 22.2;
        holding PU 2: zero variance, "met" in this release → 0, objective 2.
        """
        pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 2.0], "status": [0, 0]})
        feat = pd.DataFrame({"id": [1], "name": ["f"], "target": [5.0], "spf": [1.0],
                             "ptarget": [0.9]})
        pv = pd.DataFrame({"species": [1, 1], "pu": [1, 2], "amount": [10.0, 10.0],
                           "prob": [0.5, 1.0]})
        problem = ConservationProblem(
            planning_units=pu, features=feat, pu_vs_features=pv,
            parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 50.0, "BLM": 0.0,
                        "PROBZEROVAR": 1},
        )
        start = build_solution(problem, np.array([True, False]), 0.0)
        sol = IterativeImprovementSolver(itimptype=3).improve(problem, start)
        assert sol.selected.tolist() == [False, True]
        assert sol.objective == pytest.approx(2.0)
```

Modify `tests/pymarxan/solvers/test_score_feature.py` (apply this diff):

```diff
diff --git a/tests/pymarxan/solvers/test_score_feature.py b/tests/pymarxan/solvers/test_score_feature.py
index c0f97d8..e2b5bab 100644
--- a/tests/pymarxan/solvers/test_score_feature.py
+++ b/tests/pymarxan/solvers/test_score_feature.py
@@ -192,19 +192,23 @@ class TestRoutedThroughScoreFeature:
         assert sol.prob_penalty == pytest.approx(marxan, abs=1e-15)
 
     def test_cache_honours_the_rule(self) -> None:
+        """The solvers' chance term (``ChanceState`` over the cache, v0.47.0) follows the
+        cache's rule; the cache's own full objective excludes the term."""
         from pymarxan.solvers.cache import ProblemCache
+        from pymarxan.solvers.chance_state import ChanceState
 
         problem = _two_feature_problem()
         cache = ProblemCache.from_problem(problem)
         marxan = dataclasses.replace(cache, zero_var_rule="marxan")
         sel = np.array([True, True])
-        held = cache.compute_held(sel)
         diff = (
-            marxan.compute_full_objective(sel, held, 0.0)
-            - cache.compute_full_objective(sel, held, 0.0)
+            ChanceState.from_selection(marxan, sel).penalty_total()
+            - ChanceState.from_selection(cache, sel).penalty_total()
         )
         assert diff == pytest.approx(3.0 * (0.8 - SF4) / 0.8, abs=1e-12)
         _, pen = evaluate_solution_chance(problem, sel)
-        assert cache.compute_full_objective(sel, held, 0.0) == pytest.approx(
-            2.0 + pen, abs=1e-12,
+        assert ChanceState.from_selection(cache, sel).penalty_total() == pytest.approx(
+            pen, abs=1e-12,
         )
+        held = cache.compute_held(sel)
+        assert cache.compute_full_objective(sel, held, 0.0) == pytest.approx(2.0, abs=1e-12)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_cache.py tests/pymarxan/solvers/test_cache_probmode3.py tests/pymarxan/solvers/test_sa_chance.py tests/pymarxan/solvers/test_score_feature.py -q -m 'not bench'`
Expected: FAIL — 4 failed, 91 passed: `test_cache.py::test_chance_arrays_gated_on_chance_terms`, `test_cache_probmode3.py::TestProbmode3CacheFields::test_dense_chance_matrices_are_gone`, `TestProbmode3FullObjective::test_full_objective_plus_chance_state_is_the_evaluator_objective`, `test_score_feature.py::TestRoutedThroughScoreFeature::test_cache_honours_the_rule`. The SA / II tests in `test_sa_chance.py` pass before this task too (regression guards, mutation-checked; Rulings, Task 6).

- [ ] **Step 3: Implement**

Modify `src/pymarxan/solvers/cache.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/cache.py b/src/pymarxan/solvers/cache.py
index 0690b59..ed9d71f 100644
--- a/src/pymarxan/solvers/cache.py
+++ b/src/pymarxan/solvers/cache.py
@@ -59,7 +59,6 @@ from scipy.sparse import coo_matrix, csr_matrix
 from pymarxan.models.problem import ConservationProblem
 from pymarxan.solvers.utils import (
     _active_chance_terms,
-    _chance_terms,
     _probability_vector,
     _probmode,
     _zero_var_rule,
@@ -131,8 +130,8 @@ class ProblemCache:
     pu_feat_csr : scipy.sparse.csr_matrix
         (n_pu, n_feat) float64 — amount of each feature in each PU (the source of truth).
         ``pu_feat_matrix`` is a lazily-densified ``cached_property`` view over it, built
-        only when clumping / separation / probmode-3 / analysis touch it; a plain problem
-        never densifies. ``feat_uses_pu`` derives from the CSR's CSC form.
+        only when clumping / separation / analysis touch it; a plain problem (and a
+        PROBMODE 3 one) never densifies. ``feat_uses_pu`` derives from the CSR's CSC form.
     feat_targets : np.ndarray
         (n_feat,) float64 — target for each feature.
     feat_spf : np.ndarray
@@ -179,8 +178,10 @@ class ProblemCache:
     cost_thresh: float
     thresh_pen1: float
     thresh_pen2: float
-    # PROBMODE 3 (Z-score chance constraints) — populated for all problems
-    # but only used when probmode == 3. See solvers/probability.py for the math.
+    # Chance terms (PROBMODE 3, Z-score chance constraints). The cache holds only the
+    # immutable per-cell contributions (``chance_cells``); the running per-feature state and
+    # the penalty live in the ``ChanceState`` sidecar (solvers/chance_state.py), so neither
+    # ``compute_full_objective`` nor ``compute_delta_objective`` includes the chance term.
     probmode: int = 0
     prob_weight: float = 1.0
     # Active chance terms (``_active_chance_terms``: requested by PROBMODE and with data);
@@ -193,11 +194,6 @@ class ProblemCache:
     chance_cells: dict[str, ChanceCells] = field(
         default_factory=dict, compare=False, repr=False,
     )
-    # expected_matrix[i, j] = amount_ij · p_ij (presence); collapses to amount
-    # when there is no `prob` column or when prob == 1.
-    expected_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
-    # var_matrix[i, j] = amount_ij² · p_ij · (1 − p_ij); all-zero when no `prob`.
-    var_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
     # feat_ptarget[j] = per-feature probability target; ≤0 means disabled.
     feat_ptarget: np.ndarray = field(default_factory=lambda: np.zeros(0))
     # Phase 19 clumping (TARGET2 / CLUMPTYPE / type-4 species). All-zero /
@@ -266,11 +262,11 @@ class ProblemCache:
     @cached_property
     def pu_feat_matrix(self) -> np.ndarray:
         """Dense (n_pu, n_feat) view — densified on first access (clumping / separation /
-        probmode-3 / analysis). A plain problem never triggers this."""
+        analysis). A plain problem never triggers this."""
         if self.n_pu * self.n_feat > 20_000_000:  # ~160 MB float64 — the sparse win is lost
             warnings.warn(
                 "Densifying a large ProblemCache feature matrix "
-                f"({self.n_pu}x{self.n_feat}); clumping/separation/probmode-3 at raster "
+                f"({self.n_pu}x{self.n_feat}); clumping/separation at raster "
                 "scale is not yet sparse.",
                 stacklevel=2,
             )
@@ -372,47 +368,16 @@ class ProblemCache:
         thresh_pen2 = float(params.get("THRESHPEN2", 0.0))
 
         # PROBMODE: default 1 with a probability table (the MIP's and greedy's rule), else 0.
-        # Mode 3 state below is zero-cost when probmode != 3.
+        # Chance-term state below is zero-cost unless a chance term is active.
         probmode = _probmode(problem)
         prob_weight = float(params.get("PROBABILITYWEIGHTING", 1.0))
         chance_terms = _active_chance_terms(problem)
         zero_var_rule = _zero_var_rule(problem)
 
-        # Per-cell probability of PRESENCE — read once into a dense matrix.
-        # Marxan's PROB2D `prob` is the probability the feature is present at
-        # the cell (clumping.cpp / probability.cpp ComputeP_AllPUsSelected_2D:
-        # `ExpectedAmount2D += amount * SM.prob`). Cells without an explicit
-        # prob default to 1.0 (certain presence == deterministic), so a
-        # problem with no `prob` column reduces to the plain amount sum.
-        # NB: this is the *opposite* polarity to PROBMODE 2 (per-PU `prob`,
-        # Marxan 1D), where `prob` is the probability of loss and the held
-        # amount is `amount * (1 - prob)`.
-        # expected/var are read ONLY under probmode == 3 — build them (densely) only then;
-        # a plain problem pays zero memory for these two (n_pu, n_feat) matrices.
-        if "2d" in _chance_terms(problem):
-            dense = np.asarray(pu_feat_csr.toarray())
-            prob_matrix = np.ones_like(dense)
-            if (
-                problem.pu_vs_features is not None
-                and "prob" in problem.pu_vs_features.columns
-            ):
-                pv_pu = problem.pu_vs_features["pu"].values
-                pv_sp = problem.pu_vs_features["species"].values
-                pv_pr = problem.pu_vs_features["prob"].values
-                for k in range(len(pv_pu)):
-                    pi = pu_id_to_idx.get(int(pv_pu[k]))
-                    fj = feat_id_to_col.get(int(pv_sp[k]))
-                    if pi is not None and fj is not None:
-                        prob_matrix[pi, fj] = float(pv_pr[k])
-            # Marxan PROB2D math (probability.cpp ComputeP_AllPUsSelected_2D):
-            #   E[T_j contribution from i]   = amount_ij * p_ij          (presence)
-            #   Var[T_j contribution from i] = amount_ij^2 * p_ij * (1 - p_ij)
-            expected_matrix = dense * prob_matrix
-            var_matrix = (dense ** 2) * prob_matrix * (1.0 - prob_matrix)
-        else:
-            expected_matrix = np.zeros((0, 0))
-            var_matrix = np.zeros((0, 0))
-
+        # Per-cell contributions of the active chance terms. Marxan's PROB2D `prob` is the
+        # probability the feature is PRESENT at the cell (probability.c
+        # ComputeProbability2D: E += amount·prob, V += amount²·prob·(1 − prob)) — the
+        # opposite polarity to PROBMODE 2 (per-PU loss probability, held a·(1 − p)).
         chance_cells: dict[str, ChanceCells] = {}
         if "2d" in chance_terms:
             chance_cells["2d"] = _build_chance_cells_2d(
@@ -615,8 +580,6 @@ class ProblemCache:
             chance_terms=chance_terms,
             zero_var_rule=zero_var_rule,
             chance_cells=chance_cells,
-            expected_matrix=expected_matrix,
-            var_matrix=var_matrix,
             feat_ptarget=feat_ptarget,
             feat_target2=feat_target2,
             feat_clumptype=feat_clumptype,
@@ -681,31 +644,6 @@ class ProblemCache:
         cols = csr.indices[s:e]
         held[cols] += sign * csr.data[s:e]
 
-    def _compute_zscore_penalty(
-        self,
-        held_expected: np.ndarray,
-        held_variance: np.ndarray,
-        held_nonzero: np.ndarray,
-    ) -> float:
-        """Marxan-faithful Z-score chance-constraint penalty.
-
-        Active only when the 2D chance term is (``"2d" in self.chance_terms``); returns 0
-        otherwise. Scored by ``probability.score_features``: zero variance is
-        ``held_nonzero == 0`` (no selected cell with a variance term > 0) and follows
-        ``self.zero_var_rule``; features with ptarget ≤ 0 are disabled.
-        """
-        if "2d" not in self.chance_terms or self.n_feat == 0:
-            return 0.0
-        from pymarxan.solvers.probability import score_features  # keeps cold path cheap
-
-        # Marxan's ComputeProbability2D weights the probability shortfall only by
-        # PROBABILITYWEIGHTING — NOT by SPF (unlike the representation penalty).
-        penalty = score_features(
-            held_expected, held_variance, held_nonzero, self.feat_targets,
-            self.feat_ptarget, self.zero_var_rule,
-        )[3]
-        return self.prob_weight * float(np.sum(penalty))
-
     def compute_full_objective(
         self,
         selected: np.ndarray,
@@ -717,6 +655,9 @@ class ProblemCache:
         objective = cost + BLM * boundary + penalty [+ cost_threshold_penalty]
         [+ connectivity]
 
+        The chance terms (PROBMODE 3) are not included: callers add
+        ``ChanceState.penalty_total()`` (SA, II) — as for clumping / separation.
+
         Parameters
         ----------
         selected : np.ndarray
@@ -757,13 +698,6 @@ class ProblemCache:
                 total_cost, self.cost_thresh, self.thresh_pen1, self.thresh_pen2
             )
 
-        # PROBMODE 3 chance-constraint penalty (additive; no-op unless the 2D term is active)
-        if "2d" in self.chance_terms:
-            held_expected = self.expected_matrix[selected].sum(axis=0)
-            held_variance = self.var_matrix[selected].sum(axis=0)
-            held_nonzero = (self.var_matrix[selected] > 0).sum(axis=0)
-            obj += self._compute_zscore_penalty(held_expected, held_variance, held_nonzero)
-
         # Connectivity term (no-op when conn_mode == 0)
         if self.conn_mode:
             obj += self._compute_connectivity(selected)
@@ -888,29 +822,6 @@ class ProblemCache:
             )
             delta += new_ct_penalty - old_ct_penalty
 
-        # --- PROBMODE 3 chance-constraint delta ---
-        # Recompute the Z-score penalty before and after the flip from the
-        # full selection mask. This is O(n_feat) per flip — heavier than
-        # the deterministic delta but cheap relative to the SA outer loop.
-        # The future optimisation (sparse per-PU feature index) lands in a
-        # later patch if benchmarks demand it.
-        if "2d" in self.chance_terms:
-            held_exp_before = self.expected_matrix[selected].sum(axis=0)
-            held_var_before = self.var_matrix[selected].sum(axis=0)
-            held_exp_after = held_exp_before + sign * self.expected_matrix[idx]
-            held_var_after = held_var_before + sign * self.var_matrix[idx]
-            # Underflow guard — held_var must be non-negative
-            np.clip(held_var_after, 0.0, None, out=held_var_after)
-            nz_before = (self.var_matrix[selected] > 0).sum(axis=0)
-            nz_after = nz_before + int(sign) * (self.var_matrix[idx] > 0)
-            prob_before = self._compute_zscore_penalty(
-                held_exp_before, held_var_before, nz_before,
-            )
-            prob_after = self._compute_zscore_penalty(
-                held_exp_after, held_var_after, nz_after,
-            )
-            delta += prob_after - prob_before
-
         # --- Connectivity delta (one integer comparison when off) ---
         # O(degree), against the pre-flip selection; s = sign (+1 add, -1 remove).
         #   symmetric:  -s·w·(conn_self[idx] + Σ_out v·sel[nbr])
```

Modify `src/pymarxan/solvers/iterative_improvement.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/iterative_improvement.py b/src/pymarxan/solvers/iterative_improvement.py
index b80eb36..230718a 100644
--- a/src/pymarxan/solvers/iterative_improvement.py
+++ b/src/pymarxan/solvers/iterative_improvement.py
@@ -20,11 +20,25 @@ import numpy as np
 from pymarxan.models.problem import ConservationProblem
 from pymarxan.solvers.base import Solution, Solver, SolverConfig, require_binary
 from pymarxan.solvers.cache import ProblemCache
+from pymarxan.solvers.chance_state import ChanceState
 from pymarxan.solvers.utils import build_solution
 
 _VALID_ITIMPTYPES = frozenset(range(4))
 
 
+def _chance_state(cache: ProblemCache, selected: np.ndarray) -> ChanceState | None:
+    """A fresh ``ChanceState`` when a chance term is active, else None."""
+    if not cache.chance_terms:
+        return None
+    return ChanceState.from_selection(cache, selected)
+
+
+def _chance_from_scratch(cache: ProblemCache, selected: np.ndarray) -> float:
+    """The chance penalty of ``selected`` from scratch (0 without an active term)."""
+    state = _chance_state(cache, selected)
+    return 0.0 if state is None else state.penalty_total()
+
+
 class IterativeImprovementSolver(Solver):
     """Iterative improvement solver for conservation planning.
 
@@ -202,6 +216,8 @@ class IterativeImprovementSolver(Solver):
         if cache.separation_active:
             from pymarxan.solvers.separation import SepState
             sep_state = SepState.from_selection(cache, selected)
+        # Chance terms (PROBMODE 3): incremental E / V per pass, fresh from the selection.
+        chance_state = _chance_state(cache, selected)
 
         improved = True
         while improved:
@@ -217,6 +233,8 @@ class IterativeImprovementSolver(Solver):
                     delta += clump_state.delta_penalty(cache, i, adding=False)
                 if sep_state is not None:
                     delta += sep_state.delta_penalty(cache, i, adding=False)
+                if chance_state is not None:
+                    delta += chance_state.delta_penalty(cache, i, adding=False)
                 if delta < 0:
                     # Removing this PU improves the objective
                     selected[i] = False
@@ -226,6 +244,8 @@ class IterativeImprovementSolver(Solver):
                         clump_state.apply_flip(cache, i, adding=False)
                     if sep_state is not None:
                         sep_state.apply_flip(cache, i, adding=False)
+                    if chance_state is not None:
+                        chance_state.apply_flip(cache, i, adding=False)
                     improved = True
 
         return build_solution(
@@ -284,6 +304,8 @@ class IterativeImprovementSolver(Solver):
         if cache.separation_active:
             from pymarxan.solvers.separation import SepState
             sep_state = SepState.from_selection(cache, selected)
+        # Chance terms (PROBMODE 3): incremental E / V per pass, fresh from the selection.
+        chance_state = _chance_state(cache, selected)
 
         improved = True
         while improved:
@@ -299,6 +321,8 @@ class IterativeImprovementSolver(Solver):
                     delta += clump_state.delta_penalty(cache, i, adding=True)
                 if sep_state is not None:
                     delta += sep_state.delta_penalty(cache, i, adding=True)
+                if chance_state is not None:
+                    delta += chance_state.delta_penalty(cache, i, adding=True)
                 if delta < 0:
                     # Adding this PU improves the objective
                     selected[i] = True
@@ -308,6 +332,8 @@ class IterativeImprovementSolver(Solver):
                         clump_state.apply_flip(cache, i, adding=True)
                     if sep_state is not None:
                         sep_state.apply_flip(cache, i, adding=True)
+                    if chance_state is not None:
+                        chance_state.apply_flip(cache, i, adding=True)
                     improved = True
 
         return build_solution(
@@ -330,13 +356,15 @@ class IterativeImprovementSolver(Solver):
         """Try pairwise swaps until no improvement found.
 
         Uses compute_full_objective from cache (two PUs change at once,
-        so delta computation is not directly applicable).
+        so delta computation is not directly applicable), plus a from-scratch
+        chance term (the cache's full objective excludes it).
         """
 
         selected = solution.selected.copy()
         held = cache.compute_held(selected)
         total_cost = float(np.sum(cache.costs[selected]))
         current_obj = cache.compute_full_objective(selected, held, blm)
+        current_obj += _chance_from_scratch(cache, selected)
 
         improved = True
         while improved:
@@ -360,6 +388,7 @@ class IterativeImprovementSolver(Solver):
                     cache.apply_flip_to_held(new_held, a, 1.0)
                     new_cost = total_cost - cache.costs[r] + cache.costs[a]
                     new_obj = cache.compute_full_objective(selected, new_held, blm)
+                    new_obj += _chance_from_scratch(cache, selected)
 
                     if new_obj < current_obj:
                         held = new_held
```

Modify `src/pymarxan/solvers/simulated_annealing.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/simulated_annealing.py b/src/pymarxan/solvers/simulated_annealing.py
index 6e28b4e..f7bae14 100644
--- a/src/pymarxan/solvers/simulated_annealing.py
+++ b/src/pymarxan/solvers/simulated_annealing.py
@@ -194,6 +194,15 @@ class SimulatedAnnealingSolver(Solver):
                 )
                 current_obj += space_state.penalty_total()
 
+            # Chance terms (PROBMODE 3): ChanceState holds the incremental per-feature
+            # E / V; the cache's full objective and delta exclude the term, so its
+            # initial value is added here (as for SepState).
+            chance_state = None
+            if cache.chance_terms:
+                from pymarxan.solvers.chance_state import ChanceState
+                chance_state = ChanceState.from_selection(cache, selected)
+                current_obj += chance_state.penalty_total()
+
             # Determine initial temperature
             if start_temp_param is not None:
                 initial_temp = max(float(start_temp_param), 0.001)
@@ -218,6 +227,10 @@ class SimulatedAnnealingSolver(Solver):
                         delta += space_state.delta_penalty(
                             idx, adding=not selected[idx],
                         )
+                    if chance_state is not None:
+                        delta += chance_state.delta_penalty(
+                            cache, idx, adding=not selected[idx],
+                        )
                     if delta > 0:
                         deltas.append(delta)
 
@@ -279,6 +292,8 @@ class SimulatedAnnealingSolver(Solver):
                     delta += sep_state.delta_penalty(cache, idx, adding)
                 if space_state is not None:
                     delta += space_state.delta_penalty(idx, adding)
+                if chance_state is not None:
+                    delta += chance_state.delta_penalty(cache, idx, adding)
 
                 # Acceptance criterion
                 if delta <= 0 or (
@@ -296,6 +311,8 @@ class SimulatedAnnealingSolver(Solver):
                         sep_state.apply_flip(cache, idx, adding)
                     if space_state is not None:
                         space_state.apply_flip(idx, adding)
+                    if chance_state is not None:
+                        chance_state.apply_flip(cache, idx, adding)
 
                     # Track best
                     if current_obj < best_obj:
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_cache.py tests/pymarxan/solvers/test_cache_probmode3.py tests/pymarxan/solvers/test_sa_chance.py tests/pymarxan/solvers/test_score_feature.py -q -m 'not bench'`
Expected: 95 passed; then run the bench deliberately: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/benchmarks/bench_prob.py -m bench -q` → 6 passed (prototype ratios 4.50× / 4.72× / 4.54× / 4.61×)

- [ ] **Step 5: Gate**

Run: `~/.local/bin/ruff check src tests examples && /home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/ -q -m 'not bench' -p no:cacheprovider && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/python examples/validate_marxan_parity.py`
Expected: ruff and mypy clean; the full suite passes (prototype: 3870 passed, 101 skipped HiGHS cases which run in CI); `examples/validate_marxan_parity.py` prints MIP 35.0, SA 43.0, greedy 45.0; no golden JSON regenerated.

- [ ] **Step 6: Commit**

```bash
git add -A src/pymarxan/solvers/cache.py src/pymarxan/solvers/iterative_improvement.py src/pymarxan/solvers/simulated_annealing.py tests/benchmarks/bench_prob.py tests/pymarxan/solvers/test_cache.py tests/pymarxan/solvers/test_cache_probmode3.py tests/pymarxan/solvers/test_sa_chance.py tests/pymarxan/solvers/test_score_feature.py
git commit -F - <<'MSG'
perf(solvers): wire ChanceState into SA / II; drop the dense chance path (A3b)

SA builds a ChanceState per run, adds penalty_total() to its initial
objective (as for SepState), uses delta_penalty in the temperature-sampling
loop and per move, and apply_flip on accept. II builds one per removal and
addition pass; the swap pass adds a from-scratch chance term at both of its
compute_full_objective calls (spec §3.5 "Full objective").

ProblemCache.compute_full_objective and compute_delta_objective no longer
include the chance term, so the cache's own delta == full contract holds
for every mode; the dense (n_pu, n_feat) expected_matrix / var_matrix and
the cache's O(n_selected·n_feat) mode-3 delta are deleted (mode 3 no longer
densifies the feature matrix).

Tests rewritten per spec §3.6: dense-matrix tests move to the per-cell
arrays; the no-prob-column test becomes "term off without data"; the
vacuous full-objective test now checks cache + ChanceState == evaluator;
the mode-3 delta test moves to ChanceState; bench_prob.py is re-pointed at
ChanceState with dense and sparse risk layouts and a <= 6x mode-0 bound.
New: reported == tracked after SA, II local optimality under the reported
objective for ITIMPTYPE 1-3, and mode 3 without puvspr prob == mode 0.

Bench (-m bench, this machine; per flip = delta + commit on accept, best
of 3 x 6000 flips, ~5 cells per PU):
  2000x20   dense risk: mode0 14.5us, mode3 65.2us -> 4.50x
  2000x20   sparse    : mode0 14.4us, mode3 68.1us -> 4.72x
  10000x50  dense risk: mode0 14.6us, mode3 66.2us -> 4.54x
  10000x50  sparse    : mode0 14.5us, mode3 66.8us -> 4.61x
Before (dense-matrix delta, same layout): 17x at 2000x20, 103x at 10000x50.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01G9s5x1nmcqSTDeRUXcRPVW
MSG
```

---

### Task 7: Marxan's zero-variance rule (Z = 4) by default; `PROBZEROVAR`; docs and CHANGELOG

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `docs/USER_MANUAL.md`
- Modify: `src/pymarxan/models/problem.py`
- Modify: `src/pymarxan/solvers/base.py`
- Modify: `src/pymarxan/solvers/probability.py`
- Modify: `src/pymarxan/solvers/utils.py`
- Create: `tests/pymarxan/solvers/test_probzerovar.py`
- Modify: `tests/pymarxan/solvers/test_score_feature.py`
- Modify: `tests/pymarxan/solvers/test_zscore.py`

**Interfaces:**
- Consumes: Task 4's `_zero_var_rule`; Task 5's `ChanceState`.
- Produces: `_zero_var_rule` reads `PROBZEROVAR` (0 / absent → "marxan", 1 → "met"; other values → "marxan" and a `validate()` error: `PROBZEROVAR must be 0 (Marxan's zero-variance rule, Z = 4) or 1 (zero variance counts as met), got ...`); `probability.DEFAULT_ZERO_VAR_RULE = "marxan"`. CHANGELOG `[Unreleased]` (Changed / Fixed), USER_MANUAL PROBMODE 3 and `PROBZEROVAR` text, the docstrings spec §5 lists for release A.

- [ ] **Step 1: Write the failing tests**

Create `tests/pymarxan/solvers/test_probzerovar.py`:

```python
"""Marxan 3's zero-variance rule (Z = 4) is the default; ``PROBZEROVAR 1`` opts out.

v0.47.0 (spec §1 "Zero-variance quirk", §3.4). Marxan 3 ``ComputeProbability2D`` sets
``Z = 4`` when a feature's variance over the selection is 0, so ``P = probZUT(4) ≈ 3.17e-5``
fails any ``ptarget`` above it. ``PROBZEROVAR 1`` restores pymarxan's pre-v0.47 rule
("zero variance = met"). Includes a hand example against a test-local Python port of
Marxan 3's ``ReturnProbabilityAmounts2D`` / ``ComputeProbability2D`` / ``probZUT``
(``marxan3`` branch, ``probability.c``).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from pymarxan.io.readers import load_project
from pymarxan.io.writers import save_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.cache import ProblemCache
from pymarxan.solvers.chance_state import ChanceState
from pymarxan.solvers.iterative_improvement import IterativeImprovementSolver
from pymarxan.solvers.probability import _term_moments, evaluate_solution_chance, score_features
from pymarxan.solvers.utils import _zero_var_rule, build_solution

SF4 = float(norm.sf(4.0))


# ------------------------------------------------- Marxan 3 transcription (probability.c)


def probZUT(z: float) -> float:
    """Ibbetson (CACM Alg. 209) upper tail, Z_MAX = 5 — probability.c ``probZUT``."""
    Z_MAX = 5.0
    if z == 0.0:
        x = 0.0
    else:
        y = 0.5 * abs(z)
        if y >= Z_MAX * 0.5:
            x = 1.0
        elif y < 1.0:
            w = y * y
            x = ((((((((0.000124818987 * w
                        - 0.001075204047) * w + 0.005198775019) * w
                      - 0.019198292004) * w + 0.059054035642) * w
                    - 0.151968751364) * w + 0.319152932694) * w
                  - 0.531923007300) * w + 0.797884560593) * y * 2.0
        else:
            y -= 2.0
            x = (((((((((((((-0.000045255659 * y
                             + 0.000152529290) * y - 0.000019538132) * y
                           - 0.000676904986) * y + 0.001390604284) * y
                         - 0.000794620820) * y - 0.002034254874) * y
                       + 0.006549791214) * y - 0.010557625006) * y
                     + 0.011630447319) * y - 0.009279453341) * y
                   + 0.005353579108) * y - 0.002141268741) * y
                 + 0.000535310849) * y + 0.999936657524
    return (x + 1.0) * 0.5 if z < 0.0 else (1.0 - x) * 0.5


def marxan3_probability2d(problem: ConservationProblem, selected: np.ndarray):
    """``ReturnProbabilityAmounts2D`` over the selected PUs, then ``ComputeProbability2D``.

    Returns (per-feature P, weighted probability2D)."""
    feats = problem.features
    idx = {int(f): j for j, f in enumerate(feats["id"])}
    E = [0.0] * len(feats)
    V = [0.0] * len(feats)
    chosen = {int(p) for p, s in zip(problem.planning_units["id"], selected) if s}
    for row in problem.pu_vs_features.itertuples():
        if int(row.pu) in chosen and row.amount:
            j = idx[int(row.species)]
            E[j] += row.amount * row.prob
            V[j] += row.amount * row.amount * row.prob * (1 - row.prob)
    P, total = [], 0.0
    for j, (target, ptarget2d) in enumerate(zip(feats["target"], feats["ptarget"])):
        z = (target - E[j]) / math.sqrt(V[j]) if V[j] > 0 else 4
        p = probZUT(z) if z >= 0 else 1 - probZUT(-1 * z)
        heaviside = 1 if ptarget2d > p else 0
        shortfall = (ptarget2d - p) / ptarget2d if ptarget2d > 0 else 0
        P.append(p)
        total += heaviside * shortfall
    weight = float(problem.parameters.get("PROBABILITYWEIGHTING", 1.0))
    return P, total * weight


def hand_example(**params) -> tuple[ConservationProblem, np.ndarray]:
    """PUs 1-3 selected, PU 4 not.

    f1: q ≠ 0.5, E = 7.4 > target 5 (negative Z), ptarget 0.95.
    f2: only q = 1 cells selected → zero variance, ptarget 0.6 > probZUT(4).
    f3: unselected (its only cell is in PU 4) → zero variance, ptarget 0.3.
    f4: E = 3.4 < target 6 (positive Z), ptarget 0.8.
    """
    pu = pd.DataFrame({"id": [1, 2, 3, 4], "cost": [1.0, 2.0, 3.0, 4.0], "status": [0] * 4})
    feat = pd.DataFrame({
        "id": [1, 2, 3, 4], "name": list("abcd"), "target": [5.0, 4.0, 2.0, 6.0],
        "spf": [1.0] * 4, "ptarget": [0.95, 0.6, 0.3, 0.8],
    })
    pv = pd.DataFrame({
        "species": [1, 1, 1, 2, 2, 3, 4, 4],
        "pu": [1, 2, 3, 1, 2, 4, 2, 3],
        "amount": [4.0, 3.0, 2.0, 5.0, 1.0, 9.0, 2.0, 3.0],
        "prob": [0.8, 0.7, 0.8, 1.0, 1.0, 0.4, 0.2, 1.0],
    })
    problem = ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pv,
        parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 7.0, **params},
    )
    return problem, np.array([True, True, True, False])


class TestHandExampleAgainstMarxan3:
    def test_per_feature_probability(self) -> None:
        problem, sel = hand_example()
        P_mx, _ = marxan3_probability2d(problem, sel)
        E, V, N = _term_moments(problem, sel, "2d", problem.features["id"].values)
        z, p, _, _ = score_features(
            E, V, N, problem.features["target"].values, problem.features["ptarget"].values,
            _zero_var_rule(problem),
        )
        assert z[0] < 0 and z[3] > 0  # both signs of Z exercised
        assert N[1] == 0 and N[2] == 0  # f2 (q = 1) and f3 (unselected): zero variance
        assert np.all(np.abs(z) < 5)  # Marxan's probZUT is exact 0/1 beyond |z| = 5
        np.testing.assert_allclose(p, P_mx, rtol=0, atol=1e-6)

    def test_total_penalty_evaluator_and_chance_state(self) -> None:
        problem, sel = hand_example()
        _, total_mx = marxan3_probability2d(problem, sel)
        _, total = evaluate_solution_chance(problem, sel)
        # penalty differences are P differences / ptarget (min ptarget 0.3), times weight 7
        assert total == pytest.approx(total_mx, abs=7 * 1e-6 / 0.3)
        assert total > 7 * 0.9  # the zero-variance f2 alone costs ~ weight·1
        cache = ProblemCache.from_problem(problem)
        assert ChanceState.from_selection(cache, sel).penalty_total() == pytest.approx(
            total, abs=1e-12,
        )
        assert build_solution(problem, sel, 0.0).prob_penalty == pytest.approx(total, abs=1e-12)

    def test_zero_variance_features_pay_marxans_penalty(self) -> None:
        problem, sel = hand_example()
        shortfalls, _ = evaluate_solution_chance(problem, sel)
        assert shortfalls[2] == pytest.approx(0.6 - SF4)
        assert shortfalls[3] == pytest.approx(0.3 - SF4)

    def test_opt_out_drops_only_the_zero_variance_features(self) -> None:
        problem, sel = hand_example()
        opt_out, _ = hand_example(PROBZEROVAR=1)
        _, marxan = evaluate_solution_chance(problem, sel)
        shortfalls, met = evaluate_solution_chance(opt_out, sel)
        assert shortfalls[2] == 0.0 and shortfalls[3] == 0.0
        expected_gap = 7.0 * ((0.6 - SF4) / 0.6 + (0.3 - SF4) / 0.3)
        assert marxan - met == pytest.approx(expected_gap, abs=1e-12)


# ------------------------------------------------------------- zero-variance rule cases


def _zero_var_problem(qs: list[float], **params) -> ConservationProblem:
    pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]})
    feat = pd.DataFrame({"id": [1, 2], "name": ["a", "b"], "target": [3.0, 3.0],
                         "spf": [1.0, 1.0], "ptarget": [0.9, 0.5]})
    pv = pd.DataFrame({"species": [1, 1, 2, 2], "pu": [1, 2, 1, 2],
                       "amount": [2.0, 2.0, 2.0, 2.0], "prob": qs})
    return ConservationProblem(planning_units=pu, features=feat, pu_vs_features=pv,
                               parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 2.0,
                                           **params})


ZERO_VAR_CASES = {
    "nothing selected": ([0.5, 0.5, 0.5, 0.5], [False, False]),
    "all q = 1": ([1.0, 1.0, 1.0, 1.0], [True, True]),
    "all q = 0": ([0.0, 0.0, 0.0, 0.0], [True, True]),
    "q in {0, 1}": ([0.0, 1.0, 1.0, 0.0], [True, True]),
}


@pytest.mark.parametrize("case", sorted(ZERO_VAR_CASES))
def test_zero_variance_default_is_marxans_rule(case: str) -> None:
    qs, sel = ZERO_VAR_CASES[case]
    problem = _zero_var_problem(qs)
    sel_arr = np.array(sel)
    expected = 2.0 * ((0.9 - SF4) / 0.9 + (0.5 - SF4) / 0.5)
    assert evaluate_solution_chance(problem, sel_arr)[1] == pytest.approx(expected)
    cache = ProblemCache.from_problem(problem)
    assert cache.zero_var_rule == "marxan"
    assert ChanceState.from_selection(cache, sel_arr).penalty_total() == pytest.approx(expected)


@pytest.mark.parametrize("case", sorted(ZERO_VAR_CASES))
def test_zero_variance_opt_out_is_met(case: str) -> None:
    qs, sel = ZERO_VAR_CASES[case]
    problem = _zero_var_problem(qs, PROBZEROVAR=1)
    sel_arr = np.array(sel)
    shortfalls, penalty = evaluate_solution_chance(problem, sel_arr)
    assert penalty == 0.0 and shortfalls == {1: 0.0, 2: 0.0}
    cache = ProblemCache.from_problem(problem)
    assert cache.zero_var_rule == "met"
    assert ChanceState.from_selection(cache, sel_arr).penalty_total() == 0.0


def test_the_documented_safe_vs_risky_inversion() -> None:
    """Under Marxan's rule a safe (q = 1) PU scores worse than a risky one (CHANGELOG).

    PU 1 risky (cost 1, q = 0.5), PU 2 safe (cost 2, q = 1); amount 10, target 5,
    ptarget 0.9, weight 50. Holding PU 1: 1 + 50·(0.9 − 0.5)/0.9 ≈ 23.2; holding PU 2:
    2 + 50·(0.9 − sf(4))/0.9 ≈ 52.0 by default, 2 under PROBZEROVAR 1.
    """
    def problem(**params):
        pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 2.0], "status": [0, 0]})
        feat = pd.DataFrame({"id": [1], "name": ["f"], "target": [5.0], "spf": [1.0],
                             "ptarget": [0.9]})
        pv = pd.DataFrame({"species": [1, 1], "pu": [1, 2], "amount": [10.0, 10.0],
                           "prob": [0.5, 1.0]})
        return ConservationProblem(
            planning_units=pu, features=feat, pu_vs_features=pv,
            parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 50.0, "BLM": 0.0, **params},
        )

    risky, safe = np.array([True, False]), np.array([False, True])
    p = problem()
    assert build_solution(p, safe, 0.0).objective == pytest.approx(
        2 + 50 * (0.9 - SF4) / 0.9,
    )
    assert build_solution(p, risky, 0.0).objective < build_solution(p, safe, 0.0).objective
    start = build_solution(p, risky, 0.0)
    assert IterativeImprovementSolver(itimptype=3).improve(p, start).selected.tolist() == [
        True, False,
    ]
    q = problem(PROBZEROVAR=1)
    assert build_solution(q, safe, 0.0).objective == pytest.approx(2.0)
    sol = IterativeImprovementSolver(itimptype=3).improve(q, build_solution(q, risky, 0.0))
    assert sol.selected.tolist() == [False, True]


# ------------------------------------------------------------------ parameter handling


@pytest.mark.parametrize("value, rule", [(None, "marxan"), (0, "marxan"), (1, "met"),
                                         (1.0, "met"), ("1", "met"), (0.0, "marxan")])
def test_rule_from_parameter(value, rule: str) -> None:
    params = {} if value is None else {"PROBZEROVAR": value}
    problem, _ = hand_example(**params)
    assert _zero_var_rule(problem) == rule


@pytest.mark.parametrize("value", [2, -1, 0.5, "yes"])
def test_validate_reports_bad_probzerovar(value) -> None:
    problem, _ = hand_example(PROBZEROVAR=value)
    errors = problem.validate()
    assert any("PROBZEROVAR" in e for e in errors)
    assert _zero_var_rule(problem) == "marxan"  # pure: never raises; validate() reports


@pytest.mark.parametrize("value", [None, 0, 1])
def test_validate_accepts_probzerovar(value) -> None:
    params = {} if value is None else {"PROBZEROVAR": value}
    problem, _ = hand_example(**params)
    assert not any("PROBZEROVAR" in e for e in problem.validate())


def test_probzerovar_round_trips_through_input_dat(tmp_path) -> None:
    problem, _ = hand_example(PROBZEROVAR=1)
    save_project(problem, tmp_path)
    loaded = load_project(tmp_path)
    assert loaded.parameters["PROBZEROVAR"] == 1
    assert _zero_var_rule(loaded) == "met"
```

Modify `tests/pymarxan/solvers/test_score_feature.py` (apply this diff):

```diff
diff --git a/tests/pymarxan/solvers/test_score_feature.py b/tests/pymarxan/solvers/test_score_feature.py
index e2b5bab..5615172 100644
--- a/tests/pymarxan/solvers/test_score_feature.py
+++ b/tests/pymarxan/solvers/test_score_feature.py
@@ -135,7 +135,7 @@ class TestComputeZscorePenaltyZeroVariance:
             compute_zscore_penalty({1: 1.0}, {1: 0.5}, 1.0, frozenset())  # type: ignore[misc]
 
 
-def _two_feature_problem() -> ConservationProblem:
+def _two_feature_problem(**params) -> ConservationProblem:
     """Feature 1 uncertain (q = 0.7), feature 2 certain (q = 1: zero variance)."""
     pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]})
     feat = pd.DataFrame({
@@ -148,7 +148,7 @@ def _two_feature_problem() -> ConservationProblem:
     })
     return ConservationProblem(
         planning_units=pu, features=feat, pu_vs_features=pv,
-        parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 3.0},
+        parameters={"PROBMODE": 3, "PROBABILITYWEIGHTING": 3.0, **params},
     )
 
 
@@ -165,29 +165,33 @@ class TestEvaluateSolutionChance:
 
     def test_matches_score_feature(self) -> None:
         sel = np.array([True, True])
-        shortfalls, penalty = evaluate_solution_chance(_two_feature_problem(), sel)
+        shortfalls, penalty = evaluate_solution_chance(
+            _two_feature_problem(PROBZEROVAR=1), sel,
+        )
         e, v = 8.0 * 0.7, 2 * 16.0 * 0.7 * 0.3
         f1 = score_feature(e, v, 2, 5.0, 0.9, "met")
         assert shortfalls[1] == pytest.approx(f1.shortfall, abs=1e-15)
-        assert shortfalls[2] == 0.0  # zero variance under today's rule
+        assert shortfalls[2] == 0.0  # zero variance under PROBZEROVAR 1 (met)
         assert penalty == pytest.approx(3.0 * f1.penalty, abs=1e-15)
 
 
 class TestRoutedThroughScoreFeature:
     """The evaluator and the cache both score through the core and its rule."""
 
-    def test_evaluator_honours_the_rule(self, monkeypatch: pytest.MonkeyPatch) -> None:
-        import pymarxan.solvers.utils as utils
+    def test_evaluator_honours_the_rule(self) -> None:
+        """Default (PROBZEROVAR 0) is Marxan's Z = 4; PROBZEROVAR 1 is "met".
+
+        Updated (v0.47.0, A4): the A2 version monkeypatched the rule to "marxan" over a
+        "met" default; the rule now comes from PROBZEROVAR.
+        """
+        from pymarxan.solvers.utils import build_solution
 
         sel = np.array([True, True])
-        _, met = evaluate_solution_chance(_two_feature_problem(), sel)
-        monkeypatch.setattr(utils, "_zero_var_rule", lambda problem: "marxan")
+        _, met = evaluate_solution_chance(_two_feature_problem(PROBZEROVAR=1), sel)
         shortfalls, marxan = evaluate_solution_chance(_two_feature_problem(), sel)
         assert shortfalls[2] == pytest.approx(0.8 - SF4)
         assert marxan - met == pytest.approx(3.0 * (0.8 - SF4) / 0.8, abs=1e-12)
         # build_solution's objective goes through the same evaluator
-        from pymarxan.solvers.utils import build_solution
-
         sol = build_solution(_two_feature_problem(), sel, 0.0)
         assert sol.prob_penalty == pytest.approx(marxan, abs=1e-15)
 
@@ -197,8 +201,9 @@ class TestRoutedThroughScoreFeature:
         from pymarxan.solvers.cache import ProblemCache
         from pymarxan.solvers.chance_state import ChanceState
 
-        problem = _two_feature_problem()
+        problem = _two_feature_problem(PROBZEROVAR=1)
         cache = ProblemCache.from_problem(problem)
+        assert cache.zero_var_rule == "met"
         marxan = dataclasses.replace(cache, zero_var_rule="marxan")
         sel = np.array([True, True])
         diff = (
```

Modify `tests/pymarxan/solvers/test_zscore.py` (apply this diff):

```diff
diff --git a/tests/pymarxan/solvers/test_zscore.py b/tests/pymarxan/solvers/test_zscore.py
index 7c5dad7..ed2ff5a 100644
--- a/tests/pymarxan/solvers/test_zscore.py
+++ b/tests/pymarxan/solvers/test_zscore.py
@@ -179,18 +179,27 @@ class TestZscorePenalty:
         assert p == pytest.approx(expected, abs=1e-9)
 
     def test_sentinel_variance_means_deterministic_no_penalty(self):
-        """A zero-variance feature has no chance penalty under the "met" rule.
-
-        Rewritten (v0.47.0, A2): the zero-variance feature is now named in
-        ``zero_variance`` instead of being inferred from ``Z == 4``, so a real
-        Z of 4 is told apart from Marxan's zero-variance 4 (spec §3.4, final
-        review I-5); ``zero_var_rule="met"`` keeps this release's default.
+        """A zero-variance feature pays Marxan's Z = 4 penalty by default; 0 when opted out.
+
+        Rewritten (v0.47.0): this test pinned pymarxan's old rule, "zero variance means
+        the chance constraint is vacuously met (P = 1)". Marxan 3's ComputeProbability2D
+        sets Z = 4 for zero variance and scores it normally, so P = norm.sf(4) ≈ 3.17e-5
+        fails ptarget 0.95: penalty (ptarget − norm.sf(4))/ptarget. That is now the
+        default (PROBZEROVAR 0); ``zero_var_rule="met"`` (PROBZEROVAR 1) restores the old
+        result (spec §1 "Zero-variance quirk", §3.6 item 1).
         """
-        p = compute_zscore_penalty(
+        default = compute_zscore_penalty(
+            zscore_per_feature={1: 4.0},
+            prob_targets={1: 0.95},
+            weight=1.0,
+            zero_variance=frozenset({1}),
+        )
+        assert default == pytest.approx((0.95 - norm.sf(4.0)) / 0.95, abs=1e-15)
+        opted_out = compute_zscore_penalty(
             zscore_per_feature={1: 4.0},
             prob_targets={1: 0.95},
             weight=1.0,
             zero_variance=frozenset({1}),
             zero_var_rule="met",
         )
-        assert p == 0.0
+        assert opted_out == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_probzerovar.py tests/pymarxan/solvers/test_score_feature.py tests/pymarxan/solvers/test_zscore.py -q -m 'not bench'`
Expected: FAIL — 18 failed, 47 passed: in `test_probzerovar.py` the 4 Marxan 3 hand-example tests, `test_zero_variance_default_is_marxans_rule` ×4, `test_the_documented_safe_vs_risky_inversion`, `test_rule_from_parameter` ×3 (None / 0 / 0.0 → "marxan"), `test_validate_reports_bad_probzerovar` ×4; `test_score_feature.py::TestRoutedThroughScoreFeature::test_evaluator_honours_the_rule`; `test_zscore.py::TestZscorePenalty::test_sentinel_variance_means_deterministic_no_penalty` — all because the default is still "met".

- [ ] **Step 3: Implement**

Modify `CHANGELOG.md` (apply this diff):

```diff
diff --git a/CHANGELOG.md b/CHANGELOG.md
index a2c593d..84529be 100644
--- a/CHANGELOG.md
+++ b/CHANGELOG.md
@@ -7,6 +7,47 @@ and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0
 
 ## [Unreleased]
 
+### Changed
+
+- **Breaking (PROBMODE 3):** a feature with zero variance over the selection — nothing
+  holding it selected, or every selected cell with puvspr `prob` 0 or 1 — now follows
+  Marxan 3's rule (`probability.c` `ComputeProbability2D`): Z = 4, so its probability of
+  meeting the target is `norm.sf(4) ≈ 3.17e-5` and it pays
+  `PROBABILITYWEIGHTING × (ptarget − 3.17e-5) / ptarget` (≈ the full weighting) for any
+  `ptarget` above that. pymarxan used to treat it as met. This inverts a preference: a
+  plan holding a feature only in certain (`prob = 1`) cells now scores worse than one using
+  a slightly risky cell (target 10, ptarget 0.9, weighting 10, plans costing 1: safe
+  10.9996, risky 1.0). The new input.dat parameter **`PROBZEROVAR 1`** restores the old
+  result (`0`, the default, is Marxan's rule; `validate()` reports other values). Results
+  change for PROBMODE 3 projects whose selections leave a feature with zero variance. With
+  PROBMODE 3 and no puvspr `prob` column the chance term is off (penalty 0), as before.
+- PROBMODE 3 simulated annealing and iterative improvement are much faster: the chance term
+  moved out of `ProblemCache` into a new incremental sidecar, `ChanceState`
+  (`solvers/chance_state.py`), which keeps per-feature expected amount, variance and an exact
+  count of uncertain cells and re-scores only the flipped unit's features, with an
+  error-bounded recompute that keeps every delta equal to the full difference (1e-9).
+  Measured per flip (delta, plus commit on an accepted flip), ~5 feature cells per unit:
+  4.5–4.7× mode 0 at 2 000 × 20 and 10 000 × 50 (dense and sparse risk), against 17× and
+  103× before. The dense `(n_pu, n_feat)` `ProblemCache.expected_matrix` / `var_matrix`
+  are removed, and `ProblemCache.compute_full_objective` / `compute_delta_objective` no
+  longer include the chance term (SA and II add `ChanceState`'s; the evaluator and
+  `Solution.objective` are unchanged).
+- `compute_zscore_penalty` gains keyword-only `zero_variance` (the ids of zero-variance
+  features, so a real Z of 4 is no longer mistaken for one) and `zero_var_rule`
+  (`"marxan"` / `"met"`); `evaluate_solution_chance` gains `term="2d"`. New
+  `score_feature` / `score_features` / `FeatureChance` in `solvers/probability.py` are
+  the one copy of the Z / P / penalty arithmetic every path uses.
+- PROBMODE 3 without a puvspr `prob` column leaves `Solution.prob_shortfalls` /
+  `prob_penalty` as None (they were all-zero), since the chance term is off.
+
+### Fixed
+
+- `write_puvspr` always writes a `prob` column that is present. It dropped an all-zero
+  one, and reloading turned presence 0 into presence 1 (a missing column reads as 1).
+- `run_sweep_parallel` passes the whole problem to its workers. It rebuilt the problem
+  from five fields, dropping `probability`, `connectivity` and the subclass type, so e.g.
+  a PROBMODE 1 sweep lost its risk premium in the workers and disagreed with `run_sweep`.
+
 ## [0.46.0] — 2026-10-23
 
 ### Changed
```

Modify `docs/USER_MANUAL.md` (apply this diff):

```diff
diff --git a/docs/USER_MANUAL.md b/docs/USER_MANUAL.md
index 30e962a..a0a07d2 100644
--- a/docs/USER_MANUAL.md
+++ b/docs/USER_MANUAL.md
@@ -393,7 +393,7 @@ persistence is 1 − probability. It is attached automatically when the project
 | 0 | Probabilities ignored | — |
 | 1 (default with a table) | Risk premium `PROBABILITYWEIGHTING × Σ probability × cost` over selected units, added to the objective (not to the reported cost; off when the weighting is 0 or below) — the expected cost of protecting units that are then lost; it does not depend on the features a unit holds, and a zero-cost unit carries no premium | pymarxan extension |
 | 2 | Expected amounts: every feature amount counts as `amount × (1 − probability)`, and targets, shortfalls, penalties, representation and gap analysis are judged on them | pymarxan extension |
-| 3 | Z-score probability penalty (a soft chance target, not a hard constraint) from a per-cell probability of **presence** (a `prob` column in `puvspr.dat`) and a per-feature `ptarget` in spec.dat (Marxan's column is `ptarget2d`) | Based on Marxan's Probability 2D, except that a feature with zero variance is treated as met (Marxan 3 treats it as failing: Z = 4) |
+| 3 | Z-score probability penalty (a soft chance target, not a hard constraint) from a per-cell probability of **presence** (a `prob` column in `puvspr.dat`) and a per-feature `ptarget` in spec.dat (Marxan's column is `ptarget2d`) | Marxan's Probability 2D (Marxan 3), including its zero-variance rule (below) |
 
 Modes 1 and 2 have the same meaning in every single-zone solver: the MIP, simulated annealing and
 iterative improvement optimise them, the greedy heuristics use them in their scores (mode 1 as an
@@ -405,6 +405,20 @@ with `PROBABILITYWEIGHTING` above 0, and mode 2 — every single-zone solver and
 `ValueError` on a non-finite probability; modes 0 and 3, and mode 1 with a weighting of 0 or below,
 ignore the table. The zone solvers ignore the probability table.
 
+**PROBMODE 3 and zero variance (`PROBZEROVAR`).** A feature's variance over the selection
+is zero when no selected cell of it is uncertain: nothing that holds it is selected, or every
+selected cell has `prob` 0 or 1. Marxan 3 then sets Z = 4, so the feature's probability of
+meeting its target is `probZUT(4) ≈ 3.2e-5` and it pays almost the full penalty
+(`PROBABILITYWEIGHTING × (ptarget − 3.2e-5) / ptarget`) for any `ptarget` above that. Since
+v0.47.0 this is pymarxan's default (`PROBZEROVAR 0`). It has a counter-intuitive consequence:
+a plan holding a feature only in certain (`prob = 1`) cells scores worse than one that also
+uses a slightly risky cell — with target 10, `ptarget` 0.9, weighting 10 and plans costing 1,
+holding the feature safely scores 10.9996 and holding it in a risky unit 1.0. `PROBZEROVAR 1` in input.dat
+(or `parameters["PROBZEROVAR"] = 1`) restores the pre-v0.47 rule, "zero variance counts as
+met" (no chance penalty for that feature; a deterministic shortfall is still penalised by the
+ordinary target penalty). `validate()` reports any other value. With PROBMODE 3 but no `prob`
+column in puvspr.dat the chance term is off (penalty 0), whatever `PROBZEROVAR` says.
+
 Marxan's own **Probability 1D** (a `prob` column in `pu.dat`, a Z-score on the expected
 amount `Σ amount × (1 − p)` with variance `Σ amount² × p × (1 − p)` against `ptarget1d`) is
 **not yet implemented**; mode 2 uses the same expected amount but as a deterministic target,
@@ -416,7 +430,7 @@ at p = 0.05 with the same expected amount count the same. Use it as a risk-disco
 representation target; for a stated confidence level use a chance formulation (Watts et al.
 2021). Until Probability 1D is implemented, it can be reproduced with PROBMODE 3 by writing
 `prob = 1 − p` (the unit's persistence) on every puvspr.dat row of that unit and setting
-`ptarget` (same mean and variance; zero-variance features differ as noted above).
+`ptarget` (same mean, variance and zero-variance rule).
 
 Under PROBMODE 2 some paths still use raw amounts: clumping occupancy (`target2`), separation
 occurrence and space-target weights (simulated annealing and iterative improvement warn when
```

Modify `src/pymarxan/models/problem.py` (apply this diff):

```diff
diff --git a/src/pymarxan/models/problem.py b/src/pymarxan/models/problem.py
index 088b69a..8f93b69 100644
--- a/src/pymarxan/models/problem.py
+++ b/src/pymarxan/models/problem.py
@@ -302,6 +302,16 @@ class ConservationProblem:
                         "probability values must be in range [0, 1]"
                     )
 
+        # --- Parameter validation ---
+        if "PROBZEROVAR" in self.parameters:
+            from pymarxan.solvers.utils import _probzerovar_value
+
+            if _probzerovar_value(self.parameters["PROBZEROVAR"]) is None:
+                errors.append(
+                    "PROBZEROVAR must be 0 (Marxan's zero-variance rule, Z = 4) or 1 "
+                    f"(zero variance counts as met), got {self.parameters['PROBZEROVAR']!r}"
+                )
+
         # --- Connectivity validation ---
         if self.connectivity is not None:
             conn_required = {"id1", "id2", "value"}
```

Modify `src/pymarxan/solvers/base.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/base.py b/src/pymarxan/solvers/base.py
index 8d43726..5abf648 100644
--- a/src/pymarxan/solvers/base.py
+++ b/src/pymarxan/solvers/base.py
@@ -27,10 +27,10 @@ class Solution:
     shortfall: float = 0.0  # Total raw feature shortfall (sum of max(0, target - achieved))
     metadata: dict = field(default_factory=dict)
     zone_assignment: np.ndarray | None = None
-    # PROBMODE 3 outputs — populated when the problem has PROBMODE=3 (and
-    # any feature with ptarget > 0). Always None otherwise.
+    # PROBMODE 3 outputs — populated when the 2D chance term is active (PROBMODE 3 with a
+    # puvspr `prob` column; empty / 0 without a spec `ptarget` column). None otherwise.
     prob_shortfalls: dict[int, float] | None = None  # feature_id -> max(0, ptarget − P)
-    prob_penalty: float | None = None  # γ · Σ SPF · (ptarget − P) / ptarget
+    prob_penalty: float | None = None  # γ · Σ [ptarget > P] · (ptarget − P) / ptarget
     # TARGET2 / clumping outputs — populated when the problem has any
     # feature with target2 > 0. Always None otherwise.
     # clump_shortfalls: feature_id -> max(0, T·MISSLEVEL − held_eff)
```

Modify `src/pymarxan/solvers/probability.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/probability.py b/src/pymarxan/solvers/probability.py
index 639319a..3dc0395 100644
--- a/src/pymarxan/solvers/probability.py
+++ b/src/pymarxan/solvers/probability.py
@@ -1,35 +1,38 @@
-"""Marxan-faithful Z-score chance-constraint math (PROBMODE 3).
+"""Marxan-faithful Z-score chance-constraint math (PROBMODE 3, Marxan's Probability 2D).
 
 This is the pure-functional layer: no problem objects, no solver state.
-Inputs are plain dicts keyed by feature id; outputs are scalars or dicts.
-Solver integration lives in ``cache.py`` (SA / iterative improvement),
-``heuristic.py`` (inline), and ``mip_solver.py`` (post-hoc evaluation).
+Solver integration lives in ``chance_state.py`` (``ChanceState``: SA / iterative
+improvement, incremental per-feature E / V), ``utils.py`` (the evaluator, via
+:func:`evaluate_solution_chance`) and ``mip_solver.py`` (post-hoc report).
 
-The formulation mirrors Marxan v4's ``computation.hpp::computeProbMeasures``:
+The reference is Marxan 3 (``marxan3`` branch, ``probability.c``
+``ComputeProbability2D`` / ``probZUT``) — not Marxan v4, whose ``probZUT`` has an
+integer-division bug that inverts the penalty. Per feature ``j``, over the selected PUs:
 
-    Z_j = (T_j − E[T_j]) / sqrt(Var[T_j])    # positive = shortfall
-    P_j = probZUT(Z_j) = 1 − Φ(Z_j)          # scipy.stats.norm.sf
+    E_j = Σ amount_ij · q_ij,   V_j = Σ amount_ij² · q_ij · (1 − q_ij)
+    Z_j = (T_j − E_j) / sqrt(V_j)            # positive = shortfall
+    P_j = probZUT(Z_j) = 1 − Φ(Z_j)          # scipy.special.ndtr(−Z) = norm.sf(Z)
 
-When Var[T_j] == 0 the routine returns Marxan's sentinel ``rZ = 4``
-(``probZUT(4) ≈ 3e-5``), which short-circuits the upper-tail probability
-to "trivially met". The deterministic-target shortfall is handled
-elsewhere via the regular penalty path.
+``q`` is the per-cell probability of **presence** (puvspr ``prob``; Marxan's PROB2D),
+the opposite polarity to PROBMODE 2 / Marxan 1D, where the per-PU ``prob`` is a loss
+probability and the held amount is ``amount · (1 − prob)``. ``T_j`` is the resolved
+deterministic target (no MISSLEVEL). Marxan's ``probZUT`` is the Ibbetson polynomial
+(exact 0 / 1 for |z| ≥ 5, within ~1e-9 of ``norm.sf`` below); pymarxan keeps ``norm.sf``.
 
-Penalty per feature is normalised by ptarget (matching Marxan):
+**Zero variance** is ``nonzero_j == 0`` — no selected cell whose variance term, as
+computed in float, is > 0 — never a float threshold on ``V``. The rule is
+``zero_var_rule`` (``PROBZEROVAR`` in input.dat): ``"marxan"`` (``PROBZEROVAR 0``, the
+default) gives Marxan 3's ``Z = 4``, so ``P = probZUT(4) ≈ 3.17e-5`` fails any
+``ptarget`` above that — nothing selected, all ``q`` in {0, 1}; ``"met"``
+(``PROBZEROVAR 1``) counts such a feature as met (pymarxan before v0.47).
 
-    penalty_j = max(0, (ptarget_j − P_j) / ptarget_j)
+Penalty per feature is the Marxan step function normalised by ptarget:
 
-Marxan's ``ComputeProbability2D`` scales the probability shortfall only by
-``PROBABILITYWEIGHTING`` — **not** by SPF (the SPF weighting applies to the
-*representation* penalty, a separate objective term; see
-``score_change.cpp:611``). Features with ``ptarget_j ≤ 0`` are disabled
-(Marxan's sentinel) and contribute 0 regardless of Z.
+    penalty_j = [ptarget_j > P_j] · (ptarget_j − P_j) / ptarget_j
 
-Expected amount uses the PROB2D *presence* convention
-(``ComputeP_AllPUsSelected_2D``: ``ExpectedAmount2D += amount * prob``), i.e.
-``E[T_j] = Σ amount_ij · p_ij`` — the opposite polarity to PROBMODE 2 / Marxan
-1D, where the per-PU ``prob`` is a loss probability and held amount is
-``amount · (1 − prob)``.
+Marxan's ``ComputeProbability2D`` scales the summed penalties only by
+``PROBABILITYWEIGHTING`` — **not** by SPF (SPF weights the separate representation
+penalty). Features with ``ptarget_j ≤ 0`` are disabled (Marxan's −1) and contribute 0.
 
 References
 ----------
@@ -52,8 +55,8 @@ _MARXAN_ZERO_VARIANCE_SENTINEL = 4.0
 
 ZeroVarRule = Literal["marxan", "met"]
 _ZERO_VAR_RULES = ("marxan", "met")
-# The rule used when a caller names none. "met" keeps the pre-v0.47 behaviour.
-DEFAULT_ZERO_VAR_RULE: ZeroVarRule = "met"
+# The rule used when a caller names none: Marxan 3's (PROBZEROVAR 0).
+DEFAULT_ZERO_VAR_RULE: ZeroVarRule = "marxan"
 
 _TERMS = ("2d",)
 
```

Modify `src/pymarxan/solvers/utils.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/utils.py b/src/pymarxan/solvers/utils.py
index 12a8edf..0ab65f4 100644
--- a/src/pymarxan/solvers/utils.py
+++ b/src/pymarxan/solvers/utils.py
@@ -125,12 +125,30 @@ def _active_chance_terms(problem: ConservationProblem) -> frozenset[str]:
 
 
 def _zero_var_rule(problem: ConservationProblem) -> str:
-    """The zero-variance rule for the chance terms (spec §3.4): ``"met"`` in this release.
+    """The chance terms' zero-variance rule from ``PROBZEROVAR`` (spec §3.4).
 
-    ``"met"``: a feature with no selected cell carrying variance has no chance penalty.
-    Pure; never warns.
+    ``PROBZEROVAR 0`` (the default) → ``"marxan"``: Marxan 3's rule, a feature with no
+    selected cell carrying variance gets ``Z = 4`` (``P ≈ 3.17e-5``) and so fails any
+    ``ptarget`` above that. ``PROBZEROVAR 1`` → ``"met"``: such a feature counts as met (the
+    pre-v0.47 behaviour). Any other value means ``"marxan"`` here and is reported by
+    ``ConservationProblem.validate()``. Pure; never warns or raises.
     """
-    return "met"
+    return "met" if _probzerovar_value(problem.parameters.get("PROBZEROVAR", 0)) == 1 else (
+        "marxan"
+    )
+
+
+def _probzerovar_value(value: object) -> int | None:
+    """``PROBZEROVAR`` as 0 / 1, or None when it is neither (``validate()`` reports it)."""
+    try:
+        number = float(value)  # type: ignore[arg-type]
+    except (TypeError, ValueError):
+        return None
+    if number == 0.0:
+        return 0
+    if number == 1.0:
+        return 1
+    return None
 
 
 def _probmode2_active(problem: ConservationProblem) -> bool:
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_probzerovar.py tests/pymarxan/solvers/test_score_feature.py tests/pymarxan/solvers/test_zscore.py -q -m 'not bench'`
Expected: 65 passed

- [ ] **Step 5: Gate**

Run: `~/.local/bin/ruff check src tests examples && /home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/ -q -m 'not bench' -p no:cacheprovider && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/python examples/validate_marxan_parity.py`
Expected: ruff and mypy clean; the full suite passes (prototype: 3897 passed, 101 skipped HiGHS cases which run in CI); `examples/validate_marxan_parity.py` prints MIP 35.0, SA 43.0, greedy 45.0; no golden JSON regenerated.

- [ ] **Step 6: Commit**

```bash
git add -A CHANGELOG.md docs/USER_MANUAL.md src/pymarxan/models/problem.py src/pymarxan/solvers/base.py src/pymarxan/solvers/probability.py src/pymarxan/solvers/utils.py tests/pymarxan/solvers/test_probzerovar.py tests/pymarxan/solvers/test_score_feature.py tests/pymarxan/solvers/test_zscore.py
git commit -F - <<'MSG'
feat(probability)!: Marxan's zero-variance rule (Z = 4) by default; PROBZEROVAR (A4)

PROBMODE 3 now follows Marxan 3 (probability.c ComputeProbability2D) for a
feature with zero variance over the selection (nothing holding it selected,
or every selected cell with prob 0 or 1): Z = 4, P = norm.sf(4) ~ 3.17e-5,
so it pays ~PROBABILITYWEIGHTING for any ptarget above that (spec §1, §3.4).
The new input.dat parameter PROBZEROVAR (0 default = Marxan, 1 = "met")
restores the old result; validate() reports other values. PROBMODE 3
without a puvspr prob column keeps the chance term off (penalty 0).

Tests: the Marxan 3 hand example (a test-local port of
ReturnProbabilityAmounts2D / ComputeProbability2D / probZUT; q != 0.5, a
negative-Z feature, a zero-variance feature with ptarget > P, an unselected
feature, different ptargets; P at 1e-6); zero-variance cases (nothing
selected, q in {0, 1}) under both rules; the safe-vs-risky inversion;
PROBZEROVAR parsing, validation and save/load round trip.
test_zscore.py::test_sentinel_variance_means_deterministic_no_penalty is
rewritten with its reason (default penalty (ptarget - sf(4))/ptarget, 0
under the opt-out).

Docs: probability.py module docstring (Marxan 3, not v4; no "trivially
met"), Solution field comment, USER_MANUAL PROBMODE 3 zero-variance text and
PROBZEROVAR, CHANGELOG [Unreleased] (Changed / Fixed).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01G9s5x1nmcqSTDeRUXcRPVW
MSG
```

---

## After the last task

- Re-read spec §3.6 and check each listed test was rewritten with a written reason; nothing else in the suite changed meaning.
- Run the bench file deliberately (`-m bench`) on the final tree and record the ratios in the release notes.
- Release v0.47.0 through the `release-pymarxan` skill (minor, breaking for PROBMODE 3 zero-variance features).
