# PROBMODE 1/2 Consistency Across Solvers (v0.45.0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every solver and every report agrees with the MIP on pymarxan's PROBMODE 1 (risk premium) and PROBMODE 2 (expected amounts `amount·(1 − p)`): simulated annealing and iterative improvement optimise them, and the evaluator, representation and gap analysis report them.

**Architecture:** The evaluator's single held-amount helper (`_compute_achieved`) scales by `(1 − p)` under PROBMODE 2, and the PROBMODE 1 premium follows the MIP's `γ <= 0` rule. `ProblemCache` reads `PROBMODE` with the MIP's default (1 when a probability table is present), adds a per-unit `prob_cost` term to its full objective and O(1) delta, and keeps a build-time scaled copy `held_csr` for held amounts, leaving the raw matrix for clumping and separation. Representation and gap analysis share one expected-total helper under PROBMODE 2.

**Tech Stack:** Python 3.11+, numpy, scipy.sparse, PuLP 3.3.2 + CBC, pytest.

**Spec:** `docs/plans/2026-10-17-probmode-consistency-design.md` (approved 2026-10-17 after three review-loop rounds and a fresh-eyes final review; §4a fixes every name, type and shape). Executors read it. The spec is the authority, and this plan argues from it.

**Provenance:** every code block below is extracted (`git show`) from a prototype built task by task in a scratch worktree, one commit per task, with the full gate green at every commit. Prototype commits: Task 1 `0959b16`, Task 2 `df8bd2b`, Task 3 `b17856c`, Task 4 `81af6f2` (base `0b9bc85`). Diffs apply with `git apply --whitespace=nowarn` on the previous task's tree. Every block was replayed that way on a fresh worktree from the base, and each task's tree matched the prototype commit. Editors that strip trailing whitespace from pasted context lines break a plain `git apply`. New files are given in full.

## Global Constraints

- Python 3.11+ floor; `from __future__ import annotations` as the first import in every new file; full type hints. mypy runs as `mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports`.
- ruff rules E, F, I, UP; line length 99. **Run `ruff check` on the test files too at every task handoff.**
- Tests run ONLY under `/opt/micromamba/envs/shiny/bin/pytest`, never `.venv`. In a worktree without `.venv`, use `PYTHONPATH=src:.` with the shiny pytest, `~/.local/bin/ruff`, and the main checkout's `.venv/bin/mypy`.
- **Semantics are exactly the MIP's** (spec §3): `p_i` from `problem.probability` (missing PU → 0; duplicate PU → last row wins; values used as given); `PROBMODE` defaults to 1 when a probability table is present; mode 1 = `γ · Σ x_i·p_i·c_i`, off for `γ <= 0` or no table, separate from cost (`Solution.cost`, `total_cost`, COSTTHRESH stay base cost); mode 2 = amounts `a_ij·(1 − p_i)` wherever held amounts are computed; mode 1/2 with no table = no-op; modes 0 and 3 unchanged.
- **Raw where the spec says raw:** `pu_feat_csr`, `pu_feat_matrix`, `feat_uses_pu`, `feat_baseline_penalty`, `ClumpState`, `SepState`, `SpaceState`, presolve P6, irreplaceability, Ferrier importance and prop-target resolution are unchanged.
- **Performance model:** no per-flip allocation; `held_csr` resolved at build time; PROBMODE 1 delta behind `prob_active`. Benches comparative only (`bench` marker).
- **Goldens:** no golden JSON is regenerated. The only moving records are the Stage 3 golden's 8 `probmode2|{max_features,max_weighted_features,min_shortfall,min_largest_shortfall}|{relative,absolute}` cases, handled by `STAGE3_PROBMODE2_FIXED` (spec §4a, §5). The `tests/data/simple` anchor (MIP 35.0; SA/greedy at or above) holds.
- **SA tests** use `NUMITNS = 10_000`, `NUMTEMP = 100`, `seed=42` (`num_solutions=10` for best-of-10, marked `slow`); reported vs tracked compare with `pytest.approx(rel=1e-9, abs=1e-9)`. Pin `mip_backend="cbc"` in MIP tests.
- Commit trailer on every commit:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01AZEiop9qqxVCrMHKqSDYP8
  ```

## Review Focus

These are inputs a user will plausibly hit that the spec's acceptance list doesn't name. Each is pinned by a test in its owning task.
1. **A project folder with a `prob.dat` and no `PROBMODE` in `input.dat`** (attached automatically by `load_project`) is now mode 1 in every path, including SA. → `test_table_without_probmode_is_mode_1` (Task 2) and the default-mode parametrizations of the consistency and full = evaluator tests.
2. **A PU listed twice in the probability table** (a careless join): last row wins in every path, and `validate()` says so. → `TestValidateDuplicateProbabilityRows` (Task 1) and the duplicate-PU row in Task 2's full = evaluator / delta fixtures.
3. **`PROBABILITYWEIGHTING` set to 0 or a negative number** (a slider dragged too far): the premium is off, as in the MIP. → `test_premium_off_for_non_positive_weighting` (Task 1) and `test_mode_1_off_for_non_positive_weighting` (Task 2).
4. **A unit with probability 1 under PROBMODE 2** (certain loss): it keeps its place in the matrix with zero expected amount, so the delta's index alignment holds. → `test_mode_2_structure_and_scaling` (Task 2).
5. **A pre-existing `pu_vs_features` row for a deleted PU** under modes 0/1: representation and gap percentages do not change. → `test_representation_with_orphan_row`, `test_gap_with_orphan_row` (Task 3).

## Rulings made while prototyping

The plan-review round-1 rulings P1–P9 (synthesis `docs/plans/2026-10-17-probmode-consistency-review.md`; spec §4b) are applied in the tasks below. Where the spec was silent, these choices were made; each names its task.
- **Task 1, helpers:** `solvers/utils.py` gains `_probmode(problem)` (default 1 with a table) and `_probability_map(problem)` (last row wins), shared by the evaluator paths.
- **Task 1, golden fix:** the 4 `max_*` records enter `STAGE3_PROBMODE2_FIXED` with unchanged `targets_met` ({1:F, 2:F, 3:T}), because the §4a shape is (objective, targets_met) for every case; `_assert_same` gains an optional `keys` override rather than a second helper.
- **Task 2, cache build:** a module-level `_build_probability(...)` in the style of `_build_connectivity`; `prob_cost` is zeros of length n_pu when off.
- **Task 2, no dedicated iterative-improvement test:** II calls the same three cache methods (`compute_held`, `compute_delta_objective`, `apply_flip_to_held`) that the tests cover.
- **Task 2, test evidence:** most best-of-10 SA tests and part of delta exactness already pass before Task 2 (pinned as regression guards; the best-of-10 `[mode2]` case does fail before it); Task 2's RED comes chiefly from the consistency, full = evaluator, brute-force, iterative-improvement, NaN and combination tests. SA consistency for mode 2 goes red after Task 1 (the evaluator then reports expected amounts while the cache is still raw); the spec's "mode 2 agrees today" describes the state before Task 1.
- **Task 1, no-table default:** `_probmode` defaults to 0 without a table (every mode 1/2 path is a no-op then; the MIP's PROBMODE 3 gate only tests `== 3`).
- **Tasks 1–2, where NaN raises:** only on paths that read the table (modes 1 and 2); greedy builds its map only in modes 1 and 2; `validate()` reports non-finite values in every mode.
- **Task 2, iterative-improvement test:** ITIMPTYPE 3 only swaps and never acts from the solver's all-selected start, so it is tested through `improve()` from {PU 1}, which swaps to {PU 2} only when the premium counts.
- **Task 2, warning frequency:** the PROBMODE 2 + clumping/separation warning fires once per cache build, so a RUNMODE chaining SA into II warns twice.
- **Task 3, every mode:** representation's binary path and gap's protected amount use `compute_held` in every mode, as spec §4 says; outside mode 2 the numbers are identical to before.
- **Task 4, roadmap:** the PROBMODE backlog line is struck through as "done in v0.45" (the file's v0.43/v0.44 convention) rather than deleted.
- **Task 4, manual wording:** the greedy heuristics *use* modes 1 and 2 in their scores; they do not optimise the objective.

---

### Task 1: Evaluator: PROBMODE 2 expected amounts, the MIP's weighting rule, duplicate check, Stage 3 golden fix

**Files:**
- Modify: `src/pymarxan/models/problem.py`
- Modify: `src/pymarxan/solvers/heuristic.py`
- Modify: `src/pymarxan/solvers/mip_solver.py`
- Modify: `src/pymarxan/solvers/utils.py`
- Create: `tests/pymarxan/solvers/test_probmode_evaluator.py`
- Modify: `tests/pymarxan/solvers/test_stage3_golden_v040.py`

**Interfaces:**
- Consumes: nothing new.
- Produces (`solvers/utils.py`, the single copy of the probability semantics, plan-review P1): `_probmode(problem) -> int` (default 1 with a table, 0 without), `_probability_map(...)` (last row wins; raises `ValueError` on non-finite values, P3), `_probmode2_active(problem) -> bool`, `_probability_vector(problem, pu_id_to_idx, n_pu) -> np.ndarray`; the MIP and greedy migrate to them (pure refactor; goldens pinned); public `compute_held` applies `_coerce_selected` (P2); `_compute_achieved` scales by `(1 − p)` under mode 2 (binary and fractional), so `compute_held`, `check_targets`, shortfall, penalty and `build_solution` report expected amounts; `_compute_probmode1_risk_premium` off for `γ <= 0`. `ConservationProblem.validate()` reports `probability lists planning unit IDs more than once: [...]` and non-finite probabilities. Test-side: `STAGE3_PROBMODE2_FIXED` in `test_stage3_golden_v040.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/pymarxan/solvers/test_probmode_evaluator.py`:

```python
"""PROBMODE 1 / 2 in the evaluator (``solvers/utils.py``): the MIP's semantics (v0.45 §3).

PROBMODE 2 judges held amounts as expected amounts ``Σ amount·(1 − p)`` (binary and
fractional); the PROBMODE 1 risk premium is off for ``PROBABILITYWEIGHTING <= 0``.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, SolverConfig
from pymarxan.solvers.heuristic import HeuristicSolver
from pymarxan.solvers.mip_solver import MIPSolver
from pymarxan.solvers.utils import (
    binarize_solution,
    build_solution,
    compute_held,
    compute_probability_penalty,
)

DATA = Path(__file__).parent.parent.parent / "data"


def _simple(probability: pd.DataFrame | None = None, **params: Any) -> ConservationProblem:
    p = load_project(DATA / "simple")
    return p.copy_with(probability=probability, parameters={**p.parameters, **params})


def _half_456(**params: Any) -> ConservationProblem:
    """simple, p = 0.5 on PUs 4–6, PROBMODE 2 (§6 'Evaluator PROBMODE 2')."""
    prob = pd.DataFrame({"pu": [4, 5, 6], "probability": [0.5, 0.5, 0.5]})
    return _simple(prob, PROBMODE=2, **params)


def _mip(problem: ConservationProblem, **kw: Any) -> Solution:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # PuLP 3.3.2
        sols = MIPSolver(mip_backend="cbc", **kw).solve(problem, SolverConfig(num_solutions=1))
    assert len(sols) == 1
    return sols[0]


def _expected_held(problem: ConservationProblem, selected: np.ndarray) -> dict[int, float]:
    """Hand oracle: Σ amount·x·(1 − p) per feature (missing PU → p = 0)."""
    p_of = {int(u): float(v) for u, v in zip(
        problem.probability["pu"], problem.probability["probability"],
    )} if problem.probability is not None else {}
    x_of = {int(u): float(x) for u, x in zip(problem.planning_units["id"], selected)}
    out = {int(f): 0.0 for f in problem.features["id"]}
    for pu, sp, amt in zip(
        problem.pu_vs_features["pu"], problem.pu_vs_features["species"],
        problem.pu_vs_features["amount"],
    ):
        out[int(sp)] += float(amt) * x_of[int(pu)] * (1.0 - p_of.get(int(pu), 0.0))
    return out


class TestEvaluatorProbmode2:
    def test_max_features_met_count_matches_mip(self) -> None:
        """(i) The reported met count is the MIP's own count (expected amounts)."""
        p = _half_456(COSTBUDGET=33.0)
        s = _mip(p, objective="max_features")
        assert s.metadata["mip_objective_value"] == pytest.approx(1.0)
        assert sum(s.targets_met.values()) == 1

    def test_binarized_plan_judged_on_expected_amounts(self) -> None:
        """(ii) Feature 2 holds 22 raw but 18.5 expected against a target of 20."""
        p = _half_456()
        frac = build_solution(p, np.array([0.75, 1, 0.5, 0, 0, 1.0]), 1.0)
        b = binarize_solution(p, frac, 0.75)
        assert b.targets_met == {1: True, 2: False, 3: False}

    def test_min_set_plan_meets_every_target(self) -> None:
        """(iii) Regression pin: the MIP enforces every target on expected amounts."""
        s = _mip(_half_456())
        assert all(s.targets_met.values())

    def test_proportion_plan_meets_every_target(self) -> None:
        """(iii) Regression pin: proportion decisions, judged fraction-linearly."""
        s = _mip(_half_456(), decisions="proportion")
        assert s.selected.dtype.kind == "f"
        assert all(s.targets_met.values())

    @pytest.mark.parametrize(
        "selected",
        [
            np.array([True, False, True, True, False, True]),
            np.ones(6, dtype=bool),
            np.array([0.75, 1.0, 0.5, 0.0, 0.25, 1.0]),
        ],
    )
    def test_compute_held_is_expected_amount(self, selected: np.ndarray) -> None:
        prob = pd.DataFrame({"pu": [1, 2, 4, 5, 6], "probability": [0.1, 0.4, 0.25, 0.05, 1.0]})
        p = _simple(prob, PROBMODE=2)
        got = compute_held(p, selected)
        want = _expected_held(p, selected)
        assert got.keys() == want.keys()
        for fid in want:
            assert got[fid] == pytest.approx(want[fid], rel=1e-12, abs=1e-12)

    def test_shortfall_and_penalty_use_expected_amounts(self) -> None:
        p = _half_456()
        sel = np.array([True, True, False, False, False, True])
        s = build_solution(p, sel, 0.0)
        held = _expected_held(p, sel)  # f1 = 10+15+5 = 30, f2 = 5+10+3.5, f3 = 6+4
        targets = dict(zip(p.features["id"].astype(int), p.features["target"].astype(float)))
        want = sum(max(0.0, targets[f] - held[f]) for f in targets)
        assert s.shortfall == pytest.approx(want)
        assert s.penalty == pytest.approx(want)  # spf 1 throughout

    def test_no_table_is_a_noop(self) -> None:
        sel = np.array([True, False, True, True, False, True])
        raw = compute_held(_simple(), sel)
        for mode in (1, 2):
            p = _simple(PROBMODE=mode)
            assert compute_held(p, sel) == raw
            assert build_solution(p, sel, 1.0).objective == pytest.approx(
                build_solution(_simple(), sel, 1.0).objective
            )

    @pytest.mark.parametrize("mode", [0, 1, 3])
    def test_other_modes_keep_raw_amounts(self, mode: int) -> None:
        prob = pd.DataFrame({"pu": [4, 5, 6], "probability": [0.5, 0.5, 0.5]})
        sel = np.array([True, False, True, True, False, True])
        assert compute_held(_simple(prob, PROBMODE=mode), sel) == compute_held(_simple(), sel)


class TestProbmode1Weighting:
    @pytest.mark.parametrize("gamma", [0.0, -1.0])
    def test_premium_off_for_non_positive_weighting(self, gamma: float) -> None:
        prob = pd.DataFrame({"pu": [1, 2, 3, 4, 5, 6], "probability": np.linspace(0.1, 0.6, 6)})
        p = _simple(prob, PROBMODE=1, PROBABILITYWEIGHTING=gamma)
        sel = np.array([False, True, False, True, False, True])
        pu_index = {int(u): i for i, u in enumerate(p.planning_units["id"])}
        assert compute_probability_penalty(p, sel, pu_index) == 0.0
        assert build_solution(p, sel, 1.0).objective == pytest.approx(
            build_solution(_simple(), sel, 1.0).objective
        )

    def test_premium_on_for_positive_weighting(self) -> None:
        prob = pd.DataFrame({"pu": [2, 4], "probability": [0.5, 0.25]})
        p = _simple(prob, PROBMODE=1, PROBABILITYWEIGHTING=2.0)
        sel = np.array([False, True, False, True, False, True])
        pu_index = {int(u): i for i, u in enumerate(p.planning_units["id"])}
        assert compute_probability_penalty(p, sel, pu_index) == pytest.approx(
            2.0 * (0.5 * 15.0 + 0.25 * 12.0)
        )


class TestValidateDuplicateProbabilityRows:
    def test_duplicate_pu_reported(self) -> None:
        prob = pd.DataFrame({"pu": [2, 4, 2, 6, 6], "probability": [0.1, 0.2, 0.3, 0.4, 0.5]})
        errors = _simple(prob).validate()
        assert "probability lists planning unit IDs more than once: [2, 6]" in errors

    def test_unique_table_has_no_duplicate_error(self) -> None:
        prob = pd.DataFrame({"pu": [2, 4, 6], "probability": [0.1, 0.2, 0.3]})
        assert not any("more than once" in e for e in _simple(prob).validate())


class TestComputeHeldSelectionContract:
    def test_int_zero_one_selection_reads_as_bool(self) -> None:
        p = _simple()
        sel = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
        assert compute_held(p, sel) == {1: 37.0, 2: 20.0, 3: 18.0}
        assert compute_held(p, sel) == compute_held(p, sel.astype(bool))

    def test_other_int_values_raise(self) -> None:
        with pytest.raises(ValueError, match="0 and 1"):
            compute_held(_simple(), np.array([0, 2, 0, 1, 0, 1]))


class TestCostThresholdSeesBaseCost:
    def test_cost_and_threshold_exclude_premium(self) -> None:
        prob = pd.DataFrame({"pu": [2, 4, 6], "probability": [0.5, 0.25, 0.5]})
        params = {"COSTTHRESH": 30.0, "THRESHPEN1": 5.0, "THRESHPEN2": 1.0}
        p = _simple(prob, PROBMODE=1, PROBABILITYWEIGHTING=2.0, **params)
        sel = np.array([False, True, False, True, False, True])  # cost 35
        s = build_solution(p, sel, 0.0)
        assert s.cost == 35.0
        premium = 2.0 * (0.5 * 15.0 + 0.25 * 12.0 + 0.5 * 8.0)  # 29.0
        threshold = 5.0 + 1.0 * (35.0 - 30.0)  # on base cost, not 35 + premium
        assert s.objective == pytest.approx(35.0 + premium + threshold)  # targets met, BLM 0


class TestProbabilityHelpers:
    """The one copy of the probability semantics (shared by every solver path)."""

    def test_probmode_default(self) -> None:
        from pymarxan.solvers.utils import _probmode, _probmode2_active

        prob = pd.DataFrame({"pu": [1], "probability": [0.5]})
        assert _probmode(_simple()) == 0
        assert _probmode(_simple(prob)) == 1
        assert _probmode(_simple(prob, PROBMODE=2)) == 2
        assert _probmode2_active(_simple(prob, PROBMODE=2))
        assert not _probmode2_active(_simple(None, PROBMODE=2))

    def test_map_and_vector(self) -> None:
        from pymarxan.solvers.utils import _probability_map, _probability_vector

        prob = pd.DataFrame({"pu": [2, 99, 4, 2], "probability": [0.9, 0.3, 1.0, 0.2]})
        p = _simple(prob)
        assert _probability_map(p) == {2: 0.2, 99: 0.3, 4: 1.0}  # last row wins
        idx = {int(u): i for i, u in enumerate(p.planning_units["id"])}
        np.testing.assert_array_equal(
            _probability_vector(p, idx, 6), [0.0, 0.2, 0.0, 1.0, 0.0, 0.0],
        )


_NAN_PROB = pd.DataFrame({"pu": [1, 2, 5], "probability": [0.3, np.nan, 0.4]})


class TestNonFiniteProbabilities:
    def test_validate_reports_nan(self) -> None:
        errors = _simple(_NAN_PROB).validate()
        assert (
            "probability has non-finite values (NaN or infinite) for planning unit IDs: [2]"
            in errors
        )
        assert "probability values must be in range [0, 1]" not in errors

    @pytest.mark.parametrize("mode", [1, 2, None])
    def test_evaluator_raises(self, mode: int | None) -> None:
        params = {} if mode is None else {"PROBMODE": mode}
        p = _simple(_NAN_PROB, **params)
        with pytest.raises(ValueError, match=r"non-finite .* \[2\]"):
            build_solution(p, np.array([False, True, False, True, False, True]), 1.0)

    @pytest.mark.parametrize("mode", [1, 2])
    def test_mip_raises(self, mode: int) -> None:
        with pytest.raises(ValueError, match=r"non-finite .* \[2\]"):
            _mip(_simple(_NAN_PROB, PROBMODE=mode))

    @pytest.mark.parametrize("mode", [1, 2])
    def test_greedy_raises(self, mode: int) -> None:
        p = _simple(_NAN_PROB, PROBMODE=mode)
        with pytest.raises(ValueError, match=r"non-finite .* \[2\]"):
            HeuristicSolver().solve(p, SolverConfig(num_solutions=1, seed=1))

    def test_mode_0_ignores_the_table(self) -> None:
        p = _simple(_NAN_PROB, PROBMODE=0)
        build_solution(p, np.array([False, True, False, True, False, True]), 1.0)
```

Modify `tests/pymarxan/solvers/test_stage3_golden_v040.py` (apply this diff):

```diff
diff --git a/tests/pymarxan/solvers/test_stage3_golden_v040.py b/tests/pymarxan/solvers/test_stage3_golden_v040.py
index 6531154..e679139 100644
--- a/tests/pymarxan/solvers/test_stage3_golden_v040.py
+++ b/tests/pymarxan/solvers/test_stage3_golden_v040.py
@@ -184,17 +184,21 @@ def record() -> dict[str, Any]:
 def golden() -> dict[str, Any]:
     data: dict[str, Any] = json.loads(GOLDEN_PATH.read_text())
     return data
 
 
-def _assert_same(got: dict[str, Any], want: dict[str, Any]) -> None:
+def _assert_same(
+    got: dict[str, Any], want: dict[str, Any], keys: list[str] | None = None,
+) -> None:
+    """Compare ``got`` with a golden record; ``keys`` overrides the uniqueness rule."""
     assert got["solutions"] == want["solutions"]
     if not want["solutions"]:
         return
-    keys = ["mip_objective_value"]
-    if want["unique_optimum"]:
-        keys += ["selected", "cost", "objective", "targets_met"]
+    if keys is None:
+        keys = ["mip_objective_value"]
+        if want["unique_optimum"]:
+            keys += ["selected", "cost", "objective", "targets_met"]
     for key in keys:
         if isinstance(want[key], float):
             assert got[key] == pytest.approx(want[key], rel=1e-9, abs=1e-9), key
         else:
             assert got[key] == want[key], key
@@ -204,18 +208,53 @@ def _assert_same(got: dict[str, Any], want: dict[str, Any]) -> None:
 # pu_vs_features rows) as m >= R = target × MISSLEVEL (10 × 0.9), as relative mode, the zone
 # solver and the v0.39 max_* ghost fix do. v0.40 reported 0.0 (all real targets met).
 # Every plan ties at m = R, so only the value is pinned.
 STAGE3_FIXED = {"ghost_misslevel09|min_largest_shortfall|absolute": 9.0}
 
+# v0.45 PROBMODE consistency (spec §5): ``build_solution`` judges PROBMODE 2 plans on expected
+# amounts Σ amount·(1 − p), as the MIP's target rows do. The MIP's plans are unchanged, but
+# the reported ``objective`` and ``targets_met`` of these unique-optimum records move.
+# case -> (new objective, new targets_met); the JSON keeps the v0.44 values (not regenerated).
+_ALL_MISSED = {"1": False, "2": False, "3": False}
+_F3_MET = {"1": False, "2": False, "3": True}
+STAGE3_PROBMODE2_FIXED: dict[str, tuple[float, dict[str, bool]]] = {
+    **{
+        f"probmode2|{objective}|{shortfall}": (59.1, _F3_MET)
+        for objective in ("max_features", "max_weighted_features")
+        for shortfall in SHORTFALLS
+    },
+    **{
+        f"probmode2|{objective}|{shortfall}": (55.4, _ALL_MISSED)
+        for objective in ("min_shortfall", "min_largest_shortfall")
+        for shortfall in SHORTFALLS
+    },
+}
+_V044_PROBMODE2 = {
+    "max_features": (54.0, _F3_MET),
+    "max_weighted_features": (54.0, _F3_MET),
+    "min_shortfall": (41.0, {"1": True, "2": True, "3": False}),
+    "min_largest_shortfall": (41.0, {"1": True, "2": True, "3": False}),
+}
+
 
 @pytest.mark.parametrize("case", CASES)
 def test_mip_solver_result_unchanged(golden: dict[str, Any], case: str) -> None:
     got = _run(case)
     if case in STAGE3_FIXED:
         assert golden[case]["mip_objective_value"] == 0.0  # the v0.40 value
         assert got["mip_objective_value"] == pytest.approx(STAGE3_FIXED[case], abs=1e-9)
         return
+    if case in STAGE3_PROBMODE2_FIXED:
+        assert golden[case]["unique_optimum"]  # justifies pinning the selection below
+        old_objective, old_met = _V044_PROBMODE2[case.split("|")[1]]
+        assert golden[case]["objective"] == old_objective  # the v0.44 values
+        assert golden[case]["targets_met"] == old_met
+        new_objective, new_met = STAGE3_PROBMODE2_FIXED[case]
+        assert got["objective"] == pytest.approx(new_objective, rel=1e-9, abs=1e-9)
+        assert got["targets_met"] == new_met
+        _assert_same(got, golden[case], keys=["selected", "cost", "mip_objective_value"])
+        return
     _assert_same(got, golden[case])
 
 
 def test_simple_min_set_anchor_in_golden(golden: dict[str, Any]) -> None:
     """The correctness anchor (BLM 0): reserve {2, 4, 6} at cost 35.0."""
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_probmode_evaluator.py tests/pymarxan/solvers/test_stage3_golden_v040.py -q`
Expected: FAIL — 28 failed, 125 passed — the 8 Stage 3 probmode2 golden cases (got 54.0 / 41.0, want 59.1 / 55.4) and 20 evaluator tests (max_features met count 2 not 1; binarized feature 2 reported met; compute_held ×3 incl. int 0/1 and int 2; shortfall/penalty; γ = −1 negative premium; duplicate-PU and NaN validate; the shared helpers; NaN raising in the evaluator ×3, the MIP ×2 and greedy ×2); the regression pins (min_set, proportion, no-table, other modes, γ = 0) pass

- [ ] **Step 3: Implement**

Modify `src/pymarxan/models/problem.py` (apply this diff):

```diff
diff --git a/src/pymarxan/models/problem.py b/src/pymarxan/models/problem.py
index 1797ec1..088b69a 100644
--- a/src/pymarxan/models/problem.py
+++ b/src/pymarxan/models/problem.py
@@ -275,13 +275,31 @@ class ConservationProblem:
                 if unknown_prob_pus:
                     errors.append(
                         f"probability references planning unit IDs not in "
                         f"planning_units: {sorted(unknown_prob_pus)}"
                     )
-                # Check probability range [0, 1]
-                prob_vals = self.probability["probability"]
-                if (prob_vals < 0.0).any() or (prob_vals > 1.0).any():
+                # A PU listed twice: every path uses the last row, silently.
+                prob_pu_col = self.probability["pu"]
+                dups = {int(u) for u in prob_pu_col[prob_pu_col.duplicated()]}
+                if dups:
+                    errors.append(
+                        "probability lists planning unit IDs more than once: "
+                        f"{sorted(dups)}"
+                    )
+                # Non-finite values (a blank prob.dat cell reads as NaN): every solver
+                # raises on them, so report the PU ids here.
+                prob_vals = pd.to_numeric(self.probability["probability"], errors="coerce")
+                finite = np.isfinite(prob_vals.to_numpy(dtype=np.float64))
+                if not finite.all():
+                    bad_ids = sorted({int(u) for u in self.probability["pu"][~finite]})
+                    errors.append(
+                        "probability has non-finite values (NaN or infinite) for planning "
+                        f"unit IDs: {bad_ids}"
+                    )
+                # Check probability range [0, 1] (finite values)
+                fin_vals = prob_vals[finite]
+                if (fin_vals < 0.0).any() or (fin_vals > 1.0).any():
                     errors.append(
                         "probability values must be in range [0, 1]"
                     )
 
         # --- Connectivity validation ---
```

Modify `src/pymarxan/solvers/heuristic.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/heuristic.py b/src/pymarxan/solvers/heuristic.py
index 0f9cbc8..7f2fcb1 100644
--- a/src/pymarxan/solvers/heuristic.py
+++ b/src/pymarxan/solvers/heuristic.py
@@ -9,11 +9,11 @@ from __future__ import annotations
 import numpy as np
 
 from pymarxan.adequacy.model import SpaceSpec
 from pymarxan.models.problem import ConservationProblem
 from pymarxan.solvers.base import Solution, Solver, SolverConfig
-from pymarxan.solvers.utils import build_solution
+from pymarxan.solvers.utils import _probability_map, _probmode, build_solution
 
 _VALID_HEURTYPES = range(8)
 
 
 class HeuristicSolver(Solver):
@@ -191,20 +191,21 @@ class HeuristicSolver(Solver):
         pv_pu = problem.pu_vs_features["pu"].values
         pv_sp = problem.pu_vs_features["species"].values
         pv_am = problem.pu_vs_features["amount"].values
 
         # Probability support
-        prob_mode = int(problem.parameters.get("PROBMODE", 1))
+        # Shared helpers (solvers/utils.py): PROBMODE default 1 with a table, last row
+        # wins, non-finite values raise.
+        prob_mode = _probmode(problem)
         prob_weight = float(
             problem.parameters.get("PROBABILITYWEIGHTING", 1.0)
         )
-        prob_map: dict[int, float] = {}
-        if problem.probability is not None:
-            p_pu = problem.probability["pu"].values
-            p_val = problem.probability["probability"].values
-            for pk in range(len(p_pu)):
-                prob_map[int(p_pu[pk])] = float(p_val[pk])
+        prob_map: dict[int, float] = (
+            _probability_map(problem)
+            if problem.probability is not None and prob_mode in (1, 2)
+            else {}
+        )
 
         for k in range(len(pv_pu)):
             idx = pu_id_to_idx.get(int(pv_pu[k]))
             if idx is not None:
                 raw_amt = float(pv_am[k])
```

Modify `src/pymarxan/solvers/mip_solver.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/mip_solver.py b/src/pymarxan/solvers/mip_solver.py
index 23fd46b..c2aca57 100644
--- a/src/pymarxan/solvers/mip_solver.py
+++ b/src/pymarxan/solvers/mip_solver.py
@@ -55,11 +55,17 @@ from pymarxan.solvers.base import Solution, Solver, SolverConfig
 from pymarxan.solvers.presolve import (
     _emit_presolve,
     _presolve_data,
     _validate_presolve_flag,
 )
-from pymarxan.solvers.utils import _nonbinary_conflicts, build_solution
+from pymarxan.solvers.utils import (
+    _nonbinary_conflicts,
+    _probability_map,
+    _probmode,
+    _probmode2_active,
+    build_solution,
+)
 
 
 def _reject_nonbinary(
     problem: ConservationProblem, config: SolverConfig, decisions: str,
 ) -> None:
@@ -467,11 +473,11 @@ class MIPSolver(Solver):
         # Stage 4: settings that need 0/1 decisions fail first, before any strategy check.
         _reject_nonbinary(problem, config, self.decisions)
         binary = self.decisions == "binary"
 
         # PROBMODE 3 gating — see class docstring for rationale.
-        probmode = int(problem.parameters.get("PROBMODE", 0))
+        probmode = _probmode(problem)
         if probmode == 3:
             if self.mip_chance_strategy == "piecewise":
                 raise NotImplementedError(
                     "mip_chance_strategy='piecewise' (piecewise-linear "
                     "approximation of the chance constraint in CBC) is "
@@ -661,42 +667,31 @@ class MIPSolver(Solver):
                         f"conn_and3_{key[0]}_{key[1]}",
                     )
                     implied.append((z, _and_rule(x[key[0]], x[key[1]])))
                     conn_expr += -conn_weight * cval * z
 
-        # Probability risk premium (Mode 1). PROBMODE defaults to 1 here (0 for the
-        # PROBMODE 3 gate above): a probability table without PROBMODE is Mode 1.
+        # Probability risk premium (Mode 1; the default with a probability table). The
+        # table is read by the shared helpers in solvers/utils.py (v0.45 §3).
         prob_expr = pulp.LpAffineExpression()
-        prob_mode = int(problem.parameters.get("PROBMODE", 1))
-        if problem.probability is not None and prob_mode == 1:
+        if problem.probability is not None and probmode == 1:
             prob_weight = float(
                 problem.parameters.get("PROBABILITYWEIGHTING", 1.0)
             )
             if prob_weight > 0:
-                prob_pu = problem.probability["pu"].values
-                prob_val = problem.probability["probability"].values
-                prob_map: dict[int, float] = {}
-                for pk in range(len(prob_pu)):
-                    prob_map[int(prob_pu[pk])] = float(prob_val[pk])
+                prob_map = _probability_map(problem)
                 prob_expr = pulp.lpSum(
                     prob_weight
                     * prob_map.get(int(pu_id_arr[k]), 0.0)
                     * float(pu_cost_arr[k])
                     * x[int(pu_id_arr[k])]
                     for k in range(len(pu_id_arr))
                 )
 
-        # Build probability map for Mode 2 (persistence-adjusted amounts)
-        prob_map_m2: dict[int, float] = {}
-        if (
-            problem.probability is not None
-            and prob_mode == 2
-        ):
-            p_pu = problem.probability["pu"].values
-            p_val = problem.probability["probability"].values
-            for pk in range(len(p_pu)):
-                prob_map_m2[int(p_pu[pk])] = float(p_val[pk])
+        # Probability map for Mode 2 (expected amounts a·(1 − p) in the target rows)
+        prob_map_m2: dict[int, float] = (
+            _probability_map(problem) if _probmode2_active(problem) else {}
+        )
 
         amount_exprs = _core_amount_exprs(problem, x, prob_map_m2)
         terms = _core_target_terms(problem, x, prob_map_m2, amount_exprs)
 
         # Phase 25: no-good cuts for portfolio generation. Each forbidden
```

Modify `src/pymarxan/solvers/utils.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/utils.py b/src/pymarxan/solvers/utils.py
index af836aa..418ed54 100644
--- a/src/pymarxan/solvers/utils.py
+++ b/src/pymarxan/solvers/utils.py
@@ -85,33 +85,95 @@ def _total_cost(costs: np.ndarray, selected: np.ndarray) -> float:
     if _is_fractional(selected):
         return float(np.dot(costs.astype(np.float64), selected))
     return float(np.sum(costs[selected]))
 
 
+# --- Per-PU probability semantics (PROBMODE 1 / 2): the one copy every path uses -------
+# The evaluator, ProblemCache, the MIP and the greedy heuristic all read the probability
+# table through these helpers (v0.45 §3), so the rules cannot drift apart again.
+
+
+def _probmode(problem: ConservationProblem) -> int:
+    """``PROBMODE``; default 1 with a probability table, 0 without (v0.45 §3)."""
+    default = 1 if problem.probability is not None else 0
+    return int(problem.parameters.get("PROBMODE", default))
+
+
+def _probmode2_active(problem: ConservationProblem) -> bool:
+    """True iff PROBMODE 2 applies: a probability table and ``PROBMODE == 2``."""
+    return problem.probability is not None and _probmode(problem) == 2
+
+
+def _probability_map(problem: ConservationProblem) -> dict[int, float]:
+    """PU id -> probability from ``problem.probability`` (which must be set).
+
+    A PU listed twice: the last row wins. Values are used as given (``validate()`` reports
+    ones outside [0, 1]); a non-finite value (a blank ``prob.dat`` cell reads as NaN) raises
+    ``ValueError`` naming the PU ids, so every solver and the evaluator fail identically.
+    """
+    table = problem.probability
+    if table is None:
+        raise ValueError("problem has no probability table")
+    pids = table["pu"].to_numpy()
+    vals = np.asarray(table["probability"].to_numpy(), dtype=np.float64)
+    bad = ~np.isfinite(vals)
+    if bad.any():
+        raise ValueError(
+            "probability has non-finite values (NaN or infinite) for planning unit IDs "
+            f"{sorted({int(u) for u in pids[bad]})}"
+        )
+    p_of: dict[int, float] = {}
+    for pid, val in zip(pids, vals):
+        p_of[int(pid)] = float(val)
+    return p_of
+
+
+def _probability_vector(
+    problem: ConservationProblem, pu_id_to_idx: dict[int, int], n_pu: int,
+) -> np.ndarray:
+    """(n_pu,) probabilities by position: a PU missing from the table is 0, table rows for
+    ids that are not planning units are skipped, the last row of a duplicated PU wins."""
+    p = np.zeros(n_pu, dtype=np.float64)
+    for pid, val in _probability_map(problem).items():
+        k = pu_id_to_idx.get(pid)
+        if k is not None:
+            p[k] = val
+    return p
+
+
 def _compute_achieved(
     problem: ConservationProblem,
     selected: np.ndarray,
     pu_index: dict[int, int],
 ) -> dict[int, float]:
     """Vectorized computation of achieved amount per feature for selected PUs.
 
-    Returns a dict mapping feature id -> total achieved amount.
+    Returns a dict mapping feature id -> total achieved amount. Under PROBMODE 2 with a
+    probability table each amount is the expected amount ``amount·(1 − p)`` of its PU (the
+    MIP's target rows); ``p`` is 0 for a PU missing from the table and the last row wins
+    for a PU listed twice.
     """
     puvspr = problem.pu_vs_features
     pu_col = puvspr["pu"].values
     species_col = puvspr["species"].values
     amount_col = puvspr["amount"].values
+    if _probmode2_active(problem):
+        # p_of holds only finite values (_probability_map raises otherwise), so fillna
+        # touches exactly the PUs missing from the table (p = 0).
+        p_of = _probability_map(problem)
+        keep = 1.0 - pd.Series(pu_col).map(p_of).fillna(0.0).to_numpy(dtype=np.float64)
+        amount_col = np.asarray(amount_col, dtype=np.float64) * keep
 
     # Build boolean mask: True for rows whose PU is selected
     pu_indices = np.array(
         [pu_index.get(int(pid), -1) for pid in pu_col], dtype=np.intp
     )
     valid = pu_indices >= 0
     # Safe indexing: use 0 for invalid indices, then mask out
     safe_indices = np.where(valid, pu_indices, 0)
     if _is_fractional(selected):
-        # held_f = Σ_i r_if·x_i (raw amounts, as the binary path)
+        # held_f = Σ_i r_if·x_i (expected amounts under PROBMODE 2, as the binary path)
         weighted = pd.DataFrame({
             "species": species_col,
             "amount": amount_col.astype(np.float64) * np.where(valid, selected[safe_indices], 0.0),
         })
         result = {int(fid): 0.0 for fid in problem.features["id"]}
@@ -141,13 +203,20 @@ def compute_held(
     selected: np.ndarray,
     pu_index: dict[int, int] | None = None,
 ) -> dict[int, float]:
     """Feature id -> amount held: Σ amount over selected PUs, or Σ amount·x for proportions.
 
+    Under PROBMODE 2 with a probability table the amounts are expected amounts
+    ``amount·(1 − p)`` (as the MIP's target rows), so targets met, shortfall, penalty and
+    every report built on this agree with the MIP.
+
     Public wrapper over ``_compute_achieved`` (review R7). ``pu_index`` maps PU id to its
-    position in ``selected`` and defaults to ``planning_units`` order.
+    position in ``selected`` and defaults to ``planning_units`` order. ``selected`` must
+    follow the ``_coerce_selected`` contract: int 0/1 arrays are read as bool; other ints
+    and out-of-range proportions raise ``ValueError``.
     """
+    selected = _coerce_selected(selected, len(problem.planning_units))
     if pu_index is None:
         pu_index = {int(p): i for i, p in enumerate(problem.planning_units["id"])}
     return _compute_achieved(problem, selected, pu_index)
 
 
@@ -375,18 +444,19 @@ def compute_probability_penalty(
     pu_index: dict[int, int],
 ) -> float:
     """Dispatch on ``PROBMODE`` to the appropriate probability penalty.
 
     - ``PROBMODE == 0``: no probability term, returns 0.
-    - ``PROBMODE == 1``: Marxan risk premium
-      ``γ · Σ prob_i · cost_i · x_i`` (per-PU PROB1D).
-    - ``PROBMODE == 2``: persistence-adjusted amounts (handled upstream
-      via amount substitution); returns 0 here.
+    - ``PROBMODE == 1`` (the default): risk premium ``γ · Σ prob_i · cost_i · x_i``
+      (a pymarxan extension; per-PU probabilities). Off when ``γ <= 0`` or there is no
+      probability table, as in the MIP.
+    - ``PROBMODE == 2``: no objective term; held amounts are expected amounts
+      ``amount·(1 − prob)`` (see ``compute_held``). Returns 0 here.
     - ``PROBMODE == 3``: Marxan-faithful Z-score chance-constraint
       penalty using per-cell ``prob`` and per-feature ``ptarget``.
     """
-    prob_mode = int(problem.parameters.get("PROBMODE", 1))
+    prob_mode = _probmode(problem)
     if prob_mode == 1:
         return _compute_probmode1_risk_premium(problem, selected)
     if prob_mode == 3:
         if _is_fractional(selected):
             raise ValueError(
@@ -400,23 +470,19 @@ def compute_probability_penalty(
 
 def _compute_probmode1_risk_premium(
     problem: ConservationProblem,
     selected: np.ndarray,
 ) -> float:
-    """PROBMODE 1: γ · Σ_i x_i · prob_i · cost_i (per-PU risk premium)."""
+    """PROBMODE 1: γ · Σ_i x_i · prob_i · cost_i (per-PU risk premium); off for γ <= 0."""
     if problem.probability is None:
         return 0.0
 
     prob_weight = float(problem.parameters.get("PROBABILITYWEIGHTING", 1.0))
-    if prob_weight == 0.0:
+    if prob_weight <= 0.0:
         return 0.0
 
-    prob_pu = problem.probability["pu"].values
-    prob_val = problem.probability["probability"].values
-    prob_map: dict[int, float] = {}
-    for k in range(len(prob_pu)):
-        prob_map[int(prob_pu[k])] = float(prob_val[k])
+    prob_map = _probability_map(problem)
 
     costs = problem.planning_units["cost"].values
     pu_ids = problem.planning_units["id"].values
     fractional = _is_fractional(selected)
 
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_probmode_evaluator.py tests/pymarxan/solvers/test_stage3_golden_v040.py -q`
Expected: 153 passed

- [ ] **Step 5: Gate**

Run: `~/.local/bin/ruff check src tests && /home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/ -q -m "not bench" -p no:cacheprovider`
Expected: ruff and mypy clean; the full suite passes (prototype: 3636 passed, 101 skipped HiGHS cases which run in CI, 21–22 bench deselected); no golden JSON regenerated; no existing test changed except the Stage 3 probmode2 records via STAGE3_PROBMODE2_FIXED.

- [ ] **Step 6: Commit**

```bash
git add src/pymarxan/models/problem.py src/pymarxan/solvers/heuristic.py src/pymarxan/solvers/mip_solver.py src/pymarxan/solvers/utils.py tests/pymarxan/solvers/test_probmode_evaluator.py tests/pymarxan/solvers/test_stage3_golden_v040.py
git commit -m "fix(probability): evaluator uses PROBMODE 2 expected amounts and the MIP's weighting rule

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AZEiop9qqxVCrMHKqSDYP8"
```

---

### Task 2: Cache: SA and iterative improvement optimise PROBMODE 1 and 2

**Files:**
- Modify: `src/pymarxan/solvers/cache.py`
- Modify: `src/pymarxan/solvers/clumping.py`
- Create: `tests/pymarxan/solvers/test_cache_probmode.py`
- Create: `tests/pymarxan/solvers/test_sa_probmode.py`

**Interfaces:**
- Consumes: Task 1's `_probmode`, `_probability_map`, mode-2 `_compute_achieved` (the oracle for full = evaluator).
- Produces (`ProblemCache`, defaulted fields): `prob_cost: np.ndarray` (γ·p·c, zeros when off), `prob_active: bool`, `held_csr: csr_matrix` (`compare=False, repr=False`; scaled copy with identical `indptr` / `indices` under mode 2, otherwise `pu_feat_csr` itself; `__post_init__` fills it for a hand-built cache); module-level `_build_probability(...)`; `probmode` defaults to 1 with a table. `compute_held`, `apply_flip_to_held` and the penalty delta read `held_csr`; full objective and delta add the premium when `prob_active`; the cache build warns once (`UserWarning`) for PROBMODE 2 with active clumping or separation (P4); the p vector comes from `_probability_vector`.

- [ ] **Step 1: Write the failing tests**

Create `tests/pymarxan/solvers/test_cache_probmode.py`:

```python
"""ProblemCache under PROBMODE 1 / 2 (v0.45 §4): the MIP's and the evaluator's semantics.

Mode 1 adds ``γ·p·c`` per selected PU (``prob_cost``, read only when ``prob_active``); mode 2
reads held amounts from ``held_csr``, a copy of ``pu_feat_csr`` scaled by ``(1 − p)``. A
probability table with no ``PROBMODE`` is mode 1.
"""
from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.cache import ProblemCache
from pymarxan.solvers.utils import build_solution

DATA = Path(__file__).parent.parent.parent / "data"
LINSPACE = pd.DataFrame({"pu": [1, 2, 3, 4, 5, 6], "probability": np.linspace(0.1, 0.6, 6)})


def _simple(
    probability: pd.DataFrame | None = None, mode: int | None = None, **params: Any,
) -> ConservationProblem:
    p = load_project(DATA / "simple")
    merged = {**p.parameters, **params}
    if mode is not None:
        merged["PROBMODE"] = mode
    return p.copy_with(probability=probability, parameters=merged)


def _edge(mode: int | None, gamma: float) -> ConservationProblem:
    """§6 fixture: PU 1 and 3 missing from the table, PU 2 listed twice (0.2 wins), PU 4
    p = 1, PU 3 cost −4, PU 1 locked in, MISSLEVEL 0.8, and a cost threshold (COSTTHRESH
    30, THRESHPEN1 5, THRESHPEN2 1) that must see base cost, not cost + premium."""
    p = _simple()
    pu = p.planning_units.copy()
    pu.loc[pu["id"] == 3, "cost"] = -4.0
    pu.loc[pu["id"] == 1, "status"] = 2
    prob = pd.DataFrame({"pu": [2, 2, 4, 5, 6], "probability": [0.9, 0.2, 1.0, 0.3, 0.45]})
    params = {
        **p.parameters, "PROBABILITYWEIGHTING": gamma, "MISSLEVEL": 0.8,
        "COSTTHRESH": 30.0, "THRESHPEN1": 5.0, "THRESHPEN2": 1.0,
    }
    if mode is not None:
        params["PROBMODE"] = mode
    return p.copy_with(planning_units=pu, probability=prob, parameters=params)


FIXTURES = [
    pytest.param(mode, gamma, id=f"mode{mode}-gamma{gamma:g}")
    for mode in (1, 2, None)
    for gamma in (2.0, -1.0, 0.0)
]


class TestDefaultModeAndFlags:
    def test_table_without_probmode_is_mode_1(self) -> None:
        c = ProblemCache.from_problem(_simple(LINSPACE))
        assert c.probmode == 1
        assert c.prob_active

    def test_no_table_without_probmode_is_mode_0(self) -> None:
        c = ProblemCache.from_problem(_simple())
        assert c.probmode == 0
        assert not c.prob_active

    def test_mode_1_prob_cost(self) -> None:
        p = _simple(LINSPACE, 1, PROBABILITYWEIGHTING=2.0)
        c = ProblemCache.from_problem(p)
        assert c.prob_active
        np.testing.assert_allclose(
            c.prob_cost, 2.0 * np.linspace(0.1, 0.6, 6) * p.planning_units["cost"].to_numpy(),
        )
        np.testing.assert_array_equal(c.costs, p.planning_units["cost"].to_numpy())

    @pytest.mark.parametrize("gamma", [0.0, -1.0])
    def test_mode_1_off_for_non_positive_weighting(self, gamma: float) -> None:
        c = ProblemCache.from_problem(_simple(LINSPACE, 1, PROBABILITYWEIGHTING=gamma))
        assert not c.prob_active

    def test_mode_2_has_no_premium(self) -> None:
        c = ProblemCache.from_problem(_simple(LINSPACE, 2))
        assert not c.prob_active

    @pytest.mark.parametrize("mode", [1, 2])
    def test_no_table_is_a_noop(self, mode: int) -> None:
        c = ProblemCache.from_problem(_simple(None, mode))
        assert not c.prob_active
        assert len(c.prob_cost) in (0, c.n_pu)
        assert not np.any(c.prob_cost)
        assert c.held_csr is c.pu_feat_csr

    def test_probmode_3_unchanged(self) -> None:
        c = ProblemCache.from_problem(_simple(LINSPACE, 3))
        assert c.probmode == 3
        assert not c.prob_active
        assert c.held_csr is c.pu_feat_csr


class TestHeldCsr:
    @pytest.mark.parametrize("mode", [0, 1, 3, None])
    def test_non_mode_2_shares_the_raw_matrix(self, mode: int | None) -> None:
        c = ProblemCache.from_problem(_simple(LINSPACE, mode))
        assert c.held_csr is c.pu_feat_csr

    def test_mode_2_structure_and_scaling(self) -> None:
        p = _edge(2, 1.0)  # PU 4 has p = 1: its row keeps explicit zeros
        c = ProblemCache.from_problem(p)
        raw, held = c.pu_feat_csr, c.held_csr
        assert held is not raw
        np.testing.assert_array_equal(held.indptr, raw.indptr)
        np.testing.assert_array_equal(held.indices, raw.indices)
        p_vec = np.array([0.0, 0.2, 0.0, 1.0, 0.3, 0.45])
        np.testing.assert_allclose(
            held.data, raw.data * np.repeat(1.0 - p_vec, np.diff(raw.indptr)),
        )
        row4 = slice(raw.indptr[3], raw.indptr[4])
        assert raw.indptr[4] > raw.indptr[3]
        assert np.all(held.data[row4] == 0.0)
        # Raw matrix untouched (clumping / separation read it).
        np.testing.assert_array_equal(
            raw.toarray(), ProblemCache.from_problem(_simple()).pu_feat_csr.toarray(),
        )

    def test_hand_built_cache_falls_back_to_raw(self) -> None:
        c = ProblemCache.from_problem(_simple(LINSPACE, 2))
        rebuilt = dataclasses.replace(c, held_csr=None)
        assert rebuilt.held_csr is rebuilt.pu_feat_csr

    def test_mode_2_compute_held_is_expected_amount(self) -> None:
        prob = pd.DataFrame({"pu": [4, 5, 6], "probability": [0.5, 0.5, 0.5]})
        p = _simple(prob, 2)
        c = ProblemCache.from_problem(p)
        sel = np.array([True, False, True, True, False, True])
        # f1: 10 + 5 + 12·0.5 + 10·0.5; f2: 5 + 8 + 3·0.5 + 7·0.5; f3: 10 + 4·0.5 + 8·0.5
        np.testing.assert_allclose(c.compute_held(sel), [26.0, 18.0, 16.0])


class TestDeltaAndFullObjective:
    @pytest.mark.parametrize(("mode", "gamma"), FIXTURES)
    def test_delta_equals_full_difference(self, mode: int | None, gamma: float) -> None:
        c = ProblemCache.from_problem(_edge(mode, gamma))
        rng = np.random.default_rng(0)
        for _ in range(10):
            base = rng.random(c.n_pu) < 0.5
            for i in range(c.n_pu):
                for adding in (True, False):  # both directions
                    s = base.copy()
                    s[i] = not adding
                    h = c.compute_held(s)
                    f0 = c.compute_full_objective(s, h, 1.0)
                    d = c.compute_delta_objective(i, s, h, float(c.costs[s].sum()), 1.0)
                    s2 = s.copy()
                    s2[i] = adding
                    f1 = c.compute_full_objective(s2, c.compute_held(s2), 1.0)
                    assert d == pytest.approx(f1 - f0, rel=1e-9, abs=1e-9)

    @pytest.mark.parametrize(("mode", "gamma"), FIXTURES)
    def test_full_equals_evaluator(self, mode: int | None, gamma: float) -> None:
        p = _edge(mode, gamma)
        c = ProblemCache.from_problem(p)
        rng = np.random.default_rng(1)
        for _ in range(20):
            s = rng.random(c.n_pu) < 0.5
            full = c.compute_full_objective(s, c.compute_held(s), 1.0)
            assert full == pytest.approx(build_solution(p, s, 1.0).objective, rel=1e-9, abs=1e-9)

    @pytest.mark.parametrize("mode", [1, 2, None])
    def test_full_equals_evaluator_every_plan(self, mode: int | None) -> None:
        p = _simple(LINSPACE, mode, PROBABILITYWEIGHTING=2.0)
        c = ProblemCache.from_problem(p)
        for bits in range(2 ** c.n_pu):
            s = np.array([(bits >> k) & 1 for k in range(c.n_pu)], dtype=bool)
            full = c.compute_full_objective(s, c.compute_held(s), 1.0)
            assert full == pytest.approx(build_solution(p, s, 1.0).objective, rel=1e-9, abs=1e-9)


_NAN_PROB = pd.DataFrame({"pu": [1, 2, 5], "probability": [0.3, np.nan, 0.4]})


class TestNonFinite:
    @pytest.mark.parametrize("mode", [1, 2, None])
    def test_cache_build_raises(self, mode: int | None) -> None:
        with pytest.raises(ValueError, match=r"non-finite .* \[2\]"):
            ProblemCache.from_problem(_simple(_NAN_PROB, mode))

    def test_mode_0_ignores_the_table(self) -> None:
        assert ProblemCache.from_problem(_simple(_NAN_PROB, 0)).probmode == 0


def _clumping(p: ConservationProblem) -> ConservationProblem:
    f = p.features.copy()
    f["target2"] = [10.0, 0.0, 0.0]
    f["clumptype"] = [0, 0, 0]
    return p.copy_with(features=f)


def _separation(p: ConservationProblem) -> ConservationProblem:
    pu = p.planning_units.copy()
    pu["xloc"] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    pu["yloc"] = 0.0
    f = p.features.copy()
    f["sepdistance"] = [0.0, 0.0, 2.0]
    f["sepnum"] = [1, 1, 2]
    return p.copy_with(planning_units=pu, features=f)


_RAW_WARNING = "PROBMODE 2 expected amounts are not applied to clumping/separation"


class TestProbmode2ClumpSepWarning:
    @pytest.mark.parametrize("combine", [_clumping, _separation])
    def test_warns_under_mode_2(self, combine: Any) -> None:
        with pytest.warns(UserWarning, match=_RAW_WARNING):
            ProblemCache.from_problem(combine(_simple(LINSPACE, 2)))

    @pytest.mark.parametrize("combine", [_clumping, _separation])
    @pytest.mark.parametrize(("probability", "mode"), [(LINSPACE, 1), (LINSPACE, 0), (None, 2)])
    def test_silent_outside_mode_2(
        self, combine: Any, probability: pd.DataFrame | None, mode: int,
    ) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=_RAW_WARNING)
            ProblemCache.from_problem(combine(_simple(probability, mode)))

    def test_silent_without_clump_or_sep(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=_RAW_WARNING)
            ProblemCache.from_problem(_simple(LINSPACE, 2))
```

Create `tests/pymarxan/solvers/test_sa_probmode.py`:

```python
"""Simulated annealing optimises PROBMODE 1 and 2 (v0.45 §6).

Pinned SA parameters (probed; do not change without re-probing): ``NUMITNS = 10_000``,
``NUMTEMP = 100``; best-of-10 tests run ``SolverConfig(num_solutions=10, seed=42)``,
consistency tests ``num_solutions=1, seed=42``.
"""
from __future__ import annotations

import itertools
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, SolverConfig
from pymarxan.solvers.cache import ProblemCache
from pymarxan.solvers.clumping import ClumpState, compute_clump_penalty_from_scratch
from pymarxan.solvers.iterative_improvement import IterativeImprovementSolver
from pymarxan.solvers.mip_solver import MIPSolver
from pymarxan.solvers.separation import SepState, compute_sep_penalty_from_scratch
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
from pymarxan.solvers.utils import build_solution, compute_feature_shortfalls

DATA = Path(__file__).parent.parent.parent / "data"
SA_PARAMS = {"NUMITNS": 10_000, "NUMTEMP": 100}
LINSPACE = pd.DataFrame({"pu": [1, 2, 3, 4, 5, 6], "probability": np.linspace(0.1, 0.6, 6)})
HALF_456 = pd.DataFrame({"pu": [4, 5, 6], "probability": [0.5, 0.5, 0.5]})


def _problem(
    probability: pd.DataFrame | None,
    mode: int | None,
    *,
    big_spf: bool = False,
    **params: Any,
) -> ConservationProblem:
    p = load_project(DATA / "simple")
    merged = {**p.parameters, **SA_PARAMS, **params}
    if mode is not None:
        merged["PROBMODE"] = mode
    features = p.features.copy()
    if big_spf:  # every spf >= 10 × Σ cost: the soft optimum meets every target
        features["spf"] = 10.0 * float(p.planning_units["cost"].sum())
    return p.copy_with(features=features, probability=probability, parameters=merged)


def _sa(problem: ConservationProblem, n: int) -> list[Solution]:
    return SimulatedAnnealingSolver().solve(problem, SolverConfig(num_solutions=n, seed=42))


def _tracked(sol: Solution) -> float:
    return float(sol.metadata["history"]["best_objective"][-1])


def _plans(n: int) -> list[np.ndarray]:
    return [np.array(bits, dtype=bool) for bits in itertools.product([0, 1], repeat=n)]


class TestConsistency:
    @pytest.mark.parametrize("mode", [0, 1, 2, None])
    def test_reported_objective_equals_tracked(self, mode: int | None) -> None:
        p = _problem(LINSPACE, mode, PROBABILITYWEIGHTING=2.0)
        (sol,) = _sa(p, 1)
        assert sol.objective == pytest.approx(_tracked(sol), rel=1e-9, abs=1e-9)


class TestMode2Oracle:
    def test_raw_amounts_meet_a_target_expected_do_not(self) -> None:
        """The fixture's premise: the 35.0 anchor {2, 4, 6} misses under expected amounts."""
        anchor = np.array([False, True, False, True, False, True])
        assert all(build_solution(_problem(None, None), anchor, 0.0).targets_met.values())
        assert not all(build_solution(_problem(HALF_456, 2), anchor, 0.0).targets_met.values())

    def test_feasible_optimum_is_50(self) -> None:
        p = _problem(HALF_456, 2, big_spf=True)
        c = ProblemCache.from_problem(p)
        blm = float(p.parameters["BLM"])
        best = min(
            (c.compute_full_objective(s, c.compute_held(s), blm), tuple(s.astype(int)))
            for s in _plans(c.n_pu)
        )
        assert best[0] == pytest.approx(50.0, abs=1e-9)
        assert best[1] == (1, 1, 1, 0, 0, 0)

    @pytest.mark.slow
    def test_sa_best_of_10_reaches_feasible_optimum(self) -> None:
        p = _problem(HALF_456, 2, big_spf=True)
        sols = _sa(p, 10)
        best = min(sols, key=lambda s: s.objective)
        assert best.objective == pytest.approx(50.0, abs=1e-9)
        assert best.selected.astype(int).tolist() == [1, 1, 1, 0, 0, 0]
        assert all(best.targets_met.values())


CROSS = [pytest.param(1, 69.2, id="mode1"), pytest.param(2, 74.0, id="mode2")]
# LINSPACE with PU 2 listed twice: the first row (0.9) must lose to the last (0.2) in the
# cache, the evaluator and the MIP alike (presolve's P0 only reports it).
LINSPACE_DUP = pd.concat(
    [pd.DataFrame({"pu": [2], "probability": [0.9]}), LINSPACE], ignore_index=True,
)


def _mip_value(p: ConservationProblem) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # PuLP 3.3.2
        (s,) = MIPSolver(mip_backend="cbc").solve(p, SolverConfig(num_solutions=1))
    return float(s.metadata["mip_objective_value"])


class TestCrossSolver:
    @pytest.mark.parametrize(("mode", "optimum"), CROSS)
    def test_cache_evaluator_and_mip_agree(self, mode: int, optimum: float) -> None:
        p = _problem(LINSPACE, mode, big_spf=True, PROBABILITYWEIGHTING=2.0)
        c = ProblemCache.from_problem(p)
        blm = float(p.parameters["BLM"])
        cache_best = min(
            (c.compute_full_objective(s, c.compute_held(s), blm), tuple(s)) for s in _plans(c.n_pu)
        )
        soft = build_solution(p, np.array(cache_best[1]), blm)
        assert all(soft.targets_met.values())  # the soft optimum is target-feasible
        feasible = [
            sol.objective
            for sol in (build_solution(p, s, blm) for s in _plans(c.n_pu))
            if all(sol.targets_met.values())
        ]
        assert cache_best[0] == pytest.approx(optimum, abs=1e-9)
        assert min(feasible) == pytest.approx(optimum, abs=1e-9)
        assert _mip_value(p) == pytest.approx(optimum, abs=1e-9)

    @pytest.mark.parametrize(("mode", "optimum"), CROSS)
    def test_duplicate_pu_last_row_wins_across_solvers(self, mode: int, optimum: float) -> None:
        p = _problem(LINSPACE_DUP, mode, big_spf=True, PROBABILITYWEIGHTING=2.0)
        c = ProblemCache.from_problem(p)
        blm = float(p.parameters["BLM"])
        cache_best = min(
            c.compute_full_objective(s, c.compute_held(s), blm) for s in _plans(c.n_pu)
        )
        feasible = [
            sol.objective
            for sol in (build_solution(p, s, blm) for s in _plans(c.n_pu))
            if all(sol.targets_met.values())
        ]
        assert cache_best == pytest.approx(optimum, abs=1e-9)  # as without the duplicate
        assert min(feasible) == pytest.approx(optimum, abs=1e-9)
        assert _mip_value(p) == pytest.approx(optimum, abs=1e-9)

    @pytest.mark.slow
    @pytest.mark.parametrize(("mode", "optimum"), CROSS)
    def test_sa_best_of_10_reaches_optimum(self, mode: int, optimum: float) -> None:
        p = _problem(LINSPACE, mode, big_spf=True, PROBABILITYWEIGHTING=2.0)
        sols = _sa(p, 10)
        assert min(s.objective for s in sols) == pytest.approx(optimum, abs=1e-9)


def _with_clumping(p: ConservationProblem) -> ConservationProblem:
    f = p.features.copy()
    f["target2"] = [10.0, 0.0, 0.0]
    f["clumptype"] = [0, 0, 0]
    return p.copy_with(features=f)


def _with_separation(p: ConservationProblem) -> ConservationProblem:
    pu = p.planning_units.copy()
    pu["xloc"] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    pu["yloc"] = 0.0
    f = p.features.copy()
    f["sepdistance"] = [0.0, 0.0, 2.0]
    f["sepnum"] = [1, 1, 2]
    return p.copy_with(planning_units=pu, features=f)


COMBOS = [
    pytest.param(_with_clumping, id="clumping", marks=pytest.mark.slow),  # ~2.3 s
    pytest.param(_with_separation, id="separation"),
]


_RAW_WARNING = "PROBMODE 2 expected amounts are not applied to clumping/separation"
pytestmark = pytest.mark.filterwarnings(f"ignore:{_RAW_WARNING}:UserWarning")


class TestCombinations:
    """PROBMODE 2 with clumping / separation: their amounts stay raw (documented; the cache
    build warns)."""

    def test_clump_state_matches_from_scratch(self) -> None:
        p = _with_clumping(_problem(HALF_456, 2))
        c = ProblemCache.from_problem(p)
        rng = np.random.default_rng(3)
        sel = rng.random(c.n_pu) < 0.5
        state = ClumpState.from_selection(c, sel)
        pen = compute_clump_penalty_from_scratch(p, sel)[1]
        for idx in rng.integers(c.n_pu, size=40):
            adding = not sel[idx]
            pen += state.delta_penalty(c, int(idx), adding)
            state.apply_flip(c, int(idx), adding)
            sel[idx] = adding
        assert pen == pytest.approx(compute_clump_penalty_from_scratch(p, sel)[1], abs=1e-9)

    def test_sep_state_matches_from_scratch(self) -> None:
        p = _with_separation(_problem(HALF_456, 2))
        c = ProblemCache.from_problem(p)
        rng = np.random.default_rng(4)
        sel = rng.random(c.n_pu) < 0.5
        state = SepState.from_selection(c, sel)
        for idx in rng.integers(c.n_pu, size=40):
            adding = not sel[idx]
            state.apply_flip(c, int(idx), adding)
            sel[idx] = adding
        assert state.penalty_total() == pytest.approx(
            compute_sep_penalty_from_scratch(p, sel)[1], abs=1e-9,
        )

    @pytest.mark.parametrize("combine", COMBOS)
    def test_sa_tracked_objective_identity(self, combine: Any) -> None:
        p = combine(_problem(HALF_456, 2))
        (sol,) = _sa(p, 1)
        pu_index = {int(u): i for i, u in enumerate(p.planning_units["id"])}
        shortfalls = compute_feature_shortfalls(p, sol.selected, pu_index)
        f = p.features
        active = (f.get("target2", 0) > 0) | (
            (f.get("sepnum", 1) > 1) & (f.get("sepdistance", 0) > 0)
        )
        removed = sum(
            float(spf) * shortfalls[int(fid)]
            for fid, spf, on in zip(f["id"], f["spf"], active)
            if on
        )
        want = sol.objective - removed + (sol.clump_penalty or 0.0) + (sol.sep_penalty or 0.0)
        assert _tracked(sol) == pytest.approx(want, rel=1e-9, abs=1e-9)


def _ii_premium_problem(mode: int | None) -> ConservationProblem:
    """PU 1 (cost 4, p 0.9) alone carries the feature (target 5, spf 5). Its removal saves
    the premium 10·0.9·4 = 36 at a penalty of 25: worth it only when the premium counts."""
    pu = pd.DataFrame({"id": [1, 2], "cost": [4.0, 1.0], "status": [0, 0]})
    feat = pd.DataFrame({"id": [1], "name": ["f"], "target": [5.0], "spf": [5.0]})
    puvspr = pd.DataFrame({"species": [1], "pu": [1], "amount": [5.0]})
    prob = pd.DataFrame({"pu": [1, 2], "probability": [0.9, 0.0]})
    params: dict[str, Any] = {"PROBABILITYWEIGHTING": 10.0, "BLM": 0.0}
    if mode is not None:
        params["PROBMODE"] = mode
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=puvspr, probability=prob,
        parameters=params,
    )


class TestIterativeImprovement:
    @pytest.mark.parametrize("mode", [1, None])
    @pytest.mark.parametrize("itimptype", [1, 2])
    def test_ii_optimises_the_premium(self, mode: int | None, itimptype: int) -> None:
        p = _ii_premium_problem(mode)
        (sol,) = IterativeImprovementSolver(itimptype=itimptype).solve(
            p, SolverConfig(num_solutions=1, seed=1),
        )
        assert sol.selected.tolist() == [False, False]  # premium 36 > penalty 25
        assert sol.objective == pytest.approx(25.0)
        assert sol.objective == pytest.approx(build_solution(p, sol.selected, 0.0).objective)

    @pytest.mark.parametrize("mode", [1, None])
    def test_swap_pass_optimises_the_premium(self, mode: int | None) -> None:
        """ITIMPTYPE 3 (swaps only) from {PU 1}: swapping in PU 2 gains 36 + 4 − 25 − 1."""
        p = _ii_premium_problem(mode)
        start = build_solution(p, np.array([True, False]), 0.0)
        sol = IterativeImprovementSolver(itimptype=3).improve(p, start)
        assert sol.selected.tolist() == [False, True]
        assert sol.objective == pytest.approx(26.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_cache_probmode.py tests/pymarxan/solvers/test_sa_probmode.py -q`
Expected: FAIL — 46 failed, 27 passed — missing `prob_active` / `held_csr`; full objective ≠ evaluator for modes 1, 2 and default (now with COSTTHRESH, P7); SA reported ≠ tracked for 1, 2 and default; the oracle and cross-solver brute-force checks; the SA identity with clumping and with separation; iterative improvement ×6 (P8); duplicate PU across solvers ×2; cache NaN raising ×3; the clump/separation warning ×2. Delta exactness for the default/mode-1 cases without the premium and most best-of-10 SA tests already pass (regression guards)

- [ ] **Step 3: Implement**

Modify `src/pymarxan/solvers/cache.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/cache.py b/src/pymarxan/solvers/cache.py
index e8bf3c4..6038ca7 100644
--- a/src/pymarxan/solvers/cache.py
+++ b/src/pymarxan/solvers/cache.py
@@ -27,10 +27,25 @@ Connectivity
 ``CONNECTIVITY_WEIGHT > 0`` with a connectivity table builds CSR adjacency arrays
 (``conn_out_*``; asymmetric also ``conn_in_*``) plus symmetric self-pair rewards
 (``conn_self``), mirroring ``compute_connectivity_penalty`` (the MIP's term). Both
 ``compute_full_objective`` and ``compute_delta_objective`` read them; when ``conn_mode``
 is 0 (off) the delta pays one integer comparison.
+
+Per-PU probabilities (PROBMODE 1 / 2)
+-------------------------------------
+pymarxan extensions over ``problem.probability`` (``prob.dat``), with the MIP's semantics;
+``PROBMODE`` defaults to 1 when a probability table is present. Mode 1 adds the risk
+premium ``γ·p_i·c_i`` per selected PU from ``prob_cost`` (read only when ``prob_active``:
+mode 1, a table and ``PROBABILITYWEIGHTING`` γ > 0). Mode 2 judges targets on expected
+amounts ``a_ij·(1 − p_i)``: ``held_csr`` is a copy of ``pu_feat_csr`` with the same
+structure and scaled ``.data``, read by ``compute_held``, ``apply_flip_to_held`` and the
+penalty delta. In every other case ``held_csr`` *is* ``pu_feat_csr``, so both are resolved
+at build time and no flip pays a check. ``pu_feat_csr`` / ``pu_feat_matrix`` stay raw for
+``ClumpState`` / ``SepState`` (a documented PROBMODE 2 limitation; the build warns when
+PROBMODE 2 meets active clumping or separation). The table is read through the shared
+helpers in ``solvers/utils.py`` (``_probmode``, ``_probability_vector``), as the evaluator,
+the MIP and the greedy heuristic do.
 """
 from __future__ import annotations
 
 import warnings
 from dataclasses import dataclass, field
@@ -38,11 +53,15 @@ from functools import cached_property
 
 import numpy as np
 from scipy.sparse import coo_matrix, csr_matrix
 
 from pymarxan.models.problem import ConservationProblem
-from pymarxan.solvers.utils import compute_cost_threshold_penalty
+from pymarxan.solvers.utils import (
+    _probability_vector,
+    _probmode,
+    compute_cost_threshold_penalty,
+)
 
 
 @dataclass(frozen=True)
 class ProblemCache:
     """Precomputed arrays derived from a ConservationProblem.
@@ -165,10 +184,26 @@ class ProblemCache:
     conn_in_start: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int32))
     conn_in_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
     conn_in_val: np.ndarray = field(default_factory=lambda: np.zeros(0))
     conn_out_row: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
     conn_out_rowsum: np.ndarray = field(default_factory=lambda: np.zeros(0))
+    # PROBMODE 1 risk premium: prob_cost[i] = γ·p_i·c_i; read only when prob_active (mode 1,
+    # a probability table and γ > 0). ``costs`` stays the base cost.
+    prob_cost: np.ndarray = field(default_factory=lambda: np.zeros(0))
+    prob_active: bool = False
+    # Amounts behind held / shortfall: under PROBMODE 2 with a table a copy of pu_feat_csr
+    # with identical indptr / indices and .data scaled by (1 − p_i) (explicit zeros kept);
+    # otherwise pu_feat_csr itself. Never None after construction (__post_init__). When
+    # deriving a cache with dataclasses.replace(..., pu_feat_csr=X), pass held_csr=None so
+    # it is re-derived (an explicit held_csr is kept as given).
+    held_csr: csr_matrix = field(  # type: ignore[assignment]
+        default=None, compare=False, repr=False,
+    )
+
+    def __post_init__(self) -> None:
+        if self.held_csr is None:
+            object.__setattr__(self, "held_csr", self.pu_feat_csr)
 
     @cached_property
     def pu_feat_matrix(self) -> np.ndarray:
         """Dense (n_pu, n_feat) view — densified on first access (clumping / separation /
         probmode-3 / analysis). A plain problem never triggers this."""
@@ -274,12 +309,13 @@ class ProblemCache:
         misslevel = float(params.get("MISSLEVEL", 1.0))
         cost_thresh = float(params.get("COSTTHRESH", 0.0))
         thresh_pen1 = float(params.get("THRESHPEN1", 0.0))
         thresh_pen2 = float(params.get("THRESHPEN2", 0.0))
 
-        # PROBMODE 3 state (zero-cost when probmode != 3)
-        probmode = int(params.get("PROBMODE", 0))
+        # PROBMODE: default 1 with a probability table (the MIP's and greedy's rule), else 0.
+        # Mode 3 state below is zero-cost when probmode != 3.
+        probmode = _probmode(problem)
         prob_weight = float(params.get("PROBABILITYWEIGHTING", 1.0))
 
         # Per-cell probability of PRESENCE — read once into a dense matrix.
         # Marxan's PROB2D `prob` is the probability the feature is present at
         # the cell (clumping.cpp / probability.cpp ComputeP_AllPUsSelected_2D:
@@ -474,10 +510,22 @@ class ProblemCache:
             np.bincount(conn_out_row, weights=conn_out_val, minlength=n_pu)
             if conn_mode == 2
             else np.zeros(n_pu, dtype=np.float64)
         )
 
+        prob_cost, prob_active, held_csr = _build_probability(
+            problem, pu_id_to_idx, costs, pu_feat_csr, probmode, prob_weight,
+        )
+        if held_csr is not pu_feat_csr and (clumping_active or separation_active):
+            warnings.warn(
+                "PROBMODE 2 expected amounts are not applied to clumping/separation "
+                "features; SA optimises them on raw amounts while reports use expected "
+                "amounts.",
+                UserWarning,
+                stacklevel=2,
+            )
+
         return cls(
             n_pu=n_pu,
             n_feat=n_feat,
             costs=costs,
             statuses=statuses,
@@ -519,19 +567,24 @@ class ProblemCache:
             conn_in_start=conn_in_start,
             conn_in_idx=conn_in_idx,
             conn_in_val=conn_in_val,
             conn_out_row=conn_out_row,
             conn_out_rowsum=conn_out_rowsum,
+            prob_cost=prob_cost,
+            prob_active=prob_active,
+            held_csr=held_csr,
         )
 
     # ------------------------------------------------------------------
     # Compute methods
     # ------------------------------------------------------------------
 
     def compute_held(self, selected: np.ndarray) -> np.ndarray:
         """Compute the amount held per feature for the given selection.
 
+        Expected amounts ``a·(1 − p)`` under PROBMODE 2 with a table (``held_csr``).
+
         Parameters
         ----------
         selected : np.ndarray
             (n_pu,) boolean array indicating selected planning units.
 
@@ -539,22 +592,22 @@ class ProblemCache:
         -------
         np.ndarray
             (n_feat,) float64 — total amount held for each feature.
         """
         rows = np.flatnonzero(selected)
-        result: np.ndarray = np.asarray(self.pu_feat_csr[rows].sum(axis=0)).ravel()
+        result: np.ndarray = np.asarray(self.held_csr[rows].sum(axis=0)).ravel()
         return result
 
     def apply_flip_to_held(self, held: np.ndarray, idx: int, sign: float) -> None:
         """In-place held update for flipping PU ``idx``: ``held[cols] += sign*amounts``.
 
         Bit-identical to ``held += sign * pu_feat_matrix[idx]`` (only omits ``+ 0.0`` on
         features absent from the PU), and O(nnz in the row). Relies on the CSR having
         unique column indices per row (guaranteed by ``build_pu_feature_csr``'s
         ``sum_duplicates()`` — numpy fancy in-place add is last-wins, not accumulate).
         """
-        csr = self.pu_feat_csr
+        csr = self.held_csr
         s, e = csr.indptr[idx], csr.indptr[idx + 1]
         cols = csr.indices[s:e]
         held[cols] += sign * csr.data[s:e]
 
     def _compute_zscore_penalty(
@@ -629,10 +682,14 @@ class ProblemCache:
         shortfalls = np.maximum(0.0, effective_targets - held)
         penalty = float(np.dot(self._det_spf, shortfalls))
 
         obj = total_cost + blm * boundary + penalty
 
+        # PROBMODE 1 risk premium (separate from cost; COSTTHRESH sees base cost)
+        if self.prob_active:
+            obj += float(np.sum(self.prob_cost[selected]))
+
         # Cost threshold penalty
         if self.cost_thresh > 0:
             obj += compute_cost_threshold_penalty(
                 total_cost, self.cost_thresh, self.thresh_pen1, self.thresh_pen2
             )
@@ -729,23 +786,29 @@ class ProblemCache:
         # (round-3 H14 — same mask as compute_full_objective).
         # Only the PU's nonzero features change shortfall (absent features contribute 0),
         # so restrict to the CSR row's columns → O(nnz_row). Integer-exact vs the dense
         # full-width dot; ≤ a few ULP for arbitrary floats (the retained terms regroup in a
         # non-associative np.dot — same float-sum regime as compute_held).
-        s = self.pu_feat_csr.indptr[idx]
-        e = self.pu_feat_csr.indptr[idx + 1]
-        cols = self.pu_feat_csr.indices[s:e]
-        amts = self.pu_feat_csr.data[s:e]
+        # held_csr: expected amounts under PROBMODE 2, else pu_feat_csr (resolved at build).
+        hc = self.held_csr
+        s = hc.indptr[idx]
+        e = hc.indptr[idx + 1]
+        cols = hc.indices[s:e]
+        amts = hc.data[s:e]
         eff = self.feat_targets[cols] * self.misslevel
         old_shortfalls = np.maximum(0.0, eff - held[cols])
         new_shortfalls = np.maximum(0.0, eff - (held[cols] + sign * amts))
         penalty_delta = float(
             np.dot(self._det_spf[cols], new_shortfalls - old_shortfalls)
         )
 
         delta = cost_delta + blm * boundary_delta + penalty_delta
 
+        # --- PROBMODE 1 risk premium delta (one bool check when off) ---
+        if self.prob_active:
+            delta += sign * self.prob_cost[idx]
+
         # --- Cost threshold delta ---
         if self.cost_thresh > 0:
             new_cost = total_cost + cost_delta
             old_ct_penalty = compute_cost_threshold_penalty(
                 total_cost,
@@ -885,10 +948,40 @@ class ProblemCache:
         # Correct.
         
         return total
 
 
+def _build_probability(
+    problem: ConservationProblem,
+    pu_id_to_idx: dict[int, int],
+    costs: np.ndarray,
+    pu_feat_csr: csr_matrix,
+    probmode: int,
+    prob_weight: float,
+) -> tuple[np.ndarray, bool, csr_matrix]:
+    """``(prob_cost, prob_active, held_csr)`` for PROBMODE 1 / 2 (see the module docstring).
+
+    ``p`` comes from the shared ``_probability_vector`` (missing PU → 0, unknown ids
+    skipped, last row wins, non-finite values raise ``ValueError``).
+    """
+    n_pu = len(costs)
+    prob_cost = np.zeros(n_pu, dtype=np.float64)
+    if problem.probability is None or probmode not in (1, 2):
+        return prob_cost, False, pu_feat_csr
+    if probmode == 1 and prob_weight <= 0.0:
+        return prob_cost, False, pu_feat_csr
+    p = _probability_vector(problem, pu_id_to_idx, n_pu)
+    if probmode == 1:
+        return prob_weight * p * costs, True, pu_feat_csr
+    # Mode 2: copy and scale .data row by row, so held_csr stays structurally identical to
+    # pu_feat_csr (same indptr / indices, a p = 1 row keeps explicit zeros) — unlike
+    # diags(1 − p) @ csr, which drops those zeros and changes the per-row nnz.
+    held_csr = pu_feat_csr.copy()
+    held_csr.data = held_csr.data * np.repeat(1.0 - p, np.diff(held_csr.indptr))
+    return prob_cost, False, held_csr
+
+
 _ConnArrays = tuple[
     int, float, np.ndarray,
     np.ndarray, np.ndarray, np.ndarray,
     np.ndarray, np.ndarray, np.ndarray,
 ]
```

Modify `src/pymarxan/solvers/clumping.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/clumping.py b/src/pymarxan/solvers/clumping.py
index 1cf5ec0..70e5eaf 100644
--- a/src/pymarxan/solvers/clumping.py
+++ b/src/pymarxan/solvers/clumping.py
@@ -450,12 +450,13 @@ class ClumpState:
                 ids, occs = _build_components_for_feature(cache, selected, j)
                 comp_id_per_pu.append(ids)
                 comp_occ.append(occs)
                 held_eff[j] = _held_eff_for_feature(cache, j, occs)
             else:
-                # Non-clumping feature: held_eff is raw amount sum so callers
-                # can treat held_effective() as a single source of truth.
+                # Non-clumping feature: held_eff is the raw pu_feat_matrix sum
+                # (equals cache.compute_held except under PROBMODE 2, where
+                # compute_held holds expected amounts).
                 comp_id_per_pu.append(np.full(cache.n_pu, -1, dtype=np.int32))
                 comp_occ.append({})
                 held_eff[j] = float(
                     cache.pu_feat_matrix[selected, j].sum()
                 )
@@ -522,11 +523,11 @@ class ClumpState:
         for j in range(cache.n_feat):
             amt = float(cache.pu_feat_matrix[idx, j])
             target2 = float(cache.feat_target2[j])
             if target2 <= 0:
                 # Non-clumping: keep held_eff as the raw selected-amount sum
-                # so callers can compare against compute_held().
+                # (equals compute_held() except under PROBMODE 2).
                 if amt != 0:
                     self.held_eff[j] += sign * amt
                 continue
             if amt == 0:
                 continue  # flip doesn't touch feature j's clumps
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_cache_probmode.py tests/pymarxan/solvers/test_sa_probmode.py -q`
Expected: 73 passed

- [ ] **Step 5: Gate**

Run: `~/.local/bin/ruff check src tests && /home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/ -q -m "not bench" -p no:cacheprovider`
Expected: ruff and mypy clean; the full suite passes (prototype: 3709 passed, 101 skipped HiGHS cases which run in CI, 21–22 bench deselected); no golden JSON regenerated; no existing test changed except the Stage 3 probmode2 records via STAGE3_PROBMODE2_FIXED.

- [ ] **Step 6: Commit**

```bash
git add src/pymarxan/solvers/cache.py src/pymarxan/solvers/clumping.py tests/pymarxan/solvers/test_cache_probmode.py tests/pymarxan/solvers/test_sa_probmode.py
git commit -m "feat(cache): simulated annealing and iterative improvement optimise PROBMODE 1 and 2

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AZEiop9qqxVCrMHKqSDYP8"
```

---

### Task 3: Representation and gap analysis use PROBMODE 2 expected amounts

**Files:**
- Create: `src/pymarxan/analysis/_totals.py`
- Modify: `src/pymarxan/analysis/gap_analysis.py`
- Modify: `src/pymarxan/analysis/representation.py`
- Create: `tests/pymarxan/analysis/test_probmode_totals.py`

**Interfaces:**
- Consumes: Task 1's mode-2 `compute_held`.
- Produces: new `pymarxan/analysis/_totals.py` with `expected_feature_totals(problem) -> dict[int, float]` (spec §4a); `compute_representation` binary path via `compute_held`, denominator `expected_feature_totals`; `gap_analysis` protected amount via `compute_held` of the locked-in mask, total via `expected_feature_totals`.

- [ ] **Step 1: Write the failing tests**

Create `tests/pymarxan/analysis/test_probmode_totals.py`:

```python
"""Representation and gap analysis under PROBMODE 2 use expected amounts (v0.45 §4).

Numerators come from ``compute_held`` (``Σ a·(1 − p)``), denominators from
``expected_feature_totals`` (the same over every planning unit). Every other mode keeps
``feature_amounts()`` exactly, including ``pu_vs_features`` rows for unknown PU ids.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pymarxan.analysis._totals import expected_feature_totals
from pymarxan.analysis.gap_analysis import compute_gap_analysis
from pymarxan.analysis.representation import compute_representation
from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.utils import build_solution

DATA = Path(__file__).parent.parent.parent / "data"
PROB = pd.DataFrame({"pu": [2, 4, 5, 6], "probability": [0.4, 0.5, 0.25, 1.0]})
KEEP = {1: 1.0, 2: 0.6, 3: 1.0, 4: 0.5, 5: 0.75, 6: 0.0}  # 1 − p per PU


def _simple(
    probability: pd.DataFrame | None = PROB, mode: int | None = 2, **params: Any,
) -> ConservationProblem:
    p = load_project(DATA / "simple")
    merged = {**p.parameters, **params}
    if mode is not None:
        merged["PROBMODE"] = mode
    return p.copy_with(probability=probability, parameters=merged)


def _expected(p: ConservationProblem, pus: set[int]) -> dict[int, float]:
    out = {int(f): 0.0 for f in p.features["id"]}
    for pu, sp, amt in zip(
        p.pu_vs_features["pu"], p.pu_vs_features["species"], p.pu_vs_features["amount"],
    ):
        if int(pu) in pus:
            out[int(sp)] += float(amt) * KEEP[int(pu)]
    return out


ALL = {1, 2, 3, 4, 5, 6}


class TestExpectedTotals:
    def test_mode_2_expected_totals(self) -> None:
        p = _simple()
        got = expected_feature_totals(p)
        want = _expected(p, ALL)
        assert got.keys() == want.keys()
        for fid in want:
            assert got[fid] == pytest.approx(want[fid])

    @pytest.mark.parametrize(("probability", "mode"), [(PROB, 0), (PROB, 1), (None, 2)])
    def test_other_modes_are_feature_amounts(
        self, probability: pd.DataFrame | None, mode: int,
    ) -> None:
        p = _simple(probability, mode)
        assert expected_feature_totals(p) == {
            int(f): float(a) for f, a in p.feature_amounts().items()
        }


class TestRepresentationMode2:
    def test_expected_numerator_and_denominator(self) -> None:
        p = _simple()
        sel = np.array([True, True, False, True, False, True])
        r = compute_representation(p, build_solution(p, sel, 0.0), threshold=0.3)
        held = _expected(p, {1, 2, 4, 6})
        total = _expected(p, ALL)
        for fid in held:
            assert r.represented_amount[fid] == pytest.approx(held[fid])
            assert r.total_amount[fid] == pytest.approx(total[fid])
            assert r.pct_represented[fid] == pytest.approx(100.0 * held[fid] / total[fid])

    def test_selecting_every_unit_is_100_percent(self) -> None:
        p = _simple()
        r = compute_representation(p, build_solution(p, np.ones(6, dtype=bool), 0.0))
        for fid in r.feature_ids:
            assert r.pct_represented[fid] == pytest.approx(100.0)

    def test_binary_and_proportion_plans_agree(self) -> None:
        p = _simple()
        bits = [True, False, True, True, True, False]
        rb = compute_representation(p, build_solution(p, np.array(bits), 0.0))
        rf = compute_representation(p, build_solution(p, np.array(bits, dtype=float), 0.0))
        for fid in rb.feature_ids:
            assert rb.represented_amount[fid] == pytest.approx(rf.represented_amount[fid])
            assert rb.pct_represented[fid] == pytest.approx(rf.pct_represented[fid])

    def test_feature_without_rows_reports_zero(self) -> None:
        p = _simple()
        ghost = pd.DataFrame({"id": [4], "name": ["ghost"], "target": [1.0], "spf": [1.0]})
        p = p.copy_with(features=pd.concat([p.features, ghost], ignore_index=True))
        r = compute_representation(p, build_solution(p, np.ones(6, dtype=bool), 0.0))
        df = r.to_dataframe()
        assert df.loc[df["feature_id"] == 4, "total_amount"].item() == 0.0
        assert df.loc[df["feature_id"] == 4, "pct_represented"].item() == 0.0


def _locked(p: ConservationProblem, ids: list[int]) -> ConservationProblem:
    pu = p.planning_units.copy()
    pu.loc[pu["id"].isin(ids), "status"] = 2
    return p.copy_with(planning_units=pu)


class TestGapMode2:
    def test_expected_protected_and_total(self) -> None:
        p = _locked(_simple(), [2, 4, 5])
        g = compute_gap_analysis(p)
        held = _expected(p, {2, 4, 5})
        total = _expected(p, ALL)
        targets = dict(zip(p.features["id"].astype(int), p.features["target"].astype(float)))
        df = g.to_dataframe().set_index("feature_id")
        for fid in held:
            assert g.protected_amount[fid] == pytest.approx(held[fid])
            assert g.total_amount[fid] == pytest.approx(total[fid])
            assert g.gap[fid] == pytest.approx(max(0.0, targets[fid] - held[fid]))
            assert df.loc[fid, "percent_protected"] == pytest.approx(
                100.0 * held[fid] / total[fid]
            )


def _orphan(p: ConservationProblem) -> ConservationProblem:
    """An extra pu_vs_features row for PU 99, which is not a planning unit."""
    row = pd.DataFrame({"species": [1], "pu": [99], "amount": [40.0]})
    return p.copy_with(pu_vs_features=pd.concat([p.pu_vs_features, row], ignore_index=True))


@pytest.mark.parametrize(("probability", "mode"), [(None, None), (PROB, 0), (PROB, 1)])
class TestOtherModesUnchanged:
    def test_representation_with_orphan_row(
        self, probability: pd.DataFrame | None, mode: int | None,
    ) -> None:
        p = _orphan(_simple(probability, mode))
        sel = np.array([True, True, False, True, False, True])
        r = compute_representation(p, build_solution(p, sel, 0.0))
        # f1: raw total 60 + orphan 40 = 100; selected 10 + 15 + 12 + 10 = 47
        assert r.total_amount[1] == 100.0
        assert r.represented_amount[1] == 47.0
        assert r.pct_represented[1] == pytest.approx(47.0)
        assert r.total_amount[2] == 45.0
        assert r.represented_amount[2] == 25.0

    def test_gap_with_orphan_row(
        self, probability: pd.DataFrame | None, mode: int | None,
    ) -> None:
        p = _locked(_orphan(_simple(probability, mode)), [2, 4])
        df = compute_gap_analysis(p).to_dataframe().set_index("feature_id")
        assert df.loc[1, "total_amount"] == 100.0
        assert df.loc[1, "protected_amount"] == 27.0
        assert df.loc[1, "percent_protected"] == pytest.approx(27.0)
        assert df.loc[3, "protected_amount"] == 10.0


def test_representation_int_selection_reads_as_bool() -> None:
    """An int 0/1 ``selected`` follows the _coerce_selected contract (not fancy indexing)."""
    from pymarxan.solvers.base import Solution

    p = _simple(None, None)
    sel = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    ref = build_solution(p, sel.astype(bool), 0.0)
    as_int = Solution(
        selected=sel, cost=ref.cost, boundary=ref.boundary, objective=ref.objective,
        targets_met=ref.targets_met,
    )
    r = compute_representation(p, as_int)
    assert r.represented_amount == {1: 37.0, 2: 20.0, 3: 18.0}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/analysis/test_probmode_totals.py -q`
Expected: FAIL — collection error (`ModuleNotFoundError: pymarxan.analysis._totals`); with only the helper present, 4 failed, 12 passed (representation numerator/denominator, binary vs proportion, feature with no rows, gap protected/total); the orphan-row pins and the int-selection test (its coercion lands in Task 1) pass before and after

- [ ] **Step 3: Implement**

Create `src/pymarxan/analysis/_totals.py`:

```python
"""Feature totals shared by representation and gap analysis (v0.45 PROBMODE consistency)."""
from __future__ import annotations

import numpy as np

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.utils import _probmode2_active, compute_held


def expected_feature_totals(problem: ConservationProblem) -> dict[int, float]:
    """Feature id -> total amount: the denominator of representation and gap percentages.

    Under PROBMODE 2 with a probability table: the expected total ``Σ_i a_ij·(1 − p_i)``
    over every planning unit — ``compute_held`` of the all-selected plan, so selecting every
    unit is 100 % and the totals agree with ``build_solution``. Rows whose PU is not a
    planning unit are excluded and every feature id is a key (0.0 when it has no rows).

    Outside PROBMODE 2 these are the **raw** totals, ``feature_amounts()`` exactly as before
    (every ``pu_vs_features`` row, features with rows only).
    """
    if _probmode2_active(problem):
        held = compute_held(problem, np.ones(len(problem.planning_units), dtype=bool))
        return {int(f): float(held.get(int(f), 0.0)) for f in problem.features["id"]}
    return {int(f): float(a) for f, a in problem.feature_amounts().items()}
```

Modify `src/pymarxan/analysis/gap_analysis.py` (apply this diff):

```diff
diff --git a/src/pymarxan/analysis/gap_analysis.py b/src/pymarxan/analysis/gap_analysis.py
index b6102b2..b968de1 100644
--- a/src/pymarxan/analysis/gap_analysis.py
+++ b/src/pymarxan/analysis/gap_analysis.py
@@ -7,11 +7,13 @@ from __future__ import annotations
 
 from dataclasses import dataclass
 
 import pandas as pd
 
+from pymarxan.analysis._totals import expected_feature_totals
 from pymarxan.models.problem import ConservationProblem
+from pymarxan.solvers.utils import compute_held
 
 
 @dataclass
 class GapResult:
     """Results of a gap analysis."""
@@ -45,36 +47,28 @@ class GapResult:
         return pd.DataFrame(rows)
 
 
 def compute_gap_analysis(problem: ConservationProblem) -> GapResult:
     """Compute protection gap for each feature."""
-    protected_pu_ids = set(
-        problem.planning_units.loc[
-            problem.planning_units["status"] == 2, "id"
-        ]
-    )
-
     feature_ids = problem.features["id"].tolist()
     feature_names = problem.features["name"].tolist()
 
     targets: dict[int, float] = dict(
         zip(problem.features["id"].astype(int), problem.features["target"].astype(float))
     )
 
-    total_amount = problem.feature_amounts()
-
-    # Vectorized: filter pu_vs_features to protected PUs and groupby species
-    puvspr = problem.pu_vs_features
-    mask = puvspr["pu"].isin(protected_pu_ids)
-    protected_amount: dict[int, float]
-    if mask.any():
-        prot_totals = puvspr.loc[mask].groupby("species")["amount"].sum()
-        protected_amount = {
-            fid: float(prot_totals.get(fid, 0.0)) for fid in feature_ids
-        }
-    else:
-        protected_amount = {fid: 0.0 for fid in feature_ids}
+    # percent_protected denominator: feature_amounts(), or the expected total under
+    # PROBMODE 2 — the same helper as representation (v0.45).
+    total_amount = expected_feature_totals(problem)
+
+    # Protected amount: compute_held of the locked-in mask (expected amounts under
+    # PROBMODE 2), so it agrees with build_solution.
+    locked_in = problem.planning_units["status"].to_numpy() == 2
+    held = compute_held(problem, locked_in)
+    protected_amount: dict[int, float] = {
+        fid: float(held.get(int(fid), 0.0)) for fid in feature_ids
+    }
 
     # Apply MISSLEVEL so gap analysis agrees with solver / build_solution /
     # export_summary — previously gap was measured against the raw target.
     misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
 
```

Modify `src/pymarxan/analysis/representation.py` (apply this diff):

```diff
diff --git a/src/pymarxan/analysis/representation.py b/src/pymarxan/analysis/representation.py
index 2233e2b..6fbff0a 100644
--- a/src/pymarxan/analysis/representation.py
+++ b/src/pymarxan/analysis/representation.py
@@ -14,12 +14,14 @@ from __future__ import annotations
 
 from dataclasses import dataclass
 
 import pandas as pd
 
+from pymarxan.analysis._totals import expected_feature_totals
 from pymarxan.models.problem import ConservationProblem
 from pymarxan.solvers.base import Solution
+from pymarxan.solvers.utils import compute_held
 
 
 @dataclass
 class RepresentationResult:
     """Per-feature representation of a solution against a threshold."""
@@ -66,42 +68,31 @@ def compute_representation(
         problem: The conservation problem.
         solution: The solution (its selected units define the reserve).
         threshold: Fraction of each feature's total amount that must be
             represented to "meet" it (default 0.30 for the 30x30 goal).
 
+    Under PROBMODE 2 (with a probability table) amounts are expected amounts
+    ``amount·(1 − p)``, so ``pct_represented`` is the share of the feature's expected
+    (persistence-adjusted) total, not of its raw area or amount.
+
     Returns:
         A :class:`RepresentationResult` with per-feature totals,
         represented amounts, percentages, threshold pass/fail, and a
         summary of how many features clear the threshold.
     """
     if not 0.0 <= threshold <= 1.0:
         raise ValueError(f"threshold must be in [0, 1], got {threshold}")
 
     feature_ids = [int(f) for f in problem.features["id"]]
     feature_names = [str(n) for n in problem.features["name"]]
-    total_amount = {int(f): float(a) for f, a in problem.feature_amounts().items()}
-
-    selected_ids = {
-        int(pid)
-        for pid, sel in zip(problem.planning_units["id"], solution.selected)
-        if sel
-    }
-    puvspr = problem.pu_vs_features
-    mask = puvspr["pu"].isin(selected_ids)
-    if solution.is_fractional:
-        # Proportion decisions: represented amount Σ r·x (Stage 4 §6.1).
-        from pymarxan.solvers.utils import compute_held
-
-        held = compute_held(problem, solution.selected)
-        represented_amount = {fid: float(held.get(fid, 0.0)) for fid in feature_ids}
-    elif mask.any():
-        rep_totals = puvspr.loc[mask].groupby("species")["amount"].sum()
-        represented_amount = {
-            fid: float(rep_totals.get(fid, 0.0)) for fid in feature_ids
-        }
-    else:
-        represented_amount = {fid: 0.0 for fid in feature_ids}
+    # Denominator: feature_amounts(), or the expected total under PROBMODE 2 (v0.45).
+    total_amount = expected_feature_totals(problem)
+
+    # Represented amount Σ r·x (proportions, Stage 4 §6.1) or Σ r over selected units;
+    # expected amounts Σ r·(1 − p) under PROBMODE 2 — the amounts build_solution judges.
+    held = compute_held(problem, solution.selected)
+    represented_amount = {fid: float(held.get(fid, 0.0)) for fid in feature_ids}
 
     pct_represented: dict[int, float] = {}
     meets_threshold: dict[int, bool] = {}
     for fid in feature_ids:
         total = total_amount.get(fid, 0.0)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/analysis/test_probmode_totals.py -q`
Expected: 16 passed

- [ ] **Step 5: Gate**

Run: `~/.local/bin/ruff check src tests && /home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/ -q -m "not bench" -p no:cacheprovider`
Expected: ruff and mypy clean; the full suite passes (prototype: 3725 passed, 101 skipped HiGHS cases which run in CI, 21–22 bench deselected); no golden JSON regenerated; no existing test changed except the Stage 3 probmode2 records via STAGE3_PROBMODE2_FIXED.

- [ ] **Step 6: Commit**

```bash
git add src/pymarxan/analysis/_totals.py src/pymarxan/analysis/gap_analysis.py src/pymarxan/analysis/representation.py tests/pymarxan/analysis/test_probmode_totals.py
git commit -m "fix(analysis): representation and gap analysis use PROBMODE 2 expected amounts

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AZEiop9qqxVCrMHKqSDYP8"
```

---

### Task 4: Bench, changelog, manual, UI label, roadmap

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `docs/USER_MANUAL.md`
- Modify: `docs/plans/2026-10-11-prioritizr-parity-roadmap.md`
- Modify: `src/pymarxan/multiobjective.py`
- Modify: `src/pymarxan/objectives/base.py`
- Modify: `src/pymarxan_shiny/modules/probability/probability_config.py`
- Modify: `tests/benchmarks/bench_sa.py`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: `tests/benchmarks/bench_sa.py::TestSAProbmodeDeltaCost` (comparative, `bench` marker); CHANGELOG `[Unreleased]` Changed + Fixed; `docs/USER_MANUAL.md` PROBMODE rows as pymarxan extensions with the Marxan 1D/2D mapping and PROBMODE 2 limitations; Shiny `probability_config` Mode 2 label and tooltip; `MultiObjectiveMIP` and `objectives/base.py` docstrings; roadmap (Probability 1D stage, PROBMODE line done, raw-amount backlog).

- [ ] **Step 1: Write the comparative bench**

Modify `tests/benchmarks/bench_sa.py` (apply this diff):

```diff
diff --git a/tests/benchmarks/bench_sa.py b/tests/benchmarks/bench_sa.py
index 50e65ad..10d3d9b 100644
--- a/tests/benchmarks/bench_sa.py
+++ b/tests/benchmarks/bench_sa.py
@@ -2,10 +2,11 @@
 from __future__ import annotations
 
 import time
 
 import numpy as np
+import pandas as pd
 import pytest
 
 from pymarxan.models.problem import ConservationProblem
 from pymarxan.solvers.base import SolverConfig
 from pymarxan.solvers.cache import ProblemCache
@@ -134,5 +135,55 @@ class TestSAConnectivityDeltaCost:
             f"asymmetric {t_asym * per_call:.2f} us (x{t_asym / t_base:.3f})"
         )
         # Off must be free: <= 5% over the no-table baseline, with a 2 us/call floor
         # absorbing timer noise on fast machines.
         assert t_off <= max(1.05 * t_base, t_base + 2e-6 * self.N_CALLS)
+
+
+class TestSAProbmodeDeltaCost:
+    """v0.45.0: PROBMODE 1 / 2 in the SA delta. Comparative only (machine-relative)."""
+
+    N_CALLS = 20_000
+
+    def _problems(self) -> dict[str, ConservationProblem]:
+        """No table; table + PROBMODE 0; mode 1 γ = 0; mode 2; mode 1 γ = 1."""
+        base = make_problem(n_pu=1000, n_feat=50, density=0.3, seed=42)
+        rng = np.random.default_rng(7)
+        prob = pd.DataFrame({
+            "pu": base.planning_units["id"].to_numpy(),
+            "probability": rng.uniform(0.0, 0.6, len(base.planning_units)),
+        })
+
+        def variant(**params: float) -> ConservationProblem:
+            # copy_with is shallow: give each variant its own parameters dict.
+            return base.copy_with(probability=prob, parameters={**base.parameters, **params})
+
+        return {
+            "base": base,
+            "mode0": variant(PROBMODE=0),
+            "mode1_off": variant(PROBMODE=1, PROBABILITYWEIGHTING=0.0),
+            "mode2": variant(PROBMODE=2),
+            "mode1_on": variant(PROBMODE=1, PROBABILITYWEIGHTING=1.0),
+        }
+
+    def test_probmode_costs_nothing_when_off_or_prescaled(self) -> None:
+        problems = self._problems()
+        assert not ProblemCache.from_problem(problems["mode1_off"]).prob_active
+        assert ProblemCache.from_problem(problems["mode1_on"]).prob_active
+        c2 = ProblemCache.from_problem(problems["mode2"])
+        assert c2.held_csr is not c2.pu_feat_csr
+        t = {name: _delta_loop_time(p, self.N_CALLS) for name, p in problems.items()}
+        per_call = 1e6 / self.N_CALLS
+        t_base = t["base"]
+        print(
+            "\ndelta per call: "
+            + ", ".join(
+                f"{name} {v * per_call:.2f} us (x{v / t_base:.3f})" for name, v in t.items()
+            )
+        )
+        # Off / pre-scaled must be free. Effective bound: max(1.05x, +2 us/call) over the
+        # no-table baseline — the 2 us/call timer-noise floor dominates at ~17 us/call
+        # (about +12%), as in the v0.44.0 connectivity bench. mode1_on is print-only.
+        budget = max(1.05 * t_base, t_base + 2e-6 * self.N_CALLS)
+        assert t["mode0"] <= budget
+        assert t["mode1_off"] <= budget
+        assert t["mode2"] <= budget
```

- [ ] **Step 2: Write the docs, UI text and docstrings**

Modify `CHANGELOG.md` (apply this diff):

```diff
diff --git a/CHANGELOG.md b/CHANGELOG.md
index 1057bcb..9d097a4 100644
--- a/CHANGELOG.md
+++ b/CHANGELOG.md
@@ -5,10 +5,61 @@ All notable changes to **pymarxan** are documented in this file.
 The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
 and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
 
 ## [Unreleased]
 
+### Changed
+- **Simulated annealing and iterative improvement now optimise PROBMODE 1 and 2, so their
+  plans change on every problem with a probability table in mode 1 or 2 — including projects
+  whose `prob.dat` is picked up automatically with no `PROBMODE`, which are now mode 1.** The
+  `ProblemCache` behind SA and iterative improvement read `PROBMODE` with a default of 0 and
+  had neither the mode 1 risk premium nor the mode 2 expected amounts, so it searched a
+  different objective from the one the MIP optimised and `build_solution` reported. It now
+  uses the MIP's semantics: a table with no `PROBMODE` is mode 1; mode 1 adds
+  `PROBABILITYWEIGHTING · p · cost` per selected unit to its full objective and its per-flip
+  delta (cost and COSTTHRESH stay on base cost; off when the weighting is 0 or below); mode 2
+  holds expected amounts `amount · (1 − p)`, pre-scaled once at build so a flip costs the same.
+  SA's tracked best objective and history now equal the reported `Solution.objective`. The MIP
+  and the greedy heuristics are unchanged (they now read the table through the same shared
+  helpers; their models and plans do not move). Under PROBMODE 2 with active clumping or
+  separation, SA/II still optimise those features on raw amounts and now warn
+  (`UserWarning`) when the cache is built.
+- **PROBMODE 2 reports use expected amounts.** Targets met, shortfall, penalty and held amounts
+  (`build_solution`, `compute_held`, `write_mvbest`, the summary CSV and the Shiny summary and
+  target-met views), representation and gap analysis now count `Σ amount · (1 − p)`, the
+  amounts the MIP's target rows use; representation and gap percentages use the expected total
+  over all units as their denominator. Every solver that reports through `build_solution` —
+  including the external Marxan binary and the Zonation solver — now has its PROBMODE 2
+  targets judged on expected amounts. Other modes, and mode 2 without a table, are unchanged.
+- `validate()` reports a probability table that lists a planning unit more than once, and
+  non-finite (NaN or infinite) probabilities — a blank `prob.dat` cell reads as NaN — with
+  their planning unit ids. Callers asserting `validate() == []` will see the new messages.
+- A non-finite probability now raises `ValueError` (naming the planning unit ids) in every
+  path that reads the table — the evaluator, `ProblemCache` (SA / II), the MIP and the greedy
+  heuristics — instead of NaN objectives, a PuLP error, or a silent substitution of 0.
+- `compute_held` applies the selection contract of `build_solution`: an int 0/1 array is read
+  as bool, other ints raise `ValueError`.
+- Docs and UI: PROBMODE 1 and 2 are documented as pymarxan extensions; Marxan's Probability 1D
+  is not yet implemented, and PROBMODE 3 is described as based on Marxan's Probability 2D with
+  its two known differences (zero variance treated as met; `ptarget` rather than `ptarget2d`).
+  The probability is documented as a probability of loss throughout (the glossary and two Shiny
+  tooltips said persistence). The Shiny labels read "Risk premium on cost (expected cost lost)",
+  "Expected (persistence-adjusted) amounts" and "Z-score probability penalty", and the manual
+  explains that a target met on expected amounts has only about even odds of being met.
+
+### Fixed
+- SA and iterative improvement ignored the PROBMODE 1 risk premium (their tracked objective
+  was 44.0 where 69.2 was reported on `simple` with probabilities `linspace(0.1, 0.6)`,
+  weighting 2, selection {2, 4, 6}), defaulted to mode 0 with a probability table where the
+  MIP, the greedy heuristics and the evaluator use mode 1, and optimised raw amounts under
+  PROBMODE 2.
+- PROBMODE 2 plans were judged on raw amounts: a MIP `max_features` plan meeting one target
+  under the expected amounts it optimised reported two.
+- A negative `PROBABILITYWEIGHTING` gave a negative risk premium in the reported objective;
+  the premium is now off for any weighting of 0 or below, as in the MIP and the greedy
+  heuristics.
+
 ## [0.44.0] — 2026-10-17
 
 ### Changed
 - **Simulated annealing and iterative improvement now optimise connectivity (the
   `CONNECTIVITY_WEIGHT` term; connectivity encoded in the boundary file under `BLM` was already
```

Modify `docs/USER_MANUAL.md` (apply this diff):

```diff
diff --git a/docs/USER_MANUAL.md b/docs/USER_MANUAL.md
index 10e9ebe..e3785bd 100644
--- a/docs/USER_MANUAL.md
+++ b/docs/USER_MANUAL.md
@@ -24,10 +24,11 @@ A comprehensive guide to using the pyMarxan Shiny application for systematic con
    - [Matrix Input](#matrix-input)
    - [Connectivity Metrics](#connectivity-metrics)
 8. [Configure](#configure)
    - [Solver Selection](#solver-selection)
    - [Solver Parameters](#solver-parameters)
+   - [Probability (PROBMODE)](#probability-probmode)
    - [Partial Selection](#partial-selection)
    - [Zone Configuration](#zone-configuration)
 9. [Calibrate](#calibrate)
    - [BLM Calibration](#blm-calibration)
    - [Target Sensitivity](#target-sensitivity)
@@ -378,10 +379,57 @@ pyMarxan supports ten solver algorithms:
 | 3 | Heuristic + iterative improvement |
 | 4 | Heuristic + SA (pick best) |
 | 5 | Heuristic + SA + improvement (highest quality) |
 | 6 | Iterative improvement only |
 
+### Probability (PROBMODE)
+
+A probability table (`prob.dat`: columns `pu`, `probability`) gives, for each planning unit,
+the probability that the unit (and the features in it) is **lost** over the planning horizon;
+persistence is 1 − probability. It is attached automatically when the project folder has one.
+`PROBMODE` chooses how it is used; with a table and no `PROBMODE`, mode 1 applies.
+
+| PROBMODE | Meaning | Origin |
+|----------|---------|--------|
+| 0 | Probabilities ignored | — |
+| 1 (default with a table) | Risk premium `PROBABILITYWEIGHTING × Σ probability × cost` over selected units, added to the objective (not to the reported cost; off when the weighting is 0 or below) — the expected cost of protecting units that are then lost; it does not depend on the features a unit holds, and a zero-cost unit carries no premium | pymarxan extension |
+| 2 | Expected amounts: every feature amount counts as `amount × (1 − probability)`, and targets, shortfalls, penalties, representation and gap analysis are judged on them | pymarxan extension |
+| 3 | Z-score probability penalty (a soft chance target, not a hard constraint) from a per-cell probability of **presence** (a `prob` column in `puvspr.dat`) and a per-feature `ptarget` in spec.dat (Marxan's column is `ptarget2d`) | Based on Marxan's Probability 2D, except that a feature with zero variance is treated as met (Marxan 3 treats it as failing: Z = 4) |
+
+Modes 1 and 2 have the same meaning in every solver: the MIP, simulated annealing and
+iterative improvement optimise them, the greedy heuristics use them in their scores (mode 1 as
+an effective cost, mode 2 as scaled amounts), and every reported value
+(`Solution.objective`, targets met, the summary and target-met tables) uses them. A unit
+missing from the table has probability 0; a unit listed twice uses its last row. `validate()`
+reports duplicates, values outside [0, 1] and non-finite values (a blank cell reads as NaN);
+every solver and the evaluator raise `ValueError` on a non-finite probability.
+
+Marxan's own **Probability 1D** (a `prob` column in `pu.dat`, a Z-score on the expected
+amount `Σ amount × (1 − p)` with variance `Σ amount² × p × (1 − p)` against `ptarget1d`) is
+**not yet implemented**; mode 2 uses the same expected amount but as a deterministic target,
+without the variance or the chance constraint. Meeting a target in expectation is not the same
+as meeting it reliably: when the expected amount just equals the target, the chance of actually
+retaining the target amount is only about 50% (higher if the reserve holds more, lower with few
+high-risk units). Mode 2 also ignores how the risk is spread — one unit at p = 0.5 and ten units
+at p = 0.05 with the same expected amount count the same. Use it as a risk-discounted
+representation target; for a stated confidence level use a chance formulation (Watts et al.
+2021). Until Probability 1D is implemented, it can be reproduced with PROBMODE 3 by writing
+`prob = 1 − p` (the unit's persistence) on every puvspr.dat row of that unit and setting
+`ptarget` (same mean and variance; zero-variance features differ as noted above).
+
+Under PROBMODE 2 some paths still use raw amounts: clumping occupancy (`target2`), separation
+occurrence and space-target weights (simulated annealing warns when PROBMODE 2 meets active
+clumping or separation: it optimises those features on raw amounts while reports use expected
+amounts); the MIP presolve reachability check; irreplaceability and Ferrier importance; and
+relative (`prop`) target resolution, which is measured against raw feature totals — so a
+relative target under mode 2 asks for a share of the raw total out of expected amounts, and is
+infeasible when `prop` exceeds `Σ_i a_ij(1 − p_i) / Σ_i a_ij` (one minus the amount-weighted
+mean loss probability).
+
+Reference: Watts, M., Klein, C. J., Tulloch, V., et al. (2021). Software for prioritizing conservation actions based on probabilistic information.
+*Conservation Biology*, 35(4), 1299–1308. https://doi.org/10.1111/cobi.13681
+
 ### Partial Selection
 
 By default the MIP selects whole planning units (`decisions="binary"`). Two other decision
 types let the plan take **part of a unit** — protecting or restoring a share of its area,
 or committing a share of management effort, e.g. restoration shares or land-use fractions.
@@ -652,20 +700,20 @@ Download results as CSV files:
 | **ITIMPTYPE** | Improvement strategy for iterative improvement (0–3) |
 | **RUNMODE** | Pipeline combination of heuristic, SA, and improvement algorithms (0–6) |
 | **Selection Frequency** | How often a PU appears in the solution across multiple runs (0–100%) |
 | **Irreplaceability** | How critical a PU is for meeting targets — PUs selected in nearly all solutions are highly irreplaceable |
 | **Connectivity** | Spatial connections between PUs — encourages selecting linked PUs to maintain ecological corridors |
-| **Probability** | Per-PU persistence probability — accounts for habitat loss risk in reserve design |
-| **PROBMODE** | Probability mode: 1 = risk premium on cost, 2 = persistence-adjusted feature amounts |
+| **Probability** | Per-PU probability that the unit (and the features in it) is **lost** over the planning horizon (prob.dat). Persistence is 1 − probability. PROBMODE 3 instead reads a per-(PU, feature) probability of **presence** from puvspr.dat |
+| **PROBMODE** | Probability mode (pymarxan extensions over `prob.dat`): 1 = risk premium in the objective (default with a probability table), 2 = expected (persistence-adjusted) amounts; 3 = Z-score probability penalty based on Marxan's Probability 2D. Marxan's Probability 1D is not yet implemented — see [Probability (PROBMODE)](#probability-probmode) |
 | **Zone** | A management category (e.g. no-take, buffer, sustainable use) in multi-zone planning |
 | **Zone Contribution** | How much a PU in a given zone contributes to a feature's target (0–1 multiplier) |
 | **Zone Boundary Cost** | Penalty for adjacent PUs being in different zone types |
 | **Constraint** | An additional rule beyond targets — contiguity, minimum neighbors, budget caps, etc. |
 | **Objective** | The MIP objective: `min_set` (default), `max_features`, `max_weighted_features`, `min_shortfall`, `min_largest_shortfall`, `min_penalties` (core only); the Shiny selector does not yet drive it |
 | **puvspr.dat** | Planning Unit vs. Species file — maps features to PUs with amounts |
 | **bound.dat** | Boundary definition file — pairwise boundary lengths between adjacent PUs |
-| **prob.dat** | Probability file — per-PU habitat persistence probabilities |
+| **prob.dat** | Probability file — per-PU probability of loss (pymarxan's `PROBNAME`; not a Marxan file) |
 
 ---
 
 ## Troubleshooting
 
```

Modify `docs/plans/2026-10-11-prioritizr-parity-roadmap.md` (apply this diff):

```diff
diff --git a/docs/plans/2026-10-11-prioritizr-parity-roadmap.md b/docs/plans/2026-10-11-prioritizr-parity-roadmap.md
index b2dc87c..1e51f25 100644
--- a/docs/plans/2026-10-11-prioritizr-parity-roadmap.md
+++ b/docs/plans/2026-10-11-prioritizr-parity-roadmap.md
@@ -220,14 +220,38 @@ Fold these into a stage when one touches the same code:
   ≤ 0 ignored). Goldens' `conn_*` objectives re-recorded in place.
 - ~~SA / heuristics do not optimise connectivity~~ — done in v0.44 for SA and iterative
   improvement (`ProblemCache` connectivity arrays in the full objective and the per-flip
   delta; `docs/plans/2026-10-17-sa-connectivity-design.md`). Greedy heuristics still only
   report it.
-- SA / iterative improvement ignore the PROBMODE 1 risk premium: the `ProblemCache` objective
-  has no PROBMODE 1 term, so SA's tracked objective disagrees with the reported one (cache 44.0
-  vs reported 69.2 on `simple` with PROBMODE 1, PROBABILITYWEIGHTING 2, probabilities
-  `linspace(0.1, 0.6)`, selection {2, 4, 6}).
+- ~~SA / iterative improvement ignore the PROBMODE 1 risk premium~~ — done in v0.45 with
+  PROBMODE 1/2 consistency across solvers (`docs/plans/2026-10-17-probmode-consistency-design.md`):
+  the cache defaults to mode 1 with a table, adds the premium, and holds expected amounts under
+  mode 2; the evaluator, representation and gap analysis report expected amounts.
+- **Next stage: Marxan Probability 1D** (a `prob` column in `pu.dat` = probability of loss).
+  Per feature j: `E_j = Σ_i a_ij(1 − p_i)x_i`, `V_j = Σ_i a_ij² p_i(1 − p_i)x_i`,
+  `Z_j = (T_j − E_j)/√V_j` (Z = 4 when V_j = 0), `P_j = 1 − Φ(Z_j)`; penalty
+  `PROBABILITYWEIGHTING · Σ_j [P_j < ptarget1d_j]·(ptarget1d_j − P_j)/ptarget1d_j`, features
+  with `ptarget1d ≤ 0` (default −1) excluded; added to, not replacing, the raw-amount target
+  penalty. Reference: Marxan 3 `probability.c` (`marxan3` branch). Marxan 4.0.x's
+  `utils::probZUT` computes `1 − (1/2)·(…)` with integer `1/2 = 0`, so it always returns 1 and
+  the 1D/2D probability penalty is inverted (full penalty when the expected amount exceeds the
+  target, none when it falls short): do not use a Marxan 4 binary as the parity oracle for
+  probability. Probability 1D is algebraically PROBMODE 3 with per-cell presence
+  `q_ij = 1 − p_i` (same mean and variance), so the stage may be a loader mapping onto the
+  existing PROBMODE 3 machinery. PROBMODE 1/2 stay pymarxan extensions.
+- PROBMODE 2 raw-amount paths (documented limitations, v0.45 §7): presolve P6 reachability;
+  `analysis/irreplaceability.py` and `analysis/ferrier_importance.py`; relative / `prop` target
+  resolution against raw totals; clumping occupancy, separation occurrence and space weights
+  (`ClumpState` / `SepState` / `SpaceState` read raw `pu_feat_csr`).
+- PROBMODE 2 + clumping / separation in SA / II (v0.45 warns): derive the clump and separation
+  amounts from `held_csr` (`pu_feat_matrix`, `feat_uses_pu`, `compute_baseline_penalty` and the
+  from-scratch evaluators on expected amounts) so SA optimises what it reports.
+- PROBMODE 3 vs Marxan's Probability 2D: (a) a feature with zero variance is treated as met
+  (`solvers/probability.py`), where Marxan 3 sets Z = 4 and counts it as failing — decide and
+  fix the module docstring, which contradicts itself; (b) read Marxan's `ptarget2d` (and
+  `ptarget1d`) spec.dat columns as aliases of `ptarget` (today they are warned about and
+  ignored, so a real Marxan 2D project's PROBMODE 3 is inert).
 - One shared connectivity normaliser for the evaluator, the MIP and `ProblemCache` (today each
   re-implements the rules). Fold in two MIP gaps: a connectivity row naming an unknown planning
   unit raises `KeyError` (the evaluator and SA skip it), and negative asymmetric values leave
   the MIP unbounded / raise `IndexError` (SA and the evaluator accept them).
 - String PU ids split SA's reported and tracked objectives (pre-existing, not connectivity
```

Modify `src/pymarxan/multiobjective.py` (apply this diff):

```diff
diff --git a/src/pymarxan/multiobjective.py b/src/pymarxan/multiobjective.py
index aa6395c..98438eb 100644
--- a/src/pymarxan/multiobjective.py
+++ b/src/pymarxan/multiobjective.py
@@ -448,12 +448,13 @@ class MultiObjectiveMIP:
     ``objective`` / ``targets_met`` are filled as ``MIPSolver`` fills them (Marxan-style);
     ``metadata["sol_status"]`` is PuLP's solution status (1 optimal, 2 feasible); under the
     Gurobi backend PuLP's ``GUROBI_CMD`` reports no solution status, so a time-limited
     incumbent may show 1 rather than 2.
     ``metadata["objective_values"]`` holds each objective's value at the plan, from the
-    model's own expressions (so ``"min_cost"`` can differ from ``Solution.objective``, and
-    under PROBMODE 2 met counts use the scaled amounts). ``config.num_solutions`` is
+    model's own expressions (so ``"min_cost"`` can differ from ``Solution.objective``);
+    under PROBMODE 2 met counts use expected amounts ``amount·(1 − p)``, as
+    ``Solution.targets_met`` does since v0.45. ``config.num_solutions`` is
     ignored. ``MIP_TIME_LIMIT`` applies per stage and row, so a hierarchical sweep takes up
     to rows × n × T; with ``MIP_GAP`` > 0 or a time limit a stage's v is an incumbent's
     value. Stage 1 (or a weighted-sum row) with no solution raises ``RuntimeError``; a later
     stage with no solution returns the previous stage's plan with a ``UserWarning``.
 
```

Modify `src/pymarxan/objectives/base.py` (apply this diff):

```diff
diff --git a/src/pymarxan/objectives/base.py b/src/pymarxan/objectives/base.py
index ab5444f..0064cc3 100644
--- a/src/pymarxan/objectives/base.py
+++ b/src/pymarxan/objectives/base.py
@@ -37,12 +37,15 @@ class Objective(ABC):
         """Compute the base objective score (lower-is-better).
 
         Args:
             problem: The conservation problem instance.
             selected: Boolean array of selected planning units.
-            effective_amounts: PU-feature amount matrix (may be adjusted
-                for PROBMODE=2).
+            effective_amounts: Dense (n_pu, n_feat) PU-feature amount
+                matrix. Under PROBMODE 2 with a probability table pass the
+                expected amounts ``amount·(1 − p)``, e.g.
+                ``cache.held_csr.toarray()`` — the amounts ``build_solution``
+                judges targets on.
             pu_index: Mapping from PU ID to array index.
 
         Returns:
             Base objective value. MaxCoverage/MaxUtility negate their
             scores for the minimization convention.
```

Modify `src/pymarxan_shiny/modules/probability/probability_config.py` (apply this diff):

```diff
diff --git a/src/pymarxan_shiny/modules/probability/probability_config.py b/src/pymarxan_shiny/modules/probability/probability_config.py
index 22feec0..3475393 100644
--- a/src/pymarxan_shiny/modules/probability/probability_config.py
+++ b/src/pymarxan_shiny/modules/probability/probability_config.py
@@ -28,12 +28,12 @@ def probability_config_ui():
                     ui.input_file(
                         "prob_file",
                         "Upload prob.dat",
                         accept=[".dat", ".csv", ".tsv", ".txt"],
                     ),
-                    "Upload a file with columns 'pu' (planning unit ID) and "
-                    "'probability' (value between 0 and 1). CSV, TSV, or "
+                    "Columns 'pu' and 'probability': the chance (0–1) that the "
+                    "planning unit is lost (not that it persists). CSV, TSV, or "
                     "Marxan .dat format.",
                 ),
                 ui.hr(),
                 ui.h5("Parameters"),
                 ui.tooltip(
@@ -43,32 +43,41 @@ def probability_config_ui():
                         min=0.0,
                         max=100.0,
                         value=1.0,
                         step=0.1,
                     ),
-                    "Weight applied to the probability risk premium in the "
-                    "objective function. Higher values penalise selection of "
-                    "planning units with low persistence probability.",
+                    "Weight on the PROBMODE 1 risk premium (probability of loss × "
+                    "cost) and on the PROBMODE 3 probability shortfall. Not used by "
+                    "PROBMODE 2.",
                 ),
                 ui.tooltip(
                     ui.input_radio_buttons(
                         "prob_mode",
                         "PROBMODE",
                         choices={
-                            "1": "Mode 1 — Expected value (risk premium)",
-                            "2": "Mode 2 — Probability threshold",
-                            "3": "Mode 3 — Z-score chance constraints (PROB2D)",
+                            "1": "Mode 1 — Risk premium on cost (expected cost lost)",
+                            "2": "Mode 2 — Expected (persistence-adjusted) amounts",
+                            "3": "Mode 3 — Z-score probability penalty (soft chance "
+                            "target, based on Marxan's Probability 2D)",
                         },
                         selected="1",
                     ),
-                    "Mode 1: adds a risk premium to cost based on probability × "
-                    "cost × weight. Mode 2: persistence-adjusted amounts. "
-                    "Mode 3: per-(PU, feature) Bernoulli probability via the "
+                    "Modes 1 and 2 are pymarxan extensions over per-PU "
+                    "probabilities (prob.dat); Marxan's Probability 1D is not yet "
+                    "implemented. Mode 1 (the default with a probability table): "
+                    "adds a risk premium probability × cost × weight to the "
+                    "objective. Mode 2: each amount counts as amount × (1 − "
+                    "probability), and targets are judged on these expected "
+                    "amounts by every solver and report. This is an average, not a "
+                    "guarantee: a target met on expected amounts has roughly even "
+                    "odds of being met. "
+                    "Mode 3: per-(PU, feature) probability of presence via the "
                     "optional `prob` column on puvspr.dat and a per-feature "
-                    "`ptarget` on spec.dat; SA / heuristic / iterative-improvement "
-                    "solve the chance constraint natively, MIP drops the term "
-                    "and reports the gap post-hoc.",
+                    "`ptarget` on spec.dat (Marxan's column is `ptarget2d`); a soft "
+                    "penalty on the probability shortfall, not a hard constraint. "
+                    "SA / heuristic / iterative-improvement optimise it, MIP drops "
+                    "the term and reports the gap post-hoc.",
                 ),
                 width=350,
             ),
             ui.output_text_verbatim("prob_preview"),
         ),
```

- [ ] **Step 3: Run the tests to verify they pass**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/benchmarks/bench_sa.py -m bench -k probmode -q -s`
Expected: passes; prints the ratios (prototype, 1000 PUs × 50 features, 20k delta calls, best of 5, two runs: (a) mode 0 ×0.99–×1.00, (b) mode 1 γ = 0 ×1.00, (c) mode 2 ×0.99–×1.00 — all asserted ≤ max(1.05×, +2 µs/call) — and (d) mode 1 γ = 1 ×1.00–×1.01, print-only). Timing is machine-relative: a failure on a loaded machine is re-run, not fixed by loosening the bound.

- [ ] **Step 4: Gate**

Run: `~/.local/bin/ruff check src tests && /home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/ -q -m "not bench" -p no:cacheprovider`
Expected: ruff and mypy clean; the full suite passes (prototype: 3725 passed, 101 skipped HiGHS cases which run in CI, 21–22 bench deselected); no golden JSON regenerated; no existing test changed except the Stage 3 probmode2 records via STAGE3_PROBMODE2_FIXED.

- [ ] **Step 5: Commit**

```bash
git add CHANGELOG.md docs/USER_MANUAL.md docs/plans/2026-10-11-prioritizr-parity-roadmap.md src/pymarxan/multiobjective.py src/pymarxan/objectives/base.py src/pymarxan_shiny/modules/probability/probability_config.py tests/benchmarks/bench_sa.py
git commit -m "docs(probmode): bench, changelog, manual, UI label, roadmap

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AZEiop9qqxVCrMHKqSDYP8"
```
