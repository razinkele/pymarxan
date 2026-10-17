# PROBMODE 1/2 Consistency Across Solvers (v0.45.0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every solver and every report agrees with the MIP on pymarxan's PROBMODE 1 (risk premium) and PROBMODE 2 (expected amounts `amount·(1 − p)`): simulated annealing and iterative improvement optimise them, and the evaluator, representation and gap analysis report them.

**Architecture:** The evaluator's single held-amount helper (`_compute_achieved`) scales by `(1 − p)` under PROBMODE 2, and the PROBMODE 1 premium follows the MIP's `γ <= 0` rule. `ProblemCache` reads `PROBMODE` with the MIP's default (1 when a probability table is present), adds a per-unit `prob_cost` term to its full objective and O(1) delta, and keeps a build-time scaled copy `held_csr` for held amounts, leaving the raw matrix for clumping and separation. Representation and gap analysis share one expected-total helper under PROBMODE 2.

**Tech Stack:** Python 3.11+, numpy, scipy.sparse, PuLP 3.3.2 + CBC, pytest.

**Spec:** `docs/plans/2026-10-17-probmode-consistency-design.md` (approved 2026-10-17 after three review-loop rounds and a fresh-eyes final review; §4a fixes every name, type and shape). Executors read it. The spec is the authority, and this plan argues from it.

**Provenance:** every code block below is extracted (`git show`) from a prototype built task by task in a scratch worktree, one commit per task, with the full gate green at every commit. Prototype commits: Task 1 `d562069`, Task 2 `c913729`, Task 3 `249b7a1`, Task 4 `2a92d0b` (base `0b9bc85`). Diffs apply with `git apply --whitespace=nowarn` on the previous task's tree. Every block was replayed that way on a fresh worktree from the base, and each task's tree matched the prototype commit. Editors that strip trailing whitespace from pasted context lines break a plain `git apply`. New files are given in full.

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

Where the spec was silent, these choices were made; each names its task.
- **Task 1, helpers:** `solvers/utils.py` gains `_probmode(problem)` (default 1 with a table) and `_probability_map(problem)` (last row wins), shared by the evaluator paths.
- **Task 1, golden fix:** the 4 `max_*` records enter `STAGE3_PROBMODE2_FIXED` with unchanged `targets_met` ({1:F, 2:F, 3:T}), because the §4a shape is (objective, targets_met) for every case; `_assert_same` gains an optional `keys` override rather than a second helper.
- **Task 2, cache build:** a module-level `_build_probability(...)` in the style of `_build_connectivity`; `prob_cost` is zeros of length n_pu when off.
- **Task 2, no dedicated iterative-improvement test:** II calls the same three cache methods (`compute_held`, `compute_delta_objective`, `apply_flip_to_held`) that the tests cover.
- **Task 2, test evidence:** the best-of-10 SA tests and the delta-exactness test already pass before Task 2 (pinned anyway as regression guards); Task 2's RED comes from the consistency, full = evaluator, brute-force and combination-identity tests. SA consistency for mode 2 goes red after Task 1 (the evaluator then reports expected amounts while the cache is still raw) — the spec's "mode 2 agrees today" describes the state before Task 1.
- **Task 3, every mode:** representation's binary path and gap's protected amount use `compute_held` in every mode, as spec §4 says; outside mode 2 the numbers are identical to before.
- **Task 4, roadmap:** the PROBMODE backlog line is struck through as "done in v0.45" (the file's v0.43/v0.44 convention) rather than deleted.
- **Task 4, manual wording:** the greedy heuristics *use* modes 1 and 2 in their scores; they do not optimise the objective.

---

### Task 1: Evaluator: PROBMODE 2 expected amounts, the MIP's weighting rule, duplicate check, Stage 3 golden fix

**Files:**
- Modify: `src/pymarxan/models/problem.py`
- Modify: `src/pymarxan/solvers/utils.py`
- Create: `tests/pymarxan/solvers/test_probmode_evaluator.py`
- Modify: `tests/pymarxan/solvers/test_stage3_golden_v040.py`

**Interfaces:**
- Consumes: nothing new.
- Produces (`solvers/utils.py`): `_probmode(problem) -> int` (PROBMODE, default 1 when a probability table is present), `_probability_map(problem) -> dict[int, float]` (last row wins); `_compute_achieved` scales by `(1 − p)` under mode 2 (binary and fractional), so `compute_held`, `check_targets`, shortfall, penalty and `build_solution` report expected amounts; `_compute_probmode1_risk_premium` off for `γ <= 0`. `ConservationProblem.validate()` reports `probability lists planning unit IDs more than once: [...]`. Test-side: `STAGE3_PROBMODE2_FIXED` in `test_stage3_golden_v040.py`.

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
```

Modify `tests/pymarxan/solvers/test_stage3_golden_v040.py` (apply this diff):

```diff
diff --git a/tests/pymarxan/solvers/test_stage3_golden_v040.py b/tests/pymarxan/solvers/test_stage3_golden_v040.py
index 6531154..d743333 100644
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
@@ -204,18 +208,52 @@ def _assert_same(got: dict[str, Any], want: dict[str, Any]) -> None:
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
Expected: FAIL — 16 failed, 123 passed — 8 evaluator tests (max_features met count 2 not 1; binarized feature 2 reported met; compute_held ×3; shortfall/penalty; γ = −1 negative premium; duplicate-PU validate) and the 8 Stage 3 probmode2 golden cases (got 54.0 / 41.0, want 59.1 / 55.4); the regression pins (min_set, proportion, no-table, other modes, γ = 0) pass

- [ ] **Step 3: Implement**

Modify `src/pymarxan/models/problem.py` (apply this diff):

```diff
diff --git a/src/pymarxan/models/problem.py b/src/pymarxan/models/problem.py
index 1797ec1..d2ba49d 100644
--- a/src/pymarxan/models/problem.py
+++ b/src/pymarxan/models/problem.py
@@ -275,10 +275,18 @@ class ConservationProblem:
                 if unknown_prob_pus:
                     errors.append(
                         f"probability references planning unit IDs not in "
                         f"planning_units: {sorted(unknown_prob_pus)}"
                     )
+                # A PU listed twice: every path uses the last row, silently.
+                prob_pu_col = self.probability["pu"]
+                dups = {int(u) for u in prob_pu_col[prob_pu_col.duplicated()]}
+                if dups:
+                    errors.append(
+                        "probability lists planning unit IDs more than once: "
+                        f"{sorted(dups)}"
+                    )
                 # Check probability range [0, 1]
                 prob_vals = self.probability["probability"]
                 if (prob_vals < 0.0).any() or (prob_vals > 1.0).any():
                     errors.append(
                         "probability values must be in range [0, 1]"
```

Modify `src/pymarxan/solvers/utils.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/utils.py b/src/pymarxan/solvers/utils.py
index af836aa..8299393 100644
--- a/src/pymarxan/solvers/utils.py
+++ b/src/pymarxan/solvers/utils.py
@@ -85,23 +85,43 @@ def _total_cost(costs: np.ndarray, selected: np.ndarray) -> float:
     if _is_fractional(selected):
         return float(np.dot(costs.astype(np.float64), selected))
     return float(np.sum(costs[selected]))
 
 
+def _probmode(problem: ConservationProblem) -> int:
+    """``PROBMODE``, defaulting to 1 (the MIP's and greedy's default; v0.45 §3)."""
+    return int(problem.parameters.get("PROBMODE", 1))
+
+
+def _probability_map(probability: pd.DataFrame) -> dict[int, float]:
+    """PU id -> probability from a ``prob.dat`` table; a PU listed twice: the last row wins."""
+    p_of: dict[int, float] = {}
+    for pid, val in zip(probability["pu"].values, probability["probability"].values):
+        p_of[int(pid)] = float(val)
+    return p_of
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
+    if problem.probability is not None and _probmode(problem) == 2:
+        p_of = _probability_map(problem.probability)
+        keep = 1.0 - pd.Series(pu_col).map(p_of).fillna(0.0).to_numpy(dtype=np.float64)
+        amount_col = np.asarray(amount_col, dtype=np.float64) * keep
 
     # Build boolean mask: True for rows whose PU is selected
     pu_indices = np.array(
         [pu_index.get(int(pid), -1) for pid in pu_col], dtype=np.intp
     )
@@ -141,10 +161,14 @@ def compute_held(
     selected: np.ndarray,
     pu_index: dict[int, int] | None = None,
 ) -> dict[int, float]:
     """Feature id -> amount held: Σ amount over selected PUs, or Σ amount·x for proportions.
 
+    Under PROBMODE 2 with a probability table the amounts are expected amounts
+    ``amount·(1 − p)`` (as the MIP's target rows), so targets met, shortfall, penalty and
+    every report built on this agree with the MIP.
+
     Public wrapper over ``_compute_achieved`` (review R7). ``pu_index`` maps PU id to its
     position in ``selected`` and defaults to ``planning_units`` order.
     """
     if pu_index is None:
         pu_index = {int(p): i for i, p in enumerate(problem.planning_units["id"])}
@@ -375,18 +399,19 @@ def compute_probability_penalty(
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
@@ -400,23 +425,19 @@ def compute_probability_penalty(
 
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
+    prob_map = _probability_map(problem.probability)
 
     costs = problem.planning_units["cost"].values
     pu_ids = problem.planning_units["id"].values
     fractional = _is_fractional(selected)
 
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_probmode_evaluator.py tests/pymarxan/solvers/test_stage3_golden_v040.py -q`
Expected: 139 passed

- [ ] **Step 5: Gate**

Run: `~/.local/bin/ruff check src tests && /home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/ -q -m "not bench" -p no:cacheprovider`
Expected: ruff and mypy clean; the full suite passes (prototype: 3622 passed, 101 skipped HiGHS cases which run in CI, 21–22 bench deselected); no golden JSON regenerated; no existing test changed except the Stage 3 probmode2 records via STAGE3_PROBMODE2_FIXED.

- [ ] **Step 6: Commit**

```bash
git add src/pymarxan/models/problem.py src/pymarxan/solvers/utils.py tests/pymarxan/solvers/test_probmode_evaluator.py tests/pymarxan/solvers/test_stage3_golden_v040.py
git commit -m "fix(probability): evaluator uses PROBMODE 2 expected amounts and the MIP's weighting rule

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AZEiop9qqxVCrMHKqSDYP8"
```

---

### Task 2: Cache: SA and iterative improvement optimise PROBMODE 1 and 2

**Files:**
- Modify: `src/pymarxan/solvers/cache.py`
- Create: `tests/pymarxan/solvers/test_cache_probmode.py`
- Create: `tests/pymarxan/solvers/test_sa_probmode.py`

**Interfaces:**
- Consumes: Task 1's `_probmode`, `_probability_map`, mode-2 `_compute_achieved` (the oracle for full = evaluator).
- Produces (`ProblemCache`, defaulted fields): `prob_cost: np.ndarray` (γ·p·c, zeros when off), `prob_active: bool`, `held_csr: csr_matrix` (`compare=False, repr=False`; scaled copy with identical `indptr` / `indices` under mode 2, otherwise `pu_feat_csr` itself; `__post_init__` fills it for a hand-built cache); module-level `_build_probability(...)`; `probmode` defaults to 1 with a table. `compute_held`, `apply_flip_to_held` and the penalty delta read `held_csr`; full objective and delta add the premium when `prob_active`.

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
    p = 1, PU 3 cost −4, PU 1 locked in, MISSLEVEL 0.8."""
    p = _simple()
    pu = p.planning_units.copy()
    pu.loc[pu["id"] == 3, "cost"] = -4.0
    pu.loc[pu["id"] == 1, "status"] = 2
    prob = pd.DataFrame({"pu": [2, 2, 4, 5, 6], "probability": [0.9, 0.2, 1.0, 0.3, 0.45]})
    params = {**p.parameters, "PROBABILITYWEIGHTING": gamma, "MISSLEVEL": 0.8}
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
    pytest.param(_with_clumping, id="clumping"),
    pytest.param(_with_separation, id="separation"),
]


class TestCombinations:
    """PROBMODE 2 with clumping / separation: their amounts stay raw (documented)."""

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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_cache_probmode.py tests/pymarxan/solvers/test_sa_probmode.py -q`
Expected: FAIL — 33 failed, 19 passed — missing `prob_active` / `held_csr`; full objective ≠ evaluator for modes 1, 2 and default; SA reported ≠ tracked for 1, 2 and default; the oracle and cross-solver brute-force checks; the SA identity with clumping and with separation. The best-of-10 SA tests and delta exactness already pass (regression guards)

- [ ] **Step 3: Implement**

Modify `src/pymarxan/solvers/cache.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/cache.py b/src/pymarxan/solvers/cache.py
index e8bf3c4..b8c9f8e 100644
--- a/src/pymarxan/solvers/cache.py
+++ b/src/pymarxan/solvers/cache.py
@@ -27,10 +27,22 @@ Connectivity
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
+``ClumpState`` / ``SepState`` (a documented PROBMODE 2 limitation).
 """
 from __future__ import annotations
 
 import warnings
 from dataclasses import dataclass, field
@@ -165,10 +177,24 @@ class ProblemCache:
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
+    # otherwise pu_feat_csr itself. Never None after construction (__post_init__).
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
@@ -274,12 +300,13 @@ class ProblemCache:
         misslevel = float(params.get("MISSLEVEL", 1.0))
         cost_thresh = float(params.get("COSTTHRESH", 0.0))
         thresh_pen1 = float(params.get("THRESHPEN1", 0.0))
         thresh_pen2 = float(params.get("THRESHPEN2", 0.0))
 
-        # PROBMODE 3 state (zero-cost when probmode != 3)
-        probmode = int(params.get("PROBMODE", 0))
+        # PROBMODE: default 1 with a probability table (the MIP's and greedy's rule), else 0.
+        # Mode 3 state below is zero-cost when probmode != 3.
+        probmode = int(params.get("PROBMODE", 1 if problem.probability is not None else 0))
         prob_weight = float(params.get("PROBABILITYWEIGHTING", 1.0))
 
         # Per-cell probability of PRESENCE — read once into a dense matrix.
         # Marxan's PROB2D `prob` is the probability the feature is present at
         # the cell (clumping.cpp / probability.cpp ComputeP_AllPUsSelected_2D:
@@ -474,10 +501,14 @@ class ProblemCache:
             np.bincount(conn_out_row, weights=conn_out_val, minlength=n_pu)
             if conn_mode == 2
             else np.zeros(n_pu, dtype=np.float64)
         )
 
+        prob_cost, prob_active, held_csr = _build_probability(
+            problem, pu_id_to_idx, costs, pu_feat_csr, probmode, prob_weight,
+        )
+
         return cls(
             n_pu=n_pu,
             n_feat=n_feat,
             costs=costs,
             statuses=statuses,
@@ -519,19 +550,24 @@ class ProblemCache:
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
 
@@ -539,22 +575,22 @@ class ProblemCache:
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
@@ -629,10 +665,14 @@ class ProblemCache:
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
@@ -729,23 +769,29 @@ class ProblemCache:
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
@@ -885,10 +931,46 @@ class ProblemCache:
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
+    ``p`` is read from ``problem.probability`` in row order: PU ids that are not planning
+    units are skipped, a PU listed twice keeps its last row, a missing PU has p = 0. Values
+    are used as given (``validate()`` reports out-of-range ones).
+    """
+    n_pu = len(costs)
+    prob_cost = np.zeros(n_pu, dtype=np.float64)
+    if problem.probability is None or probmode not in (1, 2):
+        return prob_cost, False, pu_feat_csr
+    p = np.zeros(n_pu, dtype=np.float64)
+    for pid, val in zip(
+        problem.probability["pu"].values, problem.probability["probability"].values,
+    ):
+        k = pu_id_to_idx.get(int(pid))
+        if k is not None:
+            p[k] = float(val)
+    if probmode == 1:
+        if prob_weight <= 0.0:
+            return prob_cost, False, pu_feat_csr
+        return prob_weight * p * costs, True, pu_feat_csr
+    # Mode 2: scale .data row by row — NOT diags(1 − p) @ csr, which drops explicit zeros
+    # and would misalign indices with pu_feat_csr.
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

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_cache_probmode.py tests/pymarxan/solvers/test_sa_probmode.py -q`
Expected: 52 passed

- [ ] **Step 5: Gate**

Run: `~/.local/bin/ruff check src tests && /home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/ -q -m "not bench" -p no:cacheprovider`
Expected: ruff and mypy clean; the full suite passes (prototype: 3674 passed, 101 skipped HiGHS cases which run in CI, 21–22 bench deselected); no golden JSON regenerated; no existing test changed except the Stage 3 probmode2 records via STAGE3_PROBMODE2_FIXED.

- [ ] **Step 6: Commit**

```bash
git add src/pymarxan/solvers/cache.py tests/pymarxan/solvers/test_cache_probmode.py tests/pymarxan/solvers/test_sa_probmode.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/analysis/test_probmode_totals.py -q`
Expected: FAIL — collection error (`ModuleNotFoundError: pymarxan.analysis._totals`); with only the helper present, 4 failed, 11 passed (representation numerator/denominator, binary vs proportion, feature with no rows, gap protected/total); the orphan-row pins pass before and after

- [ ] **Step 3: Implement**

Create `src/pymarxan/analysis/_totals.py`:

```python
"""Feature totals shared by representation and gap analysis (v0.45 PROBMODE consistency)."""
from __future__ import annotations

import numpy as np

from pymarxan.models.problem import ConservationProblem


def expected_feature_totals(problem: ConservationProblem) -> dict[int, float]:
    """Feature id -> total amount: the denominator of representation and gap percentages.

    Under PROBMODE 2 with a probability table it is the expected total ``Σ_i a_ij·(1 − p_i)``
    over every planning unit — ``compute_held`` of the all-selected plan, so selecting every
    unit is 100 % and the totals agree with ``build_solution``. Rows whose PU is not a
    planning unit are excluded and every feature id is a key (0.0 when it has no rows).
    Otherwise ``feature_amounts()`` exactly as before (every row, features with rows only).
    """
    from pymarxan.solvers.utils import compute_held

    mode = int(problem.parameters.get("PROBMODE", 1))
    if problem.probability is not None and mode == 2:
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
index 2233e2b..6593d66 100644
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
@@ -76,32 +78,17 @@ def compute_representation(
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
Expected: 15 passed

- [ ] **Step 5: Gate**

Run: `~/.local/bin/ruff check src tests && /home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/ -q -m "not bench" -p no:cacheprovider`
Expected: ruff and mypy clean; the full suite passes (prototype: 3689 passed, 101 skipped HiGHS cases which run in CI, 21–22 bench deselected); no golden JSON regenerated; no existing test changed except the Stage 3 probmode2 records via STAGE3_PROBMODE2_FIXED.

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
index 50e65ad..389059c 100644
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
@@ -134,5 +135,54 @@ class TestSAConnectivityDeltaCost:
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
+        # Off / pre-scaled must be free: <= 5% over the no-table baseline, with a 2 us/call
+        # floor absorbing timer noise on fast machines. mode1_on is print-only.
+        budget = max(1.05 * t_base, t_base + 2e-6 * self.N_CALLS)
+        assert t["mode0"] <= budget
+        assert t["mode1_off"] <= budget
+        assert t["mode2"] <= budget
```

- [ ] **Step 2: Write the docs, UI text and docstrings**

Modify `CHANGELOG.md` (apply this diff):

```diff
diff --git a/CHANGELOG.md b/CHANGELOG.md
index 1057bcb..d8eb1ae 100644
--- a/CHANGELOG.md
+++ b/CHANGELOG.md
@@ -5,10 +5,46 @@ All notable changes to **pymarxan** are documented in this file.
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
+  and the greedy heuristics are unchanged.
+- **PROBMODE 2 reports use expected amounts.** Targets met, shortfall, penalty and held amounts
+  (`build_solution`, `compute_held`, `write_mvbest`, the summary CSV and the Shiny summary and
+  target-met views), representation and gap analysis now count `Σ amount · (1 − p)`, the
+  amounts the MIP's target rows use; representation and gap percentages use the expected total
+  over all units as their denominator. Every solver that reports through `build_solution` —
+  including the external Marxan binary and the Zonation solver — now has its PROBMODE 2
+  targets judged on expected amounts. Other modes, and mode 2 without a table, are unchanged.
+- `validate()` reports a probability table that lists a planning unit more than once.
+- Docs: PROBMODE 1 and 2 are documented as pymarxan extensions; Marxan's Probability 1D is not
+  yet implemented. The Shiny Mode 2 label reads "Expected (persistence-adjusted) amounts".
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
index 10e9ebe..2d9ca25 100644
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
@@ -378,10 +379,40 @@ pyMarxan supports ten solver algorithms:
 | 3 | Heuristic + iterative improvement |
 | 4 | Heuristic + SA (pick best) |
 | 5 | Heuristic + SA + improvement (highest quality) |
 | 6 | Iterative improvement only |
 
+### Probability (PROBMODE)
+
+A probability table (`prob.dat`: columns `pu`, `probability`, the chance that a planning unit
+is lost) is attached automatically when the project folder has one. `PROBMODE` chooses how it
+is used; with a table and no `PROBMODE`, mode 1 applies.
+
+| PROBMODE | Meaning | Origin |
+|----------|---------|--------|
+| 0 | Probabilities ignored | — |
+| 1 (default with a table) | Risk premium `PROBABILITYWEIGHTING × Σ probability × cost` over selected units, added to the objective (not to the reported cost; off when the weighting is 0 or below) | pymarxan extension |
+| 2 | Expected amounts: every feature amount counts as `amount × (1 − probability)`, and targets, shortfalls, penalties, representation and gap analysis are judged on them | pymarxan extension |
+| 3 | Z-score chance constraints from a per-cell `prob` column in `puvspr.dat` and a per-feature `ptarget` | Marxan's Probability 2D |
+
+Modes 1 and 2 have the same meaning in every solver: the MIP, simulated annealing and
+iterative improvement optimise them, the greedy heuristics use them in their scores (mode 1 as
+an effective cost, mode 2 as scaled amounts), and every reported value
+(`Solution.objective`, targets met, the summary and target-met tables) uses them. A unit
+missing from the table has probability 0; a unit listed twice uses its last row (`validate()`
+reports both duplicates and values outside [0, 1]).
+
+Marxan's own **Probability 1D** (a `prob` column in `pu.dat`, a Z-score on the expected
+amount `Σ amount × (1 − p)` with variance `Σ amount² × p × (1 − p)` against `ptarget1d`) is
+**not yet implemented**; mode 2 uses the same expected amount but as a deterministic target,
+without the variance or the chance constraint.
+
+Under PROBMODE 2 some paths still use raw amounts: clumping occupancy (`target2`), separation
+occurrence and space-target weights; the MIP presolve reachability check; irreplaceability and
+Ferrier importance; and relative (`prop`) target resolution, which is measured against raw
+feature totals.
+
 ### Partial Selection
 
 By default the MIP selects whole planning units (`decisions="binary"`). Two other decision
 types let the plan take **part of a unit** — protecting or restoring a share of its area,
 or committing a share of management effort, e.g. restoration shares or land-use fractions.
@@ -653,11 +684,11 @@ Download results as CSV files:
 | **RUNMODE** | Pipeline combination of heuristic, SA, and improvement algorithms (0–6) |
 | **Selection Frequency** | How often a PU appears in the solution across multiple runs (0–100%) |
 | **Irreplaceability** | How critical a PU is for meeting targets — PUs selected in nearly all solutions are highly irreplaceable |
 | **Connectivity** | Spatial connections between PUs — encourages selecting linked PUs to maintain ecological corridors |
 | **Probability** | Per-PU persistence probability — accounts for habitat loss risk in reserve design |
-| **PROBMODE** | Probability mode: 1 = risk premium on cost, 2 = persistence-adjusted feature amounts |
+| **PROBMODE** | Probability mode (pymarxan extensions over `prob.dat`): 1 = risk premium in the objective (default with a probability table), 2 = expected (persistence-adjusted) amounts; 3 = Marxan's Probability 2D chance constraints. Marxan's Probability 1D is not yet implemented — see [Probability (PROBMODE)](#probability-probmode) |
 | **Zone** | A management category (e.g. no-take, buffer, sustainable use) in multi-zone planning |
 | **Zone Contribution** | How much a PU in a given zone contributes to a feature's target (0–1 multiplier) |
 | **Zone Boundary Cost** | Penalty for adjacent PUs being in different zone types |
 | **Constraint** | An additional rule beyond targets — contiguity, minimum neighbors, budget caps, etc. |
 | **Objective** | The MIP objective: `min_set` (default), `max_features`, `max_weighted_features`, `min_shortfall`, `min_largest_shortfall`, `min_penalties` (core only); the Shiny selector does not yet drive it |
```

Modify `docs/plans/2026-10-11-prioritizr-parity-roadmap.md` (apply this diff):

```diff
diff --git a/docs/plans/2026-10-11-prioritizr-parity-roadmap.md b/docs/plans/2026-10-11-prioritizr-parity-roadmap.md
index b2dc87c..2525328 100644
--- a/docs/plans/2026-10-11-prioritizr-parity-roadmap.md
+++ b/docs/plans/2026-10-11-prioritizr-parity-roadmap.md
@@ -220,14 +220,21 @@ Fold these into a stage when one touches the same code:
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
+- **Next stage: Marxan Probability 1D** (a `prob` column in `pu.dat`; per-feature expected
+  `Σ a·(1 − p)`, variance `Σ a²·p·(1 − p)`, Z-score against `ptarget1d`, penalty
+  `PROBABILITYWEIGHTING · Σ shortfall`). PROBMODE 1/2 stay pymarxan extensions.
+- PROBMODE 2 raw-amount paths (documented limitations, v0.45 §7): presolve P6 reachability;
+  `analysis/irreplaceability.py` and `analysis/ferrier_importance.py`; relative / `prop` target
+  resolution against raw totals; clumping occupancy, separation occurrence and space weights
+  (`ClumpState` / `SepState` / `SpaceState` read raw `pu_feat_csr`).
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
index ab5444f..5625d9e 100644
--- a/src/pymarxan/objectives/base.py
+++ b/src/pymarxan/objectives/base.py
@@ -37,12 +37,14 @@ class Objective(ABC):
         """Compute the base objective score (lower-is-better).
 
         Args:
             problem: The conservation problem instance.
             selected: Boolean array of selected planning units.
-            effective_amounts: PU-feature amount matrix (may be adjusted
-                for PROBMODE=2).
+            effective_amounts: PU-feature amount matrix. Under PROBMODE 2
+                with a probability table pass expected amounts
+                ``amount·(1 − p)`` (``ProblemCache.held_csr``), the amounts
+                ``build_solution`` judges targets on.
             pu_index: Mapping from PU ID to array index.
 
         Returns:
             Base objective value. MaxCoverage/MaxUtility negate their
             scores for the minimization convention.
```

Modify `src/pymarxan_shiny/modules/probability/probability_config.py` (apply this diff):

```diff
diff --git a/src/pymarxan_shiny/modules/probability/probability_config.py b/src/pymarxan_shiny/modules/probability/probability_config.py
index 22feec0..e3035f2 100644
--- a/src/pymarxan_shiny/modules/probability/probability_config.py
+++ b/src/pymarxan_shiny/modules/probability/probability_config.py
@@ -53,17 +53,22 @@ def probability_config_ui():
                     ui.input_radio_buttons(
                         "prob_mode",
                         "PROBMODE",
                         choices={
                             "1": "Mode 1 — Expected value (risk premium)",
-                            "2": "Mode 2 — Probability threshold",
+                            "2": "Mode 2 — Expected (persistence-adjusted) amounts",
                             "3": "Mode 3 — Z-score chance constraints (PROB2D)",
                         },
                         selected="1",
                     ),
-                    "Mode 1: adds a risk premium to cost based on probability × "
-                    "cost × weight. Mode 2: persistence-adjusted amounts. "
+                    "Modes 1 and 2 are pymarxan extensions over per-PU "
+                    "probabilities (prob.dat); Marxan's Probability 1D is not yet "
+                    "implemented. Mode 1 (the default with a probability table): "
+                    "adds a risk premium probability × cost × weight to the "
+                    "objective. Mode 2: each amount counts as amount × (1 − "
+                    "probability), and targets are judged on these expected "
+                    "amounts by every solver and report. "
                     "Mode 3: per-(PU, feature) Bernoulli probability via the "
                     "optional `prob` column on puvspr.dat and a per-feature "
                     "`ptarget` on spec.dat; SA / heuristic / iterative-improvement "
                     "solve the chance constraint natively, MIP drops the term "
                     "and reports the gap post-hoc.",
```

- [ ] **Step 3: Run the tests to verify they pass**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/benchmarks/bench_sa.py -m bench -k probmode -q -s`
Expected: passes; prints the ratios (prototype, 1000 PUs × 50 features, 20k delta calls, best of 5, two runs: (a) mode 0 ×0.96–×1.00, (b) mode 1 γ = 0 ×0.97–×1.00, (c) mode 2 ×0.97–×0.99 — all asserted ≤ max(1.05×, +2 µs/call) — and (d) mode 1 γ = 1 ×0.97–×0.98, print-only). Timing is machine-relative: a failure on a loaded machine is re-run, not fixed by loosening the bound.

- [ ] **Step 4: Gate**

Run: `~/.local/bin/ruff check src tests && /home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/ -q -m "not bench" -p no:cacheprovider`
Expected: ruff and mypy clean; the full suite passes (prototype: 3689 passed, 101 skipped HiGHS cases which run in CI, 21–22 bench deselected); no golden JSON regenerated; no existing test changed except the Stage 3 probmode2 records via STAGE3_PROBMODE2_FIXED.

- [ ] **Step 5: Commit**

```bash
git add CHANGELOG.md docs/USER_MANUAL.md docs/plans/2026-10-11-prioritizr-parity-roadmap.md src/pymarxan/multiobjective.py src/pymarxan/objectives/base.py src/pymarxan_shiny/modules/probability/probability_config.py tests/benchmarks/bench_sa.py
git commit -m "docs(probmode): bench, changelog, manual, UI label, roadmap

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AZEiop9qqxVCrMHKqSDYP8"
```
