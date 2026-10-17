# Stage 4 (v0.42.0) Proportion and Semicontinuous Decisions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A planning unit can be partly selected (`MIPSolver(decisions="proportion"|"semicontinuous", upper=..., lower=...)`). Every number downstream of the solver is consistent with the fractions, binary results do not change, and tools that only make sense for 0/1 plans refuse fractional plans with a clear error. Nothing rounds silently.

**Architecture:** `Solution.selected` becomes `float64` for non-binary runs and stays `bool` otherwise. **The dtype is the discriminator**, so every binary code path is literally untouched and the binary goldens pin that. The MIP gets continuous `x` (plus a binary on/off `s` per PU for the true semicontinuous floor), a rejection gate at the top of `_build_base`, and `_extract_selected` with upward-only snapping. The evaluators in `solvers/utils.py` become fraction-linear with a target tolerance `1e-6 + 1e-7·|R|` (new `solvers/_tolerance.py`). Readers of `selected` fall into one of three groups: fraction-aware, guarded by `require_binary` / `FractionalSolutionError`, or internal / count-only. A completeness test keeps that classification exhaustive.

**Tech Stack:** Python 3.11+, PuLP 3.3.2 + CBC (HiGHS is not installed locally; its parametrized cases skip and run in CI), numpy, pandas, Shiny for Python; pytest.

**Spec:** `docs/plans/2026-10-17-stage4-proportion-decisions-design.md` (approved 2026-10-17 after 4 review-loop rounds). Executors read it. The spec is the authority, and this plan argues from it.

**Provenance:** every code block below is extracted (`git show`) from a prototype built task by task in a scratch worktree, one commit per task. It ran there under the commands shown, with the full gate green at the last commit. Prototype commits: Task 1 `14418d4`, Task 2 `0c9210c`, Task 3 `c8cfe78`, Task 4 `92bff83`, Task 5 `c313133`, Task 6 `816dd70`, Task 7 `4893192`, Task 8 `d395815`, Task 9 `0070744`, Task 10 `2c0fe48` (base `8b133f4`). Diffs apply with `git apply --whitespace=nowarn` (or `--3way`) on the previous task's tree. Editors that strip trailing whitespace from pasted context lines break a plain `git apply`. New files are given in full.

## Global Constraints

- Python 3.11+ floor; `from __future__ import annotations` as the first import in every new file; full type hints. mypy runs as `mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports`.
- ruff rules E, F, I, UP; line length 99. **Run `ruff check` on the test files too at every task handoff.**
- Tests run ONLY under `/opt/micromamba/envs/shiny/bin/pytest`, never `.venv`. Full gate: `PATH="/opt/micromamba/envs/shiny/bin:$HOME/.local/bin:$PWD/.venv/bin:$PATH" make check`. In a worktree without `.venv`, use `PYTHONPATH=src:.` with the shiny pytest, `~/.local/bin/ruff`, and the main checkout's `.venv/bin/mypy`.
- PuLP 3.3.2 emits `DeprecationWarning`s. Any `simplefilter("error")` must ignore `DeprecationWarning`.
- **Binary results do not change** (spec §1, §3). These goldens pass after every task without being regenerated:
  - `tests/data/golden/stage4_v041.json` (Task 1);
  - `stage3_v040.json`;
  - `stage1_v038.json`;
  - the zones v037 golden.

  "Byte-identical" binary model means identical `canonical_model` form (`tests/pymarxan/lp_canon.py`: an order-independent dump of rows, objective and variable bounds). The one intended binary change is spec §6.5: the Shiny summary table now applies MISSLEVEL.
- The `tests/data/simple` anchor holds: binary `MIPSolver` min_set at BLM 0 gives cost 35.0.
- Constants are verbatim from the spec:
  - `TARGET_ABS_TOL = 1e-6`, `TARGET_REL_TOL = 1e-7`;
  - `ZERO_TOL = 1e-9`, `SNAP_TOL = 1e-6`;
  - `decisions ∈ ("binary", "proportion", "semicontinuous")`;
  - binary requires `upper == 1.0` and `lower == 0.0`;
  - proportion requires `0 < upper ≤ 1` and `lower == 0`;
  - semicontinuous requires `0 < lower ≤ upper ≤ 1`.
- `src/pymarxan/zones/mip_solver.py` is not touched. `ZoneMIPSolver` is out of scope (spec §2).
- No Stage 4 test file imports another test file. Shared test helpers live in `tests/pymarxan/fractional_fixtures.py`, next to `lp_canon.py`.
- Seed every random test; pin results with `mip_backend="cbc"` unless the test is parametrized over installed backends.
- Commit trailer on every commit:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01AZEiop9qqxVCrMHKqSDYP8
  ```

## Review Focus

These are inputs a user will plausibly hit that the spec's acceptance list doesn't name. Each is pinned by a test in its owning task.
1. **BLM calibration with a proportion solver**: `calibrate_blm` / `pareto_frontier` must work unchanged, return fractional-typed solutions, and give a boundary that never increases as BLM grows. → `test_calibrate_blm_with_proportion_solver` (Task 8, `tests/pymarxan/calibration/test_blm_fractional.py`).
2. **Locks under the semicontinuous floor**: with `lower < upper < 1`, a locked-in PU sits at exactly `upper` and a locked-out PU at 0, not merely somewhere in the band. → `test_semicontinuous_locks` (Task 4).
3. **A PU PuLP never writes** (zero cost, no `pu_vs_features` rows, value `None`) must read 0.0 in a float64 selection, not crash or turn the dtype to object. → `test_proportion_unwritten_pu_reads_zero` (Task 4).
4. **Written fractions survive a round trip**: run and best files re-read with pandas match x to 1e-9 relative, and binary solutions still write ints. → `test_run_best_values_round_trip` (Task 7).
5. **The Shiny comparison map given a fractional plan** shows an explanation and doesn't raise out of a render function. This test found a real bug while prototyping. → `test_comparison_summary_explains_fractional` (Task 9).

## Rulings made while prototyping

Where the spec was silent, these choices were made; each names its task.
- **Tolerance module (Task 3).** The constants live in the new stdlib-only `solvers/_tolerance.py`, so `_mip_objectives.py` stays "pulp + stdlib only". `solvers.utils` re-exports `TARGET_ABS_TOL`, `TARGET_REL_TOL` and `target_tol`.
- **Shared conflict helper (Tasks 3, 4).** `utils._nonbinary_conflicts(problem) -> list[str]` (clumping / separation / space reasons) feeds both `build_solution`'s fractional `ValueError` and the MIP gate. `_total_cost` keeps the literal `np.sum(costs[selected])` for bool input.
- **PROBMODE default (Task 3).** The fractional PROBMODE 3 error uses `compute_probability_penalty`'s own default: PROBMODE absent means 1, so there is no error.
- **`binarize_solution` (Task 3)** validates the threshold first, including for binary input. It accepts finite non-bool numbers in (0, 1] and rejects NaN. It is re-exported from `pymarxan.solvers`.
- **Validation order (Task 4).** A non-zero `lower` is checked before the binary `upper` check, so `binary, lower=0.5` names `lower`. Bounds must be finite non-bool `numbers.Real`.
- **One gate message (Task 4).** `_reject_nonbinary` collects every cause into one `ValueError` ("decisions='proportion' cannot be combined with <causes>. Use decisions='binary'."). PROBMODE is read with default 0, as the existing gate does. The negative-connectivity check needs `CONNECTIVITY_WEIGHT > 0` and a connectivity table.
- **Binary construction kept literal (Task 4).** Binary `x` and symmetric `z` keep `cat="Binary"` at construction, so the canonical model is identical. The semicontinuous `s` implied rule `_on_rule` is registered right after its rows.
- **PuLP trap (Task 4).** Setting `v.cat = pulp.LpBinary` on an existing variable does not make it integer in the MPS (PuLP maps `Binary` only in `__init__`). The "connectivity reward survives" test therefore forces `z` integer with `pulp.LpInteger`, whose bounds are already [0, 1].
- **Random-problem generator (Task 4)** de-duplicates connectivity pairs. Duplicate asymmetric `(i, j)` rows already crash the binary MIP with "overlapping constraint names"; that is pre-existing and is not Stage 4's to fix (reported to the user as a follow-up).
- **The §7 "infeasible → []" case (Task 4)** is tested with an upper cap below the targets, for both proportion and semicontinuous. A floor-versus-budget case doesn't exist, because min_set has no budget and budgeted objectives can always select nothing.
- **Presolve (Task 5)** does not validate `decisions` / `upper` / `lower`, because it "never raises" (and importing the validator would create an import cycle). P6's text gains `"{upper:g} x "` only when `upper != 1`, so binary text is byte-identical.
- **MultiObjectiveMIP (Task 6)** copies `decisions` / `upper` / `lower` from its validated inner `MIPSolver`. In non-binary modes its `rounded()` closure returns `dict(zip(base.pu_ids, base.extract()))`. The proportion test uses `simple` + COSTBUDGET 30 at `rel_tol` 0, where all 3 features are met on active rows (probe: budgets 10–20 meet none, 25 meets 1).
- **Writers (Task 7):**
  - A binary `write_solution_matrix` keeps its old `np.column_stack` path. If any solution is fractional, every column goes through `_solution_column`: ints for binary columns, `.10g` strings for fractional ones.
  - `write_mvbest` and `export_summary_csv` take held amounts from `_compute_achieved` for binary too (the sums are identical for float amounts). `export_summary_csv` keeps its "no target column" fallback, where met is True.
  - The fractional ssoln `Number` is the raw float Σx. `read_ssoln` casts to int unless `Number` is a NaN-free float column with some non-integral value, so NaN or text still raise.
  - `export_solution_spatial` writes `float(format(x, ".10g"))`, a float column for GPKG/SHP.
  - In `zones/writers.py` only the `zone_assignment is None` fallbacks are guarded.
- **Guards (Task 8):**
  - The solver guards share `solvers.base._require_binary_solver(solver, context)`.
  - `IterativeImprovementSolver.improve` guards before its ITIMPTYPE 0 early return. SA has no `improve()` entry point, so the SA side of spec §6.2 is covered through `run_mode` chains.
  - `robustness` keeps its bool normalisation for non-float selections, because a hand-built 0/1 int array would otherwise be fancy-indexed. It drops that normalisation only for fractional plans.
  - The completeness test also checks that each GUARDED module, and each guarded function in a MIXED module, contains `require_binary`.
- **Shiny (Task 9):** the logic is moved into pure, testable helpers: `_target_rows`, `solution_colors`, `solution_listing`, `comparison_counts`, and `mip_decision_kwargs`.
  - The map colour is a linear RGB blend from `MAP_NOT_SEL` to `MAP_SELECTED` by x; binary keeps the palette constants.
  - For fractional plans the summary line gets ` (Σx …)`.
  - Upper and Lower inputs show only when relevant. Bad bounds surface as `MIPSolver`'s `ValueError`.
  - The comparison map catches `FractionalSolutionError` in the pure helper `comparison_summary` and shows "Comparison unavailable: …" instead of raising out of a render (Review Focus 5).
- **Docs (Task 10).** The user-guide section is "Partial Selection" in `docs/USER_MANUAL.md`, and its code example was run. `.github/copilot-instructions.md` has no solver-matrix table, so the row goes into the `solvers/` row of its Subpackages table. The roadmap gets an as-built note and backlog lines:
  - connectivity-evaluator alignment;
  - per-PU bounds;
  - fractional zone shares.
- **Not done, by decision:**
  - `Solution.metadata` gets no `decisions` key, which would also change binary metadata.
  - The Shiny panel adds no UI-level bound validation.
  - The MPS is pinned in canonical form, not as raw bytes.

---

### Task 1: Binary goldens recorded on the unmodified source

**Files:**
- Create: `tests/data/golden/stage4_v041.json` (generated)
- Create: `tests/pymarxan/solvers/test_stage4_golden_v041.py`

**Interfaces:**
- Consumes: `MIPSolver`, `MultiObjectiveMIP`, `presolve_check`, `build_solution`, `canonical_model` (`tests/pymarxan/lp_canon.py`).
- Produces: `tests/data/golden/stage4_v041.json` — canonical binary models (min_set on simple / sym-conn / asym-conn / locks / PROBMODE 1 / PROBMODE 2, min_penalties, the four budgeted objectives at COSTBUDGET 33), the hierarchical `[MaxFeatures, MinCost]` solve models, presolve P6/P9a/P9b/P11 messages, and `build_solution` outputs on 5 problems × 4 selections. Every later task must leave this file and its test green **without regeneration**.

- [ ] **Step 1: Write the golden test and recorder**

Create `tests/pymarxan/solvers/test_stage4_golden_v041.py`:

```python
"""v0.41.1 binary results that Stage 4 (proportion decisions) must not change (spec §3, §8).

GOLDEN was recorded on unmodified v0.41.1 (8b133f4, Stage 4 spec commits only) before any
source change: the binary ``MIPSolver.build_model`` models (canonical form), the models a
binary ``MultiObjectiveMIP`` hierarchical solve hands to the backend, binary
``presolve_check`` messages, and ``build_solution`` outputs for fixed binary selections
(connectivity included). Every entry must reproduce exactly.

Regenerate only with a written reason:
    PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/python \
        tests/pymarxan/solvers/test_stage4_golden_v041.py
"""
from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pulp
import pytest

from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.multiobjective import Hierarchical, MaxFeatures, MinCost, MultiObjectiveMIP
from pymarxan.solvers.mip_solver import MIP_OBJECTIVES, MIPSolver
from pymarxan.solvers.presolve import presolve_check
from pymarxan.solvers.utils import build_solution, compute_connectivity_penalty
from tests.pymarxan.lp_canon import canonical_model

DATA = Path(__file__).parent.parent.parent / "data"
GOLDEN_PATH = DATA / "golden" / "stage4_v041.json"
BUDGET = 33.0


def _simple(**params: Any) -> ConservationProblem:
    p = load_project(DATA / "simple")
    return p.copy_with(parameters={**p.parameters, **params})


def _conn(asymmetric: bool) -> ConservationProblem:
    p = _simple(CONNECTIVITY_WEIGHT=2.0, ASYMMETRIC_CONNECTIVITY=int(asymmetric))
    conn = pd.DataFrame({
        "id1": [1, 2, 3, 2, 5], "id2": [2, 1, 4, 5, 6], "value": [4.0, 3.0, 2.5, 6.0, 1.5],
    })
    return p.copy_with(connectivity=conn)


def _locks(**params: Any) -> ConservationProblem:
    p = _simple(**params)
    pu = p.planning_units.copy()
    pu.loc[pu["id"] == 3, "status"] = 2  # locked in
    pu.loc[pu["id"] == 6, "status"] = 3  # locked out
    return p.copy_with(planning_units=pu)


def _prob(probmode: int) -> ConservationProblem:
    p = _simple(PROBMODE=probmode, PROBABILITYWEIGHTING=2.0)
    prob = pd.DataFrame({
        "pu": [1, 2, 3, 4, 5, 6], "probability": [0.1, 0.4, 0.0, 0.25, 0.05, 0.3],
    })
    return p.copy_with(probability=prob)


MODEL_VARIANTS: dict[str, Callable[[], ConservationProblem]] = {
    "simple": _simple,
    "conn_sym": lambda: _conn(False),
    "conn_asym": lambda: _conn(True),
    "locks": _locks,
    "probmode1": lambda: _prob(1),
    "probmode2": lambda: _prob(2),
}
MODEL_CASES = [f"{v}|min_set" for v in MODEL_VARIANTS] + ["simple|min_penalties"] + [
    f"budget|{o}" for o in MIP_OBJECTIVES if o not in ("min_set", "min_penalties")
]

PRESOLVE_CASES: dict[str, Callable[[], ConservationProblem]] = {
    "locks_budget_low": lambda: _locks(COSTBUDGET=10.0),  # P9a: below locked-in cost 20
    "locks_budget_high": lambda: _locks(COSTBUDGET=100.0),  # P11
    "budget_cheapest": lambda: _simple(COSTBUDGET=5.0),  # P9b: below min free cost 8
    "p6_target": lambda: _locks(MISSLEVEL=1.0).copy_with(
        features=_simple().features.assign(target=[30.0, 20.0, 40.0]),
    ),
}

SELECTIONS = {"235": [0, 1, 1, 0, 1, 0], "246": [0, 1, 0, 1, 0, 1], "none": [0] * 6,
              "all": [1] * 6}
SOLUTION_VARIANTS: dict[str, Callable[[], ConservationProblem]] = {
    "simple": _simple,
    "conn_sym": lambda: _conn(False),
    "conn_asym": lambda: _conn(True),
    "probmode1": lambda: _prob(1),
    "misslevel": lambda: _simple(MISSLEVEL=0.8),
}


def _build(case: str) -> dict[str, Any]:
    variant, objective = case.split("|")
    if variant == "budget":
        problem = _simple(COSTBUDGET=BUDGET)
    else:
        problem = MODEL_VARIANTS[variant]()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # PuLP 3.3.2
        model = MIPSolver(objective=objective).build_model(problem)
    return canonical_model(model)


def _hierarchical() -> dict[str, Any]:
    """Canonical model at every backend solve of a binary [MaxFeatures, MinCost] run."""
    problem = _simple(COSTBUDGET=BUDGET)
    snaps: list[dict[str, Any]] = []
    orig = pulp.LpProblem.solve

    def spy(self: pulp.LpProblem, *args: Any, **kwargs: Any) -> Any:
        snaps.append(canonical_model(self))
        return orig(self, *args, **kwargs)

    mo = MultiObjectiveMIP([MaxFeatures(), MinCost()], Hierarchical(0.1), mip_backend="cbc")
    pulp.LpProblem.solve = spy  # type: ignore[method-assign]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sols = mo.solve(problem)
    finally:
        pulp.LpProblem.solve = orig  # type: ignore[method-assign]
    s = sols[0]
    return {
        "models": snaps,
        "selected": [int(v) for v in s.selected],
        "objective_values": {k: float(v) for k, v in s.metadata["objective_values"].items()},
        "cost": float(s.cost),
    }


def _solution(case: str) -> dict[str, Any]:
    variant, sel = case.split("|")
    problem = SOLUTION_VARIANTS[variant]()
    selected = np.array(SELECTIONS[sel], dtype=bool)
    blm = float(problem.parameters.get("BLM", 0.0))
    s = build_solution(problem, selected, blm)
    pu_index = {int(p): i for i, p in enumerate(problem.planning_units["id"])}
    return {
        "cost": float(s.cost),
        "boundary": float(s.boundary),
        "objective": float(s.objective),
        "penalty": float(s.penalty),
        "shortfall": float(s.shortfall),
        "targets_met": {str(k): bool(v) for k, v in sorted(s.targets_met.items())},
        "n_selected": int(s.n_selected),
        "connectivity": float(compute_connectivity_penalty(problem, selected, pu_index)),
    }


SOLUTION_CASES = [f"{v}|{s}" for v in SOLUTION_VARIANTS for s in SELECTIONS]


def record() -> dict[str, Any]:
    return {
        "models": {case: _build(case) for case in MODEL_CASES},
        "hierarchical": _hierarchical(),
        "presolve": {k: presolve_check(f()) for k, f in PRESOLVE_CASES.items()},
        "solutions": {case: _solution(case) for case in SOLUTION_CASES},
    }


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(GOLDEN_PATH.read_text())
    return data


@pytest.mark.parametrize("case", MODEL_CASES)
def test_binary_model_unchanged(golden: dict[str, Any], case: str) -> None:
    assert _build(case) == golden["models"][case]


def test_binary_hierarchical_models_unchanged(golden: dict[str, Any]) -> None:
    got = _hierarchical()
    want = golden["hierarchical"]
    assert got["selected"] == want["selected"]
    assert got["cost"] == pytest.approx(want["cost"], abs=1e-9)
    assert got["objective_values"] == pytest.approx(want["objective_values"], abs=1e-9)
    assert len(got["models"]) == len(want["models"])
    for g, w in zip(got["models"], want["models"], strict=True):
        assert g == w


@pytest.mark.parametrize("case", sorted(PRESOLVE_CASES))
def test_binary_presolve_messages_unchanged(golden: dict[str, Any], case: str) -> None:
    assert presolve_check(PRESOLVE_CASES[case]()) == golden["presolve"][case]


def test_presolve_fixtures_fire() -> None:
    """The golden is only useful if the budget / P6 codes actually occur."""
    codes = {k: [m.split(":")[0] for m in presolve_check(f())] for k, f in PRESOLVE_CASES.items()}
    assert "P9a" in codes["locks_budget_low"]
    assert "P11" in codes["locks_budget_high"]
    assert "P9b" in codes["budget_cheapest"]
    assert "P6" in codes["p6_target"]


@pytest.mark.parametrize("case", SOLUTION_CASES)
def test_binary_build_solution_unchanged(golden: dict[str, Any], case: str) -> None:
    got = _solution(case)
    want = golden["solutions"][case]
    for key, value in want.items():
        if isinstance(value, float):
            assert got[key] == pytest.approx(value, rel=1e-12, abs=1e-12), key
        else:
            assert got[key] == value, key


if __name__ == "__main__":
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(record(), indent=1, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN_PATH}")
```

- [ ] **Step 2: Record the golden data** (on the unmodified source)

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/python tests/pymarxan/solvers/test_stage4_golden_v041.py`
Expected: writes `tests/data/golden/stage4_v041.json` (run it on the unmodified base; no source file is touched in this task).

- [ ] **Step 3: Run the tests to verify they pass**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_stage4_golden_v041.py -q`
Expected: 37 passed

- [ ] **Step 4: Gate**

Run: `~/.local/bin/ruff check src tests && /home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/ -q -m "not bench" -p no:cacheprovider`
Expected: ruff and mypy clean; the full suite passes with no golden regenerated (prototype final: 3444 passed, 61 skipped, 20 deselected at Task 10).

- [ ] **Step 5: Commit**

```bash
git add tests/data/golden/stage4_v041.json tests/pymarxan/solvers/test_stage4_golden_v041.py
git commit -m "test(stage4): T1 record v0.41.1 binary goldens

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AZEiop9qqxVCrMHKqSDYP8"
```

---

### Task 2: `Solution` fractional properties and the `require_binary` guard

**Files:**
- Modify: `src/pymarxan/solvers/__init__.py`
- Modify: `src/pymarxan/solvers/base.py`
- Create: `tests/pymarxan/solvers/test_fractional_solution.py`

**Interfaces:**
- Consumes: nothing new.
- Produces (`pymarxan.solvers.base`, re-exported from `pymarxan.solvers`): `Solution.is_fractional -> bool` (dtype kind `"f"`), `Solution.selected_sum -> float`, `Solution.n_selected -> int` (count of `x > 0`), `class FractionalSolutionError(ValueError)`, `require_binary(solution: Solution, context: str) -> None`, and the private `_require_binary_solver(solver: object, context: str) -> None` (reads `getattr(solver, "decisions", "binary")`; used in Task 8).

- [ ] **Step 1: Write the failing tests**

Create `tests/pymarxan/solvers/test_fractional_solution.py`:

```python
"""Stage 4 (spec §5.1, §5.2): the fractional Solution model and the binary-only guard."""
from __future__ import annotations

import numpy as np
import pytest

from pymarxan.solvers import FractionalSolutionError, require_binary
from pymarxan.solvers.base import Solution


def _sol(selected: np.ndarray) -> Solution:
    return Solution(selected=selected, cost=0.0, boundary=0.0, objective=0.0, targets_met={})


def test_binary_solution_is_not_fractional() -> None:
    s = _sol(np.array([True, False, True]))
    assert not s.is_fractional
    assert s.n_selected == 2
    assert isinstance(s.n_selected, int)
    assert s.selected_sum == 2.0
    assert isinstance(s.selected_sum, float)


def test_fractional_solution_properties() -> None:
    s = _sol(np.array([0.25, 0.0, 1.0, 0.5]))
    assert s.is_fractional
    assert s.n_selected == 3  # count of x > 0, not the sum
    assert s.selected_sum == pytest.approx(1.75)


def test_dtype_is_the_discriminator_not_the_values() -> None:
    s = _sol(np.array([1.0, 0.0, 1.0]))
    assert s.is_fractional
    assert s.n_selected == 2
    assert s.selected_sum == 2.0


def test_require_binary_passes_binary() -> None:
    require_binary(_sol(np.array([True, False])), "overlap_matrix")


def test_require_binary_rejects_fractional() -> None:
    with pytest.raises(FractionalSolutionError) as exc:
        require_binary(_sol(np.array([0.5, 1.0])), "overlap_matrix")
    msg = str(exc.value)
    assert msg == (
        "overlap_matrix needs a binary solution, but this solution holds proportion "
        "decisions. Convert explicitly with binarize_solution(problem, solution, "
        "threshold=...)."
    )
    assert isinstance(exc.value, ValueError)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_fractional_solution.py -q`
Expected: FAIL — collection error (`ImportError: cannot import name 'FractionalSolutionError'`)

- [ ] **Step 3: Implement**

Modify `src/pymarxan/solvers/__init__.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/__init__.py b/src/pymarxan/solvers/__init__.py
index a18f54a..5833596 100644
--- a/src/pymarxan/solvers/__init__.py
+++ b/src/pymarxan/solvers/__init__.py
@@ -1,16 +1,23 @@
-from pymarxan.solvers.base import Solution, Solver, SolverConfig
+from pymarxan.solvers.base import (
+    FractionalSolutionError,
+    Solution,
+    Solver,
+    SolverConfig,
+    require_binary,
+)
 from pymarxan.solvers.export import write_problem
 from pymarxan.solvers.heuristic import HeuristicSolver
 from pymarxan.solvers.iterative_improvement import IterativeImprovementSolver
 from pymarxan.solvers.marxan_binary import MarxanBinarySolver
 from pymarxan.solvers.mip_solver import MIPSolver
 from pymarxan.solvers.presolve import PresolveWarning, presolve_check
 from pymarxan.solvers.run_mode import RunModePipeline
 from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
 
 __all__ = [
+    "FractionalSolutionError",
     "HeuristicSolver",
     "IterativeImprovementSolver",
     "MarxanBinarySolver",
     "MIPSolver",
     "PresolveWarning",
@@ -18,7 +25,8 @@ __all__ = [
     "SimulatedAnnealingSolver",
     "Solution",
     "Solver",
     "SolverConfig",
     "presolve_check",
+    "require_binary",
     "write_problem",
 ]
```

Modify `src/pymarxan/solvers/base.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/base.py b/src/pymarxan/solvers/base.py
index 9d74e7b..a7ebecc 100644
--- a/src/pymarxan/solvers/base.py
+++ b/src/pymarxan/solvers/base.py
@@ -11,11 +11,12 @@ from pymarxan.models.problem import ConservationProblem
 
 @dataclass
 class Solution:
     """Result of a conservation planning optimization run."""
 
-    selected: np.ndarray  # Boolean array: is planning unit i selected?
+    # bool array, or float64 proportions in [0, 1] when solved with non-binary decisions
+    selected: np.ndarray
     cost: float
     boundary: float
     objective: float  # cost + BLM * boundary + penalties
     # Feature ID -> met. Zone solvers report the overall (contribution-weighted) feature
     # targets here and the per-zone targets in metadata["zone_targets_met"].
@@ -49,13 +50,39 @@ class Solution:
 
     @property
     def all_targets_met(self) -> bool:
         return all(self.targets_met.values())
 
+    @property
+    def is_fractional(self) -> bool:
+        """Solved with proportion / semicontinuous decisions: the dtype decides, not the
+        values (an integral LP optimum is still fractional-typed)."""
+        return self.selected.dtype.kind == "f"
+
     @property
     def n_selected(self) -> int:
-        return int(self.selected.sum())
+        """Count of planning units with x > 0 (a count, not an area)."""
+        return int(np.count_nonzero(self.selected))
+
+    @property
+    def selected_sum(self) -> float:
+        """Σ x (prioritizr ``eval_n_summary``); equals ``n_selected`` for binary."""
+        return float(self.selected.sum())
+
+
+class FractionalSolutionError(ValueError):
+    """A binary-only tool was given a solution holding proportion decisions."""
+
+
+def require_binary(solution: Solution, context: str) -> None:
+    """Raise ``FractionalSolutionError`` unless ``solution.selected`` is binary (spec §5.2)."""
+    if solution.is_fractional:
+        raise FractionalSolutionError(
+            f"{context} needs a binary solution, but this solution holds proportion "
+            "decisions. Convert explicitly with binarize_solution(problem, solution, "
+            "threshold=...)."
+        )
 
 
 @dataclass
 class SolverConfig:
     """Configuration for a solver run."""
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_fractional_solution.py -q`
Expected: 5 passed

- [ ] **Step 5: Gate**

Run: `~/.local/bin/ruff check src tests && /home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/ -q -m "not bench" -p no:cacheprovider`
Expected: ruff and mypy clean; the full suite passes with no golden regenerated (prototype final: 3444 passed, 61 skipped, 20 deselected at Task 10).

- [ ] **Step 6: Commit**

```bash
git add src/pymarxan/solvers/__init__.py src/pymarxan/solvers/base.py tests/pymarxan/solvers/test_fractional_solution.py
git commit -m "feat(stage4): T2 fractional Solution properties and require_binary guard

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AZEiop9qqxVCrMHKqSDYP8"
```

---

### Task 3: Fraction-linear evaluators, target tolerance, `binarize_solution`

**Files:**
- Modify: `src/pymarxan/solvers/__init__.py`
- Create: `src/pymarxan/solvers/_tolerance.py`
- Modify: `src/pymarxan/solvers/utils.py`
- Create: `tests/pymarxan/fractional_fixtures.py`
- Create: `tests/pymarxan/solvers/test_fractional_evaluators.py`

**Interfaces:**
- Consumes: Task 2's `Solution.is_fractional`.
- Produces: new `pymarxan/solvers/_tolerance.py` with `TARGET_ABS_TOL = 1e-6`, `TARGET_REL_TOL = 1e-7`, `target_tol(v: float) -> float` (re-exported by `solvers.utils`); in `solvers/utils.py`: `_is_fractional(selected: np.ndarray) -> bool`, `_nonbinary_conflicts(problem) -> list[str]` (Task 4's gate reuses it), `_total_cost(costs, selected) -> float`, fraction-linear `_compute_achieved` / `compute_boundary` / `compute_connectivity_penalty` / PROBMODE 1 premium / `check_targets` / `compute_feature_shortfalls`, PROBMODE 3 `ValueError` in `compute_probability_penalty`, `build_solution` `ValueError` for fractional + clumping/separation/space, `binarize_solution(problem, solution, threshold=0.5) -> Solution` (re-exported from `pymarxan.solvers`); test helper module `tests/pymarxan/fractional_fixtures.py` (`three_pu`, `X`, `PU_INDEX`).

- [ ] **Step 1: Write the failing tests**

Create `tests/pymarxan/fractional_fixtures.py`:

```python
"""Shared Stage 4 test fixtures (a helper module like ``lp_canon``, not a test file)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pymarxan.models.problem import ConservationProblem

X = np.array([0.6, 0.2, 1.0])
PU_INDEX = {1: 0, 2: 1, 3: 2}


def three_pu(**params: float) -> ConservationProblem:
    """3 PUs (cost 10, 20, 30); f1 target 19 (amounts 10, 40, 5), f2 target 10 (8, -, 4)."""
    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [10.0, 20.0, 30.0], "status": [0, 0, 0]})
    feat = pd.DataFrame({
        "id": [1, 2], "name": ["a", "b"], "target": [19.0, 10.0], "spf": [1.0, 1.0],
    })
    pvf = pd.DataFrame({
        "species": [1, 1, 1, 2, 2], "pu": [1, 2, 3, 1, 3],
        "amount": [10.0, 40.0, 5.0, 8.0, 4.0],
    })
    bnd = pd.DataFrame({
        "id1": [1, 1, 2, 3], "id2": [1, 2, 3, 3], "boundary": [3.0, 2.0, 4.0, 1.0],
    })
    conn = pd.DataFrame({"id1": [1, 2, 1], "id2": [2, 1, 3], "value": [3.0, 1.0, 2.0]})
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pvf, boundary=bnd,
        parameters={"BLM": 1.0, "CONNECTIVITY_WEIGHT": 2.0, **params}, connectivity=conn,
    )
```

Create `tests/pymarxan/solvers/test_fractional_evaluators.py`:

```python
"""Stage 4 (spec §5.2, §5.3): fraction-linear evaluators, the target tolerance and
``binarize_solution``. Every number below is a hand calculation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.solvers import binarize_solution
from pymarxan.solvers.utils import (
    TARGET_ABS_TOL,
    TARGET_REL_TOL,
    _compute_achieved,
    _is_fractional,
    build_solution,
    check_targets,
    compute_boundary,
    compute_connectivity_penalty,
    compute_feature_shortfalls,
    compute_objective_terms,
    compute_probability_penalty,
    target_tol,
)
from tests.pymarxan.fractional_fixtures import PU_INDEX, X, three_pu


def test_target_tol() -> None:
    assert TARGET_ABS_TOL == 1e-6
    assert TARGET_REL_TOL == 1e-7
    assert target_tol(0.0) == 1e-6
    assert target_tol(-1e8) == pytest.approx(1e-6 + 10.0)


def test_is_fractional() -> None:
    assert _is_fractional(X)
    assert _is_fractional(np.array([1.0, 0.0]))
    assert not _is_fractional(np.array([True, False]))


def test_hand_calculation() -> None:
    p = three_pu()
    held = _compute_achieved(p, X, PU_INDEX)
    assert held == pytest.approx({1: 6.0 + 8.0 + 5.0, 2: 4.8 + 4.0})  # 19.0, 8.8
    # 3·0.6 + 2·|0.6 − 0.2| + 4·|0.2 − 1| + 1·1
    assert compute_boundary(p, X, PU_INDEX) == pytest.approx(6.8)
    # rows: (1,2,3): −3·0.2 + 3·0.4 = 0.6; (2,1,1): −0.2 + 0.4 = 0.2; (1,3,2): −1.2 + 0.8
    assert compute_connectivity_penalty(p, X, PU_INDEX) == pytest.approx(2.0 * 0.4)
    assert check_targets(p, X, PU_INDEX) == {1: True, 2: False}
    assert compute_feature_shortfalls(p, X, PU_INDEX) == pytest.approx({1: 0.0, 2: 1.2})
    s = build_solution(p, X, 1.0)
    assert s.is_fractional
    assert s.selected.dtype == np.float64
    assert s.cost == pytest.approx(6.0 + 4.0 + 30.0)
    assert s.boundary == pytest.approx(6.8)
    assert s.penalty == pytest.approx(1.2)
    assert s.shortfall == pytest.approx(1.2)
    assert s.objective == pytest.approx(40.0 + 6.8 + 1.2 + 0.8)
    assert s.n_selected == 3
    assert s.selected_sum == pytest.approx(1.8)


def test_formulas_reduce_to_binary() -> None:
    """A 0/1 float array gives the binary numbers (only the dtype differs)."""
    p = three_pu()
    b = np.array([True, False, True])
    f = b.astype(float)
    sb, sf = build_solution(p, b, 1.0), build_solution(p, f, 1.0)
    for attr in ("cost", "boundary", "objective", "penalty", "shortfall"):
        assert getattr(sf, attr) == pytest.approx(getattr(sb, attr), abs=1e-12), attr
    assert sf.targets_met == sb.targets_met
    assert compute_connectivity_penalty(p, f, PU_INDEX) == pytest.approx(
        compute_connectivity_penalty(p, b, PU_INDEX)
    )


def test_probmode1_premium_is_linear_in_x() -> None:
    p = three_pu(PROBMODE=1, PROBABILITYWEIGHTING=2.0).copy_with(
        probability=pd.DataFrame({"pu": [1, 2, 3], "probability": [0.5, 0.1, 0.0]}),
    )
    # 2 · (0.5·10·0.6 + 0.1·20·0.2 + 0)
    assert compute_probability_penalty(p, X, PU_INDEX) == pytest.approx(2.0 * (3.0 + 0.4))


@pytest.mark.parametrize("scale", [1.0, 1e4, 1e8])
def test_target_tolerance_fractional(scale: float) -> None:
    p = three_pu()
    r = 19.0 * scale
    tol = target_tol(r)
    p2 = p.copy_with(features=p.features.assign(target=[r, 10.0]))
    near = {1: r - 0.5 * tol, 2: 10.0}
    assert check_targets(p2, X, PU_INDEX, _achieved=near) == {1: True, 2: True}
    assert compute_feature_shortfalls(p2, X, PU_INDEX, _achieved=near) == {1: 0.0, 2: 0.0}
    far = {1: r - 2.0 * tol, 2: 10.0}
    assert check_targets(p2, X, PU_INDEX, _achieved=far)[1] is False
    assert compute_feature_shortfalls(p2, X, PU_INDEX, _achieved=far)[1] == pytest.approx(
        2.0 * tol
    )


def test_binary_comparison_stays_exact() -> None:
    p = three_pu()
    b = np.array([True, False, True])
    achieved = {1: 19.0 - 1e-12, 2: 10.0}
    assert check_targets(p, b, PU_INDEX, _achieved=achieved)[1] is False
    assert compute_feature_shortfalls(p, b, PU_INDEX, _achieved=achieved)[1] > 0


def test_misslevel_applies_to_fractional() -> None:
    p = three_pu(MISSLEVEL=0.8)  # f2 R = 8.0 <= 8.8
    assert check_targets(p, X, PU_INDEX) == {1: True, 2: True}


@pytest.mark.parametrize(
    ("column", "values", "word"),
    [
        ("target2", {"target2": [5.0, 0.0]}, "target2"),
        ("sep", {"sepnum": [2, 0], "sepdistance": [1.0, 0.0]}, "separation"),
        ("space", {"space_target": [0.5, 0.0]}, "space_target"),
    ],
)
def test_build_solution_rejects_fractional_combinations(
    column: str, values: dict[str, list[float]], word: str,
) -> None:
    p = three_pu()
    p = p.copy_with(features=p.features.assign(**values))
    with pytest.raises(ValueError, match=word):
        build_solution(p, X, 1.0)


def test_probmode3_rejects_fractional() -> None:
    p = three_pu(PROBMODE=3)
    with pytest.raises(ValueError, match="PROBMODE 3"):
        compute_probability_penalty(p, X, PU_INDEX)
    with pytest.raises(ValueError, match="PROBMODE 3"):
        compute_objective_terms(p, X, PU_INDEX, 1.0)
    with pytest.raises(ValueError, match="PROBMODE 3"):
        build_solution(p, X, 1.0)
    compute_probability_penalty(p, np.array([True, False, True]), PU_INDEX)  # binary: fine


def test_binarize_solution() -> None:
    p = three_pu()
    frac = build_solution(p, X, 1.0, metadata={"solver": "MIP"})
    b = binarize_solution(p, frac)
    assert b.selected.dtype == bool
    assert b.selected.tolist() == [True, False, True]
    assert b.metadata == {"solver": "MIP", "binarized_threshold": 0.5}
    assert b.cost == 40.0
    assert binarize_solution(p, frac, threshold=0.2).selected.tolist() == [True, True, True]
    assert binarize_solution(p, frac, threshold=1.0).selected.tolist() == [False, False, True]


def test_binarize_binary_returns_same_object() -> None:
    p = three_pu()
    sol = build_solution(p, np.array([True, False, True]), 1.0)
    assert binarize_solution(p, sol) is sol


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.5, float("nan")])
def test_binarize_rejects_bad_threshold(threshold: float) -> None:
    p = three_pu()
    frac = build_solution(p, X, 1.0)
    with pytest.raises(ValueError, match="threshold"):
        binarize_solution(p, frac, threshold=threshold)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/fractional_fixtures.py tests/pymarxan/solvers/test_fractional_evaluators.py -q`
Expected: FAIL — collection error (`ImportError` on `target_tol` / `binarize_solution` / the fixtures module's imports)

- [ ] **Step 3: Implement**

Modify `src/pymarxan/solvers/__init__.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/__init__.py b/src/pymarxan/solvers/__init__.py
index 5833596..a8ad3de 100644
--- a/src/pymarxan/solvers/__init__.py
+++ b/src/pymarxan/solvers/__init__.py
@@ -11,10 +11,11 @@ from pymarxan.solvers.iterative_improvement import IterativeImprovementSolver
 from pymarxan.solvers.marxan_binary import MarxanBinarySolver
 from pymarxan.solvers.mip_solver import MIPSolver
 from pymarxan.solvers.presolve import PresolveWarning, presolve_check
 from pymarxan.solvers.run_mode import RunModePipeline
 from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
+from pymarxan.solvers.utils import binarize_solution
 
 __all__ = [
     "FractionalSolutionError",
     "HeuristicSolver",
     "IterativeImprovementSolver",
@@ -24,9 +25,10 @@ __all__ = [
     "RunModePipeline",
     "SimulatedAnnealingSolver",
     "Solution",
     "Solver",
     "SolverConfig",
+    "binarize_solution",
     "presolve_check",
     "require_binary",
     "write_problem",
 ]
```

Create `src/pymarxan/solvers/_tolerance.py`:

```python
"""Target tolerance for fractional (proportion) plans (Stage 4 spec §5.3). Stdlib only, so
``solvers.utils`` and ``solvers._mip_objectives`` (pulp + stdlib) can both import it.

An LP optimum leaves every active target row at ``R − O(1e-9·|R|)`` (CBC), so an exact
``held >= R`` flips on solver noise. Binary plans keep the exact comparison.
"""
from __future__ import annotations

TARGET_ABS_TOL = 1e-6
TARGET_REL_TOL = 1e-7


def target_tol(v: float) -> float:
    """``TARGET_ABS_TOL + TARGET_REL_TOL·|v|``: absolute alone fails at large magnitudes."""
    return TARGET_ABS_TOL + TARGET_REL_TOL * abs(v)
```

Modify `src/pymarxan/solvers/utils.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/utils.py b/src/pymarxan/solvers/utils.py
index 229e875..e7131b0 100644
--- a/src/pymarxan/solvers/utils.py
+++ b/src/pymarxan/solvers/utils.py
@@ -1,15 +1,58 @@
-"""Shared utility functions for conservation planning solvers."""
+"""Shared utility functions for conservation planning solvers.
+
+Selections are bool arrays, or float64 proportions in [0, 1] (Stage 4). Every evaluator
+is linear in the fractions and reduces to the binary formula on 0/1 input; the binary
+code path keeps its exact comparisons (spec §5.3).
+"""
 from __future__ import annotations
 
+import math
+
 import numpy as np
 import pandas as pd
 
 from pymarxan.models.problem import ConservationProblem
+from pymarxan.solvers._tolerance import TARGET_ABS_TOL as TARGET_ABS_TOL
+from pymarxan.solvers._tolerance import TARGET_REL_TOL as TARGET_REL_TOL
+from pymarxan.solvers._tolerance import target_tol as target_tol
 from pymarxan.solvers.base import Solution
 
 
+def _is_fractional(selected: np.ndarray) -> bool:
+    """A proportion selection: the dtype decides, not the values (spec §5.1)."""
+    return np.asarray(selected).dtype.kind == "f"
+
+
+def _nonbinary_conflicts(problem: ConservationProblem) -> list[str]:
+    """Problem settings whose evaluators only make sense for 0/1 selections (spec §3.5).
+
+    Clumping, separation and space are evaluated post hoc by selecting PUs by truthiness.
+    PROBMODE 3 is checked separately (``compute_probability_penalty``).
+    """
+    feats = problem.features
+    out: list[str] = []
+    if "target2" in feats.columns and (feats["target2"] > 0).any():
+        out.append("clumping (a feature has target2 > 0)")
+    if (
+        "sepnum" in feats.columns
+        and "sepdistance" in feats.columns
+        and ((feats["sepnum"] > 1) & (feats["sepdistance"] > 0)).any()
+    ):
+        out.append("separation (a feature has sepdistance > 0 and sepnum > 1)")
+    if "space_target" in feats.columns and (feats["space_target"] > 0).any():
+        out.append("space targets (a feature has space_target > 0)")
+    return out
+
+
+def _total_cost(costs: np.ndarray, selected: np.ndarray) -> float:
+    """Σ c·x (fractional) or the cost of the selected PUs (binary, unchanged)."""
+    if _is_fractional(selected):
+        return float(np.dot(costs.astype(np.float64), selected))
+    return float(np.sum(costs[selected]))
+
+
 def _compute_achieved(
     problem: ConservationProblem,
     selected: np.ndarray,
     pu_index: dict[int, int],
 ) -> dict[int, float]:
@@ -27,10 +70,20 @@ def _compute_achieved(
         [pu_index.get(int(pid), -1) for pid in pu_col], dtype=np.intp
     )
     valid = pu_indices >= 0
     # Safe indexing: use 0 for invalid indices, then mask out
     safe_indices = np.where(valid, pu_indices, 0)
+    if _is_fractional(selected):
+        # held_f = Σ_i r_if·x_i (raw amounts, as the binary path)
+        weighted = pd.DataFrame({
+            "species": species_col,
+            "amount": amount_col.astype(np.float64) * np.where(valid, selected[safe_indices], 0.0),
+        })
+        result = {int(fid): 0.0 for fid in problem.features["id"]}
+        for fid, amount in weighted.groupby("species")["amount"].sum().items():
+            result[int(fid)] = float(amount)  # type: ignore[call-overload]
+        return result
     sel_mask = valid & selected[safe_indices]
 
     # Use only selected rows and groupby species
     selected_amounts = pd.DataFrame({
         "species": species_col[sel_mask],
@@ -54,13 +107,15 @@ def compute_boundary(
 ) -> float:
     """Compute total boundary length for a given selection.
 
     Diagonal entries (id1==id2) represent external boundary, added when PU is selected.
     Off-diagonal entries represent shared boundary, added when exactly one PU is selected.
+    Fractional: ``Σ b_ii·x_i + Σ b_ij·|x_i − x_j|`` (row-wise), the MIP's boundary term.
     """
     if problem.boundary is None:
         return 0.0
+    fractional = _is_fractional(selected)
 
     bnd = problem.boundary
     id1_col = bnd["id1"].values
     id2_col = bnd["id2"].values
     bval_col = bnd["boundary"].values.astype(np.float64)
@@ -72,16 +127,18 @@ def compute_boundary(
         bval = float(bval_col[i])
 
         if id1 == id2:
             idx = pu_index.get(id1)
             if idx is not None and selected[idx]:
-                total += bval
+                total += bval * float(selected[idx]) if fractional else bval
         else:
             idx1 = pu_index.get(id1)
             idx2 = pu_index.get(id2)
             if idx1 is not None and idx2 is not None:
-                if selected[idx1] != selected[idx2]:
+                if fractional:
+                    total += bval * abs(float(selected[idx1]) - float(selected[idx2]))
+                elif selected[idx1] != selected[idx2]:
                     total += bval
     return total
 
 
 def check_targets(
@@ -93,49 +150,61 @@ def check_targets(
 ) -> dict[int, bool]:
     """Check which feature targets are met by the selection.
 
     Uses the MISSLEVEL parameter (default 1.0) to allow fractional
     target achievement.  A target is considered met when the achieved
-    amount >= target * misslevel.
+    amount >= target * misslevel (exactly for binary selections; within
+    ``target_tol`` for fractional ones, spec §5.3).
     """
     misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
     achieved = (
         _achieved if _achieved is not None
         else _compute_achieved(problem, selected, pu_index)
     )
+    fractional = _is_fractional(selected)
 
     targets_met: dict[int, bool] = {}
     feat_ids = problem.features["id"].values
     feat_targets = problem.features["target"].values
     for i in range(len(feat_ids)):
         fid = int(feat_ids[i])
-        target = float(feat_targets[i])
-        targets_met[fid] = achieved.get(fid, 0.0) >= target * misslevel
+        required = float(feat_targets[i]) * misslevel
+        if fractional:
+            required -= target_tol(required)
+        targets_met[fid] = achieved.get(fid, 0.0) >= required
     return targets_met
 
 
 def compute_feature_shortfalls(
     problem: ConservationProblem,
     selected: np.ndarray,
     pu_index: dict[int, int],
     *,
     _achieved: dict[int, float] | None = None,
 ) -> dict[int, float]:
-    """Compute the shortfall for each feature (target*misslevel - achieved, min 0)."""
+    """Compute the shortfall for each feature (target*misslevel - achieved, min 0).
+
+    Fractional selections: 0 when the feature is met within ``target_tol`` (so met ⇔
+    shortfall == 0, as ``check_targets`` decides it).
+    """
     misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
     achieved = (
         _achieved if _achieved is not None
         else _compute_achieved(problem, selected, pu_index)
     )
+    fractional = _is_fractional(selected)
 
     shortfalls: dict[int, float] = {}
     feat_ids = problem.features["id"].values
     feat_targets = problem.features["target"].values
     for i in range(len(feat_ids)):
         fid = int(feat_ids[i])
         target = float(feat_targets[i]) * misslevel
-        shortfalls[fid] = max(0.0, target - achieved.get(fid, 0.0))
+        short = max(0.0, target - achieved.get(fid, 0.0))
+        if fractional and short <= target_tol(target):
+            short = 0.0
+        shortfalls[fid] = short
     return shortfalls
 
 
 def compute_penalty(
     problem: ConservationProblem,
@@ -206,10 +275,14 @@ def compute_connectivity_penalty(
     Total = CONNECTIVITY_WEIGHT * Σ edges where exactly one PU selected * value
           - CONNECTIVITY_WEIGHT * Σ edges where both PUs selected * value
 
     If ASYMMETRIC_CONNECTIVITY is set, directionality is preserved.
 
+    Fractional selections, per row: ``−v·min(x_i, x_j) + v·|x_i − x_j|`` (today's ``−v`` /
+    ``+v`` on 0/1). This evaluator differs from the MIP's connectivity term (pre-existing;
+    spec §5.3).
+
     Returns 0.0 if no connectivity data.
     """
     if problem.connectivity is None:
         return 0.0
 
@@ -222,17 +295,22 @@ def compute_connectivity_penalty(
     conn = problem.connectivity
     id1_col = conn["id1"].values
     id2_col = conn["id2"].values
     val_col = conn["value"].values.astype(np.float64)
 
+    fractional = _is_fractional(selected)
     total = 0.0
     for k in range(len(id1_col)):
         idx1 = pu_index.get(int(id1_col[k]))
         idx2 = pu_index.get(int(id2_col[k]))
         if idx1 is None or idx2 is None:
             continue
         val = float(val_col[k])
+        if fractional:
+            x1, x2 = float(selected[idx1]), float(selected[idx2])
+            total += -val * min(x1, x2) + val * abs(x1 - x2)
+            continue
         s1 = selected[idx1]
         s2 = selected[idx2]
         if s1 and s2:
             total -= val  # bonus for connected pair
         elif s1 or s2:
@@ -257,10 +335,16 @@ def compute_probability_penalty(
     """
     prob_mode = int(problem.parameters.get("PROBMODE", 1))
     if prob_mode == 1:
         return _compute_probmode1_risk_premium(problem, selected)
     if prob_mode == 3:
+        if _is_fractional(selected):
+            raise ValueError(
+                "PROBMODE 3 (Z-score chance constraints) is only defined for binary "
+                "selections; this selection holds proportion decisions. Solve with "
+                "decisions='binary' or convert with binarize_solution()."
+            )
         return _compute_probmode3_zscore_penalty(problem, selected)
     return 0.0
 
 
 def _compute_probmode1_risk_premium(
@@ -281,17 +365,21 @@ def _compute_probmode1_risk_premium(
     for k in range(len(prob_pu)):
         prob_map[int(prob_pu[k])] = float(prob_val[k])
 
     costs = problem.planning_units["cost"].values
     pu_ids = problem.planning_units["id"].values
+    fractional = _is_fractional(selected)
 
     total = 0.0
     for i in range(len(pu_ids)):
         if selected[i]:
             pid = int(pu_ids[i])
             prob = prob_map.get(pid, 0.0)
-            total += prob * float(costs[i])
+            if fractional:
+                total += prob * float(costs[i]) * float(selected[i])
+            else:
+                total += prob * float(costs[i])
     return prob_weight * total
 
 
 def _compute_probmode3_zscore_penalty(
     problem: ConservationProblem,
@@ -408,11 +496,11 @@ def compute_objective_terms(
     dict[str, float]
         Keys include ``"base"``, ``"boundary"``, ``"penalty"``,
         ``"cost_threshold"``, ``"probability"``, and ``"objective"``.
     """
     costs = np.asarray(problem.planning_units["cost"].values)
-    total_cost = float(np.sum(costs[selected]))
+    total_cost = _total_cost(costs, selected)
     total_boundary = compute_boundary(problem, selected, pu_index)
     achieved = _achieved if _achieved is not None else _compute_achieved(
         problem, selected, pu_index,
     )
     penalty = compute_penalty(problem, selected, pu_index, _achieved=achieved)
@@ -469,16 +557,28 @@ def build_solution(
 ) -> Solution:
     """Build a complete Solution from a selection array.
 
     ``space_spec`` (a :class:`~pymarxan.adequacy.model.SpaceSpec`) configures the attribute space
     for post-hoc space/adequacy reporting; defaults to geographic centroids when omitted.
+
+    A float64 ``selected`` (proportions) is evaluated fraction-linearly (spec §5.3); with
+    clumping, separation, space targets or PROBMODE 3 it raises ``ValueError``.
     """
+    if _is_fractional(selected):
+        conflicts = _nonbinary_conflicts(problem)
+        if conflicts:
+            raise ValueError(
+                "a selection holding proportion decisions cannot be evaluated with "
+                + "; ".join(conflicts)
+                + ": these are only defined for binary selections. Solve with "
+                "decisions='binary' or convert with binarize_solution()."
+            )
     pu_ids = problem.planning_units["id"].tolist()
     pu_index = {pid: i for i, pid in enumerate(pu_ids)}
 
     costs = np.asarray(problem.planning_units["cost"].values)
-    total_cost = float(np.sum(costs[selected]))
+    total_cost = _total_cost(costs, selected)
     total_boundary = compute_boundary(problem, selected, pu_index)
 
     # Compute achieved amounts once, pass to all downstream functions
     achieved = _compute_achieved(problem, selected, pu_index)
     targets_met = check_targets(problem, selected, pu_index, _achieved=achieved)
@@ -589,5 +689,26 @@ def build_solution(
         sep_shortfalls=sep_shortfalls,
         sep_penalty=sep_penalty_val,
         space_held=space_held,
         space_penalty=space_penalty_val,
     )
+
+
+def binarize_solution(
+    problem: ConservationProblem, solution: Solution, threshold: float = 0.5,
+) -> Solution:
+    """The binary plan ``x >= threshold``, re-evaluated by ``build_solution`` (spec §5.2).
+
+    The only rounding path: binary input is returned unchanged (the same object).
+    ``0 < threshold <= 1``, else ``ValueError``. ``metadata`` keeps the solution's entries
+    plus ``"binarized_threshold"``.
+    """
+    if not (isinstance(threshold, int | float) and not isinstance(threshold, bool)
+            and math.isfinite(threshold) and 0.0 < threshold <= 1.0):
+        raise ValueError(f"threshold must be in (0, 1], got {threshold!r}")
+    if not solution.is_fractional:
+        return solution
+    mask = np.asarray(solution.selected) >= threshold
+    return build_solution(
+        problem, mask, blm=float(problem.parameters.get("BLM", 0.0)),
+        metadata={**solution.metadata, "binarized_threshold": threshold},
+    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/fractional_fixtures.py tests/pymarxan/solvers/test_fractional_evaluators.py -q`
Expected: 20 passed

- [ ] **Step 5: Gate**

Run: `~/.local/bin/ruff check src tests && /home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/ -q -m "not bench" -p no:cacheprovider`
Expected: ruff and mypy clean; the full suite passes with no golden regenerated (prototype final: 3444 passed, 61 skipped, 20 deselected at Task 10).

- [ ] **Step 6: Commit**

```bash
git add src/pymarxan/solvers/__init__.py src/pymarxan/solvers/_tolerance.py src/pymarxan/solvers/utils.py tests/pymarxan/fractional_fixtures.py tests/pymarxan/solvers/test_fractional_evaluators.py
git commit -m "feat(stage4): T3 fraction-linear evaluators, target tolerance, binarize_solution

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AZEiop9qqxVCrMHKqSDYP8"
```

---

### Task 4: `MIPSolver` proportion and semicontinuous decisions

**Files:**
- Modify: `src/pymarxan/solvers/_mip_objectives.py`
- Modify: `src/pymarxan/solvers/mip_solver.py`
- Modify: `tests/pymarxan/fractional_fixtures.py`
- Create: `tests/pymarxan/solvers/test_mip_decisions.py`

**Interfaces:**
- Consumes: Task 3's `target_tol`, `_nonbinary_conflicts`, fraction-aware `build_solution`.
- Produces (`solvers/mip_solver.py`): `DECISIONS`, `ZERO_TOL = 1e-9`, `SNAP_TOL = 1e-6`, `_validate_decisions(decisions, upper, lower) -> tuple[str, float, float]`, `_reject_nonbinary(problem, config, decisions) -> None`, `_extract_selected(x_values, s_values, decisions, upper, lower) -> np.ndarray`, `_on_rule(x)`; `MIPSolver(..., decisions="binary", upper=1.0, lower=0.0)` with public `decisions`/`upper`/`lower`; `_BaseModel` / `_BuiltModel` gain `s: dict[int, LpVariable]`, `decisions`, `upper`, `lower`, and `_BaseModel.extract() -> np.ndarray`; `_and_rule` → `min`, `_src_without_sink_rule` → `max(0, ·)`. `solvers/_mip_objectives.py`: `ObjectiveContext.fractional: bool = False`, `_met_rule(term, fractional=False)`, `_slack_rule(term, fractional=False)`. Fixtures module gains `with_conn`, `Case`, `rejected_cases`, `ONE` (Task 6 and 8 import them).

- [ ] **Step 1: Write the failing tests**

Modify `tests/pymarxan/fractional_fixtures.py` (apply this diff):

```diff
diff --git a/tests/pymarxan/fractional_fixtures.py b/tests/pymarxan/fractional_fixtures.py
index d11bc9e..dd9bd88 100644
--- a/tests/pymarxan/fractional_fixtures.py
+++ b/tests/pymarxan/fractional_fixtures.py
@@ -3,10 +3,13 @@ from __future__ import annotations
 
 import numpy as np
 import pandas as pd
 
 from pymarxan.models.problem import ConservationProblem
+from pymarxan.solvers.base import SolverConfig
+
+ONE = SolverConfig(num_solutions=1)
 
 X = np.array([0.6, 0.2, 1.0])
 PU_INDEX = {1: 0, 2: 1, 3: 2}
 
 
@@ -26,5 +29,37 @@ def three_pu(**params: float) -> ConservationProblem:
     conn = pd.DataFrame({"id1": [1, 2, 1], "id2": [2, 1, 3], "value": [3.0, 1.0, 2.0]})
     return ConservationProblem(
         planning_units=pu, features=feat, pu_vs_features=pvf, boundary=bnd,
         parameters={"BLM": 1.0, "CONNECTIVITY_WEIGHT": 2.0, **params}, connectivity=conn,
     )
+
+
+def with_conn(problem: ConservationProblem, values: list[float] | None = None,
+              asymmetric: bool = False, weight: float = 2.0) -> ConservationProblem:
+    conn = pd.DataFrame({
+        "id1": [1, 2, 3, 2, 5], "id2": [2, 1, 4, 5, 6],
+        "value": values or [4.0, 3.0, 2.5, 6.0, 1.5],
+    })
+    return problem.copy_with(
+        connectivity=conn,
+        parameters={**problem.parameters, "CONNECTIVITY_WEIGHT": weight,
+                    "ASYMMETRIC_CONNECTIVITY": int(asymmetric)},
+    )
+
+
+Case = tuple[ConservationProblem, SolverConfig]
+
+
+def rejected_cases(tiny: ConservationProblem) -> dict[str, Case]:
+    """Problems / configs non-binary decisions reject (spec §3.5), keyed by the cause
+    named in the error."""
+    f = tiny.features
+    return {
+        "PROBMODE 3": (tiny.copy_with(parameters={**tiny.parameters, "PROBMODE": 3}), ONE),
+        "clumping": (tiny.copy_with(features=f.assign(target2=[1.0, 0.0, 0.0])), ONE),
+        "separation": (tiny.copy_with(features=f.assign(sepnum=[2, 0, 0],
+                                                        sepdistance=[1.0, 0.0, 0.0])), ONE),
+        "space": (tiny.copy_with(features=f.assign(space_target=[0.5, 0.0, 0.0])), ONE),
+        "forbidden_selections": (tiny, SolverConfig(
+            num_solutions=1, metadata={"forbidden_selections": [[0, 1, 0, 1, 0, 1]]})),
+        "negative connectivity": (with_conn(tiny, [4.0, -3.0, 2.5, 6.0, 1.5]), ONE),
+    }
```

Create `tests/pymarxan/solvers/test_mip_decisions.py`:

```python
"""Stage 4 (spec §2-§4, §8): MIPSolver proportion / semicontinuous decisions."""
from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pulp
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers._backends import _available_backends
from pymarxan.solvers.base import Solution, SolverConfig
from pymarxan.solvers.mip_solver import (
    SNAP_TOL,
    ZERO_TOL,
    MIPSolver,
    _extract_selected,
    _validate_decisions,
)
from tests.pymarxan.fractional_fixtures import rejected_cases, with_conn
from tests.pymarxan.lp_canon import captured_models

ONE = SolverConfig(num_solutions=1)


def _solve(problem: ConservationProblem, config: SolverConfig = ONE, **kw: Any) -> list[Solution]:
    kw.setdefault("mip_backend", "cbc")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # PuLP 3.3.2
        return MIPSolver(**kw).solve(problem, config)


# Spec §8: the targets-met tests run under every installed LP backend (HiGHS in CI).
BACKENDS = [
    pytest.param(
        b, marks=pytest.mark.skipif(not _available_backends().get(b), reason=f"{b} missing"),
    )
    for b in ("cbc", "highs")
]


def _one(problem: ConservationProblem, **kw: Any) -> Solution:
    sols = _solve(problem, **kw)
    assert sols, kw
    return sols[0]


def _obj(problem: ConservationProblem, **kw: Any) -> float:
    return float(_one(problem, **kw).metadata["mip_objective_value"])


def random_problem(
    seed: int, *, scale: float = 1.0, connectivity: str | None = None, n_pu: int = 10,
) -> ConservationProblem:
    """Seeded random problem: chain boundary, targets at 20-50% of each feature's total."""
    rng = np.random.default_rng(seed)
    ids = list(range(1, n_pu + 1))
    pu = pd.DataFrame({
        "id": ids, "cost": rng.uniform(1.0, 20.0, n_pu).round(3), "status": [0] * n_pu,
    })
    rows = []
    for f in (1, 2, 3, 4):
        for p in ids:
            if rng.random() < 0.6:
                rows.append((f, p, float(rng.uniform(0.5, 10.0)) * scale))
    pvf = pd.DataFrame(rows, columns=["species", "pu", "amount"])
    totals = pvf.groupby("species")["amount"].sum()
    feat = pd.DataFrame({
        "id": [1, 2, 3, 4], "name": list("abcd"),
        "target": [float(totals.get(f, 0.0)) * rng.uniform(0.2, 0.5) for f in (1, 2, 3, 4)],
        "spf": [1.0] * 4,
    })
    bnd = pd.DataFrame({
        "id1": ids[:-1] + ids, "id2": ids[1:] + ids,
        "boundary": list(rng.uniform(0.5, 3.0, n_pu - 1)) + list(rng.uniform(0.0, 2.0, n_pu)),
    })
    params: dict[str, Any] = {"BLM": float(rng.uniform(0.1, 2.0))}
    conn = None
    if connectivity is not None:
        pairs = rng.choice(n_pu, size=(8, 2))
        pairs = np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)  # one row per pair
        conn = pd.DataFrame({
            "id1": pairs[:, 0] + 1, "id2": pairs[:, 1] + 1,
            "value": rng.uniform(0.5, 5.0, len(pairs)),
        })
        params["CONNECTIVITY_WEIGHT"] = 1.0
        params["ASYMMETRIC_CONNECTIVITY"] = int(connectivity == "asym")
    return ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=pvf, boundary=bnd,
        parameters=params, connectivity=conn,
    )


def identity_problem() -> ConservationProblem:
    """One feature per PU (identity rij), targets = amounts: the LP optimum is integral."""
    pu = pd.DataFrame({"id": [1, 2, 3, 4], "cost": [3.0, 1.0, 4.0, 1.5], "status": [0] * 4})
    feat = pd.DataFrame({
        "id": [1, 2, 3, 4], "name": list("abcd"), "target": [2.0, 5.0, 1.0, 7.0],
        "spf": [1.0] * 4,
    })
    pvf = pd.DataFrame({"species": [1, 2, 3, 4], "pu": [1, 2, 3, 4],
                        "amount": [2.0, 5.0, 1.0, 7.0]})
    return ConservationProblem(planning_units=pu, features=feat, pu_vs_features=pvf)


# --- validation ---------------------------------------------------------------------


def test_defaults_are_binary() -> None:
    s = MIPSolver()
    assert (s.decisions, s.upper, s.lower) == ("binary", 1.0, 0.0)
    s = MIPSolver(decisions="semicontinuous", upper=0.8, lower=0.2)
    assert (s.decisions, s.upper, s.lower) == ("semicontinuous", 0.8, 0.2)
    assert _validate_decisions("proportion", 1, 0) == ("proportion", 1.0, 0.0)


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"decisions": "x"}, "decisions must be one of 'binary', 'proportion', "
                             "'semicontinuous', got 'x'"),
        ({"decisions": "binary", "upper": 0.5}, "upper only applies"),
        ({"decisions": "binary", "lower": 0.5}, "lower only applies to "
                                                "decisions='semicontinuous'"),
        ({"decisions": "proportion", "lower": 0.1}, "lower only applies"),
        ({"decisions": "proportion", "upper": 1.5}, r"upper must be in \(0, 1\], got 1.5"),
        ({"decisions": "proportion", "upper": 0.0}, r"upper must be in \(0, 1\]"),
        ({"decisions": "proportion", "upper": True}, "upper must be a finite number"),
        ({"decisions": "proportion", "upper": float("nan")}, "upper must be a finite"),
        ({"decisions": "proportion", "upper": "1"}, "upper must be a finite"),
        ({"decisions": "semicontinuous", "lower": 0.0}, r"lower must be in \(0, upper\]"),
        ({"decisions": "semicontinuous", "upper": 0.5, "lower": 0.6}, "lower must be in"),
        ({"decisions": "semicontinuous", "lower": float("inf")}, "lower must be a finite"),
    ],
)
def test_validation(kw: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        MIPSolver(**kw)


# --- rejected combinations (spec §3.5) ----------------------------------------------


@pytest.mark.parametrize("decisions", ["proportion", "semicontinuous"])
@pytest.mark.parametrize(
    "cause",
    ["PROBMODE 3", "clumping", "separation", "space", "forbidden_selections",
     "negative connectivity"],
)
def test_rejected_combinations(
    tiny_problem: ConservationProblem, decisions: str, cause: str,
) -> None:
    problem, config = rejected_cases(tiny_problem)[cause]
    kw: dict[str, Any] = {"decisions": decisions}
    if decisions == "semicontinuous":
        kw["lower"] = 0.3
    solver = MIPSolver(mip_backend="cbc", **kw)
    with pytest.raises(ValueError, match=cause) as exc:
        solver.solve(problem, config)
    assert "decisions='binary'" in str(exc.value)
    with pytest.raises(ValueError, match=cause):
        solver.build_model(problem, config)


def test_rejection_precedes_strategy_errors(tiny_problem: ConservationProblem) -> None:
    """The gate sits above the NotImplementedError strategy checks."""
    p = tiny_problem.copy_with(parameters={**tiny_problem.parameters, "PROBMODE": 3})
    with pytest.raises(ValueError, match="PROBMODE 3"):
        MIPSolver(decisions="proportion", mip_chance_strategy="socp").build_model(p)


def test_binary_allows_the_rejected_combinations(tiny_problem: ConservationProblem) -> None:
    problem, config = rejected_cases(tiny_problem)["negative connectivity"]
    assert _solve(problem, config)
    problem, config = rejected_cases(tiny_problem)["forbidden_selections"]
    assert _solve(problem, config)


def test_negative_connectivity_ignored_without_weight(tiny_problem: ConservationProblem) -> None:
    p = with_conn(tiny_problem, [4.0, -3.0, 2.5, 6.0, 1.5], weight=0.0)
    assert _one(p, decisions="proportion").is_fractional


# --- model shape --------------------------------------------------------------------


def test_proportion_model_variables(tiny_problem: ConservationProblem) -> None:
    p = with_conn(tiny_problem)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        model = MIPSolver(decisions="proportion", upper=0.7).build_model(p)
    v = {var.name: var for var in model.variables()}
    assert v["x_1"].cat == "Continuous"
    assert (v["x_1"].lowBound, v["x_1"].upBound) == (0, 0.7)
    assert v["zc_1_2"].cat == "Continuous"
    assert (v["zc_1_2"].lowBound, v["zc_1_2"].upBound) == (0, 1)
    assert not any(n.startswith("s_") for n in v)


def test_semicontinuous_model_rows(tiny_problem: ConservationProblem) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        model = MIPSolver(decisions="semicontinuous", upper=0.9, lower=0.3).build_model(
            tiny_problem,
        )
        rows = dict(model.constraints)
    v = {var.name: var for var in model.variables()}
    assert v["s_1"].cat == "Integer" and (v["s_1"].lowBound, v["s_1"].upBound) == (0, 1)
    ub, lb = rows["sc_ub_1"], rows["sc_lb_1"]
    assert ub.sense == pulp.LpConstraintLE and lb.sense == pulp.LpConstraintGE
    assert {k.name: a for k, a in ub.items()} == {"x_1": 1, "s_1": -0.9}
    assert {k.name: a for k, a in lb.items()} == {"x_1": 1, "s_1": -0.3}


# --- solves -------------------------------------------------------------------------


def test_proportion_solution_is_float(tiny_problem: ConservationProblem) -> None:
    s = _one(tiny_problem, decisions="proportion")
    assert s.is_fractional
    assert s.selected.dtype == np.float64
    assert ((s.selected >= 0) & (s.selected <= 1)).all()
    assert _obj(tiny_problem, decisions="proportion") == pytest.approx(35.583, abs=1e-3)


def test_lp_bound_tiny(tiny_problem: ConservationProblem) -> None:
    lp = _obj(tiny_problem, decisions="proportion")
    assert lp <= _obj(tiny_problem) + 1e-6
    assert _obj(tiny_problem) == pytest.approx(43.0)  # the binary BLM-1 optimum


@pytest.mark.parametrize("seed", range(21))
def test_lp_bound_random(seed: int) -> None:
    connectivity = ("sym", "asym")[seed % 2]
    p = random_problem(seed, connectivity=connectivity)
    binary = _obj(p)
    assert _obj(p, decisions="proportion") <= binary + 1e-6 * max(1.0, abs(binary))


def test_integral_lp_fixture() -> None:
    p = identity_problem()
    s = _one(p, decisions="proportion")
    assert set(s.selected.tolist()) <= {0.0, 1.0}
    assert s.metadata["mip_objective_value"] == pytest.approx(_obj(p))


@pytest.mark.parametrize("lower", [0.3, 0.6, 0.9])
def test_semicontinuous_floor(tiny_problem: ConservationProblem, lower: float) -> None:
    s = _one(tiny_problem, decisions="semicontinuous", lower=lower)
    x = s.selected
    assert s.is_fractional
    assert ((x == 0.0) | ((x >= lower) & (x <= 1.0))).all(), x
    sc = float(s.metadata["mip_objective_value"])
    prop, binary = _obj(tiny_problem, decisions="proportion"), _obj(tiny_problem)
    assert prop - 1e-6 <= sc <= binary + 1e-6


def test_upper_cap_and_locks(tiny_problem: ConservationProblem) -> None:
    pu = tiny_problem.planning_units.copy()
    pu.loc[pu["id"] == 3, "status"] = 2
    pu.loc[pu["id"] == 6, "status"] = 3
    p = tiny_problem.copy_with(
        planning_units=pu, parameters={**tiny_problem.parameters, "MISSLEVEL": 0.8},
    )
    s = _one(p, decisions="proportion", upper=0.5)
    assert (s.selected <= 0.5).all()
    assert s.selected[2] == 0.5  # locked in: as fully as allowed
    assert s.selected[5] == 0.0
    sc = _one(p, decisions="semicontinuous", upper=0.5, lower=0.2)
    assert sc.selected[2] == 0.5 and sc.selected[5] == 0.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_targets_met_at_lp_optimum_tiny(tiny_problem: ConservationProblem, backend: str) -> None:
    s = _one(tiny_problem, decisions="proportion", mip_backend=backend)
    assert s.all_targets_met
    assert s.shortfall == 0.0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("scale", [1.0, 1e4, 1e8])
@pytest.mark.parametrize("seed", range(20))
def test_targets_met_at_lp_optimum_random(seed: int, scale: float, backend: str) -> None:
    s = _one(random_problem(100 + seed, scale=scale), decisions="proportion", mip_backend=backend)
    assert s.all_targets_met, s.targets_met
    assert s.shortfall == 0.0


def test_connectivity_reward_survives(tiny_problem: ConservationProblem) -> None:
    """Fractional z > 0 occurs, and the objective beats the same model with z binary."""
    p = with_conn(tiny_problem, weight=2.0)
    with captured_models() as models:
        s = _one(p, decisions="proportion")
    z = {v.name: v.varValue for v in models[0].variables() if v.name.startswith("zc_")}
    assert any(0.0 < (val or 0.0) < 1.0 for val in z.values()), z
    lp = float(s.metadata["mip_objective_value"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        model = MIPSolver(decisions="proportion").build_model(p)
        for v in model.variables():
            if v.name.startswith("zc_"):
                v.cat = pulp.LpInteger  # bounds are already [0, 1]
        model.solve(pulp.PULP_CBC_CMD(msg=False))
    assert lp < float(pulp.value(model.objective)) - 1e-6


@pytest.mark.parametrize(
    "kw", [{"decisions": "proportion"}, {"decisions": "proportion", "upper": 0.8},
           {"decisions": "semicontinuous", "lower": 0.4}],
)
def test_objective_agreement_min_set(tiny_problem: ConservationProblem, kw: dict) -> None:
    for p in (tiny_problem, random_problem(7), random_problem(8, scale=1e4)):
        if kw.get("upper", 1.0) < 1.0:
            p = p.copy_with(parameters={**p.parameters, "MISSLEVEL": 0.5})
        s = _one(p, **kw)
        obj = float(s.metadata["mip_objective_value"])
        tol = 1e-6 * max(1.0, abs(obj)) + SNAP_TOL * float(p.planning_units["cost"].sum())
        assert abs(s.objective - obj) <= tol


def test_infeasible_returns_empty() -> None:
    """upper 0.5 cannot reach targets equal to the full amounts: [] as for binary."""
    p = identity_problem()
    assert _solve(p, decisions="proportion", upper=0.5) == []
    assert _solve(p, decisions="semicontinuous", upper=0.5, lower=0.2) == []


def test_binary_solve_unchanged(tiny_problem: ConservationProblem) -> None:
    s = _one(tiny_problem.copy_with(parameters={**tiny_problem.parameters, "BLM": 0.0}))
    assert s.selected.dtype == bool
    assert s.selected.tolist() == [False, True, False, True, False, True]
    assert s.cost == 35.0


# --- extraction ---------------------------------------------------------------------


def test_extract_binary() -> None:
    out = _extract_selected([0.0, 1.0, None, 0.9999, 0.4], [], "binary", 1.0, 0.0)
    assert out.dtype == bool
    assert out.tolist() == [False, True, False, True, False]


def test_extract_proportion_snapping() -> None:
    vals = [None, 5e-10, 1e-8, 0.4, 0.8 - 5e-7, 0.8, 0.8 + 1e-9, 0.7]
    out = _extract_selected(vals, [], "proportion", 0.8, 0.0)
    assert out.dtype == np.float64
    assert out.tolist() == [0.0, 0.0, 1e-8, 0.4, 0.8, 0.8, 0.8, 0.7]
    assert ZERO_TOL == 1e-9 and SNAP_TOL == 1e-6


def test_extract_semicontinuous_snapping() -> None:
    vals = [0.3 - 5e-7, 0.3, 0.5, 1e-7, 1.0 - 1e-7, 0.2]
    s = [1.0, 1.0, 1.0, 0.0, 1.0, 1e-9]
    out = _extract_selected(vals, s, "semicontinuous", 1.0, 0.3)
    assert out.tolist() == [0.3, 0.3, 0.5, 0.0, 1.0, 0.0]


@pytest.mark.parametrize("seed", range(5))
def test_extract_never_lowers_except_dust_and_cap(seed: int) -> None:
    rng = np.random.default_rng(seed)
    upper, lower = 0.9, 0.25
    vals = rng.uniform(0.0, 1.0, 200)
    vals[:20] = rng.uniform(0.0, 2e-9, 20)
    s = np.ones(200)
    for decisions in ("proportion", "semicontinuous"):
        out = _extract_selected(list(vals), list(s) if decisions != "proportion" else [],
                                decisions, upper, lower if decisions != "proportion" else 0.0)
        for v, o in zip(vals, out, strict=True):
            if o < v:
                assert (v <= ZERO_TOL and o == 0.0) or (v > upper and o == upper), (v, o)
            assert 0.0 <= o <= upper
            assert not math.isnan(o)


def test_semicontinuous_locks(tiny_problem: ConservationProblem) -> None:
    """Review focus: locked in sits at upper exactly, locked out at 0, with lower < upper < 1."""
    pu = tiny_problem.planning_units.copy()
    pu.loc[pu["id"] == 3, "status"] = 2
    pu.loc[pu["id"] == 6, "status"] = 3
    p = tiny_problem.copy_with(
        planning_units=pu, parameters={**tiny_problem.parameters, "MISSLEVEL": 0.6},
    )
    s = _one(p, decisions="semicontinuous", upper=0.7, lower=0.3)
    assert s.selected[2] == 0.7
    assert s.selected[5] == 0.0
    x = s.selected
    assert ((x == 0.0) | ((x >= 0.3) & (x <= 0.7))).all(), x


def test_proportion_unwritten_pu_reads_zero(tiny_problem: ConservationProblem) -> None:
    """Review focus: a zero-cost PU in no pu_vs_features row never enters the LP (None)."""
    pu = pd.concat(
        [tiny_problem.planning_units,
         pd.DataFrame({"id": [7], "cost": [0.0], "status": [0]})],
        ignore_index=True,
    )
    p = tiny_problem.copy_with(planning_units=pu, boundary=None)
    for kw in ({"decisions": "proportion"}, {"decisions": "semicontinuous", "lower": 0.3}):
        s = _one(p, **kw)
        assert s.selected.dtype == np.float64
        assert s.selected[6] == 0.0
        assert s.all_targets_met
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/fractional_fixtures.py tests/pymarxan/solvers/test_mip_decisions.py -q`
Expected: FAIL — collection error (`ImportError` on `_extract_selected` / `DECISIONS`)

- [ ] **Step 3: Implement**

Modify `src/pymarxan/solvers/_mip_objectives.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/_mip_objectives.py b/src/pymarxan/solvers/_mip_objectives.py
index 78ac539..cb7dd7e 100644
--- a/src/pymarxan/solvers/_mip_objectives.py
+++ b/src/pymarxan/solvers/_mip_objectives.py
@@ -4,21 +4,24 @@ multi-objective machinery used by ``pymarxan.multiobjective`` (Stage 3).
 Each target is one ``TargetTerm`` (prioritizr: one row per target). Builders return an
 ``ObjectiveTerm`` in the objective's natural units and sense; the solver sets the model
 objective (``set_objective``) and adds ``COSTBUDGET`` once. The four target builders work
 on any context. ``build_min_cost``, ``build_min_boundary`` and ``build_max_wtd_sum`` are
 core-only: they read ``ObjectiveContext`` fields that ``MIPSolver`` fills from
-``_build_base`` and that zone contexts leave empty. Imports only pulp and the stdlib.
+``_build_base`` and that zone contexts leave empty. Imports only pulp, the stdlib and the
+stdlib-only ``_tolerance``.
 """
 from __future__ import annotations
 
 import warnings
 from collections.abc import Callable, Iterable, Mapping, Sequence
 from dataclasses import dataclass, field
 from typing import Any, Protocol
 
 import pulp
 
+from pymarxan.solvers._tolerance import target_tol
+
 SHORTFALL_UNITS = ("relative", "absolute")
 OBJECTIVE_ALIASES = {"max_n_targets_met": "max_features"}  # prioritizr >= 9 name
 NO_SOLUTION_STATUSES = frozenset(
     {pulp.LpStatusInfeasible, pulp.LpStatusUnbounded, pulp.LpStatusUndefined}
 )
@@ -70,28 +73,39 @@ def expr_value(expr: pulp.LpAffineExpression) -> float:
     return float(expr.constant or 0.0) + sum(
         float(a) * float(v.varValue or 0.0) for v, a in expr.items()
     )
 
 
-def _slack_rule(term: TargetTerm) -> Callable[[], float]:
-    """s = max(0, R − A)."""
-    return lambda: max(0.0, term.required - expr_value(term.achieved))
+def _slack_rule(term: TargetTerm, fractional: bool = False) -> Callable[[], float]:
+    """s = max(0, R − A); fractional: zeroed when ``<= target_tol(R)`` (Stage 4 §3.6)."""
+    if not fractional:
+        return lambda: max(0.0, term.required - expr_value(term.achieved))
+
+    def rule() -> float:
+        short = max(0.0, term.required - expr_value(term.achieved))
+        return 0.0 if short <= target_tol(term.required) else short
+
+    return rule
 
 
-def _met_rule(term: TargetTerm) -> Callable[[], float]:
-    """y = 1 iff A >= R exactly: the ``check_targets`` rule."""
-    return lambda: 1.0 if expr_value(term.achieved) >= term.required else 0.0
+def _met_rule(term: TargetTerm, fractional: bool = False) -> Callable[[], float]:
+    """y = 1 iff A >= R (fractional: A >= R − target_tol(R)): the ``check_targets`` rule."""
+    if not fractional:
+        return lambda: 1.0 if expr_value(term.achieved) >= term.required else 0.0
+    required = term.required - target_tol(term.required)
+    return lambda: 1.0 if expr_value(term.achieved) >= required else 0.0
 
 
 @dataclass
 class ObjectiveContext:
     """Shared state for building objectives on one model (spec §2).
 
     ``implied`` is the implied-value registry (spec §5): each auxiliary variable a builder
     creates is registered with a rule giving its value implied by x, after every variable
     the rule reads. ``MIPSolver`` fills the expression fields from ``_build_base``; zone
-    contexts keep the defaults.
+    contexts keep the defaults. ``fractional`` (non-binary decisions) selects the tolerant
+    met / slack rules (Stage 4 §3.6).
     """
 
     model: pulp.LpProblem
     cost_expr: pulp.LpAffineExpression
     terms: list[TargetTerm]
@@ -100,29 +114,30 @@ class ObjectiveContext:
     boundary_expr: pulp.LpAffineExpression = field(default_factory=pulp.LpAffineExpression)
     conn_expr: pulp.LpAffineExpression = field(default_factory=pulp.LpAffineExpression)
     prob_expr: pulp.LpAffineExpression = field(default_factory=pulp.LpAffineExpression)
     amount_exprs: dict[int, pulp.LpAffineExpression] = field(default_factory=dict)
     implied: list[ImpliedRule] = field(default_factory=list)
+    fractional: bool = False
     _slack: dict[str, pulp.LpVariable] = field(default_factory=dict, repr=False)
     _met: dict[str, pulp.LpVariable] = field(default_factory=dict, repr=False)
 
     def slack(self, term: TargetTerm) -> pulp.LpVariable:
         """s >= 0 with s >= R - A (constraint ``{prefix}short_{key}``), created once."""
         if term.key not in self._slack:
             s = pulp.LpVariable(f"{self.prefix}s_{term.key}", lowBound=0)
             self.model += s >= term.required - term.achieved, f"{self.prefix}short_{term.key}"
             self._slack[term.key] = s
-            self.implied.append((s, _slack_rule(term)))
+            self.implied.append((s, _slack_rule(term, self.fractional)))
         return self._slack[term.key]
 
     def met(self, term: TargetTerm) -> pulp.LpVariable:
         """Binary y with A >= R·y (constraint ``{prefix}met_{key}``), created once."""
         if term.key not in self._met:
             y = pulp.LpVariable(f"{self.prefix}y_{term.key}", cat="Binary")
             self.model += term.achieved >= term.required * y, f"{self.prefix}met_{term.key}"
             self._met[term.key] = y
-            self.implied.append((y, _met_rule(term)))
+            self.implied.append((y, _met_rule(term, self.fractional)))
         return self._met[term.key]
 
 
 def _scale(term: TargetTerm, shortfall: str) -> float:
     return 1.0 / term.required if shortfall == "relative" else 1.0
```

Modify `src/pymarxan/solvers/mip_solver.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/mip_solver.py b/src/pymarxan/solvers/mip_solver.py
index f31b14a..0bee858 100644
--- a/src/pymarxan/solvers/mip_solver.py
+++ b/src/pymarxan/solvers/mip_solver.py
@@ -1,11 +1,13 @@
 """MIP solver using PuLP for exact conservation planning optimization."""
 from __future__ import annotations
 
 import copy
+import math
+import numbers
 import warnings
-from collections.abc import Callable
+from collections.abc import Callable, Sequence
 from dataclasses import dataclass
 
 import numpy as np
 import pulp
 
@@ -40,11 +42,108 @@ from pymarxan.solvers.base import Solution, Solver, SolverConfig
 from pymarxan.solvers.presolve import (
     _emit_presolve,
     _presolve_data,
     _validate_presolve_flag,
 )
-from pymarxan.solvers.utils import build_solution
+from pymarxan.solvers.utils import _nonbinary_conflicts, build_solution
+
+DECISIONS = ("binary", "proportion", "semicontinuous")
+ZERO_TOL = 1e-9  # extracted x <= ZERO_TOL is solver dust -> 0 (spec §4)
+SNAP_TOL = 1e-6  # x within SNAP_TOL below upper (or lower) snaps up to it
+
+
+def _bound(name: str, value: object) -> float:
+    if (
+        not isinstance(value, numbers.Real) or isinstance(value, bool)
+        or not math.isfinite(float(value))
+    ):
+        raise ValueError(f"{name} must be a finite number, got {value!r}")
+    return float(value)
+
+
+def _validate_decisions(
+    decisions: str, upper: float, lower: float,
+) -> tuple[str, float, float]:
+    """``(decisions, upper, lower)`` validated per the spec §2 table; ``ValueError`` naming
+    the offending argument."""
+    if decisions not in DECISIONS:
+        raise ValueError(
+            "decisions must be one of 'binary', 'proportion', 'semicontinuous', "
+            f"got {decisions!r}"
+        )
+    up, lo = _bound("upper", upper), _bound("lower", lower)
+    if decisions != "semicontinuous" and lo != 0.0:
+        raise ValueError(f"lower only applies to decisions='semicontinuous', got {lower!r}")
+    if decisions == "binary":
+        if up != 1.0:
+            raise ValueError(
+                "upper only applies to decisions='proportion' or 'semicontinuous', "
+                f"got {upper!r}"
+            )
+        return decisions, up, lo
+    if not 0.0 < up <= 1.0:
+        raise ValueError(f"upper must be in (0, 1], got {upper!r}")
+    if decisions == "semicontinuous" and not 0.0 < lo <= up:
+        raise ValueError(f"lower must be in (0, upper], got {lower!r} (upper {up!r})")
+    return decisions, up, lo
+
+
+def _reject_nonbinary(
+    problem: ConservationProblem, config: SolverConfig, decisions: str,
+) -> None:
+    """``ValueError`` for settings that only make sense with 0/1 decisions (spec §3.5)."""
+    if decisions == "binary":
+        return
+    causes: list[str] = []
+    if int(problem.parameters.get("PROBMODE", 0)) == 3:
+        causes.append("PROBMODE 3 (Z-score chance constraints)")
+    causes += _nonbinary_conflicts(problem)
+    if config.metadata and config.metadata.get("forbidden_selections"):
+        causes.append(
+            "forbidden_selections (no-good cuts are only valid for 0/1 variables)"
+        )
+    conn = problem.connectivity
+    if (
+        float(problem.parameters.get("CONNECTIVITY_WEIGHT", 0.0)) > 0
+        and conn is not None
+        and (conn["value"].to_numpy(dtype=float) < 0).any()
+    ):
+        causes.append(
+            "negative connectivity values (the linearised term would no longer equal "
+            "min(x_i, x_j))"
+        )
+    if causes:
+        raise ValueError(
+            f"decisions={decisions!r} cannot be combined with " + "; ".join(causes)
+            + ". Use decisions='binary'."
+        )
+
+
+def _extract_selected(
+    x_values: Sequence[float | None],
+    s_values: Sequence[float | None],
+    decisions: str,
+    upper: float,
+    lower: float,
+) -> np.ndarray:
+    """PU-ordered ``varValue``s -> ``Solution.selected`` (spec §4).
+
+    Binary: ``bool(round(v))``. Otherwise float64 clipped to ``[0, upper]``; snapping only
+    raises a value or removes dust (``<= ZERO_TOL`` -> 0; within ``SNAP_TOL`` below
+    ``upper`` / ``lower`` -> that bound; semicontinuous ``round(s) == 0`` -> 0), so it never
+    takes held amount away from a target.
+    """
+    if decisions == "binary":
+        return np.array([bool(round(v or 0.0)) for v in x_values], dtype=bool)
+    out = np.clip(np.array([v or 0.0 for v in x_values], dtype=np.float64), 0.0, upper)
+    out[out <= ZERO_TOL] = 0.0
+    out[(out >= upper - SNAP_TOL) & (out < upper)] = upper
+    if decisions == "semicontinuous":
+        out[(out >= lower - SNAP_TOL) & (out < lower)] = lower
+        off = np.array([round(v or 0.0) == 0 for v in s_values], dtype=bool)
+        out[off] = 0.0
+    return out
 
 
 def _validate_mip_strategy(
     name: str,
     value: str,
@@ -190,17 +289,22 @@ def _abs_rule(a: pulp.LpVariable, b: pulp.LpVariable) -> Callable[[], float]:
     """Boundary auxiliary: |x_i − x_j|, as the abs rows bound it."""
     return lambda: abs(_value(a) - _value(b))
 
 
 def _and_rule(a: pulp.LpVariable, b: pulp.LpVariable) -> Callable[[], float]:
-    """Symmetric connectivity auxiliary: x_i·x_j."""
-    return lambda: _value(a) * _value(b)
+    """Symmetric connectivity auxiliary: min(x_i, x_j) (x_i·x_j on 0/1)."""
+    return lambda: min(_value(a), _value(b))
 
 
 def _src_without_sink_rule(i: pulp.LpVariable, j: pulp.LpVariable) -> Callable[[], float]:
-    """Asymmetric connectivity auxiliary: x_i·(1 − x_j)."""
-    return lambda: _value(i) * (1.0 - _value(j))
+    """Asymmetric connectivity auxiliary: max(0, x_i − x_j) (x_i·(1 − x_j) on 0/1)."""
+    return lambda: max(0.0, _value(i) - _value(j))
+
+
+def _on_rule(x: pulp.LpVariable) -> Callable[[], float]:
+    """Semicontinuous indicator: s_i = 1 iff x_i > 0."""
+    return lambda: 1.0 if _value(x) > 0 else 0.0
 
 
 @dataclass
 class _BaseModel:
     """Result of ``MIPSolver._build_base``: the model every objective shares (spec §2)."""
@@ -217,17 +321,30 @@ class _BaseModel:
     terms: list[TargetTerm]
     implied: list[ImpliedRule]
     probmode: int
     has_target2: bool
     has_sep_active: bool
+    s: dict[int, pulp.LpVariable]  # semicontinuous on/off binaries; empty otherwise
+    decisions: str
+    upper: float
+    lower: float
 
     def context(self) -> ObjectiveContext:
         """One ``ObjectiveContext`` sharing this base's expressions and registry."""
         return ObjectiveContext(
             self.model, self.cost_expr, self.terms, blm=self.blm,
             boundary_expr=self.boundary_expr, conn_expr=self.conn_expr,
             prob_expr=self.prob_expr, amount_exprs=self.amount_exprs, implied=self.implied,
+            fractional=self.decisions != "binary",
+        )
+
+    def extract(self) -> np.ndarray:
+        """``Solution.selected`` from the current ``varValue``s (spec §4)."""
+        return _extract_selected(
+            [self.x[pid].varValue for pid in self.pu_ids],
+            [self.s[pid].varValue for pid in self.pu_ids] if self.s else [],
+            self.decisions, self.upper, self.lower,
         )
 
 
 @dataclass
 class _BuiltModel:
@@ -239,10 +356,14 @@ class _BuiltModel:
     blm: float
     objective_expr: pulp.LpAffineExpression  # natural units
     probmode: int
     has_target2: bool
     has_sep_active: bool
+    s: dict[int, pulp.LpVariable]
+    decisions: str
+    upper: float
+    lower: float
     lex: tuple[pulp.LpAffineExpression, pulp.LpAffineExpression] | None = None
 
 
 class MIPSolver(Solver):
     """Solver that formulates the Marxan minimum-set problem as a MILP.
@@ -293,10 +414,18 @@ class MIPSolver(Solver):
     ``Solution.metadata`` records ``objective``, ``mip_objective_value`` (natural sign: a
     count for ``max_*``) and ``shortfall`` — the configured units, recorded for every
     objective but meaningful only for the two shortfall objectives. ``max_features`` /
     ``max_weighted_features`` count only features with target > 0.
 
+    ``decisions`` (Stage 4): ``"binary"`` (default), ``"proportion"`` (continuous
+    ``x ∈ [0, upper]``; prioritizr's ``add_semicontinuous_decisions(u)`` is
+    ``decisions="proportion", upper=u``) or ``"semicontinuous"`` (``x = 0`` or
+    ``lower ≤ x ≤ upper``, a MILP with one on/off binary per PU). Non-binary solutions hold
+    float64 proportions; locked-in PUs are fixed at ``upper``. PROBMODE 3, clumping,
+    separation, space targets, ``forbidden_selections`` and negative connectivity values
+    are rejected with ``ValueError`` for non-binary decisions.
+
     ``presolve_check`` (default ``False``): ``True`` runs
     :func:`~pymarxan.solvers.presolve.presolve_check` plus a scan of the built model (P10)
     in ``solve`` / ``build_model`` and emits one ``PresolveWarning``; ``"strict"`` raises
     ``ValueError`` instead. The data checks run before the model is built: a non-finite
     value (P1) raises ``ValueError`` under both, since PuLP cannot build the model.
@@ -311,11 +440,15 @@ class MIPSolver(Solver):
         mip_backend: str = "auto",
         objective: str = "min_set",
         shortfall: str = "relative",
         warn_above_pu: int | None = 200_000,
         presolve_check: bool | str = False,
+        decisions: str = "binary",
+        upper: float = 1.0,
+        lower: float = 0.0,
     ) -> None:
+        self.decisions, self.upper, self.lower = _validate_decisions(decisions, upper, lower)
         _validate_mip_strategy(
             "mip_chance_strategy", mip_chance_strategy,
             ("drop", "piecewise", "socp"),
         )
         _validate_mip_strategy(
@@ -366,10 +499,14 @@ class MIPSolver(Solver):
 
         The boundary rows are built when a boundary table exists and BLM > 0 or
         ``need_boundary``. Every auxiliary variable is registered in ``implied`` with the
         value its rows imply from x (spec §5).
         """
+        # Stage 4: settings that need 0/1 decisions fail first, before any strategy check.
+        _reject_nonbinary(problem, config, self.decisions)
+        binary = self.decisions == "binary"
+
         # PROBMODE 3 gating — see class docstring for rationale.
         probmode = int(problem.parameters.get("PROBMODE", 0))
         if probmode == 3:
             if self.mip_chance_strategy == "piecewise":
                 raise NotImplementedError(
@@ -426,20 +563,33 @@ class MIPSolver(Solver):
         implied: list[ImpliedRule] = []
 
         # Build the LP model
         model = pulp.LpProblem("MarxanMIP", pulp.LpMinimize)
 
-        # Decision variables: x_i = 1 if PU i is selected
+        # Decision variables: x_i = 1 if PU i is selected (Stage 4: the selected share)
         x = {}
+        s: dict[int, pulp.LpVariable] = {}
         pu_id_arr = problem.planning_units["id"].values
         pu_st_arr = problem.planning_units["status"].values.astype(int)
         for k in range(len(pu_id_arr)):
             pid = int(pu_id_arr[k])
             status = int(pu_st_arr[k])
-            x[pid] = pulp.LpVariable(f"x_{pid}", cat="Binary")
+            if binary:
+                x[pid] = pulp.LpVariable(f"x_{pid}", cat="Binary")
+            else:
+                x[pid] = pulp.LpVariable(
+                    f"x_{pid}", lowBound=0, upBound=self.upper, cat="Continuous",
+                )
+            if self.decisions == "semicontinuous":
+                s[pid] = pulp.LpVariable(f"s_{pid}", cat="Binary")
+                model += x[pid] - self.upper * s[pid] <= 0, f"sc_ub_{pid}"
+                model += x[pid] - self.lower * s[pid] >= 0, f"sc_lb_{pid}"
+                implied.append((s[pid], _on_rule(x[pid])))
             if status == STATUS_LOCKED_IN:
-                model += x[pid] == 1, f"locked_in_{pid}"
+                # Non-binary: "selected as fully as allowed" (prioritizr's lb = 1 is
+                # infeasible when upper < 1; documented deviation, spec §3.2).
+                model += x[pid] == (1 if binary else self.upper), f"locked_in_{pid}"
             elif status == STATUS_LOCKED_OUT:
                 model += x[pid] == 0, f"locked_out_{pid}"
 
         # Cost component
         pu_cost_arr = problem.planning_units["cost"].values.astype(float)
@@ -510,21 +660,29 @@ class MIPSolver(Solver):
                     conn_expr += conn_weight * cval * z
             else:
                 # Reward both selected via binary-AND linearization:
                 #   z_ij <= x_i, z_ij <= x_j, z_ij >= x_i + x_j - 1
                 #   objective: -conn_weight * c_ij * z_ij
-                # The first row of a duplicated (i, j) / (j, i) pair wins.
+                # The first row of a duplicated (i, j) / (j, i) pair wins. Non-binary
+                # decisions: z is continuous in [0, 1] (a binary z would be forced to 0 by
+                # fractional x); the reward drives it to min(x_i, x_j).
                 seen: set[tuple[int, int]] = set()
                 for ck in range(len(c_id1)):
                     i = int(c_id1[ck])
                     j = int(c_id2[ck])
                     key = (min(i, j), max(i, j))
                     if key in seen:
                         continue
                     seen.add(key)
                     cval = float(c_val[ck])
-                    z = pulp.LpVariable(f"zc_{key[0]}_{key[1]}", cat="Binary")
+                    if binary:
+                        z = pulp.LpVariable(f"zc_{key[0]}_{key[1]}", cat="Binary")
+                    else:
+                        z = pulp.LpVariable(
+                            f"zc_{key[0]}_{key[1]}", lowBound=0, upBound=1,
+                            cat="Continuous",
+                        )
                     model += z <= x[key[0]], f"conn_and1_{key[0]}_{key[1]}"
                     model += z <= x[key[1]], f"conn_and2_{key[0]}_{key[1]}"
                     model += (
                         z >= x[key[0]] + x[key[1]] - 1,
                         f"conn_and3_{key[0]}_{key[1]}",
@@ -592,10 +750,11 @@ class MIPSolver(Solver):
         return _BaseModel(
             model=model, x=x, pu_ids=pu_ids, blm=blm, cost_expr=cost_expr,
             boundary_expr=boundary_expr, conn_expr=conn_expr, prob_expr=prob_expr,
             amount_exprs=amount_exprs, terms=terms, implied=implied, probmode=probmode,
             has_target2=bool(has_target2), has_sep_active=bool(has_sep_active),
+            s=s, decisions=self.decisions, upper=self.upper, lower=self.lower,
         )
 
     def _build(self, problem: ConservationProblem, config: SolverConfig) -> _BuiltModel:
         """Build the PuLP model (no warnings, no solve): the base plus this objective."""
         base = self._build_base(problem, config)
@@ -639,11 +798,12 @@ class MIPSolver(Solver):
             set_objective(model, term)
             model += base.cost_expr <= float(cost_budget), "cost_budget"
         return _BuiltModel(
             model=model, x=base.x, pu_ids=base.pu_ids, blm=base.blm, objective_expr=term.expr,
             probmode=base.probmode, has_target2=base.has_target2,
-            has_sep_active=base.has_sep_active, lex=lex,
+            has_sep_active=base.has_sep_active, s=base.s, decisions=base.decisions,
+            upper=base.upper, lower=base.lower, lex=lex,
         )
 
     def _strategy_metadata(
         self, probmode: int, has_target2: bool, has_sep_active: bool,
     ) -> dict[str, str]:
@@ -707,14 +867,15 @@ class MIPSolver(Solver):
         # read as 0 (S9).
         has_values = any(pulp.value(x[pid]) is not None for pid in pu_ids)
         if status in infeasible_statuses or not has_values:
             return []
 
-        # Extract solution
-        selected = np.array(
-            [bool(round(pulp.value(x[pid]) or 0.0)) for pid in pu_ids],
-            dtype=bool,
+        # Extract solution (binary: rounded bools; otherwise snapped float64, spec §4)
+        selected = _extract_selected(
+            [x[pid].varValue for pid in pu_ids],
+            [built.s[pid].varValue for pid in pu_ids] if built.s else [],
+            built.decisions, built.upper, built.lower,
         )
 
         # build_solution populates Solution.prob_shortfalls + prob_penalty
         # automatically when PROBMODE==3. Under the 'drop' strategy, the
         # MIP solved the deterministic relaxation; the chance-constraint
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/fractional_fixtures.py tests/pymarxan/solvers/test_mip_decisions.py -q`
Expected: 135 passed, 61 skipped (HiGHS cases; they run in CI)

- [ ] **Step 5: Gate**

Run: `~/.local/bin/ruff check src tests && /home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/ -q -m "not bench" -p no:cacheprovider`
Expected: ruff and mypy clean; the full suite passes with no golden regenerated (prototype final: 3444 passed, 61 skipped, 20 deselected at Task 10).

- [ ] **Step 6: Commit**

```bash
git add src/pymarxan/solvers/_mip_objectives.py src/pymarxan/solvers/mip_solver.py tests/pymarxan/fractional_fixtures.py tests/pymarxan/solvers/test_mip_decisions.py
git commit -m "feat(stage4): T4 MIPSolver proportion and semicontinuous decisions

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AZEiop9qqxVCrMHKqSDYP8"
```

---

### Task 5: Presolve data checks scale with the decision bounds

**Files:**
- Modify: `src/pymarxan/solvers/mip_solver.py`
- Modify: `src/pymarxan/solvers/presolve.py`
- Create: `tests/pymarxan/solvers/test_presolve_decisions.py`

**Interfaces:**
- Consumes: Task 4's `MIPSolver.decisions/upper/lower`.
- Produces: `presolve_check(problem, *, decisions="binary", upper=1.0, lower=0.0) -> list[str]`, `_presolve_data(flag, problem, solver_name, *, decisions="binary", upper=1.0, lower=0.0)`, `_core_checks(problem, *, decisions, upper, lower)`; `MIPSolver.solve`/`build_model` pass their bounds. Binary messages byte-identical (Task 1 golden).

- [ ] **Step 1: Write the failing tests**

Create `tests/pymarxan/solvers/test_presolve_decisions.py`:

```python
"""Stage 4 (spec §3.7): presolve data checks scale with the decision bounds."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest

from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers import MIPSolver, presolve_check
from pymarxan.solvers.presolve import _presolve_data

DATA = Path(__file__).parent.parent.parent / "data"


def _locked(**params: Any) -> ConservationProblem:
    """simple with PU 3 (cost 20) locked in and PU 6 (cost 8) locked out."""
    p = load_project(DATA / "simple")
    pu = p.planning_units.copy()
    pu.loc[pu["id"] == 3, "status"] = 2
    pu.loc[pu["id"] == 6, "status"] = 3
    return p.copy_with(planning_units=pu, parameters={**p.parameters, **params})


def _codes(msgs: list[str]) -> list[str]:
    return [m.split(":")[0] for m in msgs]


def test_binary_defaults_unchanged() -> None:
    p = _locked(COSTBUDGET=15.0)
    assert presolve_check(p) == presolve_check(p, decisions="binary", upper=1.0, lower=0.0)
    assert "P9a" in _codes(presolve_check(p))


def test_p9a_floor_scales_with_upper() -> None:
    p = _locked(COSTBUDGET=15.0)  # 0.5 · 20 <= 15 < 20
    assert "P9a" not in _codes(presolve_check(p, decisions="proportion", upper=0.5))
    msgs = presolve_check(_locked(COSTBUDGET=9.0), decisions="proportion", upper=0.5)
    assert any(m.startswith("P9a") and "(10)" in m for m in msgs), msgs


def test_p9b_skipped_for_proportion_scaled_for_semicontinuous() -> None:
    p = load_project(DATA / "simple")
    p = p.copy_with(parameters={**p.parameters, "COSTBUDGET": 5.0})  # min free cost 8
    assert "P9b" in _codes(presolve_check(p))
    assert "P9b" not in _codes(presolve_check(p, decisions="proportion"))
    # cheapest option 0.5 · 8 = 4 <= 5
    assert "P9b" not in _codes(presolve_check(p, decisions="semicontinuous", lower=0.5))
    msgs = presolve_check(p, decisions="semicontinuous", lower=0.8)  # 6.4 > 5
    assert any(m.startswith("P9b") and "(6.4)" in m for m in msgs), msgs


def test_p11_ceiling_scales_with_upper() -> None:
    p = _locked(COSTBUDGET=40.0)  # Σ cost outside locked-out = 75
    assert "P11" not in _codes(presolve_check(p))
    msgs = presolve_check(p, decisions="proportion", upper=0.5)  # 37.5 <= 40
    assert any(m.startswith("P11") and "37.5" in m for m in msgs), msgs


def test_p6_scales_with_upper() -> None:
    p = _locked()  # f1: target 30, amount outside locked-out = 50
    assert "P6" not in _codes(presolve_check(p))
    msgs = presolve_check(p, decisions="proportion", upper=0.5)  # 30 > 25
    p6 = [m for m in msgs if m.startswith("P6")]
    assert p6 and "feature 1 target 30" in p6[0], msgs


def test_presolve_data_passes_bounds() -> None:
    p = _locked(COSTBUDGET=15.0)
    assert "P9a" in _codes(_presolve_data(True, p, "MIP"))
    assert "P9a" not in _codes(
        _presolve_data(True, p, "MIP", decisions="proportion", upper=0.5)
    )


def test_strict_mip_uses_its_bounds() -> None:
    p = _locked(COSTBUDGET=15.0, MISSLEVEL=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(ValueError, match="P9a"):
            MIPSolver(objective="max_features", presolve_check="strict").solve(p)
        sols = MIPSolver(
            objective="max_features", presolve_check="strict", decisions="proportion",
            upper=0.5,
        ).solve(p)
    assert sols and sols[0].selected[2] == 0.5
    assert sols[0].cost <= 15.0 + 1e-6
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_presolve_decisions.py -q`
Expected: FAIL — 7 failed (`TypeError: presolve_check() got an unexpected keyword argument 'decisions'` and missing scaling)

- [ ] **Step 3: Implement**

Modify `src/pymarxan/solvers/mip_solver.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/mip_solver.py b/src/pymarxan/solvers/mip_solver.py
index 0bee858..f958760 100644
--- a/src/pymarxan/solvers/mip_solver.py
+++ b/src/pymarxan/solvers/mip_solver.py
@@ -826,11 +826,14 @@ class MIPSolver(Solver):
         For ``min_penalties`` this is the pass-1 model (objective Σ SPF·shortfall).
         """
         if config is None:
             config = SolverConfig()
         _warn_if_large_mip(problem.n_planning_units, self.warn_above_pu)
-        data = _presolve_data(self.presolve_check, problem, self.name())
+        data = _presolve_data(
+            self.presolve_check, problem, self.name(),
+            decisions=self.decisions, upper=self.upper, lower=self.lower,
+        )
         built = self._build(problem, config)
         _emit_presolve(self.presolve_check, data, built.model, self.name(), stacklevel=3)
         return built.model
 
     def solve(
@@ -838,11 +841,14 @@ class MIPSolver(Solver):
     ) -> list[Solution]:
         if config is None:
             config = SolverConfig()
         # S3b: heads-up before building the model when the MILP is likely too large.
         _warn_if_large_mip(problem.n_planning_units, self.warn_above_pu)
-        data = _presolve_data(self.presolve_check, problem, self.name())
+        data = _presolve_data(
+            self.presolve_check, problem, self.name(),
+            decisions=self.decisions, upper=self.upper, lower=self.lower,
+        )
         built = self._build(problem, config)
         _emit_presolve(self.presolve_check, data, built.model, self.name(), stacklevel=3)
         model, x, pu_ids = built.model, built.x, built.pu_ids
         probmode, has_target2 = built.probmode, built.has_target2
         has_sep_active, blm = built.has_sep_active, built.blm
```

Modify `src/pymarxan/solvers/presolve.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/presolve.py b/src/pymarxan/solvers/presolve.py
index 298004f..1f59c19 100644
--- a/src/pymarxan/solvers/presolve.py
+++ b/src/pymarxan/solvers/presolve.py
@@ -42,21 +42,27 @@ def _is_nonfinite(message: str) -> bool:
     """A P1 non-finite finding: PuLP refuses such data, so no model can be built."""
     return message.startswith("P1: ") and " non-finite " in message
 
 
 def _presolve_data(
-    flag: bool | str, problem: ConservationProblem, solver_name: str,
+    flag: bool | str,
+    problem: ConservationProblem,
+    solver_name: str,
+    *,
+    decisions: str = "binary",
+    upper: float = 1.0,
+    lower: float = 0.0,
 ) -> list[str]:
     """Data checks, run before the model is built (flag off: no work, ``[]``).
 
     A non-finite P1 finding raises ``ValueError`` under both ``True`` and ``"strict"``:
     PuLP would raise an opaque ``PulpError`` while building the model anyway. Every other
     finding is returned for :func:`_emit_presolve` to report together with the model scan.
     """
     if not flag:
         return []
-    messages = presolve_check(problem)
+    messages = presolve_check(problem, decisions=decisions, upper=upper, lower=lower)
     if any(_is_nonfinite(m) for m in messages):
         raise ValueError(
             f"{solver_name}: presolve check: " + " | ".join(messages)
             + " (non-finite data: the model cannot be built)"
         )
@@ -166,11 +172,17 @@ def _budget_checks(budget: float | None, floor: float, cheapest: float | None,
         out.append(f"P11: COSTBUDGET {budget:g} reaches the total cost {ceiling:g}: "
                    "the budget does not constrain the solution")
     return out
 
 
-def presolve_check(problem: ConservationProblem) -> list[str]:
+def presolve_check(
+    problem: ConservationProblem,
+    *,
+    decisions: str = "binary",
+    upper: float = 1.0,
+    lower: float = 0.0,
+) -> list[str]:
     """Data problems likely to make a solve slow, imprecise or meaningless. Never raises.
 
     Returns ``[]`` when clean; every message starts with its code (``"P5: ..."``). When
     ``problem.validate()`` fails, returns one ``"P0: ..."`` message and skips the rest.
 
@@ -181,10 +193,16 @@ def presolve_check(problem: ConservationProblem) -> list[str]:
     cheapest selectable option; P11 COSTBUDGET at or above the total cost. P10 (model
     coefficients) needs a built model and is only reported through a solver's
     ``presolve_check=`` flag. Zone problems check ``zone_costs`` (what the zone solvers
     optimise), not ``planning_units.cost``.
 
+    ``decisions`` / ``upper`` / ``lower`` (Stage 4, as ``MIPSolver`` takes them) scale the
+    core checks: P6 compares the target with ``upper ×`` the reachable amount, P9a's floor
+    and P11's ceiling are ``upper ×`` the locked-in / not-locked-out cost, and P9b's
+    cheapest option is ``lower × min cost`` for ``"semicontinuous"`` and is skipped for
+    ``"proportion"`` (any positive budget buys a sliver). Zone checks ignore them.
+
     Through a solver's flag, a non-finite P1 value (cost, amount, target, boundary, spf,
     BLM, MISSLEVEL, COSTBUDGET, zone contribution) raises ``ValueError`` even under
     ``True``, because PuLP cannot build a model from it.
     """
     try:
@@ -196,11 +214,11 @@ def presolve_check(problem: ConservationProblem) -> list[str]:
     try:
         from pymarxan.zones.model import ZonalProblem
 
         if isinstance(problem, ZonalProblem):
             return _zone_checks(problem)
-        return _core_checks(problem)
+        return _core_checks(problem, decisions=decisions, upper=upper, lower=lower)
     except Exception as exc:  # a diagnostic must never crash the caller
         return [f"P0: presolve check failed: {exc!r}"]
 
 
 def _budget(problem: ConservationProblem) -> float | None:
@@ -253,11 +271,17 @@ def _common(problem: ConservationProblem, costs: Any, cost_label: Label,
     if ghosts:
         out.append("P5: " + _first(ghosts))
     return out
 
 
-def _core_checks(problem: ConservationProblem) -> list[str]:
+def _core_checks(
+    problem: ConservationProblem,
+    *,
+    decisions: str = "binary",
+    upper: float = 1.0,
+    lower: float = 0.0,
+) -> list[str]:
     pu = problem.planning_units
     costs = pu["cost"].to_numpy(dtype=float)
     status = (pu["status"].to_numpy(dtype=int) if "status" in pu.columns
               else np.zeros(len(pu), dtype=int))
     pvf = problem.pu_vs_features
@@ -265,30 +289,37 @@ def _core_checks(problem: ConservationProblem) -> list[str]:
     budget = _budget(problem)
     out = _common(problem, costs, _labeller(("PU",), pu["id"]), [], totals, budget)
     misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
     locked_out = {int(p) for p, s in zip(pu["id"].values, status, strict=True) if s == 3}
     avail = pvf[~pvf["pu"].isin(locked_out)].groupby("species")["amount"].sum()
+    scale = "" if upper == 1.0 else f"{upper:g} x "
     short = [
-        f"feature {int(f)} target {t * misslevel:g} exceeds its total amount "
+        f"feature {int(f)} target {t * misslevel:g} exceeds {scale}its total amount "
         f"{float(avail.get(int(f), 0.0)):g} outside locked-out units"
         for f, t in zip(problem.features["id"].values,
                         problem.features["target"].to_numpy(dtype=float), strict=True)
         if t > 0 and float(totals.get(int(f), 0.0)) > 0
-        and float(avail.get(int(f), 0.0)) < t * misslevel - 1e-9
+        and upper * float(avail.get(int(f), 0.0)) < t * misslevel - 1e-9
     ]
     if short:
         out.append("P6: " + _first(short))
     if len(status) and (status == 2).all():
         out.append("P7a: every planning unit is locked in: nothing to optimise")
     if len(status) and (status == 3).all():
         out.append("P7b: every planning unit is locked out: nothing can be selected")
     free = costs[(status == 0) | (status == 1)]
+    # Stage 4: x <= upper scales the forced cost and the ceiling; a proportion plan can buy
+    # any sliver, a semicontinuous one at least lower x a unit.
+    cheapest = (
+        None if not free.size or decisions == "proportion"
+        else float(free.min()) * (lower if decisions == "semicontinuous" else 1.0)
+    )
     out += _budget_checks(
         budget,
-        floor=float(costs[status == 2].sum()),
-        cheapest=float(free.min()) if free.size else None,
-        ceiling=float(np.clip(costs[status != 3], 0, None).sum()),
+        floor=upper * float(costs[status == 2].sum()),
+        cheapest=cheapest,
+        ceiling=upper * float(np.clip(costs[status != 3], 0, None).sum()),
     )
     return out
 
 
 def _zone_checks(problem: ZonalProblem) -> list[str]:
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_presolve_decisions.py -q`
Expected: 7 passed

- [ ] **Step 5: Gate**

Run: `~/.local/bin/ruff check src tests && /home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/ -q -m "not bench" -p no:cacheprovider`
Expected: ruff and mypy clean; the full suite passes with no golden regenerated (prototype final: 3444 passed, 61 skipped, 20 deselected at Task 10).

- [ ] **Step 6: Commit**

```bash
git add src/pymarxan/solvers/mip_solver.py src/pymarxan/solvers/presolve.py tests/pymarxan/solvers/test_presolve_decisions.py
git commit -m "feat(stage4): T5 presolve checks scale with decision bounds

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AZEiop9qqxVCrMHKqSDYP8"
```

---

### Task 6: `MultiObjectiveMIP` proportion and semicontinuous decisions

**Files:**
- Modify: `src/pymarxan/multiobjective.py`
- Create: `tests/pymarxan/test_multiobjective_decisions.py`

**Interfaces:**
- Consumes: Tasks 4–5 (`MIPSolver` kwargs, `_BaseModel.extract`, `ObjectiveContext.fractional`, presolve kwargs); fixtures `rejected_cases`, `Case`.
- Produces: `MultiObjectiveMIP(objectives, approach, ..., decisions="binary", upper=1.0, lower=0.0)` with public `decisions`/`upper`/`lower`; non-binary `rounded()` returns `dict(zip(base.pu_ids, base.extract()))`; float `selected` on returned solutions.

- [ ] **Step 1: Write the failing tests**

Create `tests/pymarxan/test_multiobjective_decisions.py`:

```python
"""Stage 4 (spec §2, §4, §8): MultiObjectiveMIP with proportion / semicontinuous decisions."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.multiobjective import (
    Hierarchical,
    MaxFeatures,
    MinCost,
    MinShortfall,
    MultiObjectiveMIP,
    WeightedSum,
)
from pymarxan.solvers.base import Solution, SolverConfig
from pymarxan.solvers.utils import check_targets
from tests.pymarxan.fractional_fixtures import rejected_cases
from tests.pymarxan.lp_canon import captured_models

DATA = Path(__file__).parent.parent / "data"


def _simple(**params: Any) -> ConservationProblem:
    p = load_project(DATA / "simple")
    return p.copy_with(parameters={**p.parameters, **params})


def _solve(mo: MultiObjectiveMIP, problem: ConservationProblem,
           config: SolverConfig | None = None) -> list[Solution]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # PuLP 3.3.2
        return mo.solve(problem, config)


def test_attributes_and_validation() -> None:
    mo = MultiObjectiveMIP([MinCost()], Hierarchical(0.0))
    assert (mo.decisions, mo.upper, mo.lower) == ("binary", 1.0, 0.0)
    mo = MultiObjectiveMIP([MinCost()], Hierarchical(0.0), decisions="semicontinuous",
                           upper=0.9, lower=0.2)
    assert (mo.decisions, mo.upper, mo.lower) == ("semicontinuous", 0.9, 0.2)
    with pytest.raises(ValueError, match="upper must be in"):
        MultiObjectiveMIP([MinCost()], Hierarchical(0.0), decisions="proportion", upper=2.0)
    with pytest.raises(ValueError, match="decisions must be one of"):
        MultiObjectiveMIP([MinCost()], Hierarchical(0.0), decisions="fraction")


@pytest.mark.parametrize(
    "cause",
    ["PROBMODE 3", "clumping", "separation", "space", "forbidden_selections",
     "negative connectivity"],
)
def test_rejected_combinations(cause: str) -> None:
    problem, config = rejected_cases(_simple())[cause]
    problem = problem.copy_with(parameters={**problem.parameters, "COSTBUDGET": 30.0})
    mo = MultiObjectiveMIP([MaxFeatures(), MinCost()], Hierarchical(0.0),
                           decisions="proportion", mip_backend="cbc")
    with pytest.raises(ValueError, match=cause):
        _solve(mo, problem, config)


def test_hierarchical_max_features_then_min_cost_proportion() -> None:
    """COSTBUDGET 30: the LP optimum meets every target on active rows (held ≈ R)."""
    p = _simple(COSTBUDGET=30.0)
    mo = MultiObjectiveMIP([MaxFeatures(), MinCost()], Hierarchical(0.0),
                           decisions="proportion", mip_backend="cbc")
    with captured_models() as models:
        (s,) = _solve(mo, p)
    assert s.is_fractional
    assert s.selected.dtype == np.float64
    assert not set(s.selected.tolist()) <= {0.0, 1.0}
    assert s.metadata["stage_reached"] == 2
    count = s.metadata["objective_values"]["max_features"]
    pu_index = {int(q): i for i, q in enumerate(p.planning_units["id"])}
    met = check_targets(p, s.selected, pu_index)
    assert count > 0
    assert count == sum(met.values())
    y = sum(v.varValue or 0.0 for v in models[-1].variables() if v.name.startswith("y_f"))
    assert count >= y - 1e-9
    assert s.cost <= 30.0 + 1e-6


def test_semicontinuous_weighted_sum_floor() -> None:
    p = _simple(COSTBUDGET=40.0)
    mo = MultiObjectiveMIP([MinShortfall(), MinCost()], WeightedSum([0.9, 0.1]),
                           decisions="semicontinuous", lower=0.4, mip_backend="cbc")
    (s,) = _solve(mo, p)
    x = s.selected
    assert s.is_fractional
    assert ((x == 0.0) | (x >= 0.4)).all(), x


def test_binary_selection_stays_bool() -> None:
    mo = MultiObjectiveMIP([MaxFeatures(), MinCost()], Hierarchical(0.0), mip_backend="cbc")
    (s,) = _solve(mo, _simple(COSTBUDGET=30.0))
    assert s.selected.dtype == bool


def test_presolve_uses_bounds() -> None:
    p = _simple(COSTBUDGET=15.0)
    pu = p.planning_units.copy()
    pu.loc[pu["id"] == 3, "status"] = 2
    p = p.copy_with(planning_units=pu)
    strict = {"presolve_check": "strict", "mip_backend": "cbc"}
    with pytest.raises(ValueError, match="P9a"):
        _solve(MultiObjectiveMIP([MaxFeatures(), MinCost()], Hierarchical(0.0), **strict), p)
    (s,) = _solve(MultiObjectiveMIP([MaxFeatures(), MinCost()], Hierarchical(0.0),
                                    decisions="proportion", upper=0.5, **strict), p)
    assert s.selected[2] == 0.5
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/test_multiobjective_decisions.py -q`
Expected: FAIL — 10 failed, 1 passed (`TypeError` on `decisions=`; integral selections)

- [ ] **Step 3: Implement**

Modify `src/pymarxan/multiobjective.py` (apply this diff):

```diff
diff --git a/src/pymarxan/multiobjective.py b/src/pymarxan/multiobjective.py
index c3c0baa..a034c2a 100644
--- a/src/pymarxan/multiobjective.py
+++ b/src/pymarxan/multiobjective.py
@@ -455,10 +455,16 @@ class MultiObjectiveMIP:
     ignored. ``MIP_TIME_LIMIT`` applies per stage and row, so a hierarchical sweep takes up
     to rows × n × T; with ``MIP_GAP`` > 0 or a time limit a stage's v is an incumbent's
     value. Stage 1 (or a weighted-sum row) with no solution raises ``RuntimeError``; a later
     stage with no solution returns the previous stage's plan with a ``UserWarning``.
 
+    ``decisions`` / ``upper`` / ``lower`` are ``MIPSolver``'s (Stage 4): with non-binary
+    decisions selections are float64 proportions and the implied values use the snapped
+    proportions; met counts and slacks then use the tolerant ``check_targets`` rule
+    (A >= R − tol(R)), and the implied point is often infeasible at the ``1e-6`` row check,
+    so caps fall back to the solver's value (by design).
+
     Met counts in ``objective_values`` follow the exact ``check_targets`` rule (A >= R), so
     on data at a target's tolerance boundary (e.g. amounts 0.7 + 0.1 against 0.8) the
     reported count can be below the value a hierarchical cap kept, which the solver reached
     within its tolerance. When the implied point of a stage's plan violates a model row
     (including an earlier stage's cap), that stage's cap falls back to the solver's own
@@ -474,10 +480,13 @@ class MultiObjectiveMIP:
         mip_chance_strategy: str = "drop",
         mip_clump_strategy: str = "drop",
         mip_sep_strategy: str = "drop",
         warn_above_pu: int | None = 200_000,
         presolve_check: bool | str = False,
+        decisions: str = "binary",
+        upper: float = 1.0,
+        lower: float = 0.0,
     ) -> None:
         specs = list(objectives)
         if not specs:
             raise ValueError("objectives must hold at least one ObjectiveSpec")
         for spec in specs:
@@ -495,15 +504,19 @@ class MultiObjectiveMIP:
                 f"approach must be Hierarchical or WeightedSum, got {type(approach).__name__}"
             )
         self.objectives = specs
         self.approach = approach
         self.rows = approach.rows(len(specs))
-        # Reuses MIPSolver's strategy, backend and presolve-flag validation.
+        # Reuses MIPSolver's strategy, backend, presolve-flag and decisions validation.
         self._mip = MIPSolver(
             mip_backend=mip_backend, mip_chance_strategy=mip_chance_strategy,
             mip_clump_strategy=mip_clump_strategy, mip_sep_strategy=mip_sep_strategy,
             warn_above_pu=warn_above_pu, presolve_check=presolve_check,
+            decisions=decisions, upper=upper, lower=lower,
+        )
+        self.decisions, self.upper, self.lower = (
+            self._mip.decisions, self._mip.upper, self._mip.lower,
         )
 
     def name(self) -> str:
         return "MultiObjectiveMIP"
 
@@ -545,11 +558,14 @@ class MultiObjectiveMIP:
         if config is None:
             config = SolverConfig()
         self._check(problem)
         mip = self._mip
         _warn_if_large_mip(problem.n_planning_units, mip.warn_above_pu)
-        data = _presolve_data(mip.presolve_check, problem, self.name())
+        data = _presolve_data(
+            mip.presolve_check, problem, self.name(),
+            decisions=mip.decisions, upper=mip.upper, lower=mip.lower,
+        )
         need_boundary = any(isinstance(s, MinBoundary) for s in self.objectives)
         base = mip._build_base(problem, config, need_boundary=need_boundary)
         model, x = base.model, base.x
         budget = problem.parameters.get("COSTBUDGET")
         if budget is not None:
@@ -562,11 +578,16 @@ class MultiObjectiveMIP:
         _emit_presolve(mip.presolve_check, data, model, self.name(), stacklevel=3)
 
         solver, backend = _mip_solver_for(problem, mip.mip_backend, config.verbose)
         tracked = [*x.values(), *(v for v, _ in base.implied)]
 
+        binary = mip.decisions == "binary"
+
         def rounded() -> dict[int, float]:
+            """The plan's x: rounded (binary) or the extracted, snapped proportions."""
+            if not binary:
+                return dict(zip(base.pu_ids, (float(v) for v in base.extract()), strict=True))
             return {pid: float(round(v.varValue or 0.0)) for pid, v in x.items()}
 
         def evaluate(i: int) -> float | None:
             values, feasible = evaluate_implied(
                 model, x, rounded(), base.implied, [terms[i]],
@@ -611,11 +632,14 @@ class MultiObjectiveMIP:
                     )
                 status, sol_status = int(model.status), int(model.sol_status)
                 meta.update(approach="weighted_sum", row=r, weights=list(row))
             x_rounded = rounded()
             values, _ = evaluate_implied(model, x, x_rounded, base.implied, terms)
-            selected = np.array([bool(x_rounded[pid]) for pid in base.pu_ids], dtype=bool)
+            if binary:
+                selected = np.array([bool(x_rounded[pid]) for pid in base.pu_ids], dtype=bool)
+            else:
+                selected = np.array([x_rounded[pid] for pid in base.pu_ids], dtype=np.float64)
             meta.update(
                 status=pulp.LpStatus[status], sol_status=int(sol_status),
                 objective_values=dict(zip(keys, values, strict=True)),
                 objective_senses=dict(senses), mip_backend=backend,
             )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/test_multiobjective_decisions.py -q`
Expected: 11 passed

- [ ] **Step 5: Gate**

Run: `~/.local/bin/ruff check src tests && /home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/ -q -m "not bench" -p no:cacheprovider`
Expected: ruff and mypy clean; the full suite passes with no golden regenerated (prototype final: 3444 passed, 61 skipped, 20 deselected at Task 10).

- [ ] **Step 6: Commit**

```bash
git add src/pymarxan/multiobjective.py tests/pymarxan/test_multiobjective_decisions.py
git commit -m "feat(stage4): T6 MultiObjectiveMIP proportion and semicontinuous decisions

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AZEiop9qqxVCrMHKqSDYP8"
```

---

### Task 7: Fraction-aware writers, readers and exporters

**Files:**
- Modify: `src/pymarxan/io/exporters.py`
- Modify: `src/pymarxan/io/output.py`
- Modify: `src/pymarxan/io/readers.py`
- Modify: `src/pymarxan/io/spatial_export.py`
- Modify: `src/pymarxan/io/writers.py`
- Modify: `src/pymarxan/zones/writers.py`
- Create: `tests/pymarxan/io/test_fractional_io.py`

**Interfaces:**
- Consumes: Tasks 2–3 (`is_fractional`, `_compute_achieved`, `check_targets`, `require_binary`).
- Produces: `io/output.py` `_solution_column(solution) -> list[Any]` (ints for binary, `.10g` strings for fractional) used by `write_run` / `write_best` / `write_solution_matrix`; `write_mvbest` / `export_summary_csv` held via `_compute_achieved`, met via `check_targets`; `write_ssoln` float sums; `read_ssoln` int unless a NaN-free non-integral float column; `export_solution_spatial` / `export_frequency_spatial` floats; `zones/writers.py` `zone_assignment is None` fallbacks guarded.

- [ ] **Step 1: Write the failing tests**

Create `tests/pymarxan/io/test_fractional_io.py`:

```python
"""Stage 4 (spec §6.1, §6.2): writers / readers with fractional solutions."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from pymarxan.io.exporters import export_solution_csv, export_summary_csv
from pymarxan.io.output import OutputController
from pymarxan.io.readers import load_project, read_mvbest, read_ssoln
from pymarxan.io.spatial_export import export_frequency_spatial, export_solution_spatial
from pymarxan.io.writers import write_mvbest, write_ssoln
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers import FractionalSolutionError
from pymarxan.solvers.base import Solution
from pymarxan.solvers.utils import build_solution, check_targets
from pymarxan.zones.writers import write_zone_solution

DATA = Path(__file__).parent.parent.parent / "data"
X = np.array([0.2744680851063830, 1.0, 0.0, 0.11702127659574468, 0.10638297872340426, 1.0])


@pytest.fixture
def simple() -> ConservationProblem:
    return load_project(DATA / "simple")


@pytest.fixture
def frac(simple: ConservationProblem) -> Solution:
    return build_solution(simple, X, 1.0)


@pytest.fixture
def binary(simple: ConservationProblem) -> Solution:
    return build_solution(simple, X >= 0.5, 1.0)


def test_write_run_best_matrix(
    tmp_path: Path, simple: ConservationProblem, frac: Solution, binary: Solution,
) -> None:
    out = OutputController(params={})
    out.write_run(simple, frac, 1, tmp_path)
    out.write_best(simple, binary, tmp_path)
    out.write_solution_matrix(simple, [binary, frac], tmp_path)
    run = (tmp_path / "output_r001.csv").read_text().splitlines()
    assert run[1:3] == ["1,0.2744680851", "2,1"]
    assert (tmp_path / "output_best.csv").read_text().splitlines()[1:3] == ["1,0", "2,1"]
    matrix = (tmp_path / "output_solutionsmatrix.csv").read_text().splitlines()
    assert matrix[0] == "Planning_Unit,S1,S2"
    assert matrix[1] == "1,0,0.2744680851"
    assert matrix[4] == "4,0,0.1170212766"


def test_write_mvbest_fractional(
    tmp_path: Path, simple: ConservationProblem, frac: Solution,
) -> None:
    path = tmp_path / "mvbest.csv"
    write_mvbest(simple, frac, path)
    df = read_mvbest(path)
    pu_index = {int(p): i for i, p in enumerate(simple.planning_units["id"])}
    met = check_targets(simple, frac.selected, pu_index)
    assert [bool(v) for v in df["Target_Met"]] == [met[int(f)] for f in df["Feature_ID"]]
    assert all(df["Target_Met"])  # held ≈ R at the LP optimum: met within the tolerance
    held1 = 10 * X[0] + 15 + 12 * X[3] + 8 * X[4] + 10
    assert df.loc[df["Feature_ID"] == 1, "Amount_Held"].iloc[0] == pytest.approx(held1)
    assert (df["Shortfall"] == 0.0).all()


def test_write_mvbest_follows_misslevel_change(
    tmp_path: Path, simple: ConservationProblem, frac: Solution,
) -> None:
    """Target_Met is recomputed against the problem at write time, as for binary."""
    y = frac.selected * 0.5
    half = build_solution(simple, y, 1.0)
    write_mvbest(simple, half, tmp_path / "a.csv")
    assert not read_mvbest(tmp_path / "a.csv")["Target_Met"].any()
    simple.parameters["MISSLEVEL"] = 0.5
    write_mvbest(simple, half, tmp_path / "b.csv")
    assert read_mvbest(tmp_path / "b.csv")["Target_Met"].all()


def test_ssoln_roundtrip(
    tmp_path: Path, simple: ConservationProblem, frac: Solution, binary: Solution,
) -> None:
    path = tmp_path / "ssoln.csv"
    write_ssoln(simple, [frac, binary], path)
    df = read_ssoln(path)
    assert df["Number"].dtype == np.float64
    assert df["Number"].tolist() == pytest.approx((X + (X >= 0.5)).tolist())
    write_ssoln(simple, [binary, binary], path)
    df = read_ssoln(path)
    assert df["Number"].dtype.kind == "i"
    assert df["Number"].tolist() == [0, 2, 0, 0, 0, 2]
    assert path.read_text().splitlines()[2] == "2,2"


def test_read_ssoln_integral_floats_stay_int(tmp_path: Path) -> None:
    path = tmp_path / "s.csv"
    path.write_text("Planning_Unit,Number\n1,1.0\n2,0.0\n")
    assert read_ssoln(path)["Number"].tolist() == [1, 0]
    assert read_ssoln(path)["Number"].dtype.kind == "i"
    path.write_text("Planning_Unit,Number\n1,1.5\n2,0.0\n")
    assert read_ssoln(path)["Number"].tolist() == [1.5, 0.0]


@pytest.mark.parametrize("body", ["1,1.5\n2,\n", "1,\n2,0\n", "1,abc\n2,1\n"])
def test_read_ssoln_malformed_still_raises(tmp_path: Path, body: str) -> None:
    path = tmp_path / "s.csv"
    path.write_text("Planning_Unit,Number\n" + body)
    with pytest.raises((ValueError, TypeError)):
        read_ssoln(path)


def test_export_solution_csv(tmp_path: Path, simple: ConservationProblem, frac: Solution,
                             binary: Solution) -> None:
    export_solution_csv(simple, frac, tmp_path / "f.csv")
    export_solution_csv(simple, binary, tmp_path / "b.csv")
    assert (tmp_path / "f.csv").read_text().splitlines()[1] == "1,10.0,0.2744680851"
    assert (tmp_path / "b.csv").read_text().splitlines()[1] == "1,10.0,0"


def test_export_summary_csv(tmp_path: Path, simple: ConservationProblem, frac: Solution) -> None:
    export_summary_csv(simple, frac, tmp_path / "s.csv")
    df = pd.read_csv(tmp_path / "s.csv")
    assert df["met"].all()
    assert df.loc[0, "achieved"] == pytest.approx(10 * X[0] + 15 + 12 * X[3] + 8 * X[4] + 10)


def _gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"id": [1, 2, 3, 4, 5, 6]},
        geometry=[box(i, 0, i + 1, 1) for i in range(6)], crs="EPSG:3857",
    )


def test_spatial_exports(tmp_path: Path, frac: Solution, binary: Solution) -> None:
    export_solution_spatial(_gdf(), frac, tmp_path / "f.gpkg")
    got = gpd.read_file(tmp_path / "f.gpkg")
    assert got["selected"].tolist() == pytest.approx(X.tolist(), abs=1e-10)
    export_solution_spatial(_gdf(), binary, tmp_path / "b.gpkg")
    assert gpd.read_file(tmp_path / "b.gpkg")["selected"].tolist() == [0, 1, 0, 0, 0, 1]
    export_frequency_spatial(_gdf(), [frac, binary], tmp_path / "q.gpkg")
    q = gpd.read_file(tmp_path / "q.gpkg")
    assert q["count"].tolist() == pytest.approx((X + (X >= 0.5)).tolist())
    assert q["frequency"].tolist() == pytest.approx(((X + (X >= 0.5)) / 2).tolist())


def test_zone_writer_fallback_guarded(tmp_path: Path, frac: Solution) -> None:
    with pytest.raises(FractionalSolutionError, match="write_zone_solution"):
        write_zone_solution(frac, tmp_path / "z.csv")


def test_zone_summary_fallback_guarded(tmp_path: Path) -> None:
    import warnings

    from pymarxan.zones import load_zone_project
    from pymarxan.zones.writers import write_zone_summary

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zp = load_zone_project(DATA / "zones")
    sel = np.full(zp.n_planning_units, 0.5)
    sol = Solution(selected=sel, cost=0.0, boundary=0.0, objective=0.0, targets_met={})
    with pytest.raises(FractionalSolutionError, match="write_zone_summary"):
        write_zone_summary(zp, [sol], tmp_path / "zs.csv")


def test_run_best_values_round_trip(
    tmp_path: Path, simple: ConservationProblem, binary: Solution,
) -> None:
    """Review focus: .10g loses at most 1e-9 relative; binary files keep ints."""
    rng = np.random.default_rng(3)
    x = rng.uniform(0.0, 1.0, 6)
    x[2] = 0.0
    x[4] = 1.0 / 3.0
    sol = build_solution(simple, x, 1.0)
    out = OutputController(params={})
    out.write_run(simple, sol, 1, tmp_path)
    out.write_best(simple, sol, tmp_path)
    for name in ("output_r001.csv", "output_best.csv"):
        got = pd.read_csv(tmp_path / name)["Solution"].to_numpy()
        assert got == pytest.approx(x, rel=1e-9, abs=0.0)
    out.write_run(simple, binary, 2, tmp_path)
    col = pd.read_csv(tmp_path / "output_r002.csv")["Solution"]
    assert col.dtype.kind == "i"
    assert col.tolist() == [int(v) for v in binary.selected]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/io/test_fractional_io.py -q`
Expected: FAIL — 11 failed, 3 passed (truncated ints, no guards, NaN-free float `read_ssoln`)

- [ ] **Step 3: Implement**

Modify `src/pymarxan/io/exporters.py` (apply this diff):

```diff
diff --git a/src/pymarxan/io/exporters.py b/src/pymarxan/io/exporters.py
index b0cec85..b74dc3a 100644
--- a/src/pymarxan/io/exporters.py
+++ b/src/pymarxan/io/exporters.py
@@ -17,39 +17,41 @@ def export_solution_csv(
     path: Path | str,
 ) -> None:
     """Export a solution to CSV with planning unit details."""
     pu_ids = problem.planning_units["id"].tolist()
     costs = problem.planning_units["cost"].tolist()
+    fractional = solution.is_fractional
     rows = []
     for i, (pid, cost) in enumerate(zip(pu_ids, costs)):
         rows.append({
             "planning_unit": pid,
             "cost": cost,
-            "selected": int(solution.selected[i]),
+            "selected": (
+                format(float(solution.selected[i]), ".10g") if fractional
+                else int(solution.selected[i])
+            ),
         })
     pd.DataFrame(rows).to_csv(path, index=False)
 
 
 def export_summary_csv(
     problem: ConservationProblem,
     solution: Solution,
     path: Path | str,
 ) -> None:
     """Export target achievement summary to CSV."""
+    from pymarxan.solvers.utils import _compute_achieved, check_targets
+
     pu_ids = problem.planning_units["id"].tolist()
     id_to_idx = {pid: i for i, pid in enumerate(pu_ids)}
 
-    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
-
-    # Precompute achieved per feature in one pass
-    sel_set = {pid for pid, i in id_to_idx.items() if solution.selected[i]}
-    sel_mask = problem.pu_vs_features["pu"].isin(sel_set)
-    achieved_map = (
-        problem.pu_vs_features.loc[sel_mask]
-        .groupby("species")["amount"]
-        .sum()
-        .to_dict()
+    # Held amounts (Σ r·x for proportions) and the shared met rule (Stage 4 §6.1).
+    achieved_map = _compute_achieved(problem, solution.selected, id_to_idx)
+    met_map = (
+        check_targets(problem, solution.selected, id_to_idx, _achieved=achieved_map)
+        if "target" in problem.features.columns
+        else {}
     )
 
     feat_ids = problem.features["id"].values
     feat_names = (
         problem.features["name"].values
@@ -70,11 +72,11 @@ def export_summary_csv(
         rows.append({
             "feature_id": fid,
             "feature_name": feat_names[k],
             "target": target,
             "achieved": achieved,
-            "met": achieved >= target * misslevel,
+            "met": met_map.get(fid, True),
         })
     pd.DataFrame(rows).to_csv(path, index=False)
 
 
 def export_selection_frequency_csv(
```

Modify `src/pymarxan/io/output.py` (apply this diff):

```diff
diff --git a/src/pymarxan/io/output.py b/src/pymarxan/io/output.py
index 8ec9bca..3acf00d 100644
--- a/src/pymarxan/io/output.py
+++ b/src/pymarxan/io/output.py
@@ -6,11 +6,11 @@ parameters (SAVERUN, SAVEBEST, SAVESUMM, SAVETARGMET, etc.).
 
 from __future__ import annotations
 
 from dataclasses import dataclass, field
 from pathlib import Path
-from typing import TYPE_CHECKING
+from typing import TYPE_CHECKING, Any
 
 import numpy as np
 import pandas as pd
 
 if TYPE_CHECKING:
@@ -39,10 +39,17 @@ def _should_save(param_value: int, targets_met: bool) -> bool:
     if param_value == _SAVE_IF_UNMET:
         return not targets_met
     return True
 
 
+def _solution_column(solution: Solution) -> list[Any]:
+    """Per-PU solution values: ints (binary) or ``format(x, ".10g")`` proportions."""
+    if solution.is_fractional:
+        return [format(float(v), ".10g") for v in solution.selected]
+    return [int(v) for v in solution.selected]
+
+
 @dataclass
 class OutputController:
     """Controls which Marxan output files are written.
 
     Parameters
@@ -99,12 +106,13 @@ class OutputController:
         run_number: int,
         output_dir: Path,
     ) -> None:
         """Write a per-run solution file (output_r001.csv)."""
         pu_ids = problem.planning_units["id"].tolist()
+        values = _solution_column(solution)
         rows = [
-            {"Planning_Unit": pid, "Solution": int(solution.selected[i])}
+            {"Planning_Unit": pid, "Solution": values[i]}
             for i, pid in enumerate(pu_ids)
         ]
         fname = f"{self.scenname}_r{run_number:03d}.csv"
         pd.DataFrame(rows).to_csv(output_dir / fname, index=False)
 
@@ -114,12 +122,13 @@ class OutputController:
         solution: Solution,
         output_dir: Path,
     ) -> None:
         """Write the best solution file (output_best.csv)."""
         pu_ids = problem.planning_units["id"].tolist()
+        values = _solution_column(solution)
         rows = [
-            {"Planning_Unit": pid, "Solution": int(solution.selected[i])}
+            {"Planning_Unit": pid, "Solution": values[i]}
             for i, pid in enumerate(pu_ids)
         ]
         fname = f"{self.scenname}_best.csv"
         pd.DataFrame(rows).to_csv(output_dir / fname, index=False)
 
@@ -129,14 +138,18 @@ class OutputController:
         solutions: list[Solution],
         output_dir: Path,
     ) -> None:
         """Write the solution matrix (n_pu × n_runs)."""
         pu_ids = problem.planning_units["id"].tolist()
-        matrix = np.column_stack(
-            [sol.selected.astype(int) for sol in solutions]
-        )
-        cols = {f"S{i + 1}": matrix[:, i] for i in range(len(solutions))}
+        cols: dict[str, Any]
+        if any(sol.is_fractional for sol in solutions):
+            cols = {f"S{i + 1}": _solution_column(sol) for i, sol in enumerate(solutions)}
+        else:
+            matrix = np.column_stack(
+                [sol.selected.astype(int) for sol in solutions]
+            )
+            cols = {f"S{i + 1}": matrix[:, i] for i in range(len(solutions))}
         df = pd.DataFrame({"Planning_Unit": pu_ids, **cols})
         fname = f"{self.scenname}_solutionsmatrix.csv"
         df.to_csv(output_dir / fname, index=False)
 
     def write_outputs(
```

Modify `src/pymarxan/io/readers.py` (apply this diff):

```diff
diff --git a/src/pymarxan/io/readers.py b/src/pymarxan/io/readers.py
index 3f0ac39..f9185a3 100644
--- a/src/pymarxan/io/readers.py
+++ b/src/pymarxan/io/readers.py
@@ -383,11 +383,13 @@ def read_mvbest(path: str | Path) -> pd.DataFrame:
 
 
 def read_ssoln(path: str | Path) -> pd.DataFrame:
     """Read a Marxan output summed solution (out_ssoln.csv) file.
 
-    Casts ``Planning_Unit`` and ``Number`` to int.
+    Casts ``Planning_Unit`` to int, and ``Number`` to int unless it holds well-formed
+    non-integral floats (sums of proportion decisions), which stay float. A missing or
+    non-numeric ``Number`` raises, as the int cast always did.
 
     Parameters
     ----------
     path : str | Path
         Path to out_ssoln.csv.
@@ -397,11 +399,17 @@ def read_ssoln(path: str | Path) -> pd.DataFrame:
     pd.DataFrame
         Summed solution data.
     """
     df = pd.read_csv(path)
     df["Planning_Unit"] = df["Planning_Unit"].astype(int)
-    df["Number"] = df["Number"].astype(int)
+    number = df["Number"]
+    fractional = (
+        number.dtype.kind == "f" and bool(number.notna().all())
+        and not bool((number % 1 == 0).all())
+    )
+    # Malformed data (NaN, text) still raises from astype(int), as before Stage 4.
+    df["Number"] = number if fractional else number.astype(int)
     return df
 
 
 def read_sum(path: str | Path) -> pd.DataFrame:
     """Read a Marxan output summary (out_sum.csv) file.
```

Modify `src/pymarxan/io/spatial_export.py` (apply this diff):

```diff
diff --git a/src/pymarxan/io/spatial_export.py b/src/pymarxan/io/spatial_export.py
index 57e22be..d89238c 100644
--- a/src/pymarxan/io/spatial_export.py
+++ b/src/pymarxan/io/spatial_export.py
@@ -20,18 +20,22 @@ def export_solution_spatial(
     Parameters
     ----------
     planning_units : gpd.GeoDataFrame
         Must have ``id`` and ``geometry`` columns.
     solution : Solution
-        Solver result with ``selected`` boolean array.
+        Solver result with ``selected`` boolean array (or float proportions, written as a
+        float column rounded to 10 significant digits).
     path : Path or str
         Output file path.
     driver : str
         OGR driver name: ``"GPKG"`` (default) or ``"ESRI Shapefile"``.
     """
     gdf = planning_units.copy()
-    gdf["selected"] = solution.selected.astype(int)
+    if solution.is_fractional:
+        gdf["selected"] = [float(format(float(v), ".10g")) for v in solution.selected]
+    else:
+        gdf["selected"] = solution.selected.astype(int)
     gdf.to_file(str(path), driver=driver)
 
 
 def export_frequency_spatial(
     planning_units: gpd.GeoDataFrame,
@@ -58,13 +62,15 @@ def export_frequency_spatial(
         gdf["frequency"] = 0.0
         gdf["count"] = 0
         gdf.to_file(str(path), driver=driver)
         return
 
-    counts = np.zeros(len(planning_units), dtype=int)
+    # Proportions: count = Σ x (float), frequency = mean of x.
+    fractional = any(sol.is_fractional for sol in solutions)
+    counts = np.zeros(len(planning_units), dtype=np.float64 if fractional else int)
     for sol in solutions:
-        counts += sol.selected.astype(int)
+        counts += sol.selected.astype(np.float64 if fractional else int)
 
     gdf = planning_units.copy()
     gdf["count"] = counts
     gdf["frequency"] = counts / n_solutions
     gdf.to_file(str(path), driver=driver)
```

Modify `src/pymarxan/io/writers.py` (apply this diff):

```diff
diff --git a/src/pymarxan/io/writers.py b/src/pymarxan/io/writers.py
index d63cb8f..37caedf 100644
--- a/src/pymarxan/io/writers.py
+++ b/src/pymarxan/io/writers.py
@@ -190,25 +190,24 @@ def write_mvbest(
     solution : Solution
         The best solution to report on.
     path : str | Path
         Output file path.
     """
-    from pymarxan.solvers.utils import compute_feature_shortfalls
+    from pymarxan.solvers.utils import (
+        _compute_achieved,
+        check_targets,
+        compute_feature_shortfalls,
+    )
 
     pu_ids = problem.planning_units["id"].tolist()
     pu_index = {pid: i for i, pid in enumerate(pu_ids)}
-    misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
-    shortfalls = compute_feature_shortfalls(problem, solution.selected, pu_index)
-
-    # Precompute achieved amount per feature for selected PUs in one pass
-    sel_set = {pid for pid, i in pu_index.items() if solution.selected[i]}
-    sel_mask = problem.pu_vs_features["pu"].isin(sel_set)
-    achieved_map = (
-        problem.pu_vs_features.loc[sel_mask]
-        .groupby("species")["amount"]
-        .sum()
-        .to_dict()
+    # Held amounts (Σ r·x for proportions) and the shared met rule (exact for binary,
+    # tolerant for fractional), recomputed against the current problem (Stage 4 §6.1).
+    achieved_map = _compute_achieved(problem, solution.selected, pu_index)
+    met_map = check_targets(problem, solution.selected, pu_index, _achieved=achieved_map)
+    shortfalls = compute_feature_shortfalls(
+        problem, solution.selected, pu_index, _achieved=achieved_map,
     )
 
     feat_ids = problem.features["id"].values
     feat_names = problem.features["name"].values
     feat_targets = problem.features["target"].values.astype(float)
@@ -225,11 +224,11 @@ def write_mvbest(
     for k in range(len(feat_ids)):
         fid = int(feat_ids[k])
         target = float(feat_targets[k])
         amount_held = achieved_map.get(fid, 0.0)
         shortfall = shortfalls.get(fid, 0.0)
-        target_met = amount_held >= target * misslevel
+        target_met = met_map[fid]
         row = {
             "Feature_ID": fid,
             "Feature_Name": feat_names[k],
             "Target": target,
             "Amount_Held": amount_held,
@@ -260,11 +259,11 @@ def write_ssoln(
     path: str | Path,
 ) -> None:
     """Write summed solution showing how many times each PU was selected.
 
     Produces a CSV with one row per planning unit, counting how many
-    solutions included it.
+    solutions included it (Σ x as floats when any solution holds proportions).
 
     Columns: Planning_Unit, Number.
 
     Parameters
     ----------
@@ -274,10 +273,21 @@ def write_ssoln(
         All solutions to summarize.
     path : str | Path
         Output file path.
     """
     pu_ids = problem.planning_units["id"].tolist()
+    if any(sol.is_fractional for sol in solutions):
+        sums = np.zeros(len(pu_ids), dtype=np.float64)
+        for sol in solutions:
+            sums += sol.selected.astype(np.float64)
+        rows = [
+            {"Planning_Unit": pu_id, "Number": float(total)}
+            for pu_id, total in zip(pu_ids, sums)
+        ]
+        pd.DataFrame(rows).to_csv(path, index=False)
+        return
+
     counts = np.zeros(len(pu_ids), dtype=int)
 
     for sol in solutions:
         counts += sol.selected.astype(int)
```

Modify `src/pymarxan/zones/writers.py` (apply this diff):

```diff
diff --git a/src/pymarxan/zones/writers.py b/src/pymarxan/zones/writers.py
index 3b5bf47..998eef6 100644
--- a/src/pymarxan/zones/writers.py
+++ b/src/pymarxan/zones/writers.py
@@ -5,10 +5,11 @@ from pathlib import Path
 from typing import TYPE_CHECKING
 
 import numpy as np
 import pandas as pd
 
+from pymarxan.solvers.base import require_binary
 from pymarxan.zones.objective import (
     check_overall_targets,
     check_zone_targets,
     compute_overall_achieved,
     compute_zone_achieved,
@@ -102,10 +103,11 @@ def write_zone_solution(
         A solution with ``zone_assignment`` populated.
     path : str | Path
         Output file path.
     """
     if solution.zone_assignment is None:
+        require_binary(solution, "write_zone_solution")
         zones = np.where(solution.selected, 1, 0)
     else:
         zones = solution.zone_assignment
 
     rows = [
@@ -141,10 +143,13 @@ def write_zone_summary(
 
     Solutions without ``zone_assignment`` are read as "selected PUs in the first zone".
     """
     zone_ids = sorted(int(z) for z in problem.zones["id"].tolist())
     first_zone = zone_ids[0] if zone_ids else 1
+    for sol in solutions:
+        if sol.zone_assignment is None:
+            require_binary(sol, "write_zone_summary")
     assignments = [
         np.asarray(sol.zone_assignment, dtype=int)
         if sol.zone_assignment is not None
         else np.where(np.asarray(sol.selected, dtype=bool), first_zone, 0)
         for sol in solutions
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/io/test_fractional_io.py -q`
Expected: 14 passed

- [ ] **Step 5: Gate**

Run: `~/.local/bin/ruff check src tests && /home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/ -q -m "not bench" -p no:cacheprovider`
Expected: ruff and mypy clean; the full suite passes with no golden regenerated (prototype final: 3444 passed, 61 skipped, 20 deselected at Task 10).

- [ ] **Step 6: Commit**

```bash
git add src/pymarxan/io/exporters.py src/pymarxan/io/output.py src/pymarxan/io/readers.py src/pymarxan/io/spatial_export.py src/pymarxan/io/writers.py src/pymarxan/zones/writers.py tests/pymarxan/io/test_fractional_io.py
git commit -m "feat(stage4): T7 fraction-aware writers, readers and exporters

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AZEiop9qqxVCrMHKqSDYP8"
```

---

### Task 8: Fraction-aware analysis, binary-only guards, reader completeness test

**Files:**
- Modify: `src/pymarxan/analysis/equity.py`
- Modify: `src/pymarxan/analysis/portfolio.py`
- Modify: `src/pymarxan/analysis/portfolio_cuts.py`
- Modify: `src/pymarxan/analysis/posthoc_clusters.py`
- Modify: `src/pymarxan/analysis/rank_importance.py`
- Modify: `src/pymarxan/analysis/replacement_cost.py`
- Modify: `src/pymarxan/analysis/representation.py`
- Modify: `src/pymarxan/analysis/robustness.py`
- Modify: `src/pymarxan/analysis/scenarios.py`
- Modify: `src/pymarxan/analysis/selection_freq.py`
- Modify: `src/pymarxan/phylo/diversity.py`
- Modify: `src/pymarxan/solvers/base.py`
- Modify: `src/pymarxan/solvers/iterative_improvement.py`
- Create: `tests/pymarxan/analysis/test_fractional_analysis.py`
- Create: `tests/pymarxan/calibration/test_blm_fractional.py`
- Create: `tests/pymarxan/test_fractional_readers.py`

**Interfaces:**
- Consumes: Tasks 2–4 (`require_binary`, `_require_binary_solver`, `_compute_achieved`, `MIPSolver(decisions=...)`), fixtures.
- Produces: fraction-aware `compute_representation`, `compute_equity`, `compute_selection_frequency`, `evaluate_plans_across_scenarios`; guards in `portfolio_cuts` / `replacement_cost` (solver), `solution_diversity`, `overlap_matrix`, `posthoc_clusters`, `rank_importance`, `compute_phylogenetic_diversity`, `IterativeImprovementSolver.improve` (and so `run_mode` chains); `tests/pymarxan/test_fractional_readers.py` with tables `FRACTION_AWARE`, `GUARDED`, `INTERNAL_OR_COUNT`, `MIXED` (Task 9 removes its Shiny skip).

- [ ] **Step 1: Write the failing tests**

Create `tests/pymarxan/analysis/test_fractional_analysis.py`:

```python
"""Stage 4 (spec §6.1, §6.2, §8): analysis readers with fractional solutions."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from pymarxan.analysis.equity import compute_equity
from pymarxan.analysis.portfolio import selection_frequency, solution_diversity
from pymarxan.analysis.portfolio_cuts import generate_portfolio_cuts
from pymarxan.analysis.posthoc_clusters import compute_solution_clusters
from pymarxan.analysis.rank_importance import compute_rank_importance
from pymarxan.analysis.replacement_cost import compute_replacement_cost
from pymarxan.analysis.representation import compute_representation
from pymarxan.analysis.robustness import evaluate_plans_across_scenarios
from pymarxan.analysis.scenarios import ScenarioSet
from pymarxan.analysis.selection_freq import compute_selection_frequency
from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.phylo.diversity import compute_phylogenetic_diversity
from pymarxan.phylo.tree import PhylogeneticTree
from pymarxan.solvers import FractionalSolutionError, IterativeImprovementSolver, MIPSolver
from pymarxan.solvers.base import Solution, Solver, SolverConfig
from pymarxan.solvers.heuristic import HeuristicSolver
from pymarxan.solvers.run_mode import RunModePipeline
from pymarxan.solvers.utils import build_solution, compute_objective
from tests.pymarxan.fractional_fixtures import X, three_pu

DATA = Path(__file__).parent.parent.parent / "data"


@pytest.fixture
def simple() -> ConservationProblem:
    return load_project(DATA / "simple")


@pytest.fixture
def frac(simple: ConservationProblem) -> Solution:
    return build_solution(simple, np.array([0.3, 1.0, 0.0, 0.5, 0.1, 1.0]), 1.0)


# --- hand calculation (spec §8) -----------------------------------------------------


def test_representation_hand_calculation() -> None:
    p = three_pu()
    sol = build_solution(p, X, 1.0)
    rep = compute_representation(p, sol, threshold=0.3)
    # totals: f1 = 55, f2 = 12; held: f1 = 19, f2 = 8.8
    assert rep.represented_amount == pytest.approx({1: 19.0, 2: 8.8})
    assert rep.pct_represented == pytest.approx({1: 1900.0 / 55.0, 2: 880.0 / 12.0})
    assert rep.meets_threshold == {1: True, 2: True}


def test_equity_hand_calculation() -> None:
    p = three_pu()
    sol = build_solution(p, X, 1.0)
    eq = compute_equity(p, sol, {1: "a", 2: "a", 3: "b"})  # cost: a 0.6·10 + 0.2·20, b 30
    assert eq.group_values == pytest.approx({"a": 10.0, "b": 30.0})
    assert eq.group_shares == pytest.approx({"a": 0.25, "b": 0.75})
    count = compute_equity(p, sol, {1: "a", 2: "a", 3: "b"}, value="count")
    assert count.group_values == pytest.approx({"a": 0.8, "b": 1.0})


def test_binary_representation_and_equity_unchanged() -> None:
    p = three_pu()
    b = build_solution(p, X >= 0.5, 1.0)
    assert compute_representation(p, b).represented_amount == {1: 15.0, 2: 12.0}
    assert compute_equity(p, b, {1: "a", 2: "a", 3: "b"}).group_values == {"a": 10.0, "b": 30.0}


# --- frequencies --------------------------------------------------------------------


def test_selection_frequencies(simple: ConservationProblem, frac: Solution) -> None:
    b = build_solution(simple, frac.selected >= 0.5, 1.0)
    sf = compute_selection_frequency([frac, b])
    assert sf.counts.dtype == np.float64
    assert sf.counts.tolist() == pytest.approx([0.3, 2.0, 0.0, 1.5, 0.1, 2.0])
    assert sf.frequencies.tolist() == pytest.approx([0.15, 1.0, 0.0, 0.75, 0.05, 1.0])
    assert compute_selection_frequency([b, b]).counts.dtype.kind == "i"
    assert selection_frequency([frac, b]).tolist() == pytest.approx(sf.frequencies.tolist())


# --- robustness ---------------------------------------------------------------------


def test_robustness_no_silent_rounding(simple: ConservationProblem, frac: Solution) -> None:
    m, _, _ = evaluate_plans_across_scenarios({"s": simple}, {"p": frac}, blm=1.0)
    pu_index = {int(q): i for i, q in enumerate(simple.planning_units["id"])}
    assert m[0, 0] == pytest.approx(compute_objective(simple, frac.selected, pu_index, 1.0))
    assert m[0, 0] == pytest.approx(frac.objective)


def test_robustness_probmode3_scenario_rejects_fractional(
    simple: ConservationProblem, frac: Solution,
) -> None:
    pm3 = simple.copy_with(parameters={**simple.parameters, "PROBMODE": 3})
    with pytest.raises(ValueError, match="PROBMODE 3"):
        evaluate_plans_across_scenarios({"pm3": pm3}, {"p": frac})


# --- guards (spec §6.2) -------------------------------------------------------------


def test_guards_on_solutions(simple: ConservationProblem, frac: Solution) -> None:
    b = build_solution(simple, frac.selected >= 0.5, 1.0)
    with pytest.raises(FractionalSolutionError, match="solution_diversity"):
        solution_diversity([b, frac])
    scen = ScenarioSet()
    scen.add("a", b)
    scen.add("b", frac)
    with pytest.raises(FractionalSolutionError, match="overlap_matrix"):
        scen.overlap_matrix()
    with pytest.raises(FractionalSolutionError, match="compute_solution_clusters"):
        compute_solution_clusters(simple, frac)
    with pytest.raises(FractionalSolutionError, match="compute_rank_importance"):
        compute_rank_importance(simple, frac)
    tree = PhylogeneticTree.from_newick("((species_a:1,species_b:1):1,species_c:2);")
    with pytest.raises(FractionalSolutionError, match="compute_phylogenetic_diversity"):
        compute_phylogenetic_diversity(simple, frac, tree)
    with pytest.raises(FractionalSolutionError, match="improve"):
        IterativeImprovementSolver(itimptype=1).improve(simple, frac)
    with pytest.raises(FractionalSolutionError, match="improve"):
        IterativeImprovementSolver(itimptype=0).improve(simple, frac)


def test_binary_guarded_functions_still_work(simple: ConservationProblem, frac: Solution) -> None:
    b = build_solution(simple, frac.selected >= 0.5, 1.0)
    assert solution_diversity([b, b]) == 0.0
    scen = ScenarioSet()
    scen.add("a", b)
    assert scen.overlap_matrix().tolist() == [[1.0]]
    assert compute_solution_clusters(simple, b)["n_clusters"] >= 1


class _FractionalSA(Solver):
    """Stands in for a solver whose output feeds an improve() chain."""

    def solve(self, problem: ConservationProblem, config: SolverConfig | None = None
              ) -> list[Solution]:
        n = problem.n_planning_units
        return [build_solution(problem, np.full(n, 0.5), 0.0)]

    def name(self) -> str:
        return "fractional"

    def supports_zones(self) -> bool:
        return False


def test_run_mode_chain_propagates_guard(simple: ConservationProblem) -> None:
    with pytest.raises(FractionalSolutionError, match="improve"):
        RunModePipeline._run_pipeline(
            simple, 2, SolverConfig(num_solutions=1, seed=1), HeuristicSolver, _FractionalSA,
            IterativeImprovementSolver,
        )


@pytest.mark.parametrize("decisions", ["proportion", "semicontinuous"])
def test_solver_guards(simple: ConservationProblem, decisions: str) -> None:
    kw = {"lower": 0.3} if decisions == "semicontinuous" else {}
    solver = MIPSolver(decisions=decisions, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(FractionalSolutionError, match="generate_portfolio_cuts"):
            generate_portfolio_cuts(simple, solver=solver, k=2)
        with pytest.raises(FractionalSolutionError, match="compute_replacement_cost"):
            compute_replacement_cost(simple, solver=solver)
```

Create `tests/pymarxan/calibration/test_blm_fractional.py`:

```python
"""Stage 4 review focus: BLM calibration (and the Cohon frontier) with a proportion solver."""
from __future__ import annotations

import warnings
from pathlib import Path

from pymarxan.calibration.blm import calibrate_blm
from pymarxan.calibration.pareto import pareto_frontier
from pymarxan.io.readers import load_project
from pymarxan.solvers import MIPSolver

DATA = Path(__file__).parent.parent.parent / "data"


def test_calibrate_blm_with_proportion_solver() -> None:
    problem = load_project(DATA / "simple")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # PuLP 3.3.2
        result = calibrate_blm(
            problem, MIPSolver(decisions="proportion", mip_backend="cbc"),
            blm_values=[0.0, 0.5, 1.0, 2.0, 5.0, 20.0],
        )
    assert result.blm_values == [0.0, 0.5, 1.0, 2.0, 5.0, 20.0]
    assert all(s.is_fractional for s in result.solutions)
    assert all(s.all_targets_met for s in result.solutions)
    b = result.boundaries
    assert all(b2 <= b1 + 1e-6 for b1, b2 in zip(b, b[1:])), b
    assert pareto_frontier(result).solutions
```

Create `tests/pymarxan/test_fractional_readers.py`:

```python
"""Stage 4 (spec §6.4): every module reading ``Solution.selected`` / ``n_selected`` is
classified, so a new reader cannot slip in without deciding how it treats proportions.

Classification is per module. ``MIXED`` modules list their guarded functions (empty when
the module mixes fraction-aware and count-only readers); the per-function tests in
``tests/pymarxan/analysis/test_fractional_analysis.py`` and
``tests/pymarxan/io/test_fractional_io.py`` pin the split.
"""
from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
PATTERN = re.compile(r"\.selected\b|n_selected")

FRACTION_AWARE = {
    "pymarxan/analysis/equity.py",
    "pymarxan/analysis/representation.py",
    "pymarxan/analysis/robustness.py",
    "pymarxan/analysis/selection_freq.py",
    "pymarxan/io/exporters.py",
    "pymarxan/io/output.py",
    "pymarxan/io/spatial_export.py",
    "pymarxan/solvers/mip_solver.py",  # produces proportions
    "pymarxan/solvers/utils.py",
    "pymarxan_shiny/modules/mapping/solution_map.py",
    "pymarxan_shiny/modules/results/summary_table.py",
}
GUARDED = {
    "pymarxan/analysis/portfolio_cuts.py",
    "pymarxan/analysis/posthoc_clusters.py",
    "pymarxan/analysis/rank_importance.py",
    "pymarxan/analysis/replacement_cost.py",
    "pymarxan/phylo/diversity.py",
    "pymarxan/solvers/iterative_improvement.py",
    "pymarxan/zones/writers.py",
    "pymarxan_shiny/modules/mapping/comparison_map.py",
}
INTERNAL_OR_COUNT = {
    "pymarxan/calibration/sensitivity.py",
    "pymarxan/calibration/sweep.py",
    "pymarxan/solvers/base.py",
    "pymarxan/solvers/clumping.py",
    "pymarxan/solvers/separation.py",
    "pymarxan/solvers/space_state.py",
    "pymarxan_shiny/modules/results/export.py",
    "pymarxan_shiny/modules/run_control/run_panel.py",
    "pymarxan_shiny/modules/spatial_export/spatial_export.py",
}
MIXED: dict[str, tuple[str, ...]] = {
    "pymarxan/analysis/portfolio.py": ("solution_diversity",),
    "pymarxan/analysis/scenarios.py": ("ScenarioSet.overlap_matrix",),
    "pymarxan/io/writers.py": (),  # write_mvbest / write_ssoln aware, write_sum counts
}


def _readers() -> set[str]:
    return {
        path.relative_to(SRC).as_posix()
        for path in SRC.rglob("*.py")
        if PATTERN.search(path.read_text(encoding="utf-8"))
    }


def test_tables_are_disjoint() -> None:
    tables = [FRACTION_AWARE, GUARDED, INTERNAL_OR_COUNT, set(MIXED)]
    seen: set[str] = set()
    for table in tables:
        assert not (seen & table), seen & table
        seen |= table


def test_every_reader_is_classified() -> None:
    classified = FRACTION_AWARE | GUARDED | INTERNAL_OR_COUNT | set(MIXED)
    readers = _readers()
    assert sorted(readers - classified) == [], "classify new Solution.selected readers"
    assert sorted(classified - readers) == [], "stale entries: module no longer reads it"


def _module(rel: str) -> object:
    return importlib.import_module(rel.removesuffix(".py").replace("/", "."))


def test_guarded_modules_guard() -> None:
    for rel in GUARDED:
        if rel.startswith("pymarxan_shiny/"):
            continue  # the Shiny guard lands with the Shiny task (T9)
        text = (SRC / rel).read_text(encoding="utf-8")
        assert "require_binary" in text, rel


def test_mixed_guarded_functions_guard() -> None:
    for rel, names in MIXED.items():
        module = _module(rel)
        for name in names:
            obj: object = module
            for part in name.split("."):
                obj = getattr(obj, part)
            assert "require_binary" in inspect.getsource(obj), (rel, name)  # type: ignore[arg-type]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/analysis/test_fractional_analysis.py tests/pymarxan/calibration/test_blm_fractional.py tests/pymarxan/test_fractional_readers.py -q`
Expected: FAIL — 11 failed, 5 passed (no guards; boolean-cast representation / equity / frequency)

- [ ] **Step 3: Implement**

Modify `src/pymarxan/analysis/equity.py` (apply this diff):

```diff
diff --git a/src/pymarxan/analysis/equity.py b/src/pymarxan/analysis/equity.py
index d40c4ed..b1f05f2 100644
--- a/src/pymarxan/analysis/equity.py
+++ b/src/pymarxan/analysis/equity.py
@@ -90,15 +90,17 @@ def compute_equity(
             f"unknown value {value!r}; use 'cost', 'count', or a {{pu_id: value}} mapping"
         )
 
     # Initialize every group that appears, in first-seen order.
     group_values: dict[str, float] = {g: 0.0 for g in dict.fromkeys(groups.values())}
+    fractional = solution.is_fractional  # proportions: a group gets Σ x·value
     for pid, sel in zip(pu_ids, solution.selected):
         if sel:
             g = groups.get(int(pid))
             if g is not None:
-                group_values[g] += values_map.get(int(pid), 0.0)
+                v = values_map.get(int(pid), 0.0)
+                group_values[g] += v * float(sel) if fractional else v
 
     total = sum(group_values.values())
     if total > 0:
         group_shares = {g: v / total for g, v in group_values.items()}
     else:
```

Modify `src/pymarxan/analysis/portfolio.py` (apply this diff):

```diff
diff --git a/src/pymarxan/analysis/portfolio.py b/src/pymarxan/analysis/portfolio.py
index f8db865..f50141d 100644
--- a/src/pymarxan/analysis/portfolio.py
+++ b/src/pymarxan/analysis/portfolio.py
@@ -1,11 +1,11 @@
 """Portfolio analysis for collections of conservation planning solutions."""
 from __future__ import annotations
 
 import numpy as np
 
-from pymarxan.solvers.base import Solution
+from pymarxan.solvers.base import Solution, require_binary
 
 
 def _validate_non_empty(solutions: list[Solution]) -> None:
     """Raise ValueError when the portfolio is empty."""
     if not solutions:
@@ -51,10 +51,12 @@ def solution_diversity(solutions: list[Solution]) -> float:
 
     Returns the mean pairwise Jaccard distance.  If fewer than two
     solutions are provided the diversity is 0.0.
     """
     _validate_non_empty(solutions)
+    for sol in solutions:
+        require_binary(sol, "solution_diversity")
     n = len(solutions)
     if n < 2:
         return 0.0
 
     total = 0.0
```

Modify `src/pymarxan/analysis/portfolio_cuts.py` (apply this diff):

```diff
diff --git a/src/pymarxan/analysis/portfolio_cuts.py b/src/pymarxan/analysis/portfolio_cuts.py
index 39de501..916bc45 100644
--- a/src/pymarxan/analysis/portfolio_cuts.py
+++ b/src/pymarxan/analysis/portfolio_cuts.py
@@ -21,11 +21,11 @@ raising — same shape as ``Solver.solve`` returning an empty list on
 infeasibility.
 """
 from __future__ import annotations
 
 from pymarxan.models.problem import ConservationProblem
-from pymarxan.solvers.base import Solution, SolverConfig
+from pymarxan.solvers.base import Solution, SolverConfig, _require_binary_solver
 from pymarxan.solvers.mip_solver import MIPSolver
 
 
 def generate_portfolio_cuts(
     problem: ConservationProblem,
@@ -59,10 +59,11 @@ def generate_portfolio_cuts(
     """
     if not isinstance(k, int) or k < 1:
         raise ValueError(f"k must be a positive int; got {k!r}.")
     if solver is None:
         solver = MIPSolver()
+    _require_binary_solver(solver, "generate_portfolio_cuts")
     if config is None:
         config = SolverConfig(num_solutions=1)
 
     forbidden: list = []
     portfolio: list[Solution] = []
```

Modify `src/pymarxan/analysis/posthoc_clusters.py` (apply this diff):

```diff
diff --git a/src/pymarxan/analysis/posthoc_clusters.py b/src/pymarxan/analysis/posthoc_clusters.py
index c6a9c42..71d289e 100644
--- a/src/pymarxan/analysis/posthoc_clusters.py
+++ b/src/pymarxan/analysis/posthoc_clusters.py
@@ -10,11 +10,11 @@ For problems without ``problem.boundary`` (no edges), every selected
 PU is its own cluster (no adjacency information to merge them).
 """
 from __future__ import annotations
 
 from pymarxan.models.problem import ConservationProblem
-from pymarxan.solvers.base import Solution
+from pymarxan.solvers.base import Solution, require_binary
 
 
 def compute_solution_clusters(
     problem: ConservationProblem,
     solution: Solution,
@@ -41,10 +41,11 @@ def compute_solution_clusters(
     """
     import networkx as nx
 
     pu_ids = problem.planning_units["id"].astype(int).to_numpy()
     pu_id_to_idx = {int(pid): i for i, pid in enumerate(pu_ids)}
+    require_binary(solution, "compute_solution_clusters")
     selected_ids = [
         int(pid) for i, pid in enumerate(pu_ids) if solution.selected[i]
     ]
     n_selected = len(selected_ids)
```

Modify `src/pymarxan/analysis/rank_importance.py` (apply this diff):

```diff
diff --git a/src/pymarxan/analysis/rank_importance.py b/src/pymarxan/analysis/rank_importance.py
index e5c1ecd..02c3e2a 100644
--- a/src/pymarxan/analysis/rank_importance.py
+++ b/src/pymarxan/analysis/rank_importance.py
@@ -13,11 +13,11 @@ Evolution* 12(5): 869-877. https://doi.org/10.1111/2041-210X.13578
 from __future__ import annotations
 
 import numpy as np
 
 from pymarxan.models.problem import ConservationProblem
-from pymarxan.solvers.base import Solution
+from pymarxan.solvers.base import Solution, require_binary
 from pymarxan.solvers.utils import compute_objective_terms
 
 
 def compute_rank_importance(
     problem: ConservationProblem,
@@ -54,10 +54,11 @@ def compute_rank_importance(
         Selected PUs map to 1..n_selected (n_selected most important).
     """
     pu_ids = problem.planning_units["id"].astype(int).to_numpy()
     blm = float(problem.parameters.get("BLM", 0.0))
 
+    require_binary(solution, "compute_rank_importance")
     selected = solution.selected.copy()
     pu_index = {int(pid): i for i, pid in enumerate(pu_ids)}
 
     ranks: dict[int, float] = {int(pid): 0.0 for pid in pu_ids}
     n_selected = int(selected.sum())
```

Modify `src/pymarxan/analysis/replacement_cost.py` (apply this diff):

```diff
diff --git a/src/pymarxan/analysis/replacement_cost.py b/src/pymarxan/analysis/replacement_cost.py
index 5431cd5..7348514 100644
--- a/src/pymarxan/analysis/replacement_cost.py
+++ b/src/pymarxan/analysis/replacement_cost.py
@@ -24,11 +24,11 @@ import math
 
 from pymarxan.models.problem import (
     STATUS_LOCKED_OUT,
     ConservationProblem,
 )
-from pymarxan.solvers.base import SolverConfig
+from pymarxan.solvers.base import SolverConfig, _require_binary_solver
 from pymarxan.solvers.mip_solver import MIPSolver
 
 
 def compute_replacement_cost(
     problem: ConservationProblem,
@@ -61,10 +61,11 @@ def compute_replacement_cost(
         score 0. If lock-out makes the problem infeasible, the score is
         ``float('inf')``.
     """
     if solver is None:
         solver = MIPSolver()
+    _require_binary_solver(solver, "compute_replacement_cost")
 
     pu_ids = problem.planning_units["id"].astype(int).to_numpy()
     scores: dict[int, float] = {int(pid): 0.0 for pid in pu_ids}
 
     # Baseline solve.
```

Modify `src/pymarxan/analysis/representation.py` (apply this diff):

```diff
diff --git a/src/pymarxan/analysis/representation.py b/src/pymarxan/analysis/representation.py
index 7791ee3..a694ee2 100644
--- a/src/pymarxan/analysis/representation.py
+++ b/src/pymarxan/analysis/representation.py
@@ -85,11 +85,18 @@ def compute_representation(
         for pid, sel in zip(problem.planning_units["id"], solution.selected)
         if sel
     }
     puvspr = problem.pu_vs_features
     mask = puvspr["pu"].isin(selected_ids)
-    if mask.any():
+    if solution.is_fractional:
+        # Proportion decisions: represented amount Σ r·x (Stage 4 §6.1).
+        from pymarxan.solvers.utils import _compute_achieved
+
+        pu_index = {int(p): i for i, p in enumerate(problem.planning_units["id"])}
+        held = _compute_achieved(problem, solution.selected, pu_index)
+        represented_amount = {fid: float(held.get(fid, 0.0)) for fid in feature_ids}
+    elif mask.any():
         rep_totals = puvspr.loc[mask].groupby("species")["amount"].sum()
         represented_amount = {
             fid: float(rep_totals.get(fid, 0.0)) for fid in feature_ids
         }
     else:
```

Modify `src/pymarxan/analysis/robustness.py` (apply this diff):

```diff
diff --git a/src/pymarxan/analysis/robustness.py b/src/pymarxan/analysis/robustness.py
index 0998a4b..3b03b1d 100644
--- a/src/pymarxan/analysis/robustness.py
+++ b/src/pymarxan/analysis/robustness.py
@@ -129,11 +129,16 @@ def evaluate_plans_across_scenarios(
     for j, slabel in enumerate(scenario_labels):
         problem = problems[slabel]
         n_pu = len(problem.planning_units)
         pu_index = {int(pid): k for k, pid in enumerate(problem.planning_units["id"])}
         for i, plabel in enumerate(plan_labels):
-            selected = np.asarray(solutions[plabel].selected, dtype=bool)
+            # Proportions are evaluated as they are (no silent rounding; Stage 4 §6.1).
+            plan = solutions[plabel]
+            selected = (
+                np.asarray(plan.selected) if plan.is_fractional
+                else np.asarray(plan.selected, dtype=bool)
+            )
             if selected.shape[0] != n_pu:
                 raise ValueError(
                     f"plan {plabel!r} has {selected.shape[0]} selections but "
                     f"scenario {slabel!r} has {n_pu} planning units; scenarios "
                     "must share a common planning-unit set"
```

Modify `src/pymarxan/analysis/scenarios.py` (apply this diff):

```diff
diff --git a/src/pymarxan/analysis/scenarios.py b/src/pymarxan/analysis/scenarios.py
index 6521787..078b366 100644
--- a/src/pymarxan/analysis/scenarios.py
+++ b/src/pymarxan/analysis/scenarios.py
@@ -9,11 +9,11 @@ from dataclasses import dataclass, field
 from typing import TYPE_CHECKING
 
 import numpy as np
 import pandas as pd
 
-from pymarxan.solvers.base import Solution
+from pymarxan.solvers.base import Solution, require_binary
 
 if TYPE_CHECKING:
     from pymarxan.models.problem import ConservationProblem
     from pymarxan.solvers.base import Solver, SolverConfig
 
@@ -190,10 +190,12 @@ class ScenarioSet:
             })
         return pd.DataFrame(rows)
 
     def overlap_matrix(self) -> np.ndarray:
         """Compute pairwise selection overlap (Jaccard index)."""
+        for scenario in self._scenarios:
+            require_binary(scenario.solution, "ScenarioSet.overlap_matrix")
         n = len(self._scenarios)
         matrix = np.zeros((n, n))
         for i in range(n):
             for j in range(n):
                 si = self._scenarios[i].solution.selected
```

Modify `src/pymarxan/analysis/selection_freq.py` (apply this diff):

```diff
diff --git a/src/pymarxan/analysis/selection_freq.py b/src/pymarxan/analysis/selection_freq.py
index a983784..387141c 100644
--- a/src/pymarxan/analysis/selection_freq.py
+++ b/src/pymarxan/analysis/selection_freq.py
@@ -16,24 +16,28 @@ class SelectionFrequency:
     n_solutions: int
     best_solution: Solution | None
 
 
 def compute_selection_frequency(solutions: list[Solution]) -> SelectionFrequency:
-    """Compute how often each planning unit is selected across solutions."""
+    """Compute how often each planning unit is selected across solutions.
+
+    With proportion decisions ``counts`` is Σ x (float) and ``frequencies`` the mean x.
+    """
     if not solutions:
         return SelectionFrequency(
             frequencies=np.array([]),
             counts=np.array([]),
             n_solutions=0,
             best_solution=None,
         )
 
     n_pu = len(solutions[0].selected)
-    counts = np.zeros(n_pu, dtype=int)
+    dtype = np.float64 if any(s.is_fractional for s in solutions) else int
+    counts = np.zeros(n_pu, dtype=dtype)
 
     for sol in solutions:
-        counts += sol.selected.astype(int)
+        counts += sol.selected.astype(dtype)
 
     n = len(solutions)
     frequencies = counts / n
     best = min(solutions, key=lambda s: s.objective)
```

Modify `src/pymarxan/phylo/diversity.py` (apply this diff):

```diff
diff --git a/src/pymarxan/phylo/diversity.py b/src/pymarxan/phylo/diversity.py
index f6ceb1e..d85fbc5 100644
--- a/src/pymarxan/phylo/diversity.py
+++ b/src/pymarxan/phylo/diversity.py
@@ -6,11 +6,11 @@ from dataclasses import dataclass
 
 import pandas as pd
 
 from pymarxan.models.problem import ConservationProblem
 from pymarxan.phylo.tree import NodeId, PhylogeneticTree
-from pymarxan.solvers.base import Solution
+from pymarxan.solvers.base import Solution, require_binary
 
 
 @dataclass
 class PDResult:
     """Phylogenetic diversity a solution represents, with per-branch detail."""
@@ -83,10 +83,11 @@ def compute_phylogenetic_diversity(
     intersect the represented set (rooted PD). ``pd_representable`` is the
     ceiling reachable given which tips occur in any PU at all.
     """
     resolved = _resolve_tips(problem, tree, tip_feature_map)
 
+    require_binary(solution, "compute_phylogenetic_diversity")  # Faith PD is presence-based
     selected_ids = {
         int(pid)
         for pid, sel in zip(problem.planning_units["id"], solution.selected)
         if sel
     }
```

Modify `src/pymarxan/solvers/base.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/base.py b/src/pymarxan/solvers/base.py
index a7ebecc..af5f9bd 100644
--- a/src/pymarxan/solvers/base.py
+++ b/src/pymarxan/solvers/base.py
@@ -71,10 +71,21 @@ class Solution:
 
 class FractionalSolutionError(ValueError):
     """A binary-only tool was given a solution holding proportion decisions."""
 
 
+def _require_binary_solver(solver: object, context: str) -> None:
+    """``FractionalSolutionError`` up front for a solver configured with non-binary
+    decisions (tools that re-solve and read 0/1 plans, spec §6.2)."""
+    decisions = getattr(solver, "decisions", "binary")
+    if decisions != "binary":
+        raise FractionalSolutionError(
+            f"{context} needs a binary solver, but this solver uses "
+            f"decisions={decisions!r}. Use decisions='binary'."
+        )
+
+
 def require_binary(solution: Solution, context: str) -> None:
     """Raise ``FractionalSolutionError`` unless ``solution.selected`` is binary (spec §5.2)."""
     if solution.is_fractional:
         raise FractionalSolutionError(
             f"{context} needs a binary solution, but this solution holds proportion "
```

Modify `src/pymarxan/solvers/iterative_improvement.py` (apply this diff):

```diff
diff --git a/src/pymarxan/solvers/iterative_improvement.py b/src/pymarxan/solvers/iterative_improvement.py
index 7eb8f66..b80eb36 100644
--- a/src/pymarxan/solvers/iterative_improvement.py
+++ b/src/pymarxan/solvers/iterative_improvement.py
@@ -16,11 +16,11 @@ ITIMPTYPE modes
 from __future__ import annotations
 
 import numpy as np
 
 from pymarxan.models.problem import ConservationProblem
-from pymarxan.solvers.base import Solution, Solver, SolverConfig
+from pymarxan.solvers.base import Solution, Solver, SolverConfig, require_binary
 from pymarxan.solvers.cache import ProblemCache
 from pymarxan.solvers.utils import build_solution
 
 _VALID_ITIMPTYPES = frozenset(range(4))
 
@@ -101,11 +101,14 @@ class IterativeImprovementSolver(Solver):
 
         Returns
         -------
         Solution
             The improved solution (or the original if ITIMPTYPE=0).
+
+        Raises ``FractionalSolutionError`` for a proportion-decision solution.
         """
+        require_binary(solution, "IterativeImprovementSolver.improve")
         itimptype = int(
             problem.parameters.get("ITIMPTYPE", self._itimptype)
         )
 
         if itimptype == 0:
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/analysis/test_fractional_analysis.py tests/pymarxan/calibration/test_blm_fractional.py tests/pymarxan/test_fractional_readers.py -q`
Expected: 16 passed

- [ ] **Step 5: Gate**

Run: `~/.local/bin/ruff check src tests && /home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/ -q -m "not bench" -p no:cacheprovider`
Expected: ruff and mypy clean; the full suite passes with no golden regenerated (prototype final: 3444 passed, 61 skipped, 20 deselected at Task 10).

- [ ] **Step 6: Commit**

```bash
git add src/pymarxan/analysis/equity.py src/pymarxan/analysis/portfolio.py src/pymarxan/analysis/portfolio_cuts.py src/pymarxan/analysis/posthoc_clusters.py src/pymarxan/analysis/rank_importance.py src/pymarxan/analysis/replacement_cost.py src/pymarxan/analysis/representation.py src/pymarxan/analysis/robustness.py src/pymarxan/analysis/scenarios.py src/pymarxan/analysis/selection_freq.py src/pymarxan/phylo/diversity.py src/pymarxan/solvers/base.py src/pymarxan/solvers/iterative_improvement.py tests/pymarxan/analysis/test_fractional_analysis.py tests/pymarxan/calibration/test_blm_fractional.py tests/pymarxan/test_fractional_readers.py
git commit -m "feat(stage4): T8 fraction-aware analysis, binary-only guards, reader completeness test

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AZEiop9qqxVCrMHKqSDYP8"
```

---

### Task 9: Shiny fractional display, comparison guard, decisions inputs

**Files:**
- Modify: `src/pymarxan_app/app.py`
- Modify: `src/pymarxan_shiny/modules/mapping/comparison_map.py`
- Modify: `src/pymarxan_shiny/modules/mapping/solution_map.py`
- Modify: `src/pymarxan_shiny/modules/results/summary_table.py`
- Modify: `src/pymarxan_shiny/modules/solver_config/solver_picker.py`
- Modify: `tests/pymarxan/test_fractional_readers.py`
- Create: `tests/pymarxan_shiny/test_fractional_shiny.py`

**Interfaces:**
- Consumes: Tasks 2–4, 8.
- Produces: `results/summary_table._target_rows(problem, solution) -> list[dict]`; `mapping/solution_map.solution_colors(solution) -> list[str]`, `solution_listing(problem, solution) -> list[str]`, `_blend(c0, c1, t) -> str`; `mapping/comparison_map.comparison_counts(sol_a, sol_b) -> tuple[int, int, int]`, `comparison_summary(sol_a, sol_b) -> str`, `_require_pair`; solver picker `mip_decisions` / `mip_upper` / `mip_lower` inputs and `mip_decision_kwargs(config: dict) -> dict` used by `pymarxan_app/app.py`.

- [ ] **Step 1: Write the failing tests**

Modify `tests/pymarxan/test_fractional_readers.py` (apply this diff):

```diff
diff --git a/tests/pymarxan/test_fractional_readers.py b/tests/pymarxan/test_fractional_readers.py
index d70aed7..54e9c32 100644
--- a/tests/pymarxan/test_fractional_readers.py
+++ b/tests/pymarxan/test_fractional_readers.py
@@ -84,12 +84,10 @@ def _module(rel: str) -> object:
     return importlib.import_module(rel.removesuffix(".py").replace("/", "."))
 
 
 def test_guarded_modules_guard() -> None:
     for rel in GUARDED:
-        if rel.startswith("pymarxan_shiny/"):
-            continue  # the Shiny guard lands with the Shiny task (T9)
         text = (SRC / rel).read_text(encoding="utf-8")
         assert "require_binary" in text, rel
 
 
 def test_mixed_guarded_functions_guard() -> None:
```

Create `tests/pymarxan_shiny/test_fractional_shiny.py`:

```python
"""Stage 4 (spec §6.1, §6.2, §6.5): Shiny helpers with fractional solutions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers import FractionalSolutionError, MIPSolver
from pymarxan.solvers.base import Solution
from pymarxan.solvers.utils import build_solution
from pymarxan_shiny.modules.mapping.comparison_map import comparison_counts, comparison_summary
from pymarxan_shiny.modules.mapping.ocean_palette import MAP_NOT_SEL, MAP_SELECTED
from pymarxan_shiny.modules.mapping.solution_map import solution_colors, solution_listing
from pymarxan_shiny.modules.results.summary_table import _target_rows
from pymarxan_shiny.modules.solver_config.solver_picker import (
    mip_decision_kwargs,
    solver_picker_ui,
)

DATA = Path(__file__).parent.parent / "data"
X = np.array([0.25, 1.0, 0.0, 0.5, 0.1, 1.0])


@pytest.fixture
def simple() -> ConservationProblem:
    return load_project(DATA / "simple")


@pytest.fixture
def frac(simple: ConservationProblem) -> Solution:
    return build_solution(simple, X, 1.0)


def test_target_rows_fractional(simple: ConservationProblem, frac: Solution) -> None:
    rows = _target_rows(simple, frac)
    held1 = 10 * 0.25 + 15 + 12 * 0.5 + 8 * 0.1 + 10
    assert [r["id"] for r in rows] == [1, 2, 3]
    assert rows[0]["achieved"] == pytest.approx(held1)
    assert rows[0]["met"] == frac.targets_met[1]
    assert [r["met"] for r in rows] == [frac.targets_met[f] for f in (1, 2, 3)]
    assert rows[0]["pct"] == pytest.approx(held1 / 30.0 * 100)


def test_target_rows_binary_honours_misslevel(simple: ConservationProblem) -> None:
    """Spec §6.5: the table took met as held >= target, ignoring MISSLEVEL."""
    sel = np.array([False, True, False, True, False, False])  # f1 held 27 of 30
    p = simple.copy_with(parameters={**simple.parameters, "MISSLEVEL": 0.9})
    rows = _target_rows(p, build_solution(p, sel, 1.0))
    assert rows[0]["achieved"] == 27.0
    assert rows[0]["met"] is True


def test_solution_colors(frac: Solution, simple: ConservationProblem) -> None:
    colors = solution_colors(frac)
    assert colors[1] == MAP_SELECTED.lower() and colors[2] == MAP_NOT_SEL.lower()
    assert colors[0] not in (MAP_SELECTED.lower(), MAP_NOT_SEL.lower())
    b = build_solution(simple, X >= 0.5, 1.0)
    assert solution_colors(b) == [
        MAP_SELECTED if v else MAP_NOT_SEL for v in (X >= 0.5)
    ]


def test_solution_listing(frac: Solution, simple: ConservationProblem) -> None:
    assert solution_listing(simple, frac) == [
        "PU 1 (0.25)", "PU 2 (1)", "PU 4 (0.5)", "PU 5 (0.1)", "PU 6 (1)",
    ]
    b = build_solution(simple, X >= 0.5, 1.0)
    assert solution_listing(simple, b) == ["PU 2", "PU 4", "PU 6"]


def test_comparison_guarded(frac: Solution, simple: ConservationProblem) -> None:
    b = build_solution(simple, X >= 0.5, 1.0)
    assert comparison_counts(b, b) == (3, 0, 0)
    with pytest.raises(FractionalSolutionError, match="comparison map"):
        comparison_counts(b, frac)


def test_comparison_summary_explains_fractional(
    frac: Solution, simple: ConservationProblem,
) -> None:
    """Review focus: the comparison renders a message, not an uncaught exception."""
    b = build_solution(simple, X >= 0.5, 1.0)
    assert comparison_summary(b, b) == "Both: 3 | A only: 0 | B only: 0"
    for pair in ((b, frac), (frac, b), (frac, frac)):
        msg = comparison_summary(*pair)
        assert msg.startswith("Comparison unavailable")
        assert "binarize_solution" in msg


def test_solver_picker_has_decision_inputs() -> None:
    html = str(solver_picker_ui("sp"))
    for name in ("mip_decisions", "mip_upper", "mip_lower"):
        assert name in html


def test_mip_decision_kwargs() -> None:
    assert mip_decision_kwargs({}) == {"decisions": "binary"}
    kw = mip_decision_kwargs(
        {"mip_decisions": "proportion", "mip_upper": 0.5, "mip_lower": 0.2},
    )
    assert kw == {"decisions": "proportion", "upper": 0.5}
    kw = mip_decision_kwargs(
        {"mip_decisions": "semicontinuous", "mip_upper": 0.9, "mip_lower": 0.2},
    )
    assert kw == {"decisions": "semicontinuous", "upper": 0.9, "lower": 0.2}
    solver = MIPSolver(**kw)
    assert (solver.decisions, solver.upper, solver.lower) == ("semicontinuous", 0.9, 0.2)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/test_fractional_readers.py tests/pymarxan_shiny/test_fractional_shiny.py -q`
Expected: FAIL — collection error (`ImportError` on `_target_rows` / `comparison_summary` / `mip_decision_kwargs`)

- [ ] **Step 3: Implement**

Modify `src/pymarxan_app/app.py` (apply this diff):

```diff
diff --git a/src/pymarxan_app/app.py b/src/pymarxan_app/app.py
index f550879..ca37b36 100644
--- a/src/pymarxan_app/app.py
+++ b/src/pymarxan_app/app.py
@@ -81,10 +81,11 @@ from pymarxan_shiny.modules.run_control.run_panel import run_panel_server, run_p
 from pymarxan_shiny.modules.solver_config.objective_selector import (
     objective_selector_server,
     objective_selector_ui,
 )
 from pymarxan_shiny.modules.solver_config.solver_picker import (
+    mip_decision_kwargs,
     solver_picker_server,
     solver_picker_ui,
 )
 from pymarxan_shiny.modules.spatial.cost_upload import cost_upload_server, cost_upload_ui
 from pymarxan_shiny.modules.spatial.gadm_picker import gadm_picker_server, gadm_picker_ui
@@ -261,11 +262,11 @@ def server(input: Inputs, output: Outputs, session: Session):
     @reactive.calc
     def active_solver():
         config_dict = solver_config()
         st = config_dict.get("solver_type", "mip")
         if st == "mip":
-            return MIPSolver()
+            return MIPSolver(**mip_decision_kwargs(config_dict))
         elif st == "sa":
             return SimulatedAnnealingSolver()
         elif st == "binary":
             return MarxanBinarySolver()
         elif st == "zone_sa":
```

Modify `src/pymarxan_shiny/modules/mapping/comparison_map.py` (apply this diff):

```diff
diff --git a/src/pymarxan_shiny/modules/mapping/comparison_map.py b/src/pymarxan_shiny/modules/mapping/comparison_map.py
index 42aac89..82b740a 100644
--- a/src/pymarxan_shiny/modules/mapping/comparison_map.py
+++ b/src/pymarxan_shiny/modules/mapping/comparison_map.py
@@ -1,10 +1,11 @@
 """Comparison map Shiny module -- side-by-side solution comparison."""
 from __future__ import annotations
 
 from shiny import module, reactive, render, ui
 
+from pymarxan.solvers.base import FractionalSolutionError, Solution, require_binary
 from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup
 from pymarxan_shiny.modules.mapping.map_utils import build_pu_map, too_large_for_map
 from pymarxan_shiny.modules.mapping.ocean_palette import (
     CMP_A_ONLY,
     CMP_B_ONLY,
@@ -33,10 +34,39 @@ def comparison_color(in_a: bool, in_b: bool) -> str:
     elif in_b:
         return CMP_B_ONLY    # coral — B only
     return CMP_NEITHER       # steel-gray — neither
 
 
+def _require_pair(sol_a: Solution, sol_b: Solution) -> None:
+    """The four-colour comparison is categorical: binary plans only (Stage 4 §6.2)."""
+    require_binary(sol_a, "comparison map")
+    require_binary(sol_b, "comparison map")
+
+
+def comparison_counts(sol_a: Solution, sol_b: Solution) -> tuple[int, int, int]:
+    """(both, A only, B only) planning-unit counts of two binary plans."""
+    _require_pair(sol_a, sol_b)
+    a, b = sol_a.selected, sol_b.selected
+    both = sum(1 for i in range(len(a)) if a[i] and b[i])
+    a_only = sum(1 for i in range(len(a)) if a[i] and not b[i])
+    b_only = sum(1 for i in range(len(a)) if not a[i] and b[i])
+    return both, a_only, b_only
+
+
+def comparison_summary(sol_a: Solution, sol_b: Solution) -> str:
+    """The user-facing summary line; a fractional plan gets an explanation, not a crash."""
+    try:
+        both, a_only, b_only = comparison_counts(sol_a, sol_b)
+    except FractionalSolutionError:
+        return (
+            "Comparison unavailable: this map compares whole-unit (binary) plans, but a "
+            "selected solution holds proportion decisions. Re-run with Decisions = Binary, "
+            "or convert with binarize_solution()."
+        )
+    return f"Both: {both} | A only: {a_only} | B only: {b_only}"
+
+
 @module.ui
 def comparison_map_ui():
     sidebar = ui.sidebar(
         ui.tooltip(
             ui.input_select(
@@ -130,10 +160,12 @@ def comparison_map_server(
             if idx_a >= len(sols) or idx_b >= len(sols):
                 return None
 
             sol_a = sols[idx_a]
             sol_b = sols[idx_b]
+            if sol_a.is_fractional or sol_b.is_fractional:
+                return None  # map_summary explains why (comparison_summary)
             n_pu = len(p.planning_units)
 
             colors = [
                 comparison_color(sol_a.selected[i], sol_b.selected[i])
                 for i in range(n_pu)
@@ -156,26 +188,11 @@ def comparison_map_server(
             idx_a = int(input.sol_a())
             idx_b = int(input.sol_b())
             if idx_a >= len(sols) or idx_b >= len(sols):
                 return "Invalid solution index."
 
-            sol_a = sols[idx_a]
-            sol_b = sols[idx_b]
-            n_pu = len(p.planning_units)
-            both = sum(
-                1 for i in range(n_pu)
-                if sol_a.selected[i] and sol_b.selected[i]
-            )
-            a_only = sum(
-                1 for i in range(n_pu)
-                if sol_a.selected[i] and not sol_b.selected[i]
-            )
-            b_only = sum(
-                1 for i in range(n_pu)
-                if not sol_a.selected[i] and sol_b.selected[i]
-            )
-            return f"Both: {both} | A only: {a_only} | B only: {b_only}"
+            return comparison_summary(sols[idx_a], sols[idx_b])
 
     if not _HAS_IPYLEAFLET:
 
         @render.ui
         def cmp_content():
@@ -187,23 +204,6 @@ def comparison_map_server(
             idx_a = int(input.sol_a())
             idx_b = int(input.sol_b())
             if idx_a >= len(sols) or idx_b >= len(sols):
                 return ui.p("Invalid solution index.")
 
-            sol_a = sols[idx_a]
-            sol_b = sols[idx_b]
-            n_pu = len(p.planning_units)
-            both = sum(
-                1 for i in range(n_pu)
-                if sol_a.selected[i] and sol_b.selected[i]
-            )
-            a_only = sum(
-                1 for i in range(n_pu)
-                if sol_a.selected[i] and not sol_b.selected[i]
-            )
-            b_only = sum(
-                1 for i in range(n_pu)
-                if not sol_a.selected[i] and sol_b.selected[i]
-            )
-            return ui.div(
-                ui.p(f"Both: {both} | A only: {a_only} | B only: {b_only}"),
-            )
+            return ui.div(ui.p(comparison_summary(sols[idx_a], sols[idx_b])))
```

Modify `src/pymarxan_shiny/modules/mapping/solution_map.py` (apply this diff):

```diff
diff --git a/src/pymarxan_shiny/modules/mapping/solution_map.py b/src/pymarxan_shiny/modules/mapping/solution_map.py
index 8209c93..c8f0fd7 100644
--- a/src/pymarxan_shiny/modules/mapping/solution_map.py
+++ b/src/pymarxan_shiny/modules/mapping/solution_map.py
@@ -1,10 +1,12 @@
 """Solution map Shiny module — ipyleaflet map of selected planning units."""
 from __future__ import annotations
 
 from shiny import module, reactive, render, ui
 
+from pymarxan.models.problem import ConservationProblem
+from pymarxan.solvers.base import Solution
 from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup
 from pymarxan_shiny.modules.mapping.map_utils import build_pu_map, too_large_for_map
 from pymarxan_shiny.modules.mapping.ocean_palette import MAP_NOT_SEL, MAP_SELECTED
 
 try:
@@ -13,10 +15,36 @@ try:
     _HAS_IPYLEAFLET = True
 except ImportError:
     _HAS_IPYLEAFLET = False
 
 
+def _blend(c0: str, c1: str, t: float) -> str:
+    """Hex colour ``t`` of the way from ``c0`` to ``c1`` (``t`` clipped to [0, 1])."""
+    t = min(1.0, max(0.0, float(t)))
+    a = [int(c0[k:k + 2], 16) for k in (1, 3, 5)]
+    b = [int(c1[k:k + 2], 16) for k in (1, 3, 5)]
+    return "#" + "".join(f"{round(u + (v - u) * t):02x}" for u, v in zip(a, b, strict=True))
+
+
+def solution_colors(solution: Solution) -> list[str]:
+    """Per-PU map colours: selected / not selected, or interpolated by x for proportions."""
+    if solution.is_fractional:
+        return [_blend(MAP_NOT_SEL, MAP_SELECTED, x) for x in solution.selected]
+    return [MAP_SELECTED if v else MAP_NOT_SEL for v in solution.selected]
+
+
+def solution_listing(problem: ConservationProblem, solution: Solution) -> list[str]:
+    """``"PU 7"`` per selected PU; with proportions ``"PU 7 (0.25)"``."""
+    pu_ids = problem.planning_units["id"].tolist()
+    if solution.is_fractional:
+        return [
+            f"PU {pid} ({float(solution.selected[i]):.4g})"
+            for i, pid in enumerate(pu_ids) if solution.selected[i]
+        ]
+    return [f"PU {pid}" for i, pid in enumerate(pu_ids) if solution.selected[i]]
+
+
 @module.ui
 def solution_map_ui():
     if _HAS_IPYLEAFLET:
         return ui.card(
             help_card_header("Solution Map"),
@@ -54,17 +82,11 @@ def solution_map_server(
             p = problem()
             s = solution()
             if p is None or s is None:
                 return None
 
-            n_pu = len(p.planning_units)
-            colors = [
-                MAP_SELECTED if s.selected[i] else MAP_NOT_SEL
-                for i in range(n_pu)
-            ]
-
-            return build_pu_map(p, colors)
+            return build_pu_map(p, solution_colors(s))
 
         @render.text
         def map_summary():
             p = problem()
             s = solution()
@@ -77,12 +99,13 @@ def solution_map_server(
                 )
 
             n_pu = len(p.planning_units)
             targets_met = sum(s.targets_met.values())
             total_targets = len(s.targets_met)
+            share = f" (Σx {s.selected_sum:.2f})" if s.is_fractional else ""
             return (
-                f"Selected: {s.n_selected} / {n_pu} planning units\n"
+                f"Selected: {s.n_selected} / {n_pu} planning units{share}\n"
                 f"Cost: {s.cost:.2f} | Boundary: {s.boundary:.2f}"
                 f" | Objective: {s.objective:.2f}\n"
                 f"Targets met: {targets_met} / {total_targets}"
             )
 
@@ -94,13 +117,10 @@ def solution_map_server(
             s = solution()
             if p is None or s is None:
                 return ui.p("Run a solver to see results here.")
 
             n_pu = len(p.planning_units)
-            pu_ids = p.planning_units["id"].tolist()
-            rows = [
-                f"PU {pid}" for i, pid in enumerate(pu_ids) if s.selected[i]
-            ]
+            rows = solution_listing(p, s)
             return ui.div(
                 ui.p(f"Selected: {s.n_selected} / {n_pu}"),
                 ui.p(", ".join(rows) if rows else "No PUs selected."),
             )
```

Modify `src/pymarxan_shiny/modules/results/summary_table.py` (apply this diff):

```diff
diff --git a/src/pymarxan_shiny/modules/results/summary_table.py b/src/pymarxan_shiny/modules/results/summary_table.py
index 99cbec9..e504b64 100644
--- a/src/pymarxan_shiny/modules/results/summary_table.py
+++ b/src/pymarxan_shiny/modules/results/summary_table.py
@@ -1,13 +1,40 @@
 """Results summary Shiny module — target achievement table."""
 from __future__ import annotations
 
 from shiny import module, reactive, render, ui
 
+from pymarxan.models.problem import ConservationProblem
+from pymarxan.solvers.base import Solution
+from pymarxan.solvers.utils import _compute_achieved, check_targets
 from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup
 
 
+def _target_rows(problem: ConservationProblem, solution: Solution) -> list[dict]:
+    """One row per feature: held (Σ r·x for proportions) and met per ``check_targets``
+    (MISSLEVEL applied; tolerant for fractional plans, Stage 4 §6.1 / §6.5)."""
+    pu_index = {int(pid): i for i, pid in enumerate(problem.planning_units["id"].values)}
+    held = _compute_achieved(problem, solution.selected, pu_index)
+    met = (
+        check_targets(problem, solution.selected, pu_index, _achieved=held)
+        if "target" in problem.features.columns
+        else {}
+    )
+    rows = []
+    for _, frow in problem.features.iterrows():
+        fid = int(frow["id"])
+        fname = frow.get("name", f"Feature {fid}")
+        target = float(frow.get("target", 0.0))
+        achieved = float(held.get(fid, 0.0))
+        pct = (achieved / target * 100) if target > 0 else 100.0
+        rows.append({
+            "id": fid, "name": fname, "target": target,
+            "achieved": achieved, "pct": pct, "met": bool(met.get(fid, True)),
+        })
+    return rows
+
+
 @module.ui
 def summary_table_ui():
     return ui.card(
         help_card_header("Target Achievement"),
         ui.p(
@@ -30,35 +57,11 @@ def summary_table_server(
     def target_table():
         p = problem()
         s = solution()
         if p is None or s is None:
             return ui.p("No solution available. Run a solver first.")
-        pu_ids = p.planning_units["id"].values
-        id_to_idx = {int(pid): i for i, pid in enumerate(pu_ids)}
-
-        # Vectorized: map PU IDs to indices, filter selected, groupby feature
-        pvf = p.pu_vs_features.copy()
-        pvf["_idx"] = pvf["pu"].map(id_to_idx)
-        pvf = pvf.dropna(subset=["_idx"])
-        pvf["_idx"] = pvf["_idx"].astype(int)
-        pvf["_selected"] = pvf["_idx"].map(lambda i: bool(s.selected[i]))
-        achieved_by_feat = (
-            pvf[pvf["_selected"]].groupby("species")["amount"].sum()
-        )
-
-        rows = []
-        for _, frow in p.features.iterrows():
-            fid = int(frow["id"])
-            fname = frow.get("name", f"Feature {fid}")
-            target = float(frow.get("target", 0.0))
-            achieved = float(achieved_by_feat.get(fid, 0.0))
-            met = achieved >= target
-            pct = (achieved / target * 100) if target > 0 else 100.0
-            rows.append({
-                "id": fid, "name": fname, "target": target,
-                "achieved": achieved, "pct": pct, "met": met,
-            })
+        rows = _target_rows(p, s)
         met_count = sum(1 for r in rows if r["met"])
         total = len(rows)
         summary_line = ui.p(
             f"{met_count} of {total} targets met",
             class_="fw-bold mb-2",
```

Modify `src/pymarxan_shiny/modules/solver_config/solver_picker.py` (apply this diff):

```diff
diff --git a/src/pymarxan_shiny/modules/solver_config/solver_picker.py b/src/pymarxan_shiny/modules/solver_config/solver_picker.py
index 4c629a2..c48d7a3 100644
--- a/src/pymarxan_shiny/modules/solver_config/solver_picker.py
+++ b/src/pymarxan_shiny/modules/solver_config/solver_picker.py
@@ -5,10 +5,22 @@ from shiny import module, reactive, render, ui
 
 from pymarxan.solvers.marxan_binary import MarxanBinarySolver
 from pymarxan_shiny.modules.help.help_button import help_card_header, help_server_setup
 
 
+def mip_decision_kwargs(config: dict) -> dict:
+    """``MIPSolver`` decision kwargs from the picker config (Stage 4): ``upper`` only for
+    non-binary decisions, ``lower`` only for semicontinuous."""
+    decisions = str(config.get("mip_decisions", "binary"))
+    kwargs: dict = {"decisions": decisions}
+    if decisions != "binary":
+        kwargs["upper"] = float(config.get("mip_upper", 1.0))
+    if decisions == "semicontinuous":
+        kwargs["lower"] = float(config.get("mip_lower", 0.0))
+    return kwargs
+
+
 @module.ui
 def solver_picker_ui():
     binary_available = MarxanBinarySolver().available()
     solver_choices = {"mip": "MIP Solver (exact, PuLP/CBC)"}
     solver_choices["sa"] = "Simulated Annealing (Python)"
@@ -152,10 +164,39 @@ def solver_picker_ui():
                         ui.input_checkbox(
                             "mip_verbose", "Verbose output", value=False,
                         ),
                         "Print detailed solver progress to the console log.",
                     ),
+                    ui.tooltip(
+                        ui.input_select(
+                            "mip_decisions", "Decisions",
+                            choices={
+                                "binary": "Binary (select or not)",
+                                "proportion": "Proportion (0 to upper)",
+                                "semicontinuous": "Semicontinuous (0 or lower to upper)",
+                            },
+                            selected="binary",
+                        ),
+                        "Binary selects whole planning units. Proportion lets a unit be "
+                        "partly selected (prioritizr's add_proportion_decisions / "
+                        "add_semicontinuous_decisions(upper)). Semicontinuous requires "
+                        "at least 'lower' of any unit it selects.",
+                    ),
+                    ui.panel_conditional(
+                        "input.mip_decisions !== 'binary'",
+                        ui.input_numeric(
+                            "mip_upper", "Upper bound (max share)",
+                            value=1.0, min=0.0, max=1.0, step=0.05,
+                        ),
+                    ),
+                    ui.panel_conditional(
+                        "input.mip_decisions === 'semicontinuous'",
+                        ui.input_numeric(
+                            "mip_lower", "Lower bound (min share when selected)",
+                            value=0.25, min=0.0, max=1.0, step=0.05,
+                        ),
+                    ),
                 ),
                 ui.panel_conditional(
                     "input.solver_type === 'greedy'",
                     ui.tooltip(
                         ui.input_select(
@@ -246,10 +287,11 @@ def solver_picker_server(input, output, session, solver_config: reactive.Value):
     @reactive.event(
         input.solver_type, input.blm, input.num_solutions,
         input.seed, input.sa_iterations, input.sa_temp_steps,
         input.zone_sa_iterations, input.zone_sa_temp_steps,
         input.mip_time_limit, input.mip_gap, input.mip_verbose,
+        input.mip_decisions, input.mip_upper, input.mip_lower,
         input.heurtype, input.itimptype, input.runmode,
         input.zonation_rule, input.zonation_top_fraction,
         ignore_init=False,
     )
     def _update_config():
@@ -272,10 +314,17 @@ def solver_picker_server(input, output, session, solver_config: reactive.Value):
             config["mip_time_limit"] = int(input.mip_time_limit() or 300)
             config["mip_gap"] = float(
                 input.mip_gap() if input.mip_gap() is not None else 0.0
             )
             config["mip_verbose"] = bool(input.mip_verbose())
+            config["mip_decisions"] = input.mip_decisions() or "binary"
+            config["mip_upper"] = float(
+                input.mip_upper() if input.mip_upper() is not None else 1.0
+            )
+            config["mip_lower"] = float(
+                input.mip_lower() if input.mip_lower() is not None else 0.0
+            )
         if input.solver_type() == "greedy":
             config["heurtype"] = int(input.heurtype() or 2)
         if input.solver_type() == "iterative_improvement":
             config["itimptype"] = int(input.itimptype() or 0)
         if input.solver_type() == "pipeline":
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/test_fractional_readers.py tests/pymarxan_shiny/test_fractional_shiny.py -q`
Expected: 12 passed

- [ ] **Step 5: Gate**

Run: `~/.local/bin/ruff check src tests && /home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/ -q -m "not bench" -p no:cacheprovider`
Expected: ruff and mypy clean; the full suite passes with no golden regenerated (prototype final: 3444 passed, 61 skipped, 20 deselected at Task 10).

- [ ] **Step 6: Commit**

```bash
git add src/pymarxan_app/app.py src/pymarxan_shiny/modules/mapping/comparison_map.py src/pymarxan_shiny/modules/mapping/solution_map.py src/pymarxan_shiny/modules/results/summary_table.py src/pymarxan_shiny/modules/solver_config/solver_picker.py tests/pymarxan/test_fractional_readers.py tests/pymarxan_shiny/test_fractional_shiny.py
git commit -m "feat(stage4): T9 Shiny fractional display, comparison guard, decisions inputs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AZEiop9qqxVCrMHKqSDYP8"
```

---

### Task 10: Docs: CHANGELOG, partial-selection guide, roadmap, copilot instructions

**Files:**
- Modify: `.github/copilot-instructions.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/USER_MANUAL.md`
- Modify: `docs/plans/2026-10-11-prioritizr-parity-roadmap.md`

**Interfaces:**
- Consumes: the public API of Tasks 2–9.
- Produces: CHANGELOG `[Unreleased]` (Added / Changed / Fixed incl. spec §6.5), `docs/USER_MANUAL.md` "Partial Selection" section (its example runs), roadmap Stage 4 as-built note + backlog lines, `.github/copilot-instructions.md` updates. No code.

- [ ] **Step 1: Write the docs**

Modify `.github/copilot-instructions.md` (apply this diff):

````diff
diff --git a/.github/copilot-instructions.md b/.github/copilot-instructions.md
index 4d8a035..e5873bb 100644
--- a/.github/copilot-instructions.md
+++ b/.github/copilot-instructions.md
@@ -33,11 +33,11 @@ pymarxan is a three-layer monorepo:
 ### Subpackages
 
 | Package | Purpose |
 |---------|---------|
 | `models/` | `ConservationProblem`, `Solution`, `SolverConfig` dataclasses |
-| `solvers/` | SA, MIP, heuristic, iterative improvement, run-mode pipeline, cooling schedules |
+| `solvers/` | SA, MIP, heuristic, iterative improvement, run-mode pipeline, cooling schedules. `MIPSolver(decisions="binary" \| "proportion" \| "semicontinuous", upper=, lower=)` (also on `MultiObjectiveMIP`) is the only solver producing fractional solutions; SA / heuristic / II / zone solvers are binary-only and II refuses a fractional start (`FractionalSolutionError`) |
 | `zones/` | `ZonalProblem`, zone SA/heuristic/II/MIP solvers, zone I/O |
 | `connectivity/` | Distance decay functions, connectivity penalties |
 | `constraints/` | ABCs + contiguity, feature contiguity, neighbor, linear, budget constraints |
 | `objectives/` | **Deprecated v0.39, removal v0.42** — objectives live on `MIPSolver(objective=...)` / `ZoneMIPSolver(objective=...)` (`solvers/_mip_objectives.py`) |
 | `spatial/` | Grid generation, feature intersection, boundary generation, GeoPackage export |
@@ -85,11 +85,11 @@ class Solver(ABC):
     def name(self) -> str: ...
     def supports_zones(self) -> bool: ...
     def available(self) -> bool: return True
 ```
 
-`SolverConfig` is a dataclass (`num_solutions`, `seed`, `verbose`, `metadata`). `Solution` is a dataclass with `selected: np.ndarray`, `cost`, `boundary`, `objective`, `targets_met`, and optional `zone_assignment`.
+`SolverConfig` is a dataclass (`num_solutions`, `seed`, `verbose`, `metadata`). `Solution` is a dataclass with `selected: np.ndarray`, `cost`, `boundary`, `objective`, `targets_met`, and optional `zone_assignment`. `selected` is `bool`, or `float64` shares in [0, 1] for non-binary MIP decisions (`Solution.is_fractional` — the dtype decides). Any new reader of `selected` must be fraction-aware or call `require_binary(solution, context)`, and must be classified in `tests/pymarxan/test_fractional_readers.py`.
 
 To add a new solver: inherit `Solver`, implement `solve()`, `name()`, `supports_zones()`.
 
 ### I/O pattern (`io/readers.py`, `io/writers.py`)
````

Modify `CHANGELOG.md` (apply this diff):

```diff
diff --git a/CHANGELOG.md b/CHANGELOG.md
index b716807..43bd61c 100644
--- a/CHANGELOG.md
+++ b/CHANGELOG.md
@@ -5,10 +5,27 @@ All notable changes to **pymarxan** are documented in this file.
 The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
 and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
 
 ## [Unreleased]
 
+### Added
+
+- **Proportion and semicontinuous decisions** (Stage 4 of the prioritizr roadmap): `MIPSolver(decisions="binary" | "proportion" | "semicontinuous", upper=1.0, lower=0.0)` and the same arguments on `MultiObjectiveMIP`. `"proportion"` makes each planning unit a continuous share `x ∈ [0, upper]` (prioritizr's `add_proportion_decisions`; its `add_semicontinuous_decisions(u)` is `decisions="proportion", upper=u`, since prioritizr's version has no lower bound). `"semicontinuous"` is a true floor, `x = 0` or `lower ≤ x ≤ upper`, with one on/off binary per unit (a MILP, not faster than binary). Non-binary solutions hold `float64` proportions in `Solution.selected`; `Solution.is_fractional` and `Solution.selected_sum` (Σ x, prioritizr's `eval_n_summary`) are new, and `n_selected` stays the count of units with x > 0.
+- Fraction-linear evaluators in `pymarxan.solvers.utils`: held `Σ r·x`, cost `Σ c·x`, boundary `Σ b_ii·x_i + Σ b_ij·|x_i − x_j|` (exactly the MIP's term; prioritizr's optimiser uses the same linearisation, its evaluators use the product `x_i·x_j`), connectivity `−v·min(x_i, x_j) + v·|x_i − x_j|` per row, PROBMODE 1 premium `Σ w·p·c·x`. A fractional target is met when `held ≥ R − tol(R)`, `tol(v) = 1e-6 + 1e-7·|v|` (`TARGET_ABS_TOL`, `TARGET_REL_TOL`, `target_tol`), because LP optima leave active target rows a solver tolerance below R; binary comparisons stay exact.
+- `binarize_solution(problem, solution, threshold=0.5)`, the only rounding path, and `FractionalSolutionError` / `require_binary(solution, context)` in `pymarxan.solvers`. Tools that only make sense for 0/1 plans raise `FractionalSolutionError`: iterative improvement (and RUNMODE chains through it), `generate_portfolio_cuts` and `compute_replacement_cost` (up front, given a non-binary `MIPSolver`), `solution_diversity`, `ScenarioSet.overlap_matrix`, `compute_solution_clusters`, `compute_rank_importance`, `compute_phylogenetic_diversity` (Faith PD is presence-based) and the zone-writer fallbacks for solutions without `zone_assignment`; the Shiny comparison map shows an explanation instead of the four-colour map.
+- Fraction-aware readers: `write_run` / `write_best` / `write_solution_matrix` (values written with 10 significant digits), `write_mvbest`, `write_ssoln` / `read_ssoln` (float sums), `export_solution_csv`, `export_summary_csv`, `export_solution_spatial`, `export_frequency_spatial`, `compute_selection_frequency`, `compute_representation` (Σ r·x), `compute_equity` (Σ x·value), `evaluate_plans_across_scenarios`; the Shiny solution map shades units by x, and the solver panel has Decisions / Upper / Lower inputs.
+- `presolve_check(problem, *, decisions=, upper=, lower=)`: P6, P9a and P11 scale with `upper`; P9b uses `lower × min cost` for semicontinuous and is skipped for proportion. The solvers pass their own bounds.
+- **Deviations / limits** (deliberate): a locked-in unit is fixed at `upper` ("selected as fully as allowed"; prioritizr's lock-in is infeasible when `upper < 1`). Non-binary decisions reject PROBMODE 3, clumping (`target2`), separation, space targets, `forbidden_selections` no-good cuts and negative connectivity values with a `ValueError`. Extraction snaps solver dust (≤ 1e-9) to 0 and values within 1e-6 below `upper` / `lower` up to the bound, never down otherwise. `ZoneMIPSolver` stays binary. `compute_connectivity_penalty` still differs from the MIP's connectivity term (pre-existing: the MIP rewards both-selected pairs once, the evaluator rewards both and charges exactly-one per row), so `Solution.objective` equals the MIP objective only without connectivity.
+
+### Changed
+
+- `write_mvbest` and `export_summary_csv` compute held amounts with `_compute_achieved` and met with `check_targets` (binary output unchanged for loaded projects). `read_ssoln` keeps `Number` as int only when every value is integral. The symmetric-connectivity implied rules used by multi-objective cap checks are `min(x_i, x_j)` / `max(0, x_i − x_j)` (identical on 0/1).
+
+### Fixed
+
+- The Shiny target-achievement table ignored MISSLEVEL (`held >= target`); it now uses `check_targets`, as `write_mvbest` and `export_summary_csv` already did.
+
 ## [0.41.1] — 2026-10-16
 
 ### Fixed
 - `solve_lexicographic` (the `min_penalties` two-pass solve in `MIPSolver`) now caps pass 2 at `opt + rel_tol·|opt| + max(abs_tol, 1e-9·|opt|)`, the relative floor v0.41.0 gave the hierarchical caps. PuLP writes MPS values to 13 significant figures, so with `rel_tol=0` a primary of about 1e5 or more could exclude its own pass-1 optimum and fall back to the pass-1 plan with a warning. `MIPSolver` uses the default `rel_tol=1e-9`, which already masked this; the cap is now at most `1e-9·|opt|` looser.
```

Modify `docs/USER_MANUAL.md` (apply this diff):

````diff
diff --git a/docs/USER_MANUAL.md b/docs/USER_MANUAL.md
index 7e01713..78f2af9 100644
--- a/docs/USER_MANUAL.md
+++ b/docs/USER_MANUAL.md
@@ -24,10 +24,11 @@ A comprehensive guide to using the pyMarxan Shiny application for systematic con
    - [Matrix Input](#matrix-input)
    - [Connectivity Metrics](#connectivity-metrics)
 8. [Configure](#configure)
    - [Solver Selection](#solver-selection)
    - [Solver Parameters](#solver-parameters)
+   - [Partial Selection](#partial-selection)
    - [Zone Configuration](#zone-configuration)
 9. [Calibrate](#calibrate)
    - [BLM Calibration](#blm-calibration)
    - [Target Sensitivity](#target-sensitivity)
    - [SPF Calibration](#spf-calibration)
@@ -308,10 +309,13 @@ pyMarxan supports ten solver algorithms:
 | Parameter | Description |
 |-----------|-------------|
 | **Time Limit** | Maximum seconds before returning best-so-far solution |
 | **Optimality Gap** | Acceptable gap from proven optimum (0.0 = exact; 0.01 = within 1%) |
 | **Verbose** | Print solver progress to console |
+| **Decisions** | `binary` (default), `proportion` or `semicontinuous` — see [Partial Selection](#partial-selection) |
+| **Upper bound** | Largest share of a unit that can be selected (non-binary decisions) |
+| **Lower bound** | Smallest share of a unit that is selected at all (semicontinuous) |
 
 **Greedy parameters:**
 
 | HEURTYPE | Strategy |
 |----------|----------|
@@ -343,10 +347,62 @@ pyMarxan supports ten solver algorithms:
 | 3 | Heuristic + iterative improvement |
 | 4 | Heuristic + SA (pick best) |
 | 5 | Heuristic + SA + improvement (highest quality) |
 | 6 | Iterative improvement only |
 
+### Partial Selection
+
+By default the MIP selects whole planning units (`decisions="binary"`). Two other decision
+types let a unit be **partly** selected — partial protection, restoration shares, land-use
+fractions:
+
+| Decisions | Each unit | Model |
+|-----------|-----------|-------|
+| `binary` | selected or not | integer program |
+| `proportion` | any share `0 ≤ x ≤ upper` | linear program (fast) unless the objective counts targets met |
+| `semicontinuous` | `x = 0` or `lower ≤ x ≤ upper` | integer program (one on/off variable per unit) |
+
+```python
+from pymarxan.io.readers import load_project
+from pymarxan.solvers import MIPSolver, binarize_solution
+
+problem = load_project("tests/data/simple")
+
+sol = MIPSolver(decisions="proportion", upper=0.8).solve(problem)[0]
+sol.selected        # float64 shares in [0, 0.8]
+sol.selected_sum    # Σ x; sol.n_selected counts units with x > 0
+plan = binarize_solution(problem, sol, threshold=0.5)  # explicit 0/1 plan
+```
+
+**prioritizr mapping.** `add_proportion_decisions()` is `decisions="proportion"`.
+prioritizr's `add_semicontinuous_decisions(u)` has no lower bound — it is
+`decisions="proportion", upper=u`. `decisions="semicontinuous"` is a true floor.
+
+**How results are computed.** Every number is linear in the shares: cost `Σ c·x`, held
+amount `Σ r·x`, boundary `Σ b_ii·x_i + Σ b_ij·|x_i − x_j|` (the optimiser's own term; prioritizr's
+evaluators use `x_i·x_j` instead), connectivity `−v·min(x_i, x_j) + v·|x_i − x_j|` per row. A
+target counts as met when the held amount is within `1e-6 + 1e-7·|target|` of it, because a
+linear-program optimum leaves active targets a solver tolerance short; binary plans are judged
+exactly. The connectivity evaluator differs from the MIP's connectivity term (also for binary
+plans, a pre-existing difference), so with connectivity the reported objective is not the MIP
+objective.
+
+**Differences and limits.**
+- Locked-in units (status 2) are fixed at `upper` — "selected as fully as allowed". prioritizr
+  makes the model infeasible when `upper < 1`.
+- Non-binary decisions cannot be combined with PROBMODE 3, clumping (`target2`), separation,
+  space targets, no-good cuts or negative connectivity values; the solver raises `ValueError`.
+- Values within `1e-6` below `upper` (or `lower`) are snapped up to the bound and values at or
+  below `1e-9` to 0; nothing is rounded otherwise. A genuine optimum below `1e-9` (one unit
+  holding a billion times a target) would be lost.
+- Tools that only make sense for whole units raise `FractionalSolutionError`: iterative
+  improvement and RUNMODE chains through it, portfolio cuts, replacement cost, solution
+  diversity, scenario overlap, post-hoc clusters, rank importance, phylogenetic diversity and
+  zone writers; the solution comparison map shows an explanation instead. Convert first
+  with `binarize_solution`.
+- Zone solvers stay binary.
+
 ### Zone Configuration
 
 For multi-zone conservation planning, zones are loaded from the project files. Typical zones include:
 - **No-take zone**: Full protection, no resource extraction
 - **Buffer zone**: Limited activities allowed
````

Modify `docs/plans/2026-10-11-prioritizr-parity-roadmap.md` (apply this diff):

```diff
diff --git a/docs/plans/2026-10-11-prioritizr-parity-roadmap.md b/docs/plans/2026-10-11-prioritizr-parity-roadmap.md
index 6e0ec28..fdb2a1f 100644
--- a/docs/plans/2026-10-11-prioritizr-parity-roadmap.md
+++ b/docs/plans/2026-10-11-prioritizr-parity-roadmap.md
@@ -159,10 +159,20 @@ MIP only; CBC/HiGHS. Zones in a follow-up unless Stage 1's builders make it free
 
 ---
 
 ## Stage 4 — v0.42.0: Proportion and semi-continuous decisions
 
+**As built (v0.42.0):** see docs/plans/2026-10-17-stage4-proportion-decisions-design.md.
+`MIPSolver` / `MultiObjectiveMIP` take `decisions="binary" | "proportion" | "semicontinuous"`
+with scalar `upper` / `lower`; `"proportion", upper=u` is prioritizr's
+`add_semicontinuous_decisions(u)`, `"semicontinuous"` a true floor. Fractional `Solution`s carry
+float64 shares; evaluators are fraction-linear (boundary `|x_i − x_j|`, the optimiser's form) with
+a target tolerance `1e-6 + 1e-7·|R|`; every other `Solution.selected` reader is either
+fraction-aware or raises `FractionalSolutionError` (pinned by a completeness test).
+`binarize_solution` is the only rounding path. Locked-in units sit at `upper` (prioritizr:
+infeasible). `ZoneMIPSolver` and per-PU bounds are out of scope.
+
 **Goal:** a PU can be partly selected (0–1), or selected above a floor if selected at all. This
 covers partial protection, restoration areas and land-use shares, and pairs with
 `pymarxan.restoration`.
 
 **Scope**
@@ -198,5 +208,10 @@ Fold these into a stage when one touches the same code:
 - Neighbour penalties.
 - Shuffle portfolio. Top and extra portfolios need a solver solution pool, which CBC and HiGHS
   lack.
 - `simulate_data` / `simulate_species` / `simulate_cost` synthetic project generator, useful for
   tutorials and the JOSS paper.
+- Align `compute_connectivity_penalty` with the MIP connectivity term (symmetric: reward-only,
+  pairs de-duplicated; asymmetric: `x_i ∧ ¬x_j`, reading `ASYMMETRIC_CONNECTIVITY`). Changes
+  binary numbers, so it needs its own golden-tested release (Stage 4 §5.3).
+- Per-PU decision bounds (prioritizr `add_manual_bounded_constraints`) and fractional zone
+  shares for `ZoneMIPSolver`.
```

- [ ] **Step 2: Gate**

Run: `~/.local/bin/ruff check src tests && /home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports && PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/ -q -m "not bench" -p no:cacheprovider`
Expected: ruff and mypy clean; the full suite passes with no golden regenerated (prototype final: 3444 passed, 61 skipped, 20 deselected at Task 10).

- [ ] **Step 3: Commit**

```bash
git add .github/copilot-instructions.md CHANGELOG.md docs/USER_MANUAL.md docs/plans/2026-10-11-prioritizr-parity-roadmap.md
git commit -m "docs(stage4): T10 changelog, partial-selection guide, roadmap, copilot instructions

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AZEiop9qqxVCrMHKqSDYP8"
```
