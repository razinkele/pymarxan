# Stage 3 (v0.41.0) Exact Multi-Objective Optimisation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Planners trade cost against representation (and boundary) **exactly** on one `ConservationProblem`: "the cheapest plan among the best-representing ones" (`Hierarchical`) and "the cost-vs-representation frontier" (`WeightedSum` sweeps), with prioritizr 9's concepts, while every single-objective `MIPSolver` result stays as it is.

**Architecture:** `MIPSolver._build` splits into `_build_base` (locks, cost, unweighted boundary, connectivity, probability, amounts, target terms, no-good cuts, plus an implied-value registry for every auxiliary variable) and objective attachment through the builder table in `solvers/_mip_objectives.py`. Builders gain an `owner=` name prefix and three new entries (`min_cost`, `min_boundary`, `max_wtd_sum`). `_mip_objectives.py` also gains `evaluate_implied` (each objective's value at the implied point of a rounded selection, plus a feasibility check over every model row) and `solve_hierarchical` (caps at the better of the model and implied values). The new core module `src/pymarxan/multiobjective.py` holds the objective specs, the `Hierarchical` / `WeightedSum` approaches, prioritizr's `weights_matrix` / `rel_tol_matrix`, `nondominated`, and `MultiObjectiveMIP`, which builds the base and every term once per `solve` and runs one solve sequence per approach row.

**Tech Stack:** Python 3.11+, PuLP 3.3.2 + CBC (HiGHS is not installed locally), numpy, pandas; pytest.

**Spec:** `docs/plans/2026-10-15-stage3-multi-objective-design.md` (approved 2026-10-15 after 8 review rounds). Executors read it. The spec is the authority, and this plan argues from it.

**Provenance:** every code block below is extracted (`git show` / `git diff`) from a prototype built task by task in a scratch worktree, one commit per task, and run there under the commands shown. Prototype commits: Task 1 `0b0812a`, 2 `69a77f1`, 3 `5401951`, 4 `6926715`, 5 `fe8ff47`, 6 `69eb9fa`, 7 `b435c0d`, 8 `39618bf` (base `00d01a3`; rebuilt after two rounds of 4-lens plan review and one fresh-eyes loop round). Diffs apply with `git apply --whitespace=nowarn` (or `--3way`) on the previous task's tree; editors that strip trailing whitespace from pasted context lines break a plain `git apply`.

## Global Constraints

- Python 3.11+ floor; `from __future__ import annotations` as the first import in every new file; full type hints. mypy runs as `mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports`.
- ruff rules E, F, I, UP; line length 99. **Run `ruff check` on the test files too at every task handoff.** An E402 slipped through two reviews in v0.32.
- Tests run ONLY under `/opt/micromamba/envs/shiny/bin/pytest`, never `.venv`. Full gate: `PATH="/opt/micromamba/envs/shiny/bin:$HOME/.local/bin:$PWD/.venv/bin:$PATH" make check`. (In a worktree without `.venv`, use `PYTHONPATH=src:.` with the shiny pytest, `~/.local/bin/ruff`, and the main checkout's `.venv/bin/mypy`.)
- PuLP 3.3.2 emits `DeprecationWarning`s (`LpVariable()`, `PULP_CBC_CMD`). Any `simplefilter("error")` must ignore `DeprecationWarning`.
- Every test whose result is pinned uses `mip_backend="cbc"` (spec §9).
- `src/pymarxan/zones/mip_solver.py` is **not touched**: `ObjectiveContext(model, cost_expr, terms)` and `BUILDERS[name](ctx, shortfall)` keep working through defaults (spec §2). `tests/pymarxan/zones/` stays green after every task.
- The `tests/data/simple` anchor holds: `MIPSolver` min_set at BLM 0 selects {2, 4, 6} at cost 35.0 (pinned in Task 1's golden and by `tests/test_examples.py`).
- Golden data is recorded **before** any source change (Task 1) and regenerated only with a written reason. In `tests/data/golden/stage1_v038.json` only `core_mls_abs_relabs` and `core_mls_abs_infeasible` may change (Task 3).
- `solve_lexicographic` (used by `min_penalties`) is left untouched (spec §4.1).
- Builders are called **once per context**: they add named rows (spec §3). `MultiObjectiveMIP` builds every term once per `solve`.
- A solve in the multi-objective path is accepted only on `model.sol_status` 1 or 2; every tracked `varValue` is reset to `None` before each solve (spec §4.1).
- `objective_values` always come from `evaluate_implied` (the model's own expressions at the implied point), never from raw auxiliaries and never from the Marxan-style `compute_*` helpers (spec §5).
- `tests/pymarxan/` and `tests/pymarxan/solvers/` have `__init__.py`; new test basenames are repo-unique anyway.
- Commit trailer on every commit:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01AZEiop9qqxVCrMHKqSDYP8
  ```

## Review Focus

These are the places a plausible implementation silently goes wrong; each is pinned by a test that fails when the guard is removed (Task 7 Step 3 runs every spec §7.7 mutation).
1. **CBC returns a different optimal plan after the LP renames.** Renaming `feat_met_*`/`slack_*` reorders LP columns, so CBC may break ties differently; a selection-only golden would then fail spuriously, or be "fixed" by regenerating it. → the golden records `unique_optimum` by re-solving with the found plan forbidden, pins the value always and the selection only when unique: `test_mip_solver_result_unchanged` (Task 1).
2. **The cap value v.** Model-only loses targets after an incumbent with loose auxiliaries; recompute-only loses them at a tolerance boundary (0.7 + 0.1 < 0.8). → `test_incumbent_cap_keeps_both_targets` and `test_tolerance_boundary_cap_returns_cost_2` (Task 7), `test_incumbent_cap_uses_the_recomputed_value` and `test_tolerance_boundary_cap_keeps_the_model_target` (Task 4).
3. **The feasibility check must include earlier `hier_cap_*` rows**, or cap 2 can contradict cap 1 and a 3-stage run stops at stage 2. → `test_earlier_caps_are_checked_three_stages` (Task 4, through `evaluate_implied`) and `test_earlier_caps_are_checked_end_to_end` (Task 6, through `MultiObjectiveMIP`'s own `evaluate`, whose `if feasible else None` fallback `test_near_integral_budget_falls_back_end_to_end` also pins).
4. **Implied values must come from the model's own expressions.** The `compute_*` helpers differ for asymmetric connectivity, duplicated symmetric pairs and negative PROBABILITYWEIGHTING, which makes `v_rec` unattainable. → `test_min_cost_then_min_shortfall_reaches_stage_2[conn_sym_dupes|conn_asym|probmode1_neg]` and `test_asymmetric_connectivity_locked_endpoints_reach_stage_2` (Task 7).
5. **Weighted-sum sign and rel_tol indexing**: `s = −1` for maximised objectives, and the k-th tolerance caps the k-th stage. → `test_mixed_sense_weighted_sum` and `test_three_objectives_rel_tol_follows_stage_order` (Task 7), `test_rel_tol_is_indexed_by_stage` (Task 4).

## Rulings and spec issues found while prototyping

Each bullet names the task it applies to; the per-task `Ruling:` lines restate those that shape that task's code or tests.
- **Golden ties (Task 1).** Spec §7.1 says the refactor must reproduce the recorded results exactly. The first recording (COSTBUDGET 30) had 49 of 121 cases with tied optima, and after Task 3 four `conn_sym|max_*` selections flipped with identical values. Ruling: record `unique_optimum` per case and pin the selection fields only for unique optima; use COSTBUDGET 33, where every budgeted objective on the plain `simple` fixture has a unique optimum (enumerated over the 64 selections). 111 of 121 cases pin the selection.
- **A second ghost case changes (Task 3).** `ghost_misslevel09|min_largest_shortfall|absolute` goes from 0.0 to 9.0 (= 10 × MISSLEVEL 0.9). This is the same §2 fix on another fixture; it is pinned value-only next to the spec's 5.0 → 50.0 case.
- **`min_penalties` primary (Task 3).** "All six single objectives go through builders": its primary becomes `build_min_shortfall(ctx, "absolute")`, identical for R > 0; at MISSLEVEL 0 the always-zero slack rows disappear (LP-only change, in CHANGELOG Changed).
- **One COSTBUDGET message (Task 3).** The three legacy messages become one `objective=... requires a COSTBUDGET parameter` message; existing tests match `COSTBUDGET` / `min_shortfall.*COSTBUDGET` and pass.
- **`solve_hierarchical`'s keyword-only `tracked` is required (Task 4)**, as spec §4.1 now writes it (`*, tracked`, no default): `model.variables()` alone misses x and registry variables that the §5 tracked set needs for the reset / snapshot / restore. Values are read with `expr_value` (None → 0) and never written as 0, so a zero-cost PU in no row keeps `None`.
- **Mocking `sol_status` (Tasks 4, 6, 7).** Tests use `tests/pymarxan/scripted_solver.py`: a `pulp.LpSolver` that runs CBC and then a per-call hook. No production test hook.
- **Non-spec objectives raise `TypeError` (Task 6)**, as an unknown approach does; §6 does not list this case. So does a bare `ObjectiveSpec()` or a subclass whose `name` has no builder, at construction rather than as a `KeyError` inside `solve`.
- **Unbounded stages say so (Task 6).** When the failed solve's status is Unbounded (e.g. a negative SPF on `MinShortfall`, or negative boundary / connectivity values; `MaxWtdSum` is bounded over binary x), the `RuntimeError` says "is unbounded" instead of pointing at infeasible data.
- **Spec §5 caveats are documented (Tasks 6, 8)** in the `MultiObjectiveMIP` docstring, the CHANGELOG deviations and the tutorial: tolerance-boundary met counts can sit below a cap's v, and a cap falls back to the solver's value when the implied point breaks a model row.
- **PuLP's deprecated constraints mapping (Task 4).** PuLP 3.3.2 warns on every dict-style use of `LpProblem.constraints` (in 4.0 it becomes a method returning a list) and has no public row deletion, while `pyproject` allows `PuLP>=2.7`, where `constraints` is a plain dict. Ruling: source code goes through `constraint_map(model)` (the underlying `_constraints` dict when present, else `constraints`), which removed the per-solve warnings from `evaluate_implied` and cap deletion. Tests keep a few readable `"name" in model.constraints` checks; their warnings are PuLP's own and harmless.
- **Shared solver construction (Task 6).** `MIPSolver.solve` and `MultiObjectiveMIP.solve` share `mip_solver._mip_solver_for` (backend, `MIP_TIME_LIMIT`, `MIP_GAP`, resolved name) and `MIPSolver._strategy_metadata`; tests that script the solver patch `pymarxan.solvers.mip_solver._make_pulp_solver`.
- **`Hierarchical.rel_tol` has no default (Task 5)**, as in the spec's signature.
- **Not done, by decision (plan-wide):** `_BaseModel.context()` is not guarded against a second call (it is called once, inside `MIPSolver._build` and `MultiObjectiveMIP.solve`; a second context on the same base would rebuild named rows and PuLP raises on the duplicate name); specs and approaches stay unhashable (frozen dataclasses holding dicts/lists; nothing hashes them); test glue (`P`, `run`, `sel`) is not deduplicated across test files (each file reads on its own); Task 7 is not split (its oracle and regression files are reviewed together against the same mutation run); `solve_hierarchical` has no test for "restore without the preceding reset to None" (unreachable: restored variables are either written by the solver or overwritten by their implied rules before any read).
- **`metadata["weights"]` is the row as given (Task 6)** (not normalised); `metadata["rel_tol"]` is the row of n−1 values.
- **Specs and approaches compare by value (Task 5)** (`@dataclass(frozen=True)`, so `MaxWtdSum` weights take part in `==`). A subclass declared `eq=False` would silently inherit the base class's label-only `__eq__`.
- **Mutation "index rel_tol by objective instead of stage" (Task 7).** Priority = list order, so objective index = stage index and the literal mutation is a no-op. Ruling: test the realistic mis-indexing `rel_tols[-k]` (counted from the other end).
- **Tutorial (Task 8).** The section uses `rel_tol=0.1` with COSTBUDGET 6, as spec §8 now says; the pick trades 0.133 shortfall for one cost unit. Its test asserts backend-independent facts about the sweep, because two weight rows tie between plans on a hull edge.
- **No guard for `0·boundary_expr` (Task 2).** `build_min_cost` adds `ctx.blm * ctx.boundary_expr` unconditionally: PuLP 3.3.2 drops zero coefficients (checked on 3.3.2 only; under PuLP 2.x a zero-coefficient `y_*` would be harmless), so `MinCost` carries no dead `y_*` at BLM 0 with `need_boundary=True`.

---

### Task 1: Golden `MIPSolver` results on the unmodified source

**Files:**
- Create: `tests/pymarxan/solvers/test_stage3_golden_v040.py`
- Create (generated): `tests/data/golden/stage3_v040.json`

**Interfaces:**
- Consumes: `MIPSolver`, `MIP_OBJECTIVES`, `load_project`, `ConservationProblem.copy_with`.
- Produces: `CASES` (10 variants × 6 objectives × 2 shortfall modes = 120) plus `ghost_absolute_mls_v040`; each record holds `solutions`, `selected`, `cost`, `objective`, `targets_met`, `mip_objective_value`, `unique_optimum`. Task 3 edits this file (the two ghost cases).

Ruling: selection fields are pinned only when the recorded optimum is unique (probe: re-solve with the found plan as a no-good cut; for `min_penalties` the key is (penalty, cost)). `mip_objective_value` is pinned for every case. COSTBUDGET is 33 so that every budgeted objective on the plain `simple` fixture has a unique optimum (enumerated).

- [ ] **Step 1: Write the golden test and recorder** (no source file is touched in this task):

```python
"""v0.40 MIPSolver results that the Stage 3 ``_build`` split must not change (spec §2, §7.1).

GOLDEN was recorded on unmodified v0.40 (8f8ea76 + Stage 3 spec commits) before any source
change. Every case must reproduce exactly: ``mip_objective_value`` always, and the selection
with its cost / objective / targets_met when the recorded optimum is unique. Uniqueness is
probed at record time by re-solving with the found selection forbidden (a no-good cut): a
tie means CBC may legitimately return another optimal plan once LP names or row order change.
The absolute ``min_largest_shortfall`` ghost cases are the one deliberate change (spec §2).

Regenerate only with a written reason:
    PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/python \
        tests/pymarxan/solvers/test_stage3_golden_v040.py
"""
from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, SolverConfig
from pymarxan.solvers.mip_solver import MIP_OBJECTIVES, MIPSolver

DATA = Path(__file__).parent.parent.parent / "data"
GOLDEN_PATH = DATA / "golden" / "stage3_v040.json"
SHORTFALLS = ("relative", "absolute")
BUDGET = 33.0  # simple: total cost 83; every budgeted optimum is unique at 33 (enumerated)


def _simple(**params: Any) -> ConservationProblem:
    p = load_project(DATA / "simple")
    return p.copy_with(parameters={**p.parameters, "COSTBUDGET": BUDGET, **params})


def _no_blm() -> ConservationProblem:
    return _simple(BLM=0.0)


def _conn(asymmetric: bool) -> ConservationProblem:
    p = _simple(CONNECTIVITY_WEIGHT=2.0, ASYMMETRIC_CONNECTIVITY=int(asymmetric))
    conn = pd.DataFrame({
        "id1": [1, 2, 3, 2, 5], "id2": [2, 1, 4, 5, 6], "value": [4.0, 3.0, 2.5, 6.0, 1.5],
    })
    return p.copy_with(connectivity=conn)


def _prob(probmode: int | None, weighting: float = 2.0) -> ConservationProblem:
    params: dict[str, Any] = {"PROBABILITYWEIGHTING": weighting}
    if probmode is not None:
        params["PROBMODE"] = probmode
    p = _simple(**params)
    prob = pd.DataFrame({
        "pu": [1, 2, 3, 4, 5, 6], "probability": [0.1, 0.4, 0.0, 0.25, 0.05, 0.3],
    })
    return p.copy_with(probability=prob)


def _locks() -> ConservationProblem:
    p = _simple()
    pu = p.planning_units.copy()
    pu.loc[pu["id"] == 3, "status"] = 2  # locked in
    pu.loc[pu["id"] == 6, "status"] = 3  # locked out
    return p.copy_with(planning_units=pu)


def _ghost(misslevel: float = 0.9) -> ConservationProblem:
    """simple + feature 4 (target 10) with no pu_vs_features rows."""
    p = _simple(MISSLEVEL=misslevel)
    ghost = pd.DataFrame({"id": [4], "name": ["ghost"], "target": [10.0], "spf": [2.0]})
    return p.copy_with(features=pd.concat([p.features, ghost], ignore_index=True))


def _ghost_absolute_mls() -> ConservationProblem:
    """Spec §2: simple + a ghost with target 50 and COSTBUDGET 30."""
    p = load_project(DATA / "simple")
    ghost = pd.DataFrame({"id": [4], "name": ["ghost"], "target": [50.0], "spf": [1.0]})
    return p.copy_with(
        features=pd.concat([p.features, ghost], ignore_index=True),
        parameters={**p.parameters, "COSTBUDGET": 30.0},
    )


NOGOOD = SolverConfig(
    num_solutions=1, metadata={"forbidden_selections": [[0, 1, 0, 1, 0, 1]]},
)
ONE = SolverConfig(num_solutions=1)

VARIANTS: dict[str, tuple[Callable[[], ConservationProblem], SolverConfig]] = {
    "simple": (_simple, ONE),
    "no_blm": (_no_blm, ONE),
    "conn_sym": (lambda: _conn(False), ONE),
    "conn_asym": (lambda: _conn(True), ONE),
    "probmode1": (lambda: _prob(1), ONE),
    "probmode2": (lambda: _prob(2), ONE),
    "prob_no_probmode": (lambda: _prob(None), ONE),
    "locks": (_locks, ONE),
    "nogood": (_simple, NOGOOD),
    "ghost_misslevel09": (_ghost, ONE),
}
CASES = [
    f"{variant}|{objective}|{shortfall}"
    for variant in VARIANTS
    for objective in MIP_OBJECTIVES
    for shortfall in SHORTFALLS
]


def _solve(
    problem: ConservationProblem, config: SolverConfig, **kw: Any
) -> list[Solution]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # PuLP 3.3.2
        return MIPSolver(mip_backend="cbc", **kw).solve(problem, config)


def _key(sol: Solution, objective: str) -> tuple[float, ...]:
    """Lexicographic value the solver optimises, as a minimisation key."""
    v = float(sol.metadata["mip_objective_value"])
    if objective.startswith("max_"):
        return (-v,)
    return (v, float(sol.cost)) if objective == "min_penalties" else (v,)


def _worse(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """a is lexicographically worse than b by more than 1e-9."""
    for ai, bi in zip(a, b, strict=True):
        if abs(ai - bi) > 1e-9:
            return ai > bi
    return False


def _result(
    problem: ConservationProblem, config: SolverConfig, *, record: bool = False, **kw: Any
) -> dict[str, Any]:
    sols = _solve(problem, config, **kw)
    if not sols:
        return {"solutions": 0}
    s = sols[0]
    out: dict[str, Any] = {
        "solutions": len(sols),
        "selected": [int(v) for v in s.selected],
        "cost": float(s.cost),
        "objective": float(s.objective),
        "targets_met": {str(k): bool(v) for k, v in sorted(s.targets_met.items())},
        "mip_objective_value": float(s.metadata["mip_objective_value"]),
    }
    if record:
        forbidden = [*config.metadata.get("forbidden_selections", []), out["selected"]]
        probe = SolverConfig(num_solutions=1, metadata={"forbidden_selections": forbidden})
        other = _solve(problem, probe, **kw)
        objective = kw.get("objective", "min_set")
        out["unique_optimum"] = not other or _worse(
            _key(other[0], objective), _key(s, objective),
        )
    return out


def _run(case: str, *, record: bool = False) -> dict[str, Any]:
    variant, objective, shortfall = case.split("|")
    factory, config = VARIANTS[variant]
    return _result(factory(), config, record=record, objective=objective, shortfall=shortfall)


def record() -> dict[str, Any]:
    out: dict[str, Any] = {case: _run(case, record=True) for case in CASES}
    out["ghost_absolute_mls_v040"] = _result(
        _ghost_absolute_mls(), ONE, record=True,
        objective="min_largest_shortfall", shortfall="absolute",
    )
    return out


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(GOLDEN_PATH.read_text())
    return data


def _assert_same(got: dict[str, Any], want: dict[str, Any]) -> None:
    assert got["solutions"] == want["solutions"]
    if not want["solutions"]:
        return
    keys = ["mip_objective_value"]
    if want["unique_optimum"]:
        keys += ["selected", "cost", "objective", "targets_met"]
    for key in keys:
        if isinstance(want[key], float):
            assert got[key] == pytest.approx(want[key], rel=1e-9, abs=1e-9), key
        else:
            assert got[key] == want[key], key


@pytest.mark.parametrize("case", CASES)
def test_mip_solver_result_unchanged(golden: dict[str, Any], case: str) -> None:
    _assert_same(_run(case), golden[case])


def test_simple_min_set_anchor_in_golden(golden: dict[str, Any]) -> None:
    """The correctness anchor (BLM 0): reserve {2, 4, 6} at cost 35.0."""
    for shortfall in SHORTFALLS:
        rec = golden[f"no_blm|min_set|{shortfall}"]
        assert rec["selected"] == [0, 1, 0, 1, 0, 1]
        assert rec["cost"] == 35.0


def test_ghost_absolute_min_largest_shortfall(golden: dict[str, Any]) -> None:
    """v0.40 ignores the ghost (target 50, no rows) and reports a largest shortfall of 5.0."""
    rec = golden["ghost_absolute_mls_v040"]
    assert rec["mip_objective_value"] == 5.0
    got = _result(
        _ghost_absolute_mls(), ONE, objective="min_largest_shortfall", shortfall="absolute",
    )
    _assert_same(got, rec)


if __name__ == "__main__":
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(record(), indent=1, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN_PATH}")
```

- [ ] **Step 2: Record on the unmodified source.**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/python -W ignore tests/pymarxan/solvers/test_stage3_golden_v040.py`

Expected: `wrote .../tests/data/golden/stage3_v040.json` (2,435 lines). Check the summary:

```bash
PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/python -c "
import json; d = json.load(open('tests/data/golden/stage3_v040.json'))
print(sum(v['unique_optimum'] for v in d.values()), 'of', len(d))
print(d['ghost_absolute_mls_v040']['mip_objective_value'], d['no_blm|min_set|relative']['selected'])"
```
Expected: `111 of 121`, then `5.0 [0, 1, 0, 1, 0, 1]`. The ten tied cases are `ghost_absolute_mls_v040`, `ghost_misslevel09|min_largest_shortfall|relative`, the four `locks|max_*` cases and the four `probmode2|min_set|*` / `probmode2|min_penalties|*` cases.

- [ ] **Step 3: Run the goldens.**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_stage3_golden_v040.py -q -p no:cacheprovider`

Expected: `122 passed`. Run it twice: it is deterministic.

- [ ] **Step 4: Lint and commit.**

```bash
~/.local/bin/ruff check src tests
/home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports
git add tests/pymarxan/solvers/test_stage3_golden_v040.py tests/data/golden/stage3_v040.json
git commit -m "test(stage3): golden MIPSolver results recorded on unmodified v0.40

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AZEiop9qqxVCrMHKqSDYP8"
```

---

### Task 2: Builder protocol in `_mip_objectives.py`

**Files:**
- Modify: `src/pymarxan/solvers/_mip_objectives.py`
- Create: `tests/pymarxan/solvers/test_mip_objectives_stage3.py`

**Interfaces:**
- Consumes: `TargetTerm`, `ObjectiveTerm`, the four existing builders.
- Produces:
  - `ImpliedRule = tuple[pulp.LpVariable, Callable[[], float]]`; `expr_value(expr) -> float` (None read as 0).
  - `ObjectiveContext` gains `blm`, `boundary_expr`, `conn_expr`, `prob_expr`, `amount_exprs`, `implied` (all defaulted; `default_factory` for the expressions, dict and list). `slack()` registers `s = max(0, R − A)`; `met()` registers `y = 1 iff A ≥ R` exactly.
  - Every builder: `(ctx, shortfall, *, owner: str = "", **params: Any) -> ObjectiveTerm`. `build_min_largest_shortfall` names `f"{ctx.prefix}{owner}max_shortfall"` / `maxs_{key}` and registers `m` after its slacks.
  - `build_min_cost`, `build_min_boundary`, `build_max_wtd_sum(..., weights: Mapping[int, float] | None = None, **params)`.
  - `class BuilderFn(Protocol)` (no positional-only `/`); `BUILDERS: dict[str, BuilderFn]` with 7 entries.

Ruling: `build_min_cost` adds `ctx.blm * ctx.boundary_expr` with no guard for BLM 0: PuLP 3.3.2 drops zero coefficients (checked on 3.3.2 only; a zero-coefficient `y_*` under PuLP 2.x would be harmless).

- [ ] **Step 1: Write the failing tests:**

```python
"""Stage 3 builder protocol (spec §2): context fields, owner names, implied registry, builders."""
from __future__ import annotations

import pulp
import pytest

from pymarxan.solvers._mip_objectives import (
    BUILDERS,
    ObjectiveContext,
    TargetTerm,
    build_max_wtd_sum,
    build_min_boundary,
    build_min_cost,
    build_min_largest_shortfall,
)


def _coefs(expr: pulp.LpAffineExpression) -> dict[str, float]:
    return {v.name: float(a) for v, a in expr.items()}


def _ctx(**kw: object) -> tuple[ObjectiveContext, dict[int, pulp.LpVariable]]:
    m = pulp.LpProblem("p", pulp.LpMinimize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in (1, 2)}
    terms = [
        TargetTerm("f1", "feature", pulp.lpSum([2.0 * x[1]]), 2.0, 1.0),
        TargetTerm("f2", "feature", pulp.lpSum([4.0 * x[2]]), 4.0, 1.0),
    ]
    return ObjectiveContext(m, pulp.lpSum([x[1], 3 * x[2]]), terms, **kw), x  # type: ignore[arg-type]


def _apply(ctx: ObjectiveContext) -> None:
    for var, rule in ctx.implied:
        var.varValue = rule()


def test_context_defaults_keep_the_zone_call() -> None:
    a, _ = _ctx()
    b, _ = _ctx()
    assert a.blm == 0.0
    for expr in (a.boundary_expr, a.conn_expr, a.prob_expr):
        assert isinstance(expr, pulp.LpAffineExpression) and not expr.items()
    assert a.amount_exprs == {} and a.implied == []
    assert a.implied is not b.implied and a.boundary_expr is not b.boundary_expr


def test_slack_and_met_register_implied_rules_on_creation() -> None:
    ctx, x = _ctx()
    t1, t2 = ctx.terms
    s1, y2 = ctx.slack(t1), ctx.met(t2)
    ctx.slack(t1)  # created once: registered once
    assert [v.name for v, _ in ctx.implied] == ["s_f1", "y_f2"]
    x[1].varValue, x[2].varValue = 0.0, 1.0
    s1.varValue, y2.varValue = 7.0, 0.0  # stale solver values
    _apply(ctx)
    assert (s1.varValue, y2.varValue) == (2.0, 1.0)  # max(0, R - A); A >= R


def test_met_rule_is_exact_at_the_target() -> None:
    m = pulp.LpProblem("e", pulp.LpMinimize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in (1, 2)}
    t = TargetTerm("f1", "feature", pulp.lpSum([0.7 * x[1], 0.1 * x[2]]), 0.8, 1.0)
    ctx = ObjectiveContext(m, pulp.lpSum(x.values()), [t])
    y = ctx.met(t)
    x[1].varValue = x[2].varValue = 1.0  # 0.7 + 0.1 = 0.7999999999999999 < 0.8
    _apply(ctx)
    assert y.varValue == 0.0


def test_min_largest_shortfall_registers_m_after_its_slacks() -> None:
    ctx, x = _ctx()
    term = build_min_largest_shortfall(ctx, "relative")
    assert [v.name for v, _ in ctx.implied] == ["s_f1", "s_f2", "max_shortfall"]
    # Stale solver values: s = 3, m = 5. True values at x = (1, 0): s = (0, 4), m = 4/4 = 1.
    x[1].varValue, x[2].varValue = 1.0, 0.0
    for v, _ in ctx.implied:
        v.varValue = 3.0 if v.name.startswith("s_") else 5.0
    _apply(ctx)
    assert pulp.value(term.expr) == 1.0


def test_min_largest_shortfall_without_positive_targets_is_zero() -> None:
    m = pulp.LpProblem("z", pulp.LpMinimize)
    x = pulp.LpVariable("x_1", cat="Binary")
    ctx = ObjectiveContext(m, pulp.lpSum([x]), [TargetTerm("f0", "feature", 1 * x, 0.0, 1.0)])
    term = build_min_largest_shortfall(ctx, "absolute")
    _apply(ctx)
    assert pulp.value(term.expr) == 0.0


def test_owner_prefixes_owned_names_and_shares_slacks() -> None:
    ctx, _ = _ctx(prefix="p_")
    build_min_largest_shortfall(ctx, "relative", owner="a__")
    build_min_largest_shortfall(ctx, "absolute", owner="b__")
    names = set(ctx.model.constraints)
    assert {"p_a__maxs_f1", "p_b__maxs_f1", "p_short_f1", "p_short_f2"} <= names
    var_names = [v.name for v, _ in ctx.implied]
    assert var_names == ["p_s_f1", "p_s_f2", "p_a__max_shortfall", "p_b__max_shortfall"]


def test_default_owner_keeps_todays_names() -> None:
    ctx, _ = _ctx()
    BUILDERS["min_largest_shortfall"](ctx, "absolute")
    assert {"maxs_f1", "maxs_f2"} <= set(ctx.model.constraints)
    assert ctx.implied[-1][0].name == "max_shortfall"


def test_build_min_cost_is_min_set_objective() -> None:
    ctx, x = _ctx(blm=2.0)
    ctx.boundary_expr += 5.0 * x[1]
    ctx.conn_expr += -1.5 * x[2]
    ctx.prob_expr += 0.25 * x[1]
    term = build_min_cost(ctx, "relative")
    assert term.sense == pulp.LpMinimize
    assert _coefs(term.expr) == {"x_1": 1.0 + 10.0 + 0.25, "x_2": 3.0 - 1.5}


def test_build_min_boundary_ignores_blm() -> None:
    ctx, x = _ctx(blm=7.0)
    ctx.boundary_expr += 5.0 * x[1]
    term = build_min_boundary(ctx, "relative", owner="mb__")
    assert term.sense == pulp.LpMinimize
    assert _coefs(term.expr) == {"x_1": 5.0}
    ctx.boundary_expr += 1.0 * x[2]  # the term is a copy, not an alias
    assert _coefs(term.expr) == {"x_1": 5.0}


def test_build_max_wtd_sum_weights_and_ghosts() -> None:
    ctx, x = _ctx()
    ctx.amount_exprs = {1: pulp.lpSum([2.0 * x[1]]), 2: pulp.lpSum([4.0 * x[2]])}
    plain = build_max_wtd_sum(ctx, "relative")
    assert plain.sense == pulp.LpMaximize
    assert _coefs(plain.expr) == {"x_1": 2.0, "x_2": 4.0}
    # Feature 9 has no rows (ghost): it contributes 0, not a KeyError.
    weighted = build_max_wtd_sum(ctx, "relative", owner="w__", weights={1: -1.0, 2: 0.5, 9: 3.0})
    assert _coefs(weighted.expr) == {"x_1": -2.0, "x_2": 2.0}
    assert len(ctx.model.constraints) == 0  # owns no rows


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_every_builder_takes_owner(name: str) -> None:
    ctx, _ = _ctx()
    term = BUILDERS[name](ctx, "relative", owner=f"{name}__")
    assert isinstance(term.expr, pulp.LpAffineExpression)
    assert all(
        n.startswith(f"{name}__") or n.startswith(("short_", "met_"))
        for n in ctx.model.constraints
    )


def test_registry_has_the_stage3_builders() -> None:
    assert {"min_cost", "min_boundary", "max_wtd_sum"} <= set(BUILDERS)
```

- [ ] **Step 2: Run to verify they fail.**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_mip_objectives_stage3.py -q -p no:cacheprovider`

Expected: collection error, `ImportError: cannot import name 'build_max_wtd_sum' from 'pymarxan.solvers._mip_objectives'`.

- [ ] **Step 3: Implement.** Apply this change to `src/pymarxan/solvers/_mip_objectives.py`:

```diff
diff --git a/src/pymarxan/solvers/_mip_objectives.py b/src/pymarxan/solvers/_mip_objectives.py
index c7ca687..ffe39b9 100644
--- a/src/pymarxan/solvers/_mip_objectives.py
+++ b/src/pymarxan/solvers/_mip_objectives.py
@@ -1,14 +1,19 @@
-"""Solver-agnostic MIP objective builders shared by MIPSolver and ZoneMIPSolver (Stage 1).
+"""MIP objective builders shared by MIPSolver and ZoneMIPSolver (Stage 1), and the
+multi-objective machinery used by ``pymarxan.multiobjective`` (Stage 3).
 
 Each target is one ``TargetTerm`` (prioritizr: one row per target). Builders return an
 ``ObjectiveTerm`` in the objective's natural units and sense; the solver sets the model
-objective (``set_objective``) and adds ``COSTBUDGET`` once. Imports only pulp and the stdlib.
+objective (``set_objective``) and adds ``COSTBUDGET`` once. The four target builders work
+on any context. ``build_min_cost``, ``build_min_boundary`` and ``build_max_wtd_sum`` are
+core-only: they read ``ObjectiveContext`` fields that ``MIPSolver`` fills from
+``_build_base`` and that zone contexts leave empty. Imports only pulp and the stdlib.
 """
 from __future__ import annotations
 
 import warnings
-from collections.abc import Callable
+from collections.abc import Callable, Mapping
 from dataclasses import dataclass, field
+from typing import Any, Protocol
 
 import pulp
 
@@ -51,12 +56,50 @@ class ObjectiveTerm:
     sense: int  # pulp.LpMinimize | pulp.LpMaximize
 
 
+ImpliedRule = tuple[pulp.LpVariable, Callable[[], float]]
+
+
+def expr_value(expr: pulp.LpAffineExpression) -> float:
+    """Value of ``expr`` at the current ``varValue``s, reading ``None`` as 0.
+
+    ``pulp.value`` returns ``None`` when any variable has no value (a zero-cost PU in no
+    row is never written to the LP).
+    """
+    return float(expr.constant or 0.0) + sum(
+        float(a) * float(v.varValue or 0.0) for v, a in expr.items()
+    )
+
+
+def _slack_rule(term: TargetTerm) -> Callable[[], float]:
+    """s = max(0, R − A)."""
+    return lambda: max(0.0, term.required - expr_value(term.achieved))
+
+
+def _met_rule(term: TargetTerm) -> Callable[[], float]:
+    """y = 1 iff A >= R exactly: the ``check_targets`` rule."""
+    return lambda: 1.0 if expr_value(term.achieved) >= term.required else 0.0
+
+
 @dataclass
 class ObjectiveContext:
+    """Shared state for building objectives on one model (spec §2).
+
+    ``implied`` is the implied-value registry (spec §5): each auxiliary variable a builder
+    creates is registered with a rule giving its value implied by x, after every variable
+    the rule reads. ``MIPSolver`` fills the expression fields from ``_build_base``; zone
+    contexts keep the defaults.
+    """
+
     model: pulp.LpProblem
     cost_expr: pulp.LpAffineExpression
     terms: list[TargetTerm]
     prefix: str = ""
+    blm: float = 0.0
+    boundary_expr: pulp.LpAffineExpression = field(default_factory=pulp.LpAffineExpression)
+    conn_expr: pulp.LpAffineExpression = field(default_factory=pulp.LpAffineExpression)
+    prob_expr: pulp.LpAffineExpression = field(default_factory=pulp.LpAffineExpression)
+    amount_exprs: dict[int, pulp.LpAffineExpression] = field(default_factory=dict)
+    implied: list[ImpliedRule] = field(default_factory=list)
     _slack: dict[str, pulp.LpVariable] = field(default_factory=dict, repr=False)
     _met: dict[str, pulp.LpVariable] = field(default_factory=dict, repr=False)
 
@@ -66,6 +109,7 @@ class ObjectiveContext:
             s = pulp.LpVariable(f"{self.prefix}s_{term.key}", lowBound=0)
             self.model += s >= term.required - term.achieved, f"{self.prefix}short_{term.key}"
             self._slack[term.key] = s
+            self.implied.append((s, _slack_rule(term)))
         return self._slack[term.key]
 
     def met(self, term: TargetTerm) -> pulp.LpVariable:
@@ -74,6 +118,7 @@ class ObjectiveContext:
             y = pulp.LpVariable(f"{self.prefix}y_{term.key}", cat="Binary")
             self.model += term.achieved >= term.required * y, f"{self.prefix}met_{term.key}"
             self._met[term.key] = y
+            self.implied.append((y, _met_rule(term)))
         return self._met[term.key]
 
 
@@ -86,17 +131,23 @@ def _shortfall_terms(ctx: ObjectiveContext) -> list[TargetTerm]:
     return [t for t in ctx.terms if t.required > 0]
 
 
-def build_max_features(ctx: ObjectiveContext, shortfall: str) -> ObjectiveTerm:
+def build_max_features(
+    ctx: ObjectiveContext, shortfall: str, *, owner: str = "", **params: Any
+) -> ObjectiveTerm:
     return ObjectiveTerm(pulp.lpSum(ctx.met(t) for t in ctx.terms), pulp.LpMaximize)
 
 
-def build_max_weighted_features(ctx: ObjectiveContext, shortfall: str) -> ObjectiveTerm:
+def build_max_weighted_features(
+    ctx: ObjectiveContext, shortfall: str, *, owner: str = "", **params: Any
+) -> ObjectiveTerm:
     return ObjectiveTerm(
         pulp.lpSum(t.weight * ctx.met(t) for t in ctx.terms), pulp.LpMaximize,
     )
 
 
-def build_min_shortfall(ctx: ObjectiveContext, shortfall: str) -> ObjectiveTerm:
+def build_min_shortfall(
+    ctx: ObjectiveContext, shortfall: str, *, owner: str = "", **params: Any
+) -> ObjectiveTerm:
     """Σ w·s/R (relative, prioritizr) or Σ w·s (absolute)."""
     return ObjectiveTerm(
         pulp.lpSum(
@@ -106,19 +157,76 @@ def build_min_shortfall(ctx: ObjectiveContext, shortfall: str) -> ObjectiveTerm:
     )
 
 
-def build_min_largest_shortfall(ctx: ObjectiveContext, shortfall: str) -> ObjectiveTerm:
-    """m >= s/R (or s) for every term; minimise m. Unweighted, as in prioritizr."""
-    m = pulp.LpVariable(f"{ctx.prefix}max_shortfall", lowBound=0)
-    for t in _shortfall_terms(ctx):
-        ctx.model += m >= _scale(t, shortfall) * ctx.slack(t), f"{ctx.prefix}maxs_{t.key}"
+def build_min_largest_shortfall(
+    ctx: ObjectiveContext, shortfall: str, *, owner: str = "", **params: Any
+) -> ObjectiveTerm:
+    """m >= s/R (or s) for every term with R > 0; minimise m. Unweighted, as in prioritizr.
+
+    ``m`` is registered in ``ctx.implied`` after the loop, so its rule reads slacks whose
+    rules have already run (spec §5).
+    """
+    m = pulp.LpVariable(f"{ctx.prefix}{owner}max_shortfall", lowBound=0)
+    short = _shortfall_terms(ctx)
+    for t in short:
+        ctx.model += m >= _scale(t, shortfall) * ctx.slack(t), f"{ctx.prefix}{owner}maxs_{t.key}"
+    slacks = [(_scale(t, shortfall), ctx.slack(t)) for t in short]
+    ctx.implied.append(
+        (m, lambda: max((k * float(s.varValue or 0.0) for k, s in slacks), default=0.0))
+    )
     return ObjectiveTerm(pulp.lpSum([m]), pulp.LpMinimize)
 
 
-BUILDERS: dict[str, Callable[[ObjectiveContext, str], ObjectiveTerm]] = {
+def build_min_cost(
+    ctx: ObjectiveContext, shortfall: str, *, owner: str = "", **params: Any
+) -> ObjectiveTerm:
+    """cost + BLM·boundary + connectivity + probability: what ``min_set`` minimises."""
+    return ObjectiveTerm(
+        ctx.cost_expr + ctx.blm * ctx.boundary_expr + ctx.conn_expr + ctx.prob_expr,
+        pulp.LpMinimize,
+    )
+
+
+def build_min_boundary(
+    ctx: ObjectiveContext, shortfall: str, *, owner: str = "", **params: Any
+) -> ObjectiveTerm:
+    """The boundary term alone, unweighted by BLM (compactness as its own objective)."""
+    return ObjectiveTerm(ctx.boundary_expr.copy(), pulp.LpMinimize)
+
+
+def build_max_wtd_sum(
+    ctx: ObjectiveContext,
+    shortfall: str,
+    *,
+    owner: str = "",
+    weights: Mapping[int, float] | None = None,
+    **params: Any,
+) -> ObjectiveTerm:
+    """Σ_j w_j·A_j, the weighted amount held (prioritizr ``max_wtd_sum``); w = 1 by default.
+
+    A feature without ``pu_vs_features`` rows contributes 0.
+    """
+    if weights is None:
+        weights = {fid: 1.0 for fid in ctx.amount_exprs}
+    expr = pulp.LpAffineExpression()
+    for fid, w in weights.items():
+        expr += float(w) * ctx.amount_exprs.get(fid, pulp.LpAffineExpression())
+    return ObjectiveTerm(expr, pulp.LpMaximize)
+
+
+class BuilderFn(Protocol):
+    def __call__(
+        self, ctx: ObjectiveContext, shortfall: str, *, owner: str = "", **params: Any
+    ) -> ObjectiveTerm: ...
+
+
+BUILDERS: dict[str, BuilderFn] = {
     "max_features": build_max_features,
     "max_weighted_features": build_max_weighted_features,
     "min_shortfall": build_min_shortfall,
     "min_largest_shortfall": build_min_largest_shortfall,
+    "min_cost": build_min_cost,
+    "min_boundary": build_min_boundary,
+    "max_wtd_sum": build_max_wtd_sum,
 }
 
 
```

The rules are factories (`_slack_rule`, `_met_rule`), not default-argument lambdas: mypy reports `Cannot infer type of lambda` for `lambda t=term: ...`.

- [ ] **Step 4: Run the new tests, the existing builder tests and the zone solver.**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_mip_objectives_stage3.py tests/pymarxan/solvers/test_mip_objectives_module.py -q -p no:cacheprovider`

Expected: `36 passed`.

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/ tests/pymarxan/zones/ -q -p no:cacheprovider`

Expected: all pass (1,475 at the prototype; `zones/mip_solver.py` unchanged).

- [ ] **Step 5: Lint, type-check, commit.**

```bash
~/.local/bin/ruff check src tests
/home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports
git add src/pymarxan/solvers/_mip_objectives.py tests/pymarxan/solvers/test_mip_objectives_stage3.py
git commit -m "feat(mip): builder protocol with owner names, implied registry, min_cost/min_boundary/max_wtd_sum

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AZEiop9qqxVCrMHKqSDYP8"
```

---

### Task 3: Split `MIPSolver._build` into `_build_base` plus builders

**Files:**
- Modify: `src/pymarxan/solvers/mip_solver.py` (`_build`, the helper `_feat_met_vars` is removed)
- Modify: `tests/pymarxan/solvers/test_stage3_golden_v040.py` (the two ghost cases)
- Modify: `tests/pymarxan/solvers/test_stage1_golden_v038.py` (docstring: the written reason)
- Regenerate: `tests/data/golden/stage1_v038.json` (two entries)
- Create: `tests/pymarxan/solvers/test_mip_build_base.py`
- Modify: `CHANGELOG.md` (`[Unreleased]` Changed / Fixed)

**Interfaces:**
- Consumes: Task 2's builders, `ObjectiveContext` fields, `ImpliedRule`.
- Produces:
  - `_BaseModel` dataclass (`model, x, pu_ids, blm, cost_expr, boundary_expr, conn_expr, prob_expr, amount_exprs, terms, implied, probmode, has_target2, has_sep_active`) with `context() -> ObjectiveContext` sharing the registry.
  - `MIPSolver._build_base(problem, config, *, need_boundary=False) -> _BaseModel`: strategy gating, x + locks, cost, unweighted boundary (BLM > 0 **or** `need_boundary`), connectivity, PROBMODE 1 / 2 (defaults 0 for the PROBMODE-3 gate and 1 for the probability term, exactly as today), amounts, target terms, no-good cuts. No hard targets, ghost bounds, budget row or objective. Registers `|x_i − x_j|`, `x_i·x_j` and `x_i·(1 − x_j)` rules.
  - `_build` = base + `build_min_cost` and hard `target_{id}` rows (min_set; features with rows only), `build_min_shortfall(ctx, "absolute")` as the `min_penalties` primary, or `BUILDERS[objective]` + `cost_budget` (budgeted objectives).

Ruling: a second ghost case changes (`ghost_misslevel09|min_largest_shortfall|absolute`, 0.0 → 9.0); it is the same §2 fix and is pinned value-only. Ruling: `min_penalties`' primary is the absolute `min_shortfall` builder (no R ≤ 0 slack rows). Ruling: the budgeted objectives share one COSTBUDGET error message.

- [ ] **Step 1: Write the failing base-model tests:**

```python
"""MIPSolver._build_base (Stage 3 spec §2): the objective-free base model."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver

DATA = Path(__file__).parent.parent.parent / "data"


def _simple(**params: object) -> ConservationProblem:
    p = load_project(DATA / "simple")
    return p.copy_with(parameters={**p.parameters, **params})


def test_base_has_no_targets_budget_or_objective() -> None:
    base = MIPSolver()._build_base(_simple(COSTBUDGET=30.0), SolverConfig())
    names = set(base.model.constraints)
    assert not any(n.startswith(("target_", "cost_budget", "short_", "met_")) for n in names)
    assert base.model.objective is None or not base.model.objective.items()
    assert [t.key for t in base.terms] == ["f1", "f2", "f3"]
    assert set(base.amount_exprs) == {1, 2, 3}


def test_context_shares_the_registry() -> None:
    base = MIPSolver()._build_base(_simple(COSTBUDGET=30.0), SolverConfig())
    ctx = base.context()
    ctx.slack(base.terms[0])
    assert ctx.implied is base.implied and base.implied[-1][0].name == "s_f1"


def test_need_boundary_builds_the_unweighted_boundary_at_blm_zero() -> None:
    p = _simple(BLM=0.0)
    without = MIPSolver()._build_base(p, SolverConfig())
    assert not without.boundary_expr.items()
    assert not any(n.startswith("abs") for n in without.model.constraints)
    base = MIPSolver()._build_base(p, SolverConfig(), need_boundary=True)
    assert {"abs1_1_2", "abs2_5_6"} <= set(base.model.constraints)
    coefs = {v.name: a for v, a in base.boundary_expr.items()}
    assert coefs["x_1"] == 2.0 and coefs["y_1_2"] == 1.0  # self-boundary, shared edge


def test_base_registers_boundary_and_connectivity_auxiliaries() -> None:
    conn = pd.DataFrame({"id1": [1, 2], "id2": [2, 3], "value": [1.0, 2.0]})
    for asym in (0, 1):
        p = _simple(CONNECTIVITY_WEIGHT=1.0, ASYMMETRIC_CONNECTIVITY=asym).copy_with(
            connectivity=conn,
        )
        base = MIPSolver()._build_base(p, SolverConfig())
        names = [v.name for v, _ in base.implied]
        assert names[:5] == ["y_1_2", "y_2_3", "y_3_4", "y_4_5", "y_5_6"]
        assert names[5:] == ["zc_1_2", "zc_2_3"]
        base.x[1].varValue, base.x[2].varValue, base.x[3].varValue = 1.0, 0.0, 1.0
        rules = dict((v.name, rule) for v, rule in base.implied)
        assert rules["y_1_2"]() == 1.0
        # symmetric: x1·x2 = 0; asymmetric: x2·(1 − x3) = 0 and x1·(1 − x2) = 1
        assert rules["zc_1_2"]() == (1.0 if asym else 0.0)
        assert rules["zc_2_3"]() == 0.0


def test_min_penalties_primary_is_absolute_min_shortfall_builder() -> None:
    """Ruling (plan Task 3): min_penalties' primary is build_min_shortfall(ctx, "absolute"),
    so a target with R = target × MISSLEVEL <= 0 gets no slack row."""
    model = MIPSolver(objective="min_penalties").build_model(_simple(MISSLEVEL=0.0))
    assert not any(n.startswith("short_") for n in model.constraints)
    model = MIPSolver(objective="min_penalties").build_model(_simple())
    assert {"short_f1", "short_f2", "short_f3"} <= set(model.constraints)
```

- [ ] **Step 2: Pin the deliberate ghost change in the Task 1 golden test:**

```diff
diff --git a/tests/pymarxan/solvers/test_stage3_golden_v040.py b/tests/pymarxan/solvers/test_stage3_golden_v040.py
index f3b65d4..c06ed6f 100644
--- a/tests/pymarxan/solvers/test_stage3_golden_v040.py
+++ b/tests/pymarxan/solvers/test_stage3_golden_v040.py
@@ -196,9 +196,21 @@ def _assert_same(got: dict[str, Any], want: dict[str, Any]) -> None:
             assert got[key] == want[key], key
 
 
+# Spec §2 Fixed: absolute min_largest_shortfall now counts a ghost feature (target > 0, no
+# pu_vs_features rows) as m >= R = target × MISSLEVEL (10 × 0.9), as relative mode, the zone
+# solver and the v0.39 max_* ghost fix do. v0.40 reported 0.0 (all real targets met).
+# Every plan ties at m = R, so only the value is pinned.
+STAGE3_FIXED = {"ghost_misslevel09|min_largest_shortfall|absolute": 9.0}
+
+
 @pytest.mark.parametrize("case", CASES)
 def test_mip_solver_result_unchanged(golden: dict[str, Any], case: str) -> None:
-    _assert_same(_run(case), golden[case])
+    got = _run(case)
+    if case in STAGE3_FIXED:
+        assert golden[case]["mip_objective_value"] == 0.0  # the v0.40 value
+        assert got["mip_objective_value"] == pytest.approx(STAGE3_FIXED[case], abs=1e-9)
+        return
+    _assert_same(got, golden[case])
 
 
 def test_simple_min_set_anchor_in_golden(golden: dict[str, Any]) -> None:
@@ -210,13 +222,16 @@ def test_simple_min_set_anchor_in_golden(golden: dict[str, Any]) -> None:
 
 
 def test_ghost_absolute_min_largest_shortfall(golden: dict[str, Any]) -> None:
-    """v0.40 ignores the ghost (target 50, no rows) and reports a largest shortfall of 5.0."""
-    rec = golden["ghost_absolute_mls_v040"]
-    assert rec["mip_objective_value"] == 5.0
+    """Spec §2 Fixed: v0.40 ignored the ghost (target 50, no rows) and reported 5.0.
+
+    The builder counts it (m >= 50). The ghost holds m at 50, so every plan ties and CBC
+    returns the empty plan where v0.40 returned a cost-23 one: pin only the value.
+    """
+    assert golden["ghost_absolute_mls_v040"]["mip_objective_value"] == 5.0
     got = _result(
         _ghost_absolute_mls(), ONE, objective="min_largest_shortfall", shortfall="absolute",
     )
-    _assert_same(got, rec)
+    assert got["mip_objective_value"] == 50.0
 
 
 if __name__ == "__main__":
```

- [ ] **Step 3: Run to verify they fail.**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_mip_build_base.py tests/pymarxan/solvers/test_stage3_golden_v040.py -q -p no:cacheprovider`

Expected: `6 failed, 120 passed`: three base tests with `AttributeError: 'MIPSolver' object has no attribute '_build_base'`, `test_min_penalties_primary_is_absolute_min_shortfall_builder` with `assert not True` (v0.40 adds `short_*` rows at MISSLEVEL 0), `test_mip_solver_result_unchanged[ghost_misslevel09|min_largest_shortfall|absolute]` (`assert 0.0 == 9.0 ± ...`) and `test_ghost_absolute_min_largest_shortfall` (`assert 5.0 == 50.0`).

- [ ] **Step 4: Implement the split.** Apply this change to `src/pymarxan/solvers/mip_solver.py`. `_build_base` is the old `_build` body moved verbatim, with the expressions started as `pulp.LpAffineExpression()` (not `0`), the boundary condition widened to `(blm > 0 or need_boundary)`, one `implied.append(...)` per auxiliary, and the no-good cuts moved in; every objective-specific block is deleted.

```diff
diff --git a/src/pymarxan/solvers/mip_solver.py b/src/pymarxan/solvers/mip_solver.py
index b6f1e02..e464cda 100644
--- a/src/pymarxan/solvers/mip_solver.py
+++ b/src/pymarxan/solvers/mip_solver.py
@@ -3,6 +3,7 @@ from __future__ import annotations
 
 import copy
 import warnings
+from collections.abc import Callable
 from dataclasses import dataclass
 
 import numpy as np
@@ -24,8 +25,12 @@ from pymarxan.solvers._backends import (
 )
 from pymarxan.solvers._mip_objectives import (
     BUILDERS,
+    ImpliedRule,
     ObjectiveContext,
     TargetTerm,
+    build_min_cost,
+    build_min_shortfall,
+    expr_value,
     resolve_objective,
     set_objective,
     solve_lexicographic,
@@ -95,18 +100,6 @@ def _core_spf(problem: ConservationProblem) -> dict[int, float]:
     }
 
 
-def _feat_met_vars(problem: ConservationProblem) -> dict[int, pulp.LpVariable]:
-    """Binary met-indicator per feature with raw target > 0 (spec §1; as the zone
-    solver's ``_zone_target_terms``): a target-0 feature is no term and is not counted."""
-    ids = problem.features["id"].values
-    targets = problem.features["target"].values.astype(float)
-    return {
-        int(f): pulp.LpVariable(f"feat_met_{int(f)}", cat="Binary")
-        for f, t in zip(ids, targets, strict=True)
-        if t > 0
-    }
-
-
 MIP_OBJECTIVES = (
     "min_set",
     "max_features",
@@ -165,6 +158,53 @@ def _core_target_terms(
     return terms
 
 
+def _value(var: pulp.LpVariable) -> float:
+    return float(var.varValue or 0.0)
+
+
+def _abs_rule(a: pulp.LpVariable, b: pulp.LpVariable) -> Callable[[], float]:
+    """Boundary auxiliary: |x_i − x_j|, as the abs rows bound it."""
+    return lambda: abs(_value(a) - _value(b))
+
+
+def _and_rule(a: pulp.LpVariable, b: pulp.LpVariable) -> Callable[[], float]:
+    """Symmetric connectivity auxiliary: x_i·x_j."""
+    return lambda: _value(a) * _value(b)
+
+
+def _src_without_sink_rule(i: pulp.LpVariable, j: pulp.LpVariable) -> Callable[[], float]:
+    """Asymmetric connectivity auxiliary: x_i·(1 − x_j)."""
+    return lambda: _value(i) * (1.0 - _value(j))
+
+
+@dataclass
+class _BaseModel:
+    """Result of ``MIPSolver._build_base``: the model every objective shares (spec §2)."""
+
+    model: pulp.LpProblem
+    x: dict[int, pulp.LpVariable]
+    pu_ids: list[int]
+    blm: float
+    cost_expr: pulp.LpAffineExpression
+    boundary_expr: pulp.LpAffineExpression  # unweighted
+    conn_expr: pulp.LpAffineExpression
+    prob_expr: pulp.LpAffineExpression
+    amount_exprs: dict[int, pulp.LpAffineExpression]
+    terms: list[TargetTerm]
+    implied: list[ImpliedRule]
+    probmode: int
+    has_target2: bool
+    has_sep_active: bool
+
+    def context(self) -> ObjectiveContext:
+        """One ``ObjectiveContext`` sharing this base's expressions and registry."""
+        return ObjectiveContext(
+            self.model, self.cost_expr, self.terms, blm=self.blm,
+            boundary_expr=self.boundary_expr, conn_expr=self.conn_expr,
+            prob_expr=self.prob_expr, amount_exprs=self.amount_exprs, implied=self.implied,
+        )
+
+
 @dataclass
 class _BuiltModel:
     """Result of ``MIPSolver._build``: the PuLP model plus what ``solve`` needs."""
@@ -291,11 +331,19 @@ class MIPSolver(Solver):
     def available(self) -> bool:
         return True  # pulp is a core dependency
 
-    def _build(self, problem: ConservationProblem, config: SolverConfig) -> _BuiltModel:
-        """Build the PuLP model (no warnings, no solve)."""
-        objective_expr: pulp.LpAffineExpression | None = None
-        lex: tuple[pulp.LpAffineExpression, pulp.LpAffineExpression] | None = None
+    def _build_base(
+        self,
+        problem: ConservationProblem,
+        config: SolverConfig,
+        *,
+        need_boundary: bool = False,
+    ) -> _BaseModel:
+        """Everything every objective shares (spec §2); no targets, budget or objective.
 
+        The boundary rows are built when a boundary table exists and BLM > 0 or
+        ``need_boundary``. Every auxiliary variable is registered in ``implied`` with the
+        value its rows imply from x (spec §5).
+        """
         # PROBMODE 3 gating — see class docstring for rationale.
         probmode = int(problem.parameters.get("PROBMODE", 0))
         if probmode == 3:
@@ -351,6 +399,7 @@ class MIPSolver(Solver):
 
         blm = float(problem.parameters.get("BLM", 0.0))
         pu_ids = problem.planning_units["id"].tolist()
+        implied: list[ImpliedRule] = []
 
         # Build the LP model
         model = pulp.LpProblem("MarxanMIP", pulp.LpMinimize)
@@ -368,17 +417,17 @@ class MIPSolver(Solver):
             elif status == STATUS_LOCKED_OUT:
                 model += x[pid] == 0, f"locked_out_{pid}"
 
-        # Objective: cost component
+        # Cost component
         pu_cost_arr = problem.planning_units["cost"].values.astype(float)
         cost_expr = pulp.lpSum(
             float(pu_cost_arr[k]) * x[int(pu_id_arr[k])]
             for k in range(len(pu_id_arr))
         )
 
-        # Boundary component with auxiliary variables
-        boundary_expr = 0
+        # Unweighted boundary component with auxiliary variables
+        boundary_expr = pulp.LpAffineExpression()
         y = {}
-        if blm > 0 and problem.boundary is not None:
+        if (blm > 0 or need_boundary) and problem.boundary is not None:
             b_id1 = problem.boundary["id1"].values
             b_id2 = problem.boundary["id2"].values
             b_val = problem.boundary["boundary"].values.astype(float)
@@ -406,10 +455,11 @@ class MIPSolver(Solver):
                             y_var >= x[key[1]] - x[key[0]],
                             f"abs2_{key[0]}_{key[1]}",
                         )
+                        implied.append((y_var, _abs_rule(x[key[0]], x[key[1]])))
                     boundary_expr += bval * y[key]
 
         # Connectivity component with auxiliary variables
-        conn_expr = 0
+        conn_expr = pulp.LpAffineExpression()
         conn_weight = float(problem.parameters.get("CONNECTIVITY_WEIGHT", 0.0))
         asymmetric = bool(
             int(problem.parameters.get("ASYMMETRIC_CONNECTIVITY", 0))
@@ -432,11 +482,13 @@ class MIPSolver(Solver):
                         f"zc_{i}_{j}", lowBound=0, cat="Continuous",
                     )
                     model += z >= x[i] - x[j], f"conn_asym_{i}_{j}"
+                    implied.append((z, _src_without_sink_rule(x[i], x[j])))
                     conn_expr += conn_weight * cval * z
             else:
                 # Reward both selected via binary-AND linearization:
                 #   z_ij <= x_i, z_ij <= x_j, z_ij >= x_i + x_j - 1
                 #   objective: -conn_weight * c_ij * z_ij
+                # The first row of a duplicated (i, j) / (j, i) pair wins.
                 seen: set[tuple[int, int]] = set()
                 for ck in range(len(c_id1)):
                     i = int(c_id1[ck])
@@ -453,10 +505,12 @@ class MIPSolver(Solver):
                         z >= x[key[0]] + x[key[1]] - 1,
                         f"conn_and3_{key[0]}_{key[1]}",
                     )
+                    implied.append((z, _and_rule(x[key[0]], x[key[1]])))
                     conn_expr += -conn_weight * cval * z
 
-        # Probability risk premium (Mode 1)
-        prob_expr = 0
+        # Probability risk premium (Mode 1). PROBMODE defaults to 1 here (0 for the
+        # PROBMODE 3 gate above): a probability table without PROBMODE is Mode 1.
+        prob_expr = pulp.LpAffineExpression()
         prob_mode = int(problem.parameters.get("PROBMODE", 1))
         if problem.probability is not None and prob_mode == 1:
             prob_weight = float(
@@ -488,162 +542,7 @@ class MIPSolver(Solver):
                 prob_map_m2[int(p_pu[pk])] = float(p_val[pk])
 
         amount_exprs = _core_amount_exprs(problem, x, prob_map_m2)
-
-        # Phase 23: select objective formulation. Default "min_set" preserves
-        # pre-Phase-23 behaviour.
-        feat_met: dict[int, pulp.LpVariable] = {}
-        slack: dict[int, pulp.LpVariable] = {}
-        t_var: pulp.LpVariable | None = None
-        feat_ids_init = problem.features["id"].values
-        builder_path = self.objective == "min_shortfall" or (
-            self.objective == "min_largest_shortfall" and self.shortfall == "relative"
-        )
-        if builder_path:
-            cost_budget = problem.parameters.get("COSTBUDGET")
-            if cost_budget is None:
-                raise ValueError(
-                    f"objective={self.objective!r} requires a COSTBUDGET parameter "
-                    "(problem.parameters['COSTBUDGET'] = ...); without a budget every "
-                    "shortfall is 0 by selecting all planning units."
-                )
-            terms = _core_target_terms(problem, x, prob_map_m2, amount_exprs)
-            ctx = ObjectiveContext(model, cost_expr, terms)
-            term = BUILDERS[self.objective](ctx, self.shortfall)
-            set_objective(model, term)
-            model += cost_expr <= float(cost_budget), "cost_budget"
-            objective_expr = term.expr
-        elif self.objective == "max_features":
-            # Binary z_j per feature; relaxed target ≥ target · z_j; cost
-            # cap from COSTBUDGET. Maximise Σ z_j (LpMinimize → negate).
-            cost_budget = problem.parameters.get("COSTBUDGET")
-            if cost_budget is None:
-                raise ValueError(
-                    "objective='max_features' requires a COSTBUDGET parameter "
-                    "on the problem (problem.parameters['COSTBUDGET'] = ...). "
-                    "Without a budget the formulation is degenerate — every "
-                    "feature target is trivially met by selecting all PUs."
-                )
-            cost_budget = float(cost_budget)
-            feat_met = _feat_met_vars(problem)
-            model += (
-                -pulp.lpSum(feat_met[fid] for fid in feat_met),
-                "objective",
-            )
-            model += cost_expr <= cost_budget, "cost_budget"
-        elif self.objective == "max_weighted_features":
-            # Binary z_j per feature; relaxed target ≥ target · z_j; cost cap
-            # from COSTBUDGET. Maximise Σ spf_j·z_j (LpMinimize → negate) — the
-            # spf-weighted count of targets met. With spf = branch length this
-            # maximises phylogenetic diversity. This weights the *target-met*
-            # indicator; it is NOT the objectives/ MaxUtility class (which sums
-            # held amounts). Like max_features, a feature with no pu_vs_features
-            # rows can never be met (feat_met is forced to 0 below). Solution.objective
-            # is recomputed min_set-style and is NOT the weighted objective value.
-            cost_budget = problem.parameters.get("COSTBUDGET")
-            if cost_budget is None:
-                raise ValueError(
-                    "objective='max_weighted_features' requires a COSTBUDGET "
-                    "parameter on the problem "
-                    "(problem.parameters['COSTBUDGET'] = ...). Without a budget "
-                    "the formulation is degenerate — every target is trivially "
-                    "met by selecting all PUs."
-                )
-            cost_budget = float(cost_budget)
-            spf_by_id = _core_spf(problem)
-            feat_met = _feat_met_vars(problem)
-            model += (
-                -pulp.lpSum(spf_by_id[fid] * feat_met[fid] for fid in feat_met),
-                "objective",
-            )
-            model += cost_expr <= cost_budget, "cost_budget"
-        elif self.objective == "min_largest_shortfall":
-            # Auxiliary slack_j ≥ 0, slack_j ≥ target_j - Σ amount·x;
-            # t ≥ slack_j for every feature. Minimise t.
-            cost_budget = problem.parameters.get("COSTBUDGET")
-            if cost_budget is None:
-                raise ValueError(
-                    "objective='min_largest_shortfall' requires a COSTBUDGET "
-                    "parameter (problem.parameters['COSTBUDGET'] = ...). "
-                    "Without a budget the objective is degenerate (buying "
-                    "every PU drives every shortfall to 0)."
-                )
-            cost_budget = float(cost_budget)
-            t_var = pulp.LpVariable("max_shortfall", lowBound=0)
-            for fi in range(len(feat_ids_init)):
-                fid = int(feat_ids_init[fi])
-                slack[fid] = pulp.LpVariable(f"slack_{fid}", lowBound=0)
-                # Coupling slack ≤ t handled below via t ≥ slack.
-            model += (t_var, "objective")
-            model += cost_expr <= cost_budget, "cost_budget"
-        elif self.objective == "min_penalties":
-            # Exact lexicographic (S2): pass 1 min Σ SPF·shortfall, pass 2 min cost at that
-            # optimum (solve_lexicographic). v0.38's big-M (Σ cost + 1) only dominated when
-            # SPF·Δshortfall >= 1, so fractional data could trade targets for cost. Not
-            # prioritizr's add_min_penalties_objective (penalty-only, hard targets).
-            terms = _core_target_terms(problem, x, prob_map_m2, amount_exprs)
-            ctx = ObjectiveContext(model, cost_expr, terms)
-            primary = pulp.lpSum(t.weight * ctx.slack(t) for t in ctx.terms)
-            model.setObjective(primary)  # build_model exports the pass-1 model
-            lex = (primary, cost_expr)
-            objective_expr = primary
-        else:
-            model += (
-                cost_expr + blm * boundary_expr + conn_expr + prob_expr,
-                "objective",
-            )
-
-        # Constraints: feature targets
-        misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
-
-        # Pre-group feature data for O(1) lookup instead of O(R) filtering
-        feat_groups = problem.pu_vs_features.groupby("species")
-
-        feat_ids = problem.features["id"].values
-        feat_targets = problem.features["target"].values.astype(float)
-        for fi in range(len(feat_ids)):
-            fid = int(feat_ids[fi])
-            target = float(feat_targets[fi]) * misslevel
-
-            if fid in feat_groups.groups and not (
-                builder_path or self.objective == "min_penalties"
-            ):
-                amount_expr = amount_exprs[fid]
-                if self.objective in ("max_features", "max_weighted_features"):
-                    if fid not in feat_met:
-                        continue  # target 0: no term (spec §1), no row
-                    # Relaxed target: only enforced when feat_met_j = 1.
-                    # Maximising Σ (weight·)feat_met_j therefore picks as many
-                    # (weighted) features as the cost budget will pay for.
-                    model += (
-                        amount_expr >= target * feat_met[fid],
-                        f"target_{fid}",
-                    )
-                elif self.objective == "min_largest_shortfall":
-                    # slack_j ≥ target_j − amount; t ≥ slack_j.
-                    model += (
-                        slack[fid] >= target - amount_expr,
-                        f"slack_{fid}",
-                    )
-                    assert t_var is not None  # for mypy: set in init block above
-                    model += t_var >= slack[fid], f"max_shortfall_{fid}"
-                else:
-                    model += amount_expr >= target, f"target_{fid}"
-
-        if feat_met:
-            # Ghost features (target > 0, no pu_vs_features rows) can never be met; v0.38
-            # left feat_met unconstrained so the maximiser counted them for free.
-            # feat_met exists for raw target > 0 (spec §1), but a ghost is forced to 0 only
-            # when target × MISSLEVEL > 0: at MISSLEVEL 0 the required amount is 0, so the
-            # term is met (the zone solver's 0 >= 0·y agrees) and the two conditions differ
-            # deliberately.
-            for fi in range(len(feat_ids)):
-                fid = int(feat_ids[fi])
-                if (
-                    fid in feat_met
-                    and fid not in feat_groups.groups
-                    and float(feat_targets[fi]) * misslevel > 0
-                ):
-                    feat_met[fid].upBound = 0
+        terms = _core_target_terms(problem, x, prob_map_m2, amount_exprs)
 
         # Phase 25: no-good cuts for portfolio generation. Each forbidden
         # selection vector contributes one constraint forcing at least
@@ -666,14 +565,59 @@ class MIPSolver(Solver):
                     diff_terms.append(x[pid])
             model += pulp.lpSum(diff_terms) >= 1, f"nogood_cut_{forbid_idx}"
 
-        if objective_expr is None:  # natural units: min_set / max_* / absolute minimax
-            objective_expr = model.objective
-            if self.objective in ("max_features", "max_weighted_features"):
-                objective_expr = -model.objective
+        return _BaseModel(
+            model=model, x=x, pu_ids=pu_ids, blm=blm, cost_expr=cost_expr,
+            boundary_expr=boundary_expr, conn_expr=conn_expr, prob_expr=prob_expr,
+            amount_exprs=amount_exprs, terms=terms, implied=implied, probmode=probmode,
+            has_target2=bool(has_target2), has_sep_active=bool(has_sep_active),
+        )
+
+    def _build(self, problem: ConservationProblem, config: SolverConfig) -> _BuiltModel:
+        """Build the PuLP model (no warnings, no solve): the base plus this objective."""
+        base = self._build_base(problem, config)
+        model = base.model
+        ctx = base.context()
+        lex: tuple[pulp.LpAffineExpression, pulp.LpAffineExpression] | None = None
+        if self.objective == "min_set":
+            term = build_min_cost(ctx, self.shortfall)
+            set_objective(model, term)
+            # Hard targets for every feature with pu_vs_features rows (a ghost adds none).
+            misslevel = float(problem.parameters.get("MISSLEVEL", 1.0))
+            for fid_raw, target in zip(
+                problem.features["id"].values,
+                problem.features["target"].values.astype(float),
+                strict=True,
+            ):
+                fid = int(fid_raw)
+                if fid in base.amount_exprs:
+                    model += (
+                        base.amount_exprs[fid] >= float(target) * misslevel,
+                        f"target_{fid}",
+                    )
+        elif self.objective == "min_penalties":
+            # Exact lexicographic (S2): pass 1 min Σ SPF·shortfall, pass 2 min cost at that
+            # optimum (solve_lexicographic). v0.38's big-M (Σ cost + 1) only dominated when
+            # SPF·Δshortfall >= 1, so fractional data could trade targets for cost. Not
+            # prioritizr's add_min_penalties_objective (penalty-only, hard targets).
+            term = build_min_shortfall(ctx, "absolute")
+            model.setObjective(term.expr)  # build_model exports the pass-1 model
+            lex = (term.expr, base.cost_expr)
+        else:
+            cost_budget = problem.parameters.get("COSTBUDGET")
+            if cost_budget is None:
+                raise ValueError(
+                    f"objective={self.objective!r} requires a COSTBUDGET parameter "
+                    "(problem.parameters['COSTBUDGET'] = ...); without a budget the "
+                    "objective is degenerate (selecting every planning unit meets "
+                    "every target)."
+                )
+            term = BUILDERS[self.objective](ctx, self.shortfall)
+            set_objective(model, term)
+            model += base.cost_expr <= float(cost_budget), "cost_budget"
         return _BuiltModel(
-            model=model, x=x, pu_ids=pu_ids, blm=blm, objective_expr=objective_expr,
-            probmode=probmode, has_target2=bool(has_target2),
-            has_sep_active=bool(has_sep_active), lex=lex,
+            model=model, x=base.x, pu_ids=base.pu_ids, blm=base.blm, objective_expr=term.expr,
+            probmode=base.probmode, has_target2=base.has_target2,
+            has_sep_active=base.has_sep_active, lex=lex,
         )
 
     def build_model(
@@ -754,12 +698,8 @@ class MIPSolver(Solver):
         # automatically when PROBMODE==3. Under the 'drop' strategy, the
         # MIP solved the deterministic relaxation; the chance-constraint
         # gap is what build_solution reports post-hoc.
-        # pulp.value() is None when any variable in the expression has no value (zero-cost
-        # x never written to the LP); evaluate the objective reading those as 0.
-        obj_expr = built.objective_expr
-        obj_value = float(obj_expr.constant or 0.0) + sum(
-            float(coef) * float(v.varValue or 0.0) for v, coef in obj_expr.items()
-        )
+        # A zero-cost x never written to the LP has no value: expr_value reads it as 0.
+        obj_value = expr_value(built.objective_expr)
         meta = {
             "solver": self.name(),
             "status": pulp.LpStatus[status],
```

- [ ] **Step 5: Regenerate the v0.38 golden and check that only the two entries changed.**

```bash
PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/python -W ignore tests/pymarxan/solvers/test_stage1_golden_v038.py
PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/python - <<'EOF'
import json, subprocess
new = json.load(open("tests/data/golden/stage1_v038.json"))
old = json.loads(subprocess.check_output(["git", "show", "HEAD:tests/data/golden/stage1_v038.json"]))
print(sorted(k for k in old if old[k] != new[k]), new["core_mls_abs_relabs_selected"])
EOF
```
Expected: `['core_mls_abs_infeasible', 'core_mls_abs_relabs'] [1, 0, 0]`. In both entries the constraints become `cost_budget, maxs_f1, maxs_f2, short_f1, short_f2` (were `max_shortfall_1/2`, `slack_1/2`), the slack variables `s_f1, s_f2`, and the objective is still `[["max_shortfall", 1.0]]`. If any other key is listed, the refactor is wrong; do not regenerate around it.

Record the reason in the v0.38 file's docstring:

```diff
diff --git a/tests/pymarxan/solvers/test_stage1_golden_v038.py b/tests/pymarxan/solvers/test_stage1_golden_v038.py
index dc5782d..0644790 100644
--- a/tests/pymarxan/solvers/test_stage1_golden_v038.py
+++ b/tests/pymarxan/solvers/test_stage1_golden_v038.py
@@ -1,6 +1,15 @@
 """v0.38 MIP models and selections that Stage 1 must not change (spec Testing 4, 5, 12).
 
-GOLDEN was recorded on unmodified v0.38 (e5a066c). Regenerate only with a written reason:
+GOLDEN was recorded on unmodified v0.38 (e5a066c). Regenerate only with a written reason.
+
+Regenerated for v0.41 (Stage 3, spec §2): ``core_mls_abs_relabs`` and
+``core_mls_abs_infeasible`` only. Absolute ``min_largest_shortfall`` moved onto
+``build_min_largest_shortfall``, which renames its rows and variables (``slack_{id}`` ->
+``s_f{id}`` / ``short_f{id}``, ``max_shortfall_{id}`` -> ``maxs_f{id}``) and drops R <= 0
+slack rows. The pinned selection ``[1, 0, 0]`` is unchanged; every other entry is
+byte-identical.
+
+Regenerate with:
     PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/python \
         tests/pymarxan/solvers/test_stage1_golden_v038.py
 """
```

- [ ] **Step 6: CHANGELOG** (`[Unreleased]`):

```diff
diff --git a/CHANGELOG.md b/CHANGELOG.md
index 6934325..3564a67 100644
--- a/CHANGELOG.md
+++ b/CHANGELOG.md
@@ -7,6 +7,14 @@ and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0
 
 ## [Unreleased]
 
+### Changed
+
+- `MIPSolver` internals are refactored onto the objective builders in `solvers/_mip_objectives.py`: `_build` is now a shared base model (`_build_base`: locks, cost, boundary, connectivity, probability, amounts, no-good cuts) plus one builder per objective. Solutions are unchanged (pinned by `tests/data/golden/stage3_v040.json`), except the ghost-feature fix below. LP variable and constraint names in exported models (`build_model`, `write_problem`) change for `max_features` / `max_weighted_features` / absolute `min_largest_shortfall`: `feat_met_{id}` → `y_f{id}`, `slack_{id}` → `s_f{id}`, the relaxed `target_{id}` → `met_f{id}`, `slack_{id}` rows → `short_f{id}`, `max_shortfall_{id}` → `maxs_f{id}`; the objective is no longer named `"objective"`. `min_set`'s hard `target_{id}` rows keep their names. The missing-`COSTBUDGET` error for `max_features`, `max_weighted_features` and absolute `min_largest_shortfall` now uses the shared `objective=... requires a COSTBUDGET parameter` text. Absolute `min_largest_shortfall` and `min_penalties` no longer add slack rows for a feature with target × MISSLEVEL ≤ 0, and `max_*` ghost features get a `met_f{id}` row instead of a fixed `upBound`.
+
+### Fixed
+
+- Absolute `min_largest_shortfall` now counts a ghost feature (target > 0, no `pu_vs_features` rows): its whole target is unmet, so the largest shortfall is at least target × MISSLEVEL, as in relative mode, `ZoneMIPSolver` and the v0.39 `max_*` ghost fix. v0.40 ignored it: the simple fixture plus a ghost with target 50 and COSTBUDGET 30 reported 5.0 (now 50.0), and with a ghost of target 10 at MISSLEVEL 0.9 and COSTBUDGET 33 it reported 0.0 (now 9.0). With a ghost present every plan ties on the minimax, so the returned plan degenerates (CBC may return the empty plan); `presolve_check` (P5) flags ghost features.
+
 ## [0.40.0] — 2026-10-15
 
 ### Added
```

- [ ] **Step 7: Run the goldens, the solver and zone suites, and the full suite** (this is the step that can break the whole tree).

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_mip_build_base.py tests/pymarxan/solvers/test_stage3_golden_v040.py tests/pymarxan/solvers/test_stage1_golden_v038.py -q -p no:cacheprovider`

Expected: `131 passed`.

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/ -q -m 'not bench' -p no:cacheprovider`

Expected: all pass (3,048 = 2,903 baseline + 122 + 18 + 5; the prototype ran this suite at 3,047, before the registry-sharing test was added).

- [ ] **Step 8: Lint, type-check, commit.**

```bash
~/.local/bin/ruff check src tests
/home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports
git add src/pymarxan/solvers/mip_solver.py tests/pymarxan/solvers/test_mip_build_base.py tests/pymarxan/solvers/test_stage3_golden_v040.py tests/pymarxan/solvers/test_stage1_golden_v038.py tests/data/golden/stage1_v038.json CHANGELOG.md
git commit -m "refactor(mip): split MIPSolver._build into _build_base plus objective builders

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AZEiop9qqxVCrMHKqSDYP8"
```

---

### Task 4: `evaluate_implied`, `solve_hierarchical`, `HierarchicalResult`

**Files:**
- Modify: `src/pymarxan/solvers/_mip_objectives.py` (append)
- Create: `tests/pymarxan/scripted_solver.py` (test helper: the `sol_status` mock seam)
- Create: `tests/pymarxan/solvers/test_mip_hierarchical.py`

**Interfaces:**
- Consumes: `ObjectiveTerm`, `set_objective`, `expr_value`, `ImpliedRule`; in tests, `MIPSolver._build_base` and `_BaseModel.context()`.
- Produces:
  - `ACCEPTED_SOL_STATUSES = {LpSolutionOptimal, LpSolutionIntegerFeasible}`.
  - `evaluate_implied(model, x, x_rounded, implied, terms) -> tuple[list[float], bool]`: snapshot the tracked set (`model.variables() ∪ x ∪ registry`), set x, apply rules in order, read each `term.expr`, check **every** row with `valid(1e-6)`, restore.
  - `HierarchicalResult(status, sol_status, stage_reached)` (frozen).
  - `solve_hierarchical(model, terms, rel_tols, abs_tol, solver, evaluate, *, tracked) -> HierarchicalResult`: reset → solve → accept on sol_status 1/2 → snapshot → cap `hier_cap_k` at `min/max(v_model, v_rec)` (v_rec None → v_model) with `r_k·|v| + abs_tol`; restore the previous snapshot on a failed stage; caps deleted in `finally`. Raises nothing.
  - `constraint_map(model) -> dict[str, LpConstraint]` (rows without PuLP 3.3's deprecated mapping access).
  - `tests/pymarxan/scripted_solver.py`: `Scripted(hooks)` (CBC, then `hooks[call](lp)`, recording each solved model's `(sense, rhs)` per row) and three hooks: `fail` (no solution, and junk values: every x at 1, the rest None, so a missing restore is caught), `optimal_without_solution` (status Optimal, sol_status 0: rejected) and `not_solved_with_incumbent` (status NotSolved, sol_status 2: accepted).

Ruling: tests mock `sol_status` through `tests/pymarxan/scripted_solver.py` (a `pulp.LpSolver` that runs CBC, then a per-call hook), with no production test hook. Ruling: keyword-only `tracked` (x and the registry) is required, with no default, as in spec §4.1; `model.variables()` alone misses those variables. Ruling: rows are read and deleted through `constraint_map`, not PuLP 3.3's deprecated `constraints` mapping. Ruling: the module-level test helper is `build_case` (pytest < 8 nose-compat would call a module-level `setup`).

- [ ] **Step 1: Write the test helper and the failing tests.** The test cases port spec-review rounds 4–7 (tolerance boundary, incumbent, near-integral budget, connectivity with locked endpoints, stale slacks, cap contradiction) and the status snapshot.

`tests/pymarxan/scripted_solver.py`:

```python
"""A CBC wrapper whose per-call hooks rewrite the result: the seam for mocking ``sol_status``."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pulp

from pymarxan.solvers._mip_objectives import constraint_map

Hook = Callable[[pulp.LpProblem], None]


class Scripted(pulp.LpSolver):
    """Solve with CBC, record each model's rows, then run ``hooks[call]`` if given.

    ``models[i]`` maps every constraint name to ``(sense, rhs)`` as solved in call i + 1.
    """

    def __init__(self, hooks: dict[int, Hook] | None = None) -> None:
        super().__init__(msg=False)
        self.inner = pulp.PULP_CBC_CMD(msg=False)
        self.hooks = hooks or {}
        self.calls = 0
        self.models: list[dict[str, tuple[int, float]]] = []

    def actualSolve(self, lp: pulp.LpProblem, **kwargs: Any) -> int:  # noqa: N802
        self.calls += 1
        self.inner.actualSolve(lp)
        self.models.append({
            n: (int(c.sense), -float(c.constant)) for n, c in constraint_map(lp).items()
        })
        hook = self.hooks.get(self.calls)
        if hook is not None:
            hook(lp)
        return int(lp.status)


def fail(lp: pulp.LpProblem) -> None:
    """Hook: the solve found no solution, and left junk values (every x at 1, the rest
    None) so that a caller that forgets to restore the previous stage is caught."""
    lp.status, lp.sol_status = pulp.LpStatusNotSolved, pulp.LpSolutionNoSolutionFound
    for v in lp.variables():
        v.varValue = 1.0 if v.name.startswith("x_") else None


def optimal_without_solution(lp: pulp.LpProblem) -> None:
    """Hook: status Optimal but sol_status NoSolutionFound: must count as no solution."""
    lp.status, lp.sol_status = pulp.LpStatusOptimal, pulp.LpSolutionNoSolutionFound


def not_solved_with_incumbent(lp: pulp.LpProblem) -> None:
    """Hook: status NotSolved (CBC on a time limit) but sol_status IntegerFeasible: an
    incumbent, which must be accepted."""
    lp.status, lp.sol_status = pulp.LpStatusNotSolved, pulp.LpSolutionIntegerFeasible
```

`tests/pymarxan/solvers/test_mip_hierarchical.py`:

```python
"""evaluate_implied and solve_hierarchical (Stage 3 spec §4.1, §5, §7.4, §7.5).

Ports the spec-review cases: round 4 (tolerance boundary), round 5 (incumbent), round 6
(near-integral budget, asymmetric connectivity with locked endpoints), round 7 (stale slacks,
cap contradiction) and the status snapshot after a failed stage.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import pandas as pd
import pulp
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers._mip_objectives import (
    HierarchicalResult,
    ObjectiveContext,
    ObjectiveTerm,
    build_max_features,
    build_max_weighted_features,
    build_min_cost,
    build_min_largest_shortfall,
    build_min_shortfall,
    constraint_map,
    evaluate_implied,
    expr_value,
    solve_hierarchical,
)
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver, _BaseModel
from tests.pymarxan.scripted_solver import (
    Scripted,
    fail,
    not_solved_with_incumbent,
    optimal_without_solution,
)

Builder = Callable[..., ObjectiveTerm]


def P(pu: dict[str, list[Any]], feat: dict[str, list[Any]], pvf: dict[str, list[Any]],
      **kw: Any) -> ConservationProblem:
    params = kw.pop("params", {})
    return ConservationProblem(
        planning_units=pd.DataFrame(pu), features=pd.DataFrame(feat),
        pu_vs_features=pd.DataFrame(pvf), parameters=params, **kw,
    )


def build_case(
    problem: ConservationProblem, builders: list[Builder]
) -> tuple[_BaseModel, ObjectiveContext, list[ObjectiveTerm]]:
    base = MIPSolver()._build_base(problem, SolverConfig(), need_boundary=True)
    if "COSTBUDGET" in problem.parameters:
        base.model += base.cost_expr <= float(problem.parameters["COSTBUDGET"]), "cost_budget"
    ctx = base.context()
    return base, ctx, [b(ctx, "relative") for b in builders]


def hier(
    base: _BaseModel, terms: list[ObjectiveTerm], rel_tols: list[float], *,
    solver: pulp.LpSolver | None = None, abs_tol: float = 1e-9,
) -> HierarchicalResult:
    def evaluate(i: int) -> float | None:
        rounded = {pid: float(round(v.varValue or 0.0)) for pid, v in base.x.items()}
        vals, feasible = evaluate_implied(base.model, base.x, rounded, base.implied, [terms[i]])
        return vals[0] if feasible else None

    tracked = [*base.x.values(), *(v for v, _ in base.implied)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # PuLP 3.3.2
        return solve_hierarchical(
            base.model, terms, rel_tols, abs_tol, solver or Scripted(), evaluate,
            tracked=tracked,
        )


def xs(base: _BaseModel) -> list[int]:
    return [int(round(v.varValue or 0.0)) for v in base.x.values()]


def caps(model: pulp.LpProblem) -> list[str]:
    return [n for n in constraint_map(model) if n.startswith("hier_cap_")]


TOL = P(  # round 4: 0.7 + 0.1 = 0.7999999999999999 < 0.8, but CBC calls the target met
    {"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]},
    {"id": [1], "name": ["a"], "target": [0.8]},
    {"species": [1, 1], "pu": [1, 2], "amount": [0.7, 0.1]},
)


# --- evaluate_implied -------------------------------------------------------------


def test_evaluate_implied_values_feasibility_and_restore() -> None:
    p = P({"id": [1, 2, 3], "cost": [1.0, 2.0, 0.0], "status": [0, 0, 0]},
          {"id": [1, 2], "name": ["a", "b"], "target": [2.0, 4.0]},
          {"species": [1, 2], "pu": [1, 2], "amount": [2.0, 4.0]})
    base, ctx, terms = build_case(p, [build_min_cost, build_min_largest_shortfall,
                                 build_max_features])
    assert "x_3" not in {v.name for v in base.model.variables()}  # zero cost, no rows
    for v, _ in base.implied:
        v.varValue = 9.0  # stale
    base.x[1].varValue, base.x[2].varValue, base.x[3].varValue = 0.9999999, 0.0, None
    before = {v.name: v.varValue for v in [*base.x.values(), *(v for v, _ in base.implied)]}
    vals, feasible = evaluate_implied(
        base.model, base.x, {1: 1.0, 2: 0.0, 3: 0.0}, base.implied, terms,
    )
    assert feasible
    assert vals == [1.0, 1.0, 1.0]  # cost 1; f2 fully short (relative 1.0); f1 met
    after = {v.name: v.varValue for v in [*base.x.values(), *(v for v, _ in base.implied)]}
    assert after == before  # every tracked value restored, x_3 still None


def test_evaluate_implied_infeasible_when_rounding_breaks_the_budget() -> None:
    # round 6: x1 = 1 − 5e-8 fits budget 1e7·(1 − 5e-8); x1 rounded to 1 does not.
    c1, eps = 1e7, 5e-8
    p = P({"id": [1, 2], "cost": [c1, 1.0], "status": [0, 0]},
          {"id": [1], "name": ["a"], "target": [10.0]},
          {"species": [1], "pu": [1], "amount": [10.0]},
          params={"COSTBUDGET": c1 * (1 - eps)})
    base, _, terms = build_case(p, [build_max_features])
    vals, feasible = evaluate_implied(base.model, base.x, {1: 1.0, 2: 0.0}, base.implied, terms)
    assert vals == [1.0] and not feasible


def test_evaluate_implied_m_reads_fresh_slacks() -> None:
    # round 7 C5: stale slacks s = 3, m = 5; at x = (1, 1) the true largest shortfall is 0.
    p = P({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]},
          {"id": [1, 2], "name": ["a", "b"], "target": [2.0, 4.0]},
          {"species": [1, 2], "pu": [1, 2], "amount": [2.0, 4.0]})
    base, _, terms = build_case(p, [build_min_cost, build_min_largest_shortfall])
    for v, _ in base.implied:
        v.varValue = 3.0 if v.name.startswith("s_") else 5.0
    vals, _ = evaluate_implied(base.model, base.x, {1: 1.0, 2: 1.0}, base.implied, terms)
    assert vals == [2.0, 0.0]
    vals, _ = evaluate_implied(base.model, base.x, {1: 1.0, 2: 0.0}, base.implied, terms)
    assert vals == [1.0, 1.0]


# --- solve_hierarchical: the cap ---------------------------------------------------


@pytest.mark.parametrize("rel_tol", [0.0, 0.5])
def test_min_cap_is_v_plus_r_abs_v_plus_abs_tol(rel_tol: float) -> None:
    # Stage 1 min MinCost with a cost of -4 locked in: v = -4 < 0, so |v| matters.
    p = P({"id": [1, 2], "cost": [-4.0, 1.0], "status": [2, 0]},
          {"id": [1], "name": ["a"], "target": [1.0]},
          {"species": [1], "pu": [2], "amount": [1.0]})
    base, _, terms = build_case(p, [build_min_cost, build_min_shortfall])
    solver = Scripted()
    res = hier(base, terms, [rel_tol], solver=solver, abs_tol=0.25)
    assert res == HierarchicalResult(pulp.LpStatusOptimal, 1, 2)
    sense, rhs = solver.models[1]["hier_cap_1"]
    assert sense == pulp.LpConstraintLE
    assert rhs == pytest.approx(-4.0 + rel_tol * 4.0 + 0.25)
    assert caps(base.model) == []  # removed after the run


def test_max_cap_is_v_minus_r_abs_v_minus_abs_tol() -> None:
    p = P({"id": [1, 2, 3], "cost": [1.0, 1.0, 1.0], "status": [0, 0, 0]},
          {"id": [1, 2], "name": ["a", "b"], "target": [1.0, 1.0]},
          {"species": [1, 2], "pu": [1, 2], "amount": [1.0, 1.0]})
    base, _, terms = build_case(p, [build_max_features, build_min_cost])
    solver = Scripted()
    hier(base, terms, [0.25], solver=solver, abs_tol=0.1)
    sense, rhs = solver.models[1]["hier_cap_1"]
    assert sense == pulp.LpConstraintGE
    assert rhs == pytest.approx(2.0 - (0.25 * 2.0 + 0.1))
    assert xs(base) == [1, 1, 0]  # Σ y >= 1.4 needs both targets: PU1 and PU2


def test_rel_tol_is_indexed_by_stage() -> None:
    p = P({"id": [1, 2, 3], "cost": [1.0, 2.0, 3.0], "status": [0, 0, 0]},
          {"id": [1, 2], "name": ["a", "b"], "target": [1.0, 1.0]},
          {"species": [1, 2], "pu": [1, 2], "amount": [1.0, 1.0]})
    base, _, terms = build_case(p, [build_max_features, build_min_shortfall, build_min_cost])
    solver = Scripted()
    hier(base, terms, [0.5, 0.0], solver=solver, abs_tol=0.0)
    assert solver.models[2]["hier_cap_1"][1] == pytest.approx(2.0 - 0.5 * 2.0)
    assert solver.models[2]["hier_cap_2"][1] == pytest.approx(0.0)


# --- solve_hierarchical: v = better of v_model and v_rec ------------------------------


def test_tolerance_boundary_cap_keeps_the_model_target() -> None:
    # round 4 (§7.4): v_model = 1, v_rec = 0. Recompute-only caps at 0 and returns the
    # empty plan; best-of-both keeps the target and costs 2.
    base, _, terms = build_case(TOL, [build_max_features, build_min_cost])
    res = hier(base, terms, [0.0])
    assert res.stage_reached == 2
    assert xs(base) == [1, 1]


def test_incumbent_cap_uses_the_recomputed_value() -> None:
    # round 5 (§7.4): a stage-1 incumbent (sol_status 2) meets both targets but leaves
    # y_f2 = 0 (v_model = 1, v_rec = 2). The cap must use 2, so stage 2 keeps both targets.
    p = P({"id": [1, 2, 3], "cost": [1.0, 1.0, 5.0], "status": [0, 0, 0]},
          {"id": [1, 2], "name": ["a", "b"], "target": [1.0, 1.0]},
          {"species": [1, 2, 1, 2], "pu": [1, 2, 3, 3], "amount": [1.0, 1.0, 1.0, 1.0]},
          params={"COSTBUDGET": 5.0})
    base, _, terms = build_case(p, [build_max_features, build_min_cost])

    def loose(lp: pulp.LpProblem) -> None:
        lp.sol_status = pulp.LpSolutionIntegerFeasible
        values = {"x_1": 1.0, "x_2": 1.0, "x_3": 0.0, "y_f1": 1.0, "y_f2": 0.0}
        for v in lp.variables():
            if v.name in values:
                v.varValue = values[v.name]

    solver = Scripted({1: loose})
    res = hier(base, terms, [0.0], solver=solver)
    assert solver.models[1]["hier_cap_1"][1] == pytest.approx(2.0 - 1e-9)
    assert res.stage_reached == 2
    assert xs(base) == [1, 1, 0]  # both targets kept at cost 2


def test_near_integral_budget_falls_back_to_the_model_value() -> None:
    # round 6 (§7.5): rounding x1 = 1 − 5e-8 breaks COSTBUDGET, so v = v_model = 0.
    c1, eps = 1e7, 5e-8
    p = P({"id": [1, 2], "cost": [c1, 1.0], "status": [0, 0]},
          {"id": [1], "name": ["a"], "target": [10.0]},
          {"species": [1], "pu": [1], "amount": [10.0]},
          params={"COSTBUDGET": c1 * (1 - eps)})
    base, _, terms = build_case(p, [build_max_features, build_min_cost])

    def near(lp: pulp.LpProblem) -> None:
        lp.sol_status = pulp.LpSolutionIntegerFeasible
        values = {"x_1": 1 - eps, "x_2": 0.0, "y_f1": 0.0}
        for v in lp.variables():
            if v.name in values:
                v.varValue = values[v.name]

    solver = Scripted({1: near})
    res = hier(base, terms, [0.0], solver=solver)
    assert solver.models[1]["hier_cap_1"][1] == pytest.approx(-1e-9)
    assert res.stage_reached == 2


def test_earlier_caps_are_checked_three_stages() -> None:
    # round 7 (§7.5): negative SPF on tolerance-boundary amounts. Skipping hier_cap_* rows
    # in the feasibility check lets cap 2 contradict cap 1 (stage_reached 2).
    p = P({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]},
          {"id": [1], "name": ["a"], "target": [0.8], "spf": [-1.0]},
          {"species": [1, 1], "pu": [1, 2], "amount": [0.7, 0.1]})
    base, _, terms = build_case(p, [build_max_features, build_max_weighted_features,
                               build_min_cost])
    res = hier(base, terms, [0.0, 0.0])
    assert res.stage_reached == 3


@pytest.mark.parametrize(("asym", "weighting"), [(1, None), (0, None), (0, -1.0), (0, 0.7)])
def test_connectivity_with_locked_endpoints_reaches_stage_2(
    asym: int, weighting: float | None,
) -> None:
    # round 6 (§7.5): v_rec is the model's own expression at the implied point.
    params: dict[str, Any] = {"CONNECTIVITY_WEIGHT": 1.0, "ASYMMETRIC_CONNECTIVITY": asym}
    kw: dict[str, Any] = {}
    if weighting is not None:
        params["PROBABILITYWEIGHTING"] = weighting
        kw["probability"] = pd.DataFrame({"pu": [1, 2, 3], "probability": [0.5] * 3})
    p = P({"id": [1, 2, 3], "cost": [1.0] * 3, "status": [2, 2, 0]},
          {"id": [1], "name": ["a"], "target": [5.0]},
          {"species": [1], "pu": [3], "amount": [5.0]},
          connectivity=pd.DataFrame({"id1": [1], "id2": [2], "value": [10.0]}),
          params=params, **kw)
    base, _, terms = build_case(p, [build_min_cost, build_min_shortfall])
    assert hier(base, terms, [0.0]).stage_reached == 2


def test_zero_cost_pu_in_no_row_does_not_break_v() -> None:
    p = P({"id": [1, 2], "cost": [1.0, 0.0], "status": [0, 0]},
          {"id": [1], "name": ["a"], "target": [1.0]},
          {"species": [1], "pu": [1], "amount": [1.0]})
    base, _, terms = build_case(p, [build_min_cost, build_min_shortfall])
    res = hier(base, terms, [0.0])
    assert res.stage_reached == 2 and xs(base) == [0, 0]
    assert base.x[2].varValue is None


# --- solve_hierarchical: failures --------------------------------------------------


def test_stage_2_failure_restores_stage_1_and_its_status() -> None:
    # Budget 1: the best relative shortfall is 0.5, from PU1 only (PU2 costs 2). The failed
    # stage 2 leaves every x at 1 (shortfall 0): only a restore gives back 0.5 and [1, 0].
    p = P({"id": [1, 2], "cost": [1.0, 2.0], "status": [0, 0]},
          {"id": [1], "name": ["a"], "target": [2.0]},
          {"species": [1, 1], "pu": [1, 2], "amount": [1.0, 1.0]},
          params={"COSTBUDGET": 1.0})
    base, _, terms = build_case(p, [build_min_shortfall, build_min_cost])
    res = hier(base, terms, [0.0], solver=Scripted({2: fail}))
    assert res == HierarchicalResult(pulp.LpStatusOptimal, pulp.LpSolutionOptimal, 1)
    assert xs(base) == [1, 0]
    assert expr_value(terms[0].expr) == pytest.approx(0.5)
    assert caps(base.model) == []


def test_acceptance_follows_sol_status_not_status() -> None:
    p = P({"id": [1, 2], "cost": [1.0, 2.0], "status": [0, 0]},
          {"id": [1], "name": ["a"], "target": [2.0]},
          {"species": [1, 1], "pu": [1, 2], "amount": [1.0, 1.0]},
          params={"COSTBUDGET": 1.0})
    base, _, terms = build_case(p, [build_min_shortfall, build_min_cost])
    # status Optimal, sol_status NoSolutionFound: no solution.
    res = hier(base, terms, [0.0], solver=Scripted({2: optimal_without_solution}))
    assert res.stage_reached == 1
    # status NotSolved, sol_status IntegerFeasible (a time-limit incumbent): accepted.
    res = hier(base, terms, [0.0], solver=Scripted({2: not_solved_with_incumbent}))
    assert res == HierarchicalResult(pulp.LpStatusNotSolved, pulp.LpSolutionIntegerFeasible, 2)


def test_stage_1_failure_reports_stage_0() -> None:
    base, _, terms = build_case(TOL, [build_min_cost, build_min_shortfall])
    res = hier(base, terms, [0.0], solver=Scripted({1: fail}))
    assert res.stage_reached == 0 and caps(base.model) == []


def test_values_are_reset_before_each_solve() -> None:
    # A failed stage 2 that leaves no values must not see stage-1 values as a solution.
    base, _, terms = build_case(TOL, [build_min_cost, build_min_shortfall])
    seen: list[set[Any]] = []

    def record(lp: pulp.LpProblem) -> None:
        seen.append({v.varValue for v in lp.variables()})
        fail(lp)

    class Blank(Scripted):
        def actualSolve(self, lp: pulp.LpProblem, **kwargs: Any) -> int:  # noqa: N802
            if self.calls == 0:
                return super().actualSolve(lp)
            self.calls += 1
            record(lp)  # no inner solve: values are whatever the helper left
            return int(lp.status)

    res = hier(base, terms, [0.0], solver=Blank())
    assert seen == [{None}]
    assert res.stage_reached == 1
```

- [ ] **Step 2: Run to verify they fail.**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_mip_hierarchical.py -q -p no:cacheprovider`

Expected: collection error, `ImportError: cannot import name 'HierarchicalResult' from 'pymarxan.solvers._mip_objectives'`.

- [ ] **Step 3: Implement** (appended to `_mip_objectives.py`; `solve_lexicographic` is unchanged):

```diff
diff --git a/src/pymarxan/solvers/_mip_objectives.py b/src/pymarxan/solvers/_mip_objectives.py
index ffe39b9..a4f64e6 100644
--- a/src/pymarxan/solvers/_mip_objectives.py
+++ b/src/pymarxan/solvers/_mip_objectives.py
@@ -11,7 +11,7 @@ core-only: they read ``ObjectiveContext`` fields that ``MIPSolver`` fills from
 from __future__ import annotations
 
 import warnings
-from collections.abc import Callable, Mapping
+from collections.abc import Callable, Iterable, Mapping, Sequence
 from dataclasses import dataclass, field
 from typing import Any, Protocol
 
@@ -22,6 +22,8 @@ OBJECTIVE_ALIASES = {"max_n_targets_met": "max_features"}  # prioritizr >= 9 nam
 NO_SOLUTION_STATUSES = frozenset(
     {pulp.LpStatusInfeasible, pulp.LpStatusUnbounded, pulp.LpStatusUndefined}
 )
+# A multi-objective solve is accepted only on these solution statuses (spec §4.1).
+ACCEPTED_SOL_STATUSES = frozenset({pulp.LpSolutionOptimal, pulp.LpSolutionIntegerFeasible})
 
 
 def resolve_objective(name: str, allowed: tuple[str, ...]) -> str:
@@ -283,3 +285,136 @@ def solve_lexicographic(
         )
         return status1
     return int(model.status)
+
+
+def constraint_map(model: pulp.LpProblem) -> dict[str, pulp.LpConstraint]:
+    """Name -> row of ``model``, without PuLP 3.3's deprecated mapping access.
+
+    PuLP 3.3 warns on every dict-style use of ``LpProblem.constraints`` (PuLP 4.0 makes
+    it a method returning a list) and has no public row deletion; PuLP 2.x stores the
+    dict on ``constraints`` itself. ``pyproject`` allows ``PuLP>=2.7,<4``.
+    """
+    rows = getattr(model, "_constraints", None)
+    if isinstance(rows, dict):
+        return rows
+    legacy: dict[str, pulp.LpConstraint] = model.constraints
+    return legacy
+
+
+def _tracked(
+    model: pulp.LpProblem, extra: Iterable[pulp.LpVariable]
+) -> list[pulp.LpVariable]:
+    """``model.variables()`` ∪ ``extra``, deduplicated (spec §5 tracked variable set).
+
+    A PU with zero cost and no rows, and an auxiliary in an empty row, are not in
+    ``model.variables()``; callers pass x and the registry as ``extra``.
+    """
+    seen: dict[int, pulp.LpVariable] = {}
+    for v in [*model.variables(), *extra]:
+        seen.setdefault(id(v), v)
+    return list(seen.values())
+
+
+def evaluate_implied(
+    model: pulp.LpProblem,
+    x: Mapping[int, pulp.LpVariable],
+    x_rounded: Mapping[int, float],
+    implied: Sequence[ImpliedRule],
+    terms: Sequence[ObjectiveTerm],
+) -> tuple[list[float], bool]:
+    """Each term's value at the implied point of ``x_rounded``, and whether that point is
+    feasible (spec §5).
+
+    Sets x to ``x_rounded``, applies every implied rule in registry order, reads each
+    ``term.expr``, checks **every** model row (``hier_cap_*`` included) with
+    ``valid(1e-6)``, and restores every tracked ``varValue``.
+    """
+    tracked = _tracked(model, [*x.values(), *(v for v, _ in implied)])
+    snapshot = [(v, v.varValue) for v in tracked]
+    try:
+        for pid, var in x.items():
+            var.varValue = float(x_rounded[pid])
+        for var, rule in implied:
+            var.varValue = rule()
+        values = [expr_value(t.expr) for t in terms]
+        feasible = all(c.valid(1e-6) for c in constraint_map(model).values())
+    finally:
+        for var, value in snapshot:
+            var.varValue = value
+    return values, feasible
+
+
+@dataclass(frozen=True)
+class HierarchicalResult:
+    """``status`` / ``sol_status`` of the last accepted stage; ``stage_reached`` = accepted
+    stages (n on full success, 0 when stage 1 has no solution)."""
+
+    status: int
+    sol_status: int
+    stage_reached: int
+
+
+def solve_hierarchical(
+    model: pulp.LpProblem,
+    terms: Sequence[ObjectiveTerm],
+    rel_tols: Sequence[float],
+    abs_tol: float,
+    solver: pulp.LpSolver,
+    evaluate: Callable[[int], float | None],
+    *,
+    tracked: Iterable[pulp.LpVariable],
+) -> HierarchicalResult:
+    """Lexicographic optimisation of ``terms`` in list order (spec §4.1).
+
+    Stage k < n is capped (row ``hier_cap_{k}``) at ``f ≤ v + r_k·|v| + abs_tol``
+    (minimise) or ``f ≥ v − (r_k·|v| + abs_tol)`` (maximise), ``r_k = rel_tols[k-1]``.
+    ``v`` is the better, in the stage's sense, of the model value of ``term.expr`` and
+    ``evaluate(k-1)`` (the implied value, ``None`` when the implied point is infeasible).
+    Values are reset to ``None`` before each solve, and a solve is accepted only when
+    ``model.sol_status`` is 1 or 2. On a failed stage the previous stage's values are
+    restored and the loop stops. Cap rows are removed before returning, also on failure.
+    ``tracked`` adds variables missing from ``model.variables()`` (x, the registry) to the
+    reset / snapshot / restore. Raises nothing itself.
+    """
+    extra = list(tracked)
+    caps: list[str] = []
+    snapshot: list[tuple[pulp.LpVariable, float | None]] | None = None
+    accepted = HierarchicalResult(0, 0, 0)
+    try:
+        for k, term in enumerate(terms, start=1):
+            variables = _tracked(model, extra)
+            for var in variables:
+                var.varValue = None
+            set_objective(model, term)
+            model.solve(solver)
+            if model.sol_status not in ACCEPTED_SOL_STATUSES:
+                if snapshot is None:
+                    return HierarchicalResult(int(model.status), int(model.sol_status), 0)
+                for var in variables:
+                    var.varValue = None
+                for var, value in snapshot:
+                    var.varValue = value
+                return accepted
+            snapshot = [(var, var.varValue) for var in _tracked(model, extra)]
+            accepted = HierarchicalResult(int(model.status), int(model.sol_status), k)
+            if k == len(terms):
+                break
+            v_model = expr_value(term.expr)
+            v_rec = evaluate(k - 1)
+            if v_rec is None:
+                v = v_model
+            elif term.sense == pulp.LpMinimize:
+                v = min(v_model, v_rec)
+            else:
+                v = max(v_model, v_rec)
+            slack = float(rel_tols[k - 1]) * abs(v) + abs_tol
+            name = f"hier_cap_{k}"
+            if term.sense == pulp.LpMinimize:
+                model += term.expr <= v + slack, name
+            else:
+                model += term.expr >= v - slack, name
+            caps.append(name)
+        return accepted
+    finally:
+        for name in caps:
+            del constraint_map(model)[name]
```

- [ ] **Step 4: Run.**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_mip_hierarchical.py -q -p no:cacheprovider`

Expected: `20 passed`. Then `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/ tests/pymarxan/zones/ -q -p no:cacheprovider`: all pass.

- [ ] **Step 5: Lint, type-check, commit.**

```bash
~/.local/bin/ruff check src tests
/home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports
git add src/pymarxan/solvers/_mip_objectives.py tests/pymarxan/scripted_solver.py tests/pymarxan/solvers/test_mip_hierarchical.py
git commit -m "feat(mip): evaluate_implied and solve_hierarchical with best-of-both caps

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AZEiop9qqxVCrMHKqSDYP8"
```

---

### Task 5: `pymarxan.multiobjective`: specs, approaches, matrices, `nondominated`

**Files:**
- Create: `src/pymarxan/multiobjective.py`
- Create: `tests/pymarxan/test_multiobjective_core.py`

**Interfaces:**
- Consumes: `BUILDERS`, `ObjectiveContext`, `ObjectiveTerm`, `validate_shortfall`, `Solution`.
- Produces:
  - `ObjectiveSpec` (frozen; `name` / `sense` ClassVars; `label` kw-only, must be an identifier; `key = label or name`; `build(ctx)` calls the builder with `owner=f"{key}__"`). Subclasses `MinCost`, `MinBoundary`, `MinShortfall(shortfall="relative")`, `MinLargestShortfall(shortfall="relative")`, `MaxFeatures`, `MaxWeightedFeatures`, `MaxWtdSum(weights=None)` (finite, any sign; stored as `{int: float}`). An unknown `shortfall` raises at construction.
  - `Hierarchical(rel_tol, abs_tol=1e-9)` (`rel_tol` required, as in the spec) and `WeightedSum(weights)`: values validated at construction; `.rows(n)` returns one row per run and checks the length (n−1 / n).
  - `weights_matrix(n_objectives, n_values, *, include_zeros=True, include_extremes=True)` and `rel_tol_matrix(n_objectives, n_values, max, *, include_zeros=True, order=True)` (prioritizr row order: first column fastest), `list[list[float]]`.
  - `nondominated(solutions) -> list[Solution]`.
  - `__all__` (Task 6 adds `MultiObjectiveMIP`).

Ruling: `Hierarchical.rel_tol` is required (no default), as in the spec's signature. Ruling: specs and approaches compare by value (`@dataclass(frozen=True)` on every class: a subclass declared `eq=False` would silently inherit the base class's label-only `__eq__`). Ruling: the matrix helpers accept numpy integers and reject `bool`.

- [ ] **Step 1: Write the failing tests:**

```python
"""pymarxan.multiobjective pure parts (Stage 3 spec §3, §4.1-4.3, §5 nondominated, §6)."""
from __future__ import annotations

import math

import numpy as np
import pulp
import pytest

from pymarxan.multiobjective import (
    Hierarchical,
    MaxFeatures,
    MaxWeightedFeatures,
    MaxWtdSum,
    MinBoundary,
    MinCost,
    MinLargestShortfall,
    MinShortfall,
    WeightedSum,
    nondominated,
    rel_tol_matrix,
    weights_matrix,
)
from pymarxan.solvers._mip_objectives import ObjectiveContext, TargetTerm
from pymarxan.solvers.base import Solution

# --- specs ----------------------------------------------------------------------


@pytest.mark.parametrize(("spec", "name", "sense"), [
    (MinCost(), "min_cost", "min"),
    (MinBoundary(), "min_boundary", "min"),
    (MinShortfall(), "min_shortfall", "min"),
    (MinLargestShortfall(), "min_largest_shortfall", "min"),
    (MaxFeatures(), "max_features", "max"),
    (MaxWeightedFeatures(), "max_weighted_features", "max"),
    (MaxWtdSum(), "max_wtd_sum", "max"),
])
def test_spec_name_sense_and_key(spec: object, name: str, sense: str) -> None:
    assert (spec.name, spec.sense, spec.key) == (name, sense, name)  # type: ignore[attr-defined]


def test_label_is_the_key_and_must_be_an_identifier() -> None:
    assert MinShortfall(label="rep").key == "rep"
    assert MinShortfall("absolute").shortfall == "absolute"  # shortfall is positional
    for bad in ("two words", "a-b", "1st", ""):
        with pytest.raises(ValueError, match="label"):
            MinCost(label=bad)


def test_unknown_shortfall_raises_at_construction() -> None:
    with pytest.raises(ValueError, match="shortfall"):
        MinShortfall(shortfall="percent")
    with pytest.raises(ValueError, match="shortfall"):
        MinLargestShortfall(shortfall="percent")


def test_max_wtd_sum_weights_finite_any_sign() -> None:
    assert MaxWtdSum(weights={1: -2, 2: 0.5}).weights == {1: -2.0, 2: 0.5}
    assert MaxWtdSum(weights={1: 1}) != MaxWtdSum(weights={1: 2})  # weights take part in ==
    for bad in (math.nan, math.inf, "x"):
        with pytest.raises(ValueError, match="weights"):
            MaxWtdSum(weights={1: bad})  # type: ignore[dict-item]


def test_spec_build_prefixes_owned_names_with_the_key() -> None:
    m = pulp.LpProblem("o", pulp.LpMinimize)
    x = pulp.LpVariable("x_1", cat="Binary")
    ctx = ObjectiveContext(m, pulp.lpSum([x]), [TargetTerm("f1", "feature", 2 * x, 2.0, 1.0)])
    MinLargestShortfall(label="worst").build(ctx)
    MinLargestShortfall("absolute").build(ctx)
    assert {"worst__maxs_f1", "min_largest_shortfall__maxs_f1", "short_f1"} <= set(
        m.constraints
    )


# --- approaches -------------------------------------------------------------------


def test_hierarchical_rows() -> None:
    assert Hierarchical(0.1).rows(3) == [[0.1, 0.1]]
    assert Hierarchical([0.0, 2.0]).rows(3) == [[0.0, 2.0]]  # values > 1 allowed
    assert Hierarchical([[0.0, 0.5], [0.5, 0.0]]).rows(3) == [[0.0, 0.5], [0.5, 0.0]]
    assert Hierarchical(np.array([[0.1], [0.2]])).rows(2) == [[0.1], [0.2]]
    assert Hierarchical(0.3).rows(1) == [[]]
    assert Hierarchical(0.0).abs_tol == 1e-9
    assert Hierarchical(0.1) == Hierarchical(0.1) and WeightedSum([1, 2]) == WeightedSum([1, 2])


@pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf, [0.1, -1.0], [[0.1], [math.nan]]])
def test_hierarchical_rel_tol_values(bad: object) -> None:
    with pytest.raises(ValueError, match="rel_tol"):
        Hierarchical(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [-1e-9, math.inf])
def test_hierarchical_abs_tol(bad: float) -> None:
    with pytest.raises(ValueError, match="abs_tol"):
        Hierarchical(0.0, abs_tol=bad)


def test_hierarchical_row_length() -> None:
    with pytest.raises(ValueError, match="rel_tol row 1"):
        Hierarchical([[0.1, 0.2], [0.1]]).rows(3)


def test_weighted_sum_rows_and_errors() -> None:
    assert WeightedSum([1, 3]).rows(2) == [[1.0, 3.0]]
    assert WeightedSum([[1, 0], [0, 1]]).rows(2) == [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(ValueError, match="weights row 0 has 3"):
        WeightedSum([1, 1, 1]).rows(2)
    with pytest.raises(ValueError, match="weights row 1 sums to 0"):
        WeightedSum([[1, 0], [0, 0]])
    for bad in ([-1, 1], [math.nan, 1], [[1, math.inf]]):
        with pytest.raises(ValueError, match="weights"):
            WeightedSum(bad)  # type: ignore[arg-type]


# --- prioritizr matrices (spec §4.3) -------------------------------------------------


def test_weights_matrix_2_5_is_prioritizrs_15_rows() -> None:
    assert weights_matrix(2, 5) == [
        [1, 1], [1, 0], [0, 1], [.5, .25], [.75, .25], [1, .25], [.25, .5], [.75, .5],
        [1, .5], [.25, .75], [.5, .75], [1, .75], [.25, 1], [.5, 1], [.75, 1],
    ]


def test_weights_matrix_counts_and_options() -> None:
    assert len(weights_matrix(3, 3)) == 22
    assert weights_matrix(np.int64(2), np.int64(5)) == weights_matrix(2, 5)  # numpy ints
    assert len(weights_matrix(3, 5)) == 112
    no_ext = weights_matrix(2, 5, include_extremes=False)
    assert no_ext[0] == [1, 1] and [1, 0] not in no_ext and len(no_ext) == 13
    # Without zeros: 4 points (.25 … 1) per column; the extremes still hold zeros.
    no_zero = weights_matrix(2, 4, include_zeros=False)
    assert no_zero[:3] == [[1, 1], [1, 0], [0, 1]] and no_zero[3] == [.5, .25]


def test_rel_tol_matrix_pins() -> None:
    assert rel_tol_matrix(3, 3, 1) == [[0, 0], [.5, 0], [1, 0], [.5, .5], [1, .5], [1, 1]]
    assert len(rel_tol_matrix(3, 5, 1)) == 15
    assert len(rel_tol_matrix(3, 5, 1, order=False)) == 25
    assert rel_tol_matrix(2, 3, 0.2) == [[0.0], [0.1], [0.2]]
    assert rel_tol_matrix(2, 2, 1, include_zeros=False) == [[0.5], [1.0]]


@pytest.mark.parametrize("args", [(1, 5), (2, 1), (2.0, 5)])
def test_matrix_helpers_reject_small_sizes(args: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="n_"):
        weights_matrix(*args)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_"):
        rel_tol_matrix(*args, 1.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_rel_tol_matrix_rejects_bad_max(bad: float) -> None:
    with pytest.raises(ValueError, match="max"):
        rel_tol_matrix(3, 3, bad)


# --- nondominated (spec §5, §7.2) ---------------------------------------------------


def _sol(values: dict[str, float], senses: dict[str, str] | None = None) -> Solution:
    senses = senses or {k: "min" for k in values}
    return Solution(
        selected=np.zeros(1, dtype=bool), cost=0.0, boundary=0.0, objective=0.0,
        targets_met={}, metadata={"objective_values": values, "objective_senses": senses},
    )


def test_nondominated_ties_keep_the_first() -> None:
    a, b, c = _sol({"c": 1, "s": 2}), _sol({"c": 1, "s": 2}), _sol({"c": 2, "s": 1})
    b.cost = 99.0  # b equals a on every objective but is a different Solution
    kept = nondominated([a, b, c])
    assert len(kept) == 2 and kept[0] is a and kept[1] is c
    kept = nondominated([b, a, c])
    assert len(kept) == 2 and kept[0] is b and kept[1] is c


def test_nondominated_compares_exactly() -> None:
    # 1e-9 apart on both keys, in opposite directions: neither dominates, and they are not
    # identical, so both are kept (no tolerance).
    a, b = _sol({"c": 1.0, "s": 2.0}), _sol({"c": 1.0 + 1e-9, "s": 2.0 - 1e-9})
    kept = nondominated([a, b])
    assert len(kept) == 2 and kept[0] is a and kept[1] is b


def test_nondominated_drops_weakly_dominated() -> None:
    a, b = _sol({"c": 1, "s": 2}), _sol({"c": 1, "s": 3})
    assert nondominated([b, a]) == [a]


def test_nondominated_mixed_senses() -> None:
    senses = {"cost": "min", "held": "max"}
    a = _sol({"cost": 1, "held": 5}, senses)
    b = _sol({"cost": 2, "held": 4}, senses)  # dominated: costlier and holds less
    c = _sol({"cost": 3, "held": 9}, senses)
    assert nondominated([a, b, c]) == [a, c]


def test_nondominated_empty_and_errors() -> None:
    assert nondominated([]) == []
    with pytest.raises(ValueError, match="different"):
        nondominated([_sol({"c": 1}), _sol({"s": 1})])
    with pytest.raises(ValueError, match="different"):
        nondominated([_sol({"c": 1}), _sol({"c": 1}, {"c": "max"})])
    bare = _sol({"c": 1})
    bare.metadata = {}
    with pytest.raises(ValueError, match="lacks"):
        nondominated([bare])
```

- [ ] **Step 2: Run to verify they fail.**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/test_multiobjective_core.py -q -p no:cacheprovider`

Expected: collection error, `ModuleNotFoundError: No module named 'pymarxan.multiobjective'`.

- [ ] **Step 3: Implement `src/pymarxan/multiobjective.py`:**

```python
"""Exact multi-objective planning on one ``ConservationProblem`` (Stage 3, prioritizr 9 parity).

Objectives are specs (``MinCost``, ``MinBoundary``, ``MinShortfall``, ``MinLargestShortfall``,
``MaxFeatures``, ``MaxWeightedFeatures``, ``MaxWtdSum``) combined by an approach:

- ``Hierarchical(rel_tol, abs_tol=1e-9)``: lexicographic, in list order. Each stage caps the
  objective just solved at ``v + r·|v| + abs_tol`` (minimise) or ``v − (r·|v| + abs_tol)``
  (maximise) and solves the next. At ``v = 0`` only ``abs_tol`` is allowed.
- ``WeightedSum(weights)``: one solve of ``Σ (w/Σw)·s·f`` per weight row (s = −1 for a
  maximised objective). Objectives are **not** scaled: objectives in different units need
  weights scaled to their magnitudes (Das & Dennis 1997,
  https://doi.org/10.1007/BF01197559).

Targets are soft (only representation objectives use them) and ``COSTBUDGET`` is optional.
``weights_matrix`` / ``rel_tol_matrix`` build prioritizr's parameter sweeps and
``nondominated`` filters the resulting solutions. Deviations from prioritizr 9.0.1 are listed
in the CHANGELOG for v0.41.0 (Hanson et al. 2024, https://doi.org/10.1111/cobi.14376;
Williams & Kendall 2017, https://doi.org/10.1016/j.ecolmodel.2016.10.010).
"""
from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import Any, ClassVar

from pymarxan.solvers._mip_objectives import (
    BUILDERS,
    ObjectiveContext,
    ObjectiveTerm,
    validate_shortfall,
)
from pymarxan.solvers.base import Solution

__all__ = [
    "Hierarchical",
    "MaxFeatures",
    "MaxWeightedFeatures",
    "MaxWtdSum",
    "MinBoundary",
    "MinCost",
    "MinLargestShortfall",
    "MinShortfall",
    "ObjectiveSpec",
    "WeightedSum",
    "nondominated",
    "rel_tol_matrix",
    "weights_matrix",
]


# --- objective specs -------------------------------------------------------------


@dataclass(frozen=True)
class ObjectiveSpec:
    """One objective. ``key`` (``label or name``) names its reported value and owns its rows.

    ``label`` must be a Python identifier: PuLP rewrites other characters in names, so two
    distinct labels could collide.
    """

    name: ClassVar[str] = ""
    sense: ClassVar[str] = "min"
    label: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.label is not None and not (
            isinstance(self.label, str) and self.label.isidentifier()
        ):
            raise ValueError(f"label must be a Python identifier, got {self.label!r}")

    @property
    def key(self) -> str:
        return self.label or self.name

    def _shortfall(self) -> str:
        return "relative"

    def _params(self) -> dict[str, Any]:
        return {}

    def build(self, ctx: ObjectiveContext) -> ObjectiveTerm:
        """The builder's term; owned names are prefixed ``f"{key}__"``. Call once per ctx."""
        return BUILDERS[self.name](
            ctx, self._shortfall(), owner=f"{self.key}__", **self._params(),
        )


@dataclass(frozen=True)
class MinCost(ObjectiveSpec):
    """cost + BLM·boundary + connectivity + probability penalty (what ``min_set`` minimises).

    With BLM > 0 it already counts boundary, so a weighted sum with ``MinBoundary`` counts
    boundary twice.
    """

    name: ClassVar[str] = "min_cost"
    sense: ClassVar[str] = "min"


@dataclass(frozen=True)
class MinBoundary(ObjectiveSpec):
    """The boundary length alone, unweighted by BLM. Needs a boundary table."""

    name: ClassVar[str] = "min_boundary"
    sense: ClassVar[str] = "min"


@dataclass(frozen=True)
class MinShortfall(ObjectiveSpec):
    """Σ SPF·shortfall, relative (÷ target × MISSLEVEL, prioritizr) or absolute."""

    name: ClassVar[str] = "min_shortfall"
    sense: ClassVar[str] = "min"
    shortfall: str = "relative"

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_shortfall(self.shortfall)

    def _shortfall(self) -> str:
        return self.shortfall


@dataclass(frozen=True)
class MinLargestShortfall(ObjectiveSpec):
    """The largest (relative or absolute) shortfall over targets, unweighted."""

    name: ClassVar[str] = "min_largest_shortfall"
    sense: ClassVar[str] = "min"
    shortfall: str = "relative"

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_shortfall(self.shortfall)

    def _shortfall(self) -> str:
        return self.shortfall


@dataclass(frozen=True)
class MaxFeatures(ObjectiveSpec):
    """Count of targets met (features with target > 0)."""

    name: ClassVar[str] = "max_features"
    sense: ClassVar[str] = "max"


@dataclass(frozen=True)
class MaxWeightedFeatures(ObjectiveSpec):
    """SPF-weighted count of targets met."""

    name: ClassVar[str] = "max_weighted_features"
    sense: ClassVar[str] = "max"


@dataclass(frozen=True)
class MaxWtdSum(ObjectiveSpec):
    """Σ w_j · amount held_j (prioritizr ``max_wtd_sum``).

    ``weights``: ``None`` (every feature 1) or ``{feature_id: w}`` covering every feature in
    the problem (finite, any sign); coverage is checked at ``solve``.

    Caution (as prioritizr notes for its weighted-sum objectives): a weighted sum of amounts
    ignores complementarity. More of an already well-held feature scores as much as the
    same amount of a missing one, so the plan can leave features unrepresented; pair it
    with a shortfall objective when representation matters.
    """

    name: ClassVar[str] = "max_wtd_sum"
    sense: ClassVar[str] = "max"
    weights: Mapping[int, float] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.weights is not None:
            bad = sorted(k for k, w in self.weights.items() if not _finite(w))
            if bad:
                raise ValueError(f"weights must be finite numbers; bad feature ids {bad}")
            object.__setattr__(
                self, "weights", {int(k): float(w) for k, w in self.weights.items()},
            )

    def _params(self) -> dict[str, Any]:
        return {"weights": self.weights}


# --- approaches -----------------------------------------------------------------


def _finite(v: object) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, bool) and math.isfinite(v)


def _nonneg(name: str, v: object) -> float:
    if not _finite(v) or float(v) < 0:  # type: ignore[arg-type]
        raise ValueError(f"{name} values must be finite and >= 0, got {v!r}")
    return float(v)  # type: ignore[arg-type]


def _as_rows(name: str, value: object) -> list[list[float]]:
    """A float -> one row of one value; a 1-D sequence -> one row; a 2-D -> its rows."""
    if hasattr(value, "tolist"):  # numpy array or scalar
        value = value.tolist()
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return [[_nonneg(name, value)]]
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(f"{name} must be a number, a sequence or a 2-D sequence")
    items = list(value)
    if all(isinstance(v, Sequence) and not isinstance(v, str) for v in items) and items:
        return [[_nonneg(name, v) for v in row] for row in items]
    return [[_nonneg(name, v) for v in items]]


@dataclass(frozen=True)
class Hierarchical:
    """Lexicographic optimisation in list order (spec §4.1).

    ``rel_tol``: a float (every stage), n−1 floats, or a 2-D sequence (one row of n−1 per
    run); finite, >= 0, values > 1 allowed. The k-th tolerance caps the k-th stage.
    """

    rel_tol: float | Sequence[float] | Sequence[Sequence[float]]
    abs_tol: float = 1e-9

    def __post_init__(self) -> None:
        _as_rows("rel_tol", self.rel_tol)
        _nonneg("abs_tol", self.abs_tol)

    def rows(self, n: int) -> list[list[float]]:
        """One rel_tol row of n−1 values per run."""
        if isinstance(self.rel_tol, numbers.Real):
            return [[float(self.rel_tol)] * (n - 1)]
        rows = _as_rows("rel_tol", self.rel_tol)
        for i, row in enumerate(rows):
            if len(row) != n - 1:
                raise ValueError(
                    f"rel_tol row {i} has {len(row)} values; {n} objectives need {n - 1}"
                )
        return rows


@dataclass(frozen=True)
class WeightedSum:
    """One solve of ``Σ (w/Σw)·s·f`` per row (spec §4.2); no objective scaling.

    ``weights``: n floats, or a 2-D sequence (one row of n per run); finite, >= 0, and no
    row may sum to 0.
    """

    weights: Sequence[float] | Sequence[Sequence[float]]

    def __post_init__(self) -> None:
        for i, row in enumerate(_as_rows("weights", self.weights)):
            if sum(row) == 0:
                raise ValueError(f"weights row {i} sums to 0")

    def rows(self, n: int) -> list[list[float]]:
        rows = _as_rows("weights", self.weights)
        for i, row in enumerate(rows):
            if len(row) != n:
                raise ValueError(f"weights row {i} has {len(row)} values; expected {n}")
        return rows


# --- prioritizr parameter matrices ------------------------------------------------------


def _check_matrix_args(n_objectives: int, n_values: int) -> tuple[int, int]:
    """Both sizes as ``int`` (numpy integers accepted, ``bool`` rejected)."""
    for name, v in (("n_objectives", n_objectives), ("n_values", n_values)):
        if not isinstance(v, numbers.Integral) or isinstance(v, bool) or v < 2:
            raise ValueError(f"{name} must be an integer >= 2, got {v!r}")
    return int(n_objectives), int(n_values)


def _grid(values: list[float], n_cols: int) -> list[list[float]]:
    """R ``expand.grid`` row order: the first column varies fastest."""
    return [list(reversed(combo)) for combo in product(values, repeat=n_cols)]


def _points(top: float, n_values: int, include_zeros: bool) -> list[float]:
    if include_zeros:
        return [top * i / (n_values - 1) for i in range(n_values)]
    return [top * i / n_values for i in range(1, n_values + 1)]


def weights_matrix(
    n_objectives: int,
    n_values: int,
    *,
    include_zeros: bool = True,
    include_extremes: bool = True,
) -> list[list[float]]:
    """prioritizr ``approach_weights_matrix`` (9.0.1), row for row.

    A grid of ``n_values`` points in [0, 1] (with ``include_zeros=False``, the
    ``n_values`` non-zero points of an ``n_values + 1`` grid); rows whose values are all
    equal, or with fewer than two entries > 1e-6, are dropped; the identity rows are
    prepended if ``include_extremes`` and a row of ones always comes first. Despite
    prioritizr's documentation, proportional duplicates such as (0.5, 0.25) and (1, 0.5)
    are kept.
    """
    n_objectives, n_values = _check_matrix_args(n_objectives, n_values)
    rows = [
        row for row in _grid(_points(1.0, n_values, include_zeros), n_objectives)
        if len(set(row)) > 1 and sum(v > 1e-6 for v in row) >= 2
    ]
    if include_extremes:
        identity = [
            [1.0 if i == j else 0.0 for j in range(n_objectives)] for i in range(n_objectives)
        ]
        rows = identity + rows
    return [[1.0] * n_objectives] + rows


def rel_tol_matrix(
    n_objectives: int,
    n_values: int,
    max: float,  # prioritizr's argument name
    *,
    include_zeros: bool = True,
    order: bool = True,
) -> list[list[float]]:
    """prioritizr ``approach_rel_tol_matrix`` (9.0.1), row for row.

    A grid over n−1 columns of ``n_values`` points in [0, max] (with
    ``include_zeros=False``, the ``n_values`` non-zero points of an ``n_values + 1`` grid).
    With ``order``, only rows that are non-increasing across columns are kept, so an
    earlier stage gets at least the tolerance of a later one.
    """
    n_objectives, n_values = _check_matrix_args(n_objectives, n_values)
    if not _finite(max) or max < 0:
        raise ValueError(f"max must be finite and >= 0, got {max!r}")
    rows = _grid(_points(float(max), n_values, include_zeros), n_objectives - 1)
    if order:
        rows = [r for r in rows if all(r[i - 1] >= r[i] for i in range(1, len(r)))]
    return rows


# --- nondominated -------------------------------------------------------------------


def nondominated(solutions: Sequence[Solution]) -> list[Solution]:
    """The solutions no other solution dominates, in input order.

    Uses ``metadata["objective_values"]`` and ``["objective_senses"]``. A solution is
    dominated if another is at least as good on every key and strictly better on one.
    Values are compared exactly. Of identical value vectors only the first is kept.
    """
    if not solutions:
        return []
    keys: list[str] | None = None
    senses: dict[str, str] = {}
    vectors: list[list[float]] = []
    for i, sol in enumerate(solutions):
        values = sol.metadata.get("objective_values")
        sol_senses = sol.metadata.get("objective_senses")
        if not isinstance(values, Mapping) or not isinstance(sol_senses, Mapping):
            raise ValueError(
                f"solution {i} lacks metadata 'objective_values' / 'objective_senses'"
            )
        if keys is None:
            keys, senses = sorted(values), dict(sol_senses)
        if sorted(values) != keys or dict(sol_senses) != senses or set(senses) != set(keys):
            raise ValueError(f"solution {i} has different objective keys or senses")
        # Orient every key as a minimisation.
        vectors.append([
            float(values[k]) if senses[k] == "min" else -float(values[k]) for k in keys
        ])

    def dominates(a: list[float], b: list[float]) -> bool:
        return all(p <= q for p, q in zip(a, b, strict=True)) and a != b

    kept: list[Solution] = []
    for i, vec in enumerate(vectors):
        if any(dominates(other, vec) for other in vectors):
            continue
        if any(vectors[j] == vec for j in range(i)):
            continue
        kept.append(solutions[i])
    return kept
```

- [ ] **Step 4: Run.**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/test_multiobjective_core.py -q -p no:cacheprovider`

Expected: `35 passed`.

- [ ] **Step 5: Lint, type-check, commit.**

```bash
~/.local/bin/ruff check src tests
/home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports
git add src/pymarxan/multiobjective.py tests/pymarxan/test_multiobjective_core.py
git commit -m "feat(multiobjective): objective specs, approaches, prioritizr matrices, nondominated

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AZEiop9qqxVCrMHKqSDYP8"
```

---

### Task 6: `MultiObjectiveMIP.solve`

**Files:**
- Modify: `src/pymarxan/multiobjective.py`
- Modify: `src/pymarxan/solvers/mip_solver.py` (shared `_mip_solver_for` and `MIPSolver._strategy_metadata`, used by both solvers)
- Create: `tests/pymarxan/test_multiobjective_solve.py`

**Interfaces:**
- Consumes: `MIPSolver` (strategy/backend/presolve validation, `_build_base`), `_warn_if_large_mip`, `_presolve_data` / `_emit_presolve`, `evaluate_implied`, `solve_hierarchical`, `set_objective`, `ACCEPTED_SOL_STATUSES`, `build_solution`.
- Produces (in `src/pymarxan/solvers/mip_solver.py`, also used by `MIPSolver.solve`): `_mip_solver_for(problem, backend, verbose) -> (solver, backend_name)` and `MIPSolver._strategy_metadata(probmode, has_target2, has_sep_active) -> dict[str, str]`.
- Produces: `MultiObjectiveMIP(objectives, approach, *, mip_backend="auto", mip_chance_strategy="drop", mip_clump_strategy="drop", mip_sep_strategy="drop", warn_above_pu=200_000, presolve_check=False)` with `.rows`, `.name()` and `.solve(problem, config=None) -> list[Solution]` (one per row). Metadata keys: `solver`, `approach`, `row`, `rel_tol` / `weights`, `stage_reached` (hierarchical), `status`, `sol_status`, `objective_values`, `objective_senses`, `mip_backend` (+ the `mip_*_strategy` keys exactly when `MIPSolver` adds them). No `metadata["objective"]`.
- Errors (spec §6): empty objectives, duplicate keys, shape mismatch → `ValueError` at construction; unknown approach / non-spec objective / spec without a builder / `ZonalProblem` → `TypeError`; `MinBoundary` without a boundary table and `MaxWtdSum` weights not covering the features → `ValueError` at `solve`; stage 1 or a weighted-sum row with no solution → `RuntimeError` (saying "is unbounded" when the status is Unbounded); a later stage with no solution → one `UserWarning` and the previous stage's plan.
- `presolve_check` runs once per `solve`, with the first row's objective set. `config.num_solutions` is ignored.

Ruling: an objective that is not an `ObjectiveSpec`, or has no builder, raises `TypeError` (not listed in §6). Ruling: an Unbounded stage or row raises `RuntimeError` saying "is unbounded". Ruling: the two spec §5 caveats (tolerance-boundary met counts; a cap falling back to the solver's value when the implied point breaks a row) are documented in the `MultiObjectiveMIP` docstring. Ruling: `metadata["weights"]` is the row as given, not normalised. Ruling: solver construction and the strategy metadata are factored into `mip_solver.py` and shared with `MIPSolver.solve`.

- [ ] **Step 1: Write the failing tests:**

```python
"""MultiObjectiveMIP.solve (Stage 3 spec §3, §5, §6, §7.4 failure paths, §7.6)."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import pulp
import pytest

import pymarxan.multiobjective as mo
import pymarxan.solvers.mip_solver as ms
from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.multiobjective import (
    Hierarchical,
    MaxFeatures,
    MaxWeightedFeatures,
    MaxWtdSum,
    MinBoundary,
    MinCost,
    MinLargestShortfall,
    MinShortfall,
    MultiObjectiveMIP,
    ObjectiveSpec,
    WeightedSum,
)
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver
from pymarxan.zones import load_zone_project
from tests.pymarxan.scripted_solver import (
    Scripted,
    fail,
    not_solved_with_incumbent,
    optimal_without_solution,
)

DATA = Path(__file__).parent.parent / "data"


def simple(**params: Any) -> ConservationProblem:
    p = load_project(DATA / "simple")
    return p.copy_with(parameters={**p.parameters, **params})


def run(objectives: list[Any], approach: Any, problem: ConservationProblem,
        config: SolverConfig | None = None, **kw: Any) -> list[Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # PuLP 3.3.2
        return MultiObjectiveMIP(objectives, approach, mip_backend="cbc", **kw).solve(
            problem, config,
        )


def test_public_names() -> None:
    assert set(mo.__all__) == {
        "MultiObjectiveMIP", "Hierarchical", "WeightedSum", "ObjectiveSpec", "MinCost",
        "MinBoundary", "MinShortfall", "MinLargestShortfall", "MaxFeatures",
        "MaxWeightedFeatures", "MaxWtdSum", "weights_matrix", "rel_tol_matrix", "nondominated",
    }


# --- construction errors (spec §6) ---------------------------------------------------


def test_construction_errors() -> None:
    with pytest.raises(ValueError, match="objectives"):
        MultiObjectiveMIP([], Hierarchical(0.0))
    with pytest.raises(ValueError, match="duplicate.*min_shortfall"):
        MultiObjectiveMIP([MinShortfall(), MinShortfall(label="min_shortfall")], Hierarchical(0.0))
    with pytest.raises(TypeError, match="approach"):
        MultiObjectiveMIP([MinCost()], "hierarchical")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ObjectiveSpec"):
        MultiObjectiveMIP(["min_cost"], Hierarchical(0.0))  # type: ignore[list-item]
    with pytest.raises(TypeError, match="builder"):
        MultiObjectiveMIP([ObjectiveSpec()], Hierarchical(0.0))  # base class: no builder
    with pytest.raises(ValueError, match="rel_tol row 0 has 2"):
        MultiObjectiveMIP([MinCost(), MinShortfall()], Hierarchical([0.1, 0.1]))
    with pytest.raises(ValueError, match="weights row 0 has 3"):
        MultiObjectiveMIP([MinCost(), MinShortfall()], WeightedSum([1, 1, 1]))
    with pytest.raises(ValueError, match="mip_backend"):
        MultiObjectiveMIP([MinCost()], Hierarchical(0.0), mip_backend="nope")
    with pytest.raises(ValueError, match="mip_sep_strategy"):
        MultiObjectiveMIP([MinCost()], Hierarchical(0.0), mip_sep_strategy="socp")


def test_zonal_problem_is_rejected() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zp = load_zone_project(DATA / "zones")
    with pytest.raises(TypeError, match="ZonalProblem"):
        MultiObjectiveMIP([MinCost()], Hierarchical(0.0)).solve(zp)


def test_min_boundary_needs_a_boundary_table() -> None:
    p = simple()
    p = p.copy_with(boundary=None)
    with pytest.raises(ValueError, match="MinBoundary.*boundary"):
        MultiObjectiveMIP([MinBoundary()], Hierarchical(0.0)).solve(p)


def test_max_wtd_sum_weights_must_cover_the_features() -> None:
    with pytest.raises(ValueError, match=r"weights.*missing \[3\]"):
        run([MaxWtdSum(weights={1: 1, 2: 1})], Hierarchical(0.0), simple(COSTBUDGET=30))
    with pytest.raises(ValueError, match=r"weights.*unknown \[9\]"):
        run([MaxWtdSum(weights={1: 1, 2: 1, 3: 1, 9: 1})], Hierarchical(0.0),
            simple(COSTBUDGET=30))


# --- outputs (spec §5) ---------------------------------------------------------------


def test_hierarchical_metadata() -> None:
    sols = run([MinShortfall(), MinCost(label="cost")], Hierarchical([[0.0], [0.5]]),
               simple(COSTBUDGET=30.0), config=SolverConfig(num_solutions=7))
    assert len(sols) == 2  # one per rel_tol row; num_solutions is ignored
    meta = sols[1].metadata
    assert set(meta) == {
        "solver", "approach", "row", "rel_tol", "stage_reached", "status", "sol_status",
        "objective_values", "objective_senses", "mip_backend",
    }
    assert meta["solver"] == "MultiObjectiveMIP" and meta["approach"] == "hierarchical"
    assert (meta["row"], meta["rel_tol"], meta["stage_reached"]) == (1, [0.5], 2)
    assert (meta["status"], meta["sol_status"], meta["mip_backend"]) == ("Optimal", 1, "cbc")
    assert meta["objective_senses"] == {"min_shortfall": "min", "cost": "min"}
    assert set(meta["objective_values"]) == {"min_shortfall", "cost"}


def test_weighted_sum_metadata_and_solution_fields() -> None:
    p = simple(COSTBUDGET=30.0)
    sols = run([MinCost(), MaxFeatures()], WeightedSum([[1, 0], [1, 1]]), p)
    meta = sols[1].metadata
    assert set(meta) == {
        "solver", "approach", "row", "weights", "status", "sol_status",
        "objective_values", "objective_senses", "mip_backend",
    }
    assert (meta["approach"], meta["row"], meta["weights"]) == ("weighted_sum", 1, [1.0, 1.0])
    # Marxan-style fields exactly as MIPSolver fills them.
    ref = MIPSolver(mip_backend="cbc").solve(p, SolverConfig(num_solutions=1))[0]
    assert ref.metadata.keys() >= {"objective"} and "objective" not in meta
    s = sols[0]
    assert sols[0].metadata["objective_values"] == {"min_cost": 0.0, "max_features": 0.0}
    assert int(s.selected.sum()) == 0 and s.cost == 0.0 and s.boundary == 0.0


def test_costbudget_optional_and_honoured() -> None:
    unbounded = run([MinShortfall()], Hierarchical(0.0), simple())[0]
    assert unbounded.metadata["objective_values"]["min_shortfall"] == 0.0
    budgeted = run([MinShortfall(), MinCost()], Hierarchical(0.0), simple(COSTBUDGET=30.0))[0]
    assert budgeted.cost <= 30.0
    # Soft targets: at the best shortfall under the budget some target is unmet.
    assert budgeted.metadata["objective_values"]["min_shortfall"] > 0
    assert not all(budgeted.targets_met.values())


def test_min_boundary_is_non_trivial_at_blm_zero() -> None:
    p = simple(BLM=0.0, COSTBUDGET=45.0)
    alone = run([MinShortfall(), MinCost()], Hierarchical(0.0), p)[0]
    compact = run([MinShortfall(), MinBoundary(), MinCost()], Hierarchical(0.0), p)[0]
    vals = compact.metadata["objective_values"]
    assert vals["min_shortfall"] == alone.metadata["objective_values"]["min_shortfall"]
    assert vals["min_boundary"] == pytest.approx(compact.boundary)
    assert compact.boundary < alone.boundary


def test_two_min_largest_shortfall_labels_build() -> None:
    sols = run(
        [MinLargestShortfall(label="rel"), MinLargestShortfall("absolute", label="abs")],
        WeightedSum([1, 1]), simple(COSTBUDGET=30.0),
    )
    vals = sols[0].metadata["objective_values"]
    assert set(vals) == {"rel", "abs"} and vals["abs"] >= vals["rel"]


def test_presolve_runs_once_on_the_first_rows_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, float]] = []

    def spy(flag: Any, data: Any, model: pulp.LpProblem, name: str, *, stacklevel: int) -> None:
        seen.append({v.name: a for v, a in model.objective.items()})

    monkeypatch.setattr(mo, "_emit_presolve", spy)
    run([MinCost(), MaxFeatures()], WeightedSum([[3, 1], [1, 3]]), simple(COSTBUDGET=30.0),
        presolve_check=True)
    assert len(seen) == 1
    assert seen[0]["x_6"] == pytest.approx(0.75 * (8.0 + 2.0))  # (3/4)·(cost + BLM·self-edge)
    assert seen[0]["y_f1"] == pytest.approx(-0.25)  # row 0: −(1/4)·met


# --- failures (spec §4.1, §7.4, §7.5) -----------------------------------------------


def _scripted(monkeypatch: pytest.MonkeyPatch, solver: Scripted) -> None:
    # MultiObjectiveMIP builds its solver through mip_solver._mip_solver_for.
    monkeypatch.setattr(ms, "_make_pulp_solver", lambda *a, **k: solver)


def test_stage_1_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _scripted(monkeypatch, Scripted({1: fail}))
    with pytest.raises(RuntimeError, match=r"stage 1 \(min_shortfall\).*presolve_check"):
        run([MinShortfall(), MinCost()], Hierarchical(0.0), simple(COSTBUDGET=30.0))


def test_later_stage_failure_warns_and_returns_the_previous_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = Scripted({2: fail})
    _scripted(monkeypatch, solver)
    with pytest.warns(UserWarning, match="stage 2"):
        sols = run([MinShortfall(), MinCost()], Hierarchical([[0.0], [0.0]]),
                   simple(COSTBUDGET=30.0))
    first, second = sols
    assert first.metadata["stage_reached"] == 1
    # The restored stage-1 plan: best relative shortfall under COSTBUDGET 30 (the failed
    # stage left every x at 1, which would read as shortfall 0 at cost 83).
    assert first.metadata["objective_values"]["min_shortfall"] == pytest.approx(0.38333, abs=1e-4)
    assert first.cost <= 30.0
    assert (first.metadata["status"], first.metadata["sol_status"]) == ("Optimal", 1)
    # Row 2 is a fresh run: no cap from row 1 remains, and it reaches stage 2.
    assert second.metadata["stage_reached"] == 2
    assert not any(n.startswith("hier_cap_") for n in solver.models[2])
    assert "hier_cap_1" in solver.models[3]


def test_weighted_sum_row_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _scripted(monkeypatch, Scripted({2: fail}))
    with pytest.raises(RuntimeError, match="weighted-sum row 1"):
        run([MinCost(), MaxFeatures()], WeightedSum([[1, 1], [1, 2]]), simple(COSTBUDGET=30.0))


def test_infeasible_base_raises_for_hierarchical() -> None:
    p = simple(COSTBUDGET=30.0)
    pu = p.planning_units.copy()
    pu["status"] = 2  # every PU locked in: cost 83 > budget 30
    with pytest.raises(RuntimeError, match="stage 1"):
        run([MinCost()], Hierarchical(0.0), p.copy_with(planning_units=pu))


def test_ghost_feature_is_reported() -> None:
    p = simple(COSTBUDGET=30.0)
    ghost = pd.DataFrame({"id": [4], "name": ["g"], "target": [5.0], "spf": [1.0]})
    p = p.copy_with(features=pd.concat([p.features, ghost], ignore_index=True))
    sol = run([MaxWtdSum(weights={1: 1, 2: 1, 3: 1, 4: 5}), MinCost()], Hierarchical(0.0), p)[0]
    assert sol.metadata["objective_values"]["max_wtd_sum"] > 0


def test_unbounded_stage_says_so() -> None:
    p = simple()
    features = p.features.copy()
    features.loc[0, "spf"] = -1.0  # minimising −s/R: s grows without limit
    p = p.copy_with(features=features)
    with pytest.raises(RuntimeError, match="stage 1 .* is unbounded"):
        run([MinShortfall()], Hierarchical(0.0), p)
    with pytest.raises(RuntimeError, match="weighted-sum row 0 .* is unbounded"):
        run([MinShortfall()], WeightedSum([1]), p)


@pytest.mark.parametrize("approach", [Hierarchical(0.0), WeightedSum([1, 1])])
def test_acceptance_follows_sol_status(
    approach: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = [MinShortfall(), MinCost()]
    last = 2 if isinstance(approach, Hierarchical) else 1  # the last solve of row 0
    _scripted(monkeypatch, Scripted({last: not_solved_with_incumbent}))
    sol = run(specs, approach, simple(COSTBUDGET=30.0))[0]
    assert (sol.metadata["status"], sol.metadata["sol_status"]) == ("Not Solved", 2)
    _scripted(monkeypatch, Scripted({1: optimal_without_solution}))
    with pytest.raises(RuntimeError, match="found no solution"):
        run(specs, approach, simple(COSTBUDGET=30.0))


def test_weighted_sum_resets_values_before_each_row(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[set[Any]] = []

    class Blank(Scripted):
        def actualSolve(self, lp: pulp.LpProblem, **kwargs: Any) -> int:  # noqa: N802
            if self.calls == 0:
                return super().actualSolve(lp)
            self.calls += 1
            seen.append({v.varValue for v in lp.variables()})  # row 1, no inner solve
            lp.status, lp.sol_status = pulp.LpStatusNotSolved, pulp.LpSolutionNoSolutionFound
            return int(lp.status)

    _scripted(monkeypatch, Blank())
    with pytest.raises(RuntimeError, match="weighted-sum row 1"):
        run([MinCost(), MinShortfall()], WeightedSum([[1, 1], [1, 2]]), simple(COSTBUDGET=30.0))
    assert seen == [{None}]


def test_near_integral_x_is_rounded(monkeypatch: pytest.MonkeyPatch) -> None:
    specs = [MinShortfall(), MinCost()]
    p = simple(COSTBUDGET=30.0)
    exact = run(specs, WeightedSum([20, 1]), p)[0]
    assert exact.selected.any()  # fuzz a selected PU: PUs 2 and 6 (cost 23; MinCost 29 at BLM 1)

    def fuzz(lp: pulp.LpProblem) -> None:
        for v in lp.variables():
            if v.name.startswith("x_") and v.varValue is not None:
                v.varValue = 0.9999999 if v.varValue > 0.5 else 1e-7

    _scripted(monkeypatch, Scripted({1: fuzz}))
    fuzzy = run(specs, WeightedSum([20, 1]), p)[0]
    assert fuzzy.selected.tolist() == exact.selected.tolist()
    assert fuzzy.metadata["objective_values"] == exact.metadata["objective_values"]


def test_strategy_metadata_keys() -> None:
    p = simple()
    features = p.features.assign(target2=[1.0, 0, 0], sepnum=[2, 0, 0], sepdistance=[1.0, 0, 0])
    p = p.copy_with(features=features, parameters={**p.parameters, "PROBMODE": 3})
    meta = run([MinCost()], Hierarchical(0.0), p)[0].metadata
    assert {k: meta[k] for k in meta if k.endswith("_strategy")} == {
        "mip_chance_strategy": "drop", "mip_clump_strategy": "drop", "mip_sep_strategy": "drop",
    }
    plain = run([MinCost()], Hierarchical(0.0), simple())[0].metadata
    assert not any(k.endswith("_strategy") for k in plain)


def test_stage_1_failure_in_a_later_row_names_the_row(monkeypatch: pytest.MonkeyPatch) -> None:
    _scripted(monkeypatch, Scripted({3: fail}))  # row 1, stage 1
    with pytest.raises(RuntimeError, match=r"hierarchical row 1: stage 1 \(min_shortfall\)"):
        run([MinShortfall(), MinCost()], Hierarchical([[0.0], [0.0]]), simple(COSTBUDGET=30.0))


def test_single_objective_equals_mip_solver() -> None:
    """n = 1 is one solve; the Solution is filled exactly as MIPSolver fills it."""
    p = simple(COSTBUDGET=30.0)
    ref = MIPSolver(objective="min_shortfall", mip_backend="cbc").solve(
        p, SolverConfig(num_solutions=1),
    )[0]
    for approach in (Hierarchical(0.0), WeightedSum([1])):
        sol = run([MinShortfall()], approach, p)[0]
        assert sol.selected.any() and sol.selected.tolist() == ref.selected.tolist()
        assert (sol.cost, sol.boundary, sol.objective) == (ref.cost, ref.boundary, ref.objective)
        assert sol.targets_met == ref.targets_met
        assert sol.metadata["objective_values"]["min_shortfall"] == pytest.approx(
            ref.metadata["mip_objective_value"], abs=1e-9,
        )


def P(pu: dict[str, list[Any]], feat: dict[str, list[Any]], pvf: dict[str, list[Any]],
      **kw: Any) -> ConservationProblem:
    params = kw.pop("params", {})
    return ConservationProblem(
        planning_units=pd.DataFrame(pu), features=pd.DataFrame(feat),
        pu_vs_features=pd.DataFrame(pvf), parameters=params, **kw,
    )


def test_near_integral_budget_falls_back_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """Round 6 through MultiObjectiveMIP: v_rec's implied point breaks COSTBUDGET, so the
    cap must use v_model and stage 2 solves."""
    c1, eps = 1e7, 5e-8
    p = P({"id": [1, 2], "cost": [c1, 1.0], "status": [0, 0]},
          {"id": [1], "name": ["a"], "target": [10.0]},
          {"species": [1], "pu": [1], "amount": [10.0]},
          params={"COSTBUDGET": c1 * (1 - eps)})

    def near(lp: pulp.LpProblem) -> None:
        lp.sol_status = pulp.LpSolutionIntegerFeasible
        values = {"x_1": 1 - eps, "x_2": 0.0, "y_f1": 0.0}
        for v in lp.variables():
            if v.name in values:
                v.varValue = values[v.name]

    _scripted(monkeypatch, Scripted({1: near}))
    sol = run([MaxFeatures(), MinCost()], Hierarchical(0.0), p)[0]
    assert sol.metadata["stage_reached"] == 2


def test_earlier_caps_are_checked_end_to_end() -> None:
    """Round 7 through MultiObjectiveMIP: negative SPF on tolerance-boundary amounts."""
    p = P({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]},
          {"id": [1], "name": ["a"], "target": [0.8], "spf": [-1.0]},
          {"species": [1, 1], "pu": [1, 2], "amount": [0.7, 0.1]})
    sol = run([MaxFeatures(), MaxWeightedFeatures(), MinCost()], Hierarchical(0.0), p)[0]
    assert sol.metadata["stage_reached"] == 3


def test_hierarchical_passes_x_and_the_registry_as_tracked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[set[str]] = []
    real = mo.solve_hierarchical

    def spy(*args: Any, tracked: Any, **kwargs: Any) -> Any:
        tracked = list(tracked)
        seen.append({v.name for v in tracked})
        return real(*args, tracked=tracked, **kwargs)

    monkeypatch.setattr(mo, "solve_hierarchical", spy)
    p = simple(COSTBUDGET=30.0).copy_with(boundary=None)  # x_1 … x_6, no y_* auxiliaries
    run([MinShortfall(), MinCost()], Hierarchical(0.0), p)
    assert seen == [{f"x_{i}" for i in range(1, 7)} | {"s_f1", "s_f2", "s_f3"}]
```

- [ ] **Step 2: Run to verify they fail.**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/test_multiobjective_solve.py -q -p no:cacheprovider`

Expected: collection error, `ImportError: cannot import name 'MultiObjectiveMIP' from 'pymarxan.multiobjective'`.

- [ ] **Step 3: Implement:**

```diff
diff --git a/src/pymarxan/multiobjective.py b/src/pymarxan/multiobjective.py
index e57ff66..7dfbea6 100644
--- a/src/pymarxan/multiobjective.py
+++ b/src/pymarxan/multiobjective.py
@@ -21,18 +21,30 @@ from __future__ import annotations
 
 import math
 import numbers
+import warnings
 from collections.abc import Mapping, Sequence
 from dataclasses import dataclass, field
 from itertools import product
 from typing import Any, ClassVar
 
+import numpy as np
+import pulp
+
+from pymarxan.models.problem import ConservationProblem
 from pymarxan.solvers._mip_objectives import (
+    ACCEPTED_SOL_STATUSES,
     BUILDERS,
     ObjectiveContext,
     ObjectiveTerm,
+    evaluate_implied,
+    set_objective,
+    solve_hierarchical,
     validate_shortfall,
 )
-from pymarxan.solvers.base import Solution
+from pymarxan.solvers.base import Solution, SolverConfig
+from pymarxan.solvers.mip_solver import MIPSolver, _mip_solver_for, _warn_if_large_mip
+from pymarxan.solvers.presolve import _emit_presolve, _presolve_data
+from pymarxan.solvers.utils import build_solution
 
 __all__ = [
     "Hierarchical",
@@ -43,6 +55,7 @@ __all__ = [
     "MinCost",
     "MinLargestShortfall",
     "MinShortfall",
+    "MultiObjectiveMIP",
     "ObjectiveSpec",
     "WeightedSum",
     "nondominated",
@@ -383,3 +396,206 @@ def nondominated(solutions: Sequence[Solution]) -> list[Solution]:
             continue
         kept.append(solutions[i])
     return kept
+
+
+# --- the solver -------------------------------------------------------------------
+
+
+def _no_solution(status: int) -> str:
+    """Why a solve has no accepted solution, for the RuntimeError text."""
+    if status == pulp.LpStatusUnbounded:
+        return (
+            "is unbounded: an objective can improve without limit (e.g. a negative SPF "
+            "on a shortfall objective, or negative boundary / connectivity values)"
+        )
+    return (
+        f"found no solution (status {pulp.LpStatus.get(status, status)}); run "
+        "presolve_check(problem) to look for infeasible data"
+    )
+
+
+class MultiObjectiveMIP:
+    """Exact multi-objective MIP over one single-zone ``ConservationProblem`` (spec §5).
+
+    ``objectives`` are specs; their effective keys (``label or name``) must be unique.
+    ``approach`` is ``Hierarchical`` or ``WeightedSum``. The base model is
+    ``MIPSolver``'s (locks, boundary, connectivity, probability, no-good cuts); targets are
+    soft and ``COSTBUDGET``, when set, binds every stage and run. Each objective's term is
+    built once per ``solve``.
+
+    ``solve`` returns one ``Solution`` per approach row. ``cost`` / ``boundary`` /
+    ``objective`` / ``targets_met`` are filled as ``MIPSolver`` fills them (Marxan-style);
+    ``metadata["objective_values"]`` holds each objective's value at the plan, from the
+    model's own expressions (so ``"min_cost"`` can differ from ``Solution.objective``, and
+    under PROBMODE 2 met counts use the scaled amounts). ``config.num_solutions`` is
+    ignored. ``MIP_TIME_LIMIT`` applies per stage and row, so a hierarchical sweep takes up
+    to rows × n × T; with ``MIP_GAP`` > 0 or a time limit a stage's v is an incumbent's
+    value. Stage 1 (or a weighted-sum row) with no solution raises ``RuntimeError``; a later
+    stage with no solution returns the previous stage's plan with a ``UserWarning``.
+
+    Met counts in ``objective_values`` follow the exact ``check_targets`` rule (A >= R), so
+    on data at a target's tolerance boundary (e.g. amounts 0.7 + 0.1 against 0.8) the
+    reported count can be below the value a hierarchical cap kept, which the solver reached
+    within its tolerance. When the implied point of a stage's plan violates a model row
+    (including an earlier stage's cap), that stage's cap falls back to the solver's own
+    value, and later stages can as well.
+    """
+
+    def __init__(
+        self,
+        objectives: Sequence[ObjectiveSpec],
+        approach: Hierarchical | WeightedSum,
+        *,
+        mip_backend: str = "auto",
+        mip_chance_strategy: str = "drop",
+        mip_clump_strategy: str = "drop",
+        mip_sep_strategy: str = "drop",
+        warn_above_pu: int | None = 200_000,
+        presolve_check: bool | str = False,
+    ) -> None:
+        specs = list(objectives)
+        if not specs:
+            raise ValueError("objectives must hold at least one ObjectiveSpec")
+        for spec in specs:
+            if not isinstance(spec, ObjectiveSpec) or spec.name not in BUILDERS:
+                raise TypeError(
+                    f"objectives must be ObjectiveSpec subclasses with a builder "
+                    f"(MinCost, MinShortfall, ...), got {spec!r}"
+                )
+        keys = [s.key for s in specs]
+        dupes = sorted({k for k in keys if keys.count(k) > 1})
+        if dupes:
+            raise ValueError(f"objectives have duplicate keys (label or name): {dupes}")
+        if not isinstance(approach, Hierarchical | WeightedSum):
+            raise TypeError(
+                f"approach must be Hierarchical or WeightedSum, got {type(approach).__name__}"
+            )
+        self.objectives = specs
+        self.approach = approach
+        self.rows = approach.rows(len(specs))
+        # Reuses MIPSolver's strategy, backend and presolve-flag validation.
+        self._mip = MIPSolver(
+            mip_backend=mip_backend, mip_chance_strategy=mip_chance_strategy,
+            mip_clump_strategy=mip_clump_strategy, mip_sep_strategy=mip_sep_strategy,
+            warn_above_pu=warn_above_pu, presolve_check=presolve_check,
+        )
+
+    def name(self) -> str:
+        return "MultiObjectiveMIP"
+
+    def _check(self, problem: ConservationProblem) -> None:
+        from pymarxan.zones.model import ZonalProblem
+
+        if isinstance(problem, ZonalProblem):
+            raise TypeError(
+                "MultiObjectiveMIP takes a single-zone ConservationProblem; ZonalProblem "
+                "support is a follow-on (use ZoneMIPSolver)"
+            )
+        if problem.boundary is None and any(isinstance(s, MinBoundary) for s in self.objectives):
+            raise ValueError(
+                "MinBoundary needs a boundary table (problem.boundary is None): its "
+                "objective would be the constant 0"
+            )
+        ids = {int(f) for f in problem.features["id"].values}
+        for spec in self.objectives:
+            if isinstance(spec, MaxWtdSum) and spec.weights is not None:
+                missing = sorted(ids - set(spec.weights))
+                unknown = sorted(set(spec.weights) - ids)
+                if missing or unknown:
+                    raise ValueError(
+                        f"MaxWtdSum {spec.key!r} weights must cover every problem feature: "
+                        f"missing {missing}, unknown {unknown}"
+                    )
+
+    def _weighted(self, terms: list[ObjectiveTerm], row: list[float]) -> ObjectiveTerm:
+        total = sum(row)
+        expr = pulp.LpAffineExpression()
+        for w, term in zip(row, terms, strict=True):
+            sign = 1.0 if term.sense == pulp.LpMinimize else -1.0
+            expr += (w / total) * sign * term.expr
+        return ObjectiveTerm(expr, pulp.LpMinimize)
+
+    def solve(
+        self, problem: ConservationProblem, config: SolverConfig | None = None
+    ) -> list[Solution]:
+        if config is None:
+            config = SolverConfig()
+        self._check(problem)
+        mip = self._mip
+        _warn_if_large_mip(problem.n_planning_units, mip.warn_above_pu)
+        data = _presolve_data(mip.presolve_check, problem, self.name())
+        need_boundary = any(isinstance(s, MinBoundary) for s in self.objectives)
+        base = mip._build_base(problem, config, need_boundary=need_boundary)
+        model, x = base.model, base.x
+        budget = problem.parameters.get("COSTBUDGET")
+        if budget is not None:
+            model += base.cost_expr <= float(budget), "cost_budget"
+        ctx = base.context()
+        terms = [spec.build(ctx) for spec in self.objectives]  # once per solve (spec §3)
+        hierarchical = isinstance(self.approach, Hierarchical)
+        # P10 scans the first row's objective only (spec §9).
+        set_objective(model, terms[0] if hierarchical else self._weighted(terms, self.rows[0]))
+        _emit_presolve(mip.presolve_check, data, model, self.name(), stacklevel=3)
+
+        solver, backend = _mip_solver_for(problem, mip.mip_backend, config.verbose)
+        tracked = [*x.values(), *(v for v, _ in base.implied)]
+
+        def rounded() -> dict[int, float]:
+            return {pid: float(round(v.varValue or 0.0)) for pid, v in x.items()}
+
+        def evaluate(i: int) -> float | None:
+            values, feasible = evaluate_implied(
+                model, x, rounded(), base.implied, [terms[i]],
+            )
+            return values[0] if feasible else None
+
+        keys = [s.key for s in self.objectives]
+        senses = {s.key: s.sense for s in self.objectives}
+        solutions: list[Solution] = []
+        for r, row in enumerate(self.rows):
+            meta: dict[str, Any] = {"solver": self.name()}
+            if isinstance(self.approach, Hierarchical):
+                res = solve_hierarchical(
+                    model, terms, row, self.approach.abs_tol, solver, evaluate,
+                    tracked=tracked,
+                )
+                if res.stage_reached == 0:
+                    raise RuntimeError(
+                        f"hierarchical row {r}: stage 1 ({keys[0]}) {_no_solution(res.status)}"
+                    )
+                if res.stage_reached < len(terms):
+                    k = res.stage_reached + 1
+                    warnings.warn(
+                        f"hierarchical row {r}: stage {k} ({keys[k - 1]}) found no "
+                        f"solution; returning the stage {k - 1} plan",
+                        UserWarning,
+                        stacklevel=2,
+                    )
+                status, sol_status = res.status, res.sol_status
+                meta.update(approach="hierarchical", row=r, rel_tol=list(row),
+                            stage_reached=res.stage_reached)
+            else:
+                for var in [*model.variables(), *tracked]:
+                    var.varValue = None
+                set_objective(model, self._weighted(terms, row))
+                model.solve(solver)
+                if model.sol_status not in ACCEPTED_SOL_STATUSES:
+                    raise RuntimeError(
+                        f"weighted-sum row {r} (weights {list(row)}) "
+                        f"{_no_solution(int(model.status))}; the sweep is aborted"
+                    )
+                status, sol_status = int(model.status), int(model.sol_status)
+                meta.update(approach="weighted_sum", row=r, weights=list(row))
+            x_rounded = rounded()
+            values, _ = evaluate_implied(model, x, x_rounded, base.implied, terms)
+            selected = np.array([bool(x_rounded[pid]) for pid in base.pu_ids], dtype=bool)
+            meta.update(
+                status=pulp.LpStatus[status], sol_status=int(sol_status),
+                objective_values=dict(zip(keys, values, strict=True)),
+                objective_senses=senses, mip_backend=backend,
+            )
+            meta.update(
+                mip._strategy_metadata(base.probmode, base.has_target2, base.has_sep_active)
+            )
+            solutions.append(build_solution(problem, selected, base.blm, metadata=meta))
+        return solutions
```

And the shared helpers in `src/pymarxan/solvers/mip_solver.py`:

```diff
diff --git a/src/pymarxan/solvers/mip_solver.py b/src/pymarxan/solvers/mip_solver.py
index e464cda..f31b14a 100644
--- a/src/pymarxan/solvers/mip_solver.py
+++ b/src/pymarxan/solvers/mip_solver.py
@@ -89,6 +89,30 @@ def _warn_if_large_mip(n_pu: int, threshold: int | None) -> None:
         )
 
 
+def _mip_solver_for(
+    problem: ConservationProblem, backend: str, verbose: bool,
+) -> tuple[pulp.LpSolver, str]:
+    """The PuLP solver for ``backend`` with the problem's ``MIP_TIME_LIMIT`` (default 300 s)
+    and ``MIP_GAP`` (default 0), and the resolved backend name for metadata.
+
+    Phase 21: CBC is the default under ``"auto"`` but HiGHS wins if available, so the
+    concrete name is read from the solver, not from ``backend``. Shared by ``MIPSolver``
+    and ``MultiObjectiveMIP``.
+    """
+    solver = _make_pulp_solver(
+        backend,
+        time_limit=int(problem.parameters.get("MIP_TIME_LIMIT", 300)),
+        gap=float(problem.parameters.get("MIP_GAP", 0.0)),
+        verbose=verbose,
+    )
+    name = (
+        "highs" if isinstance(solver, pulp.HiGHS_CMD)
+        else "gurobi" if isinstance(solver, pulp.GUROBI_CMD)
+        else "cbc"
+    )
+    return solver, name
+
+
 def _core_spf(problem: ConservationProblem) -> dict[int, float]:
     """feature id -> spf; 1.0 when the column is absent (as the zone MIP's _spf_lookup)."""
     ids = problem.features["id"].values
@@ -620,6 +644,20 @@ class MIPSolver(Solver):
             has_sep_active=base.has_sep_active, lex=lex,
         )
 
+    def _strategy_metadata(
+        self, probmode: int, has_target2: bool, has_sep_active: bool,
+    ) -> dict[str, str]:
+        """The ``mip_*_strategy`` keys for the gaps this problem has (shared with
+        ``MultiObjectiveMIP``)."""
+        meta: dict[str, str] = {}
+        if probmode == 3:
+            meta["mip_chance_strategy"] = self.mip_chance_strategy
+        if has_target2:
+            meta["mip_clump_strategy"] = self.mip_clump_strategy
+        if has_sep_active:
+            meta["mip_sep_strategy"] = self.mip_sep_strategy
+        return meta
+
     def build_model(
         self, problem: ConservationProblem, config: SolverConfig | None = None
     ) -> pulp.LpProblem:
@@ -649,24 +687,7 @@ class MIPSolver(Solver):
         probmode, has_target2 = built.probmode, built.has_target2
         has_sep_active, blm = built.has_sep_active, built.blm
 
-        # Solve
-        time_limit = int(problem.parameters.get("MIP_TIME_LIMIT", 300))
-        gap = float(problem.parameters.get("MIP_GAP", 0.0))
-        verbose = config.verbose
-        # Phase 21: dispatch through the backend factory. CBC remains default
-        # when mip_backend="auto" but HiGHS wins if available.
-        solver = _make_pulp_solver(
-            self.mip_backend,
-            time_limit=time_limit, gap=gap, verbose=verbose,
-        )
-        # Record the resolved backend in metadata so users can confirm
-        # which solver actually ran. "auto" gets resolved here, so capture
-        # the concrete name post-factory rather than self.mip_backend.
-        resolved_backend = (
-            "highs" if isinstance(solver, pulp.HiGHS_CMD)
-            else "gurobi" if isinstance(solver, pulp.GUROBI_CMD)
-            else "cbc"
-        )
+        solver, resolved_backend = _mip_solver_for(problem, self.mip_backend, config.verbose)
         if built.lex is not None:
             status = solve_lexicographic(model, built.lex[0], built.lex[1], solver)
         else:
@@ -708,11 +729,6 @@ class MIPSolver(Solver):
             "mip_objective_value": obj_value,
             "shortfall": self.shortfall,
         }
-        if probmode == 3:
-            meta["mip_chance_strategy"] = self.mip_chance_strategy
-        if has_target2:
-            meta["mip_clump_strategy"] = self.mip_clump_strategy
-        if has_sep_active:
-            meta["mip_sep_strategy"] = self.mip_sep_strategy
+        meta.update(self._strategy_metadata(probmode, has_target2, has_sep_active))
         sol = build_solution(problem, selected, blm, metadata=meta)
         return [copy.deepcopy(sol) for _ in range(config.num_solutions)]
```

- [ ] **Step 4: Run.**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/test_multiobjective_solve.py tests/pymarxan/test_multiobjective_core.py -q -p no:cacheprovider`

Expected: `62 passed`.

- [ ] **Step 5: Lint, type-check, commit.**

```bash
~/.local/bin/ruff check src tests
/home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports
git add src/pymarxan/multiobjective.py src/pymarxan/solvers/mip_solver.py tests/pymarxan/test_multiobjective_solve.py
git commit -m "feat(multiobjective): MultiObjectiveMIP hierarchical and weighted-sum solves

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AZEiop9qqxVCrMHKqSDYP8"
```

---

### Task 7: Enumeration oracles, prioritizr parity, regressions, mutation spot-checks

**Files:**
- Create: `tests/pymarxan/test_multiobjective_oracles.py` (spec §7.2, §7.3)
- Create: `tests/pymarxan/test_multiobjective_regressions.py` (spec §7.4, §7.5)

**Interfaces:**
- Consumes: the public API of Tasks 5–6, `evaluate_implied` / `_build_base` (one §5 invariant test), `Scripted` / `fail`.
- Produces: tests only. These tests verify Task 6, so they pass when written; their strength is shown by Step 3, where each spec §7.7 mutation must fail at least one of them.

Every expected value in the oracle file is derived by brute force over the 2^7 selections of a 7-PU toy (or the 2^6 of prioritizr's fixture), inside the test. The weighted-sum hull test derives one weight row strictly inside each hull vertex's normal cone (endpoint vertices: between the axis and the single edge normal); grid rows assert only the optimal weighted value.

Ruling: the §7.7 mutation "index `rel_tol` by objective instead of stage" is a no-op when priority = list order, so the spot-check uses `rel_tols[-k]`. Beyond §7.7 the script also runs R6 (no snapshot restore after a failed later stage), S1/S2 (accept on `status` instead of `sol_status`, hierarchical and weighted-sum paths), R8 (round x by `int()` truncation), P9 (`MultiObjectiveMIP`'s `evaluate` ignores `feasible`), N5 (`nondominated` with a 1e-6 tolerance) and N6 (raise on a stage-1 failure only in sweep row 0); the script asserts that every mutation fails a test. Ruling: the r > 0 oracle runs in both orders; cost-first locks PU1 in so v = 4 ≠ 0, and the locks / no-good test uses COSTBUDGET 8, which binds (best shortfall 0.9), so r = 0.5 changes the plan.

- [ ] **Step 1: Write the oracle and parity tests:**

```python
"""Enumeration oracles and prioritizr parity for MultiObjectiveMIP (Stage 3 spec §7.2, §7.3).

Every expected value is derived by brute force over all 2^7 selections of a 7-PU toy, so a
formulation mistake shows up as an oracle mismatch rather than a silent pass.
"""
from __future__ import annotations

import itertools
import warnings
from collections.abc import Sequence
from typing import Any

import pandas as pd
import pytest

import pymarxan.solvers.mip_solver as ms
from pymarxan.models.problem import ConservationProblem
from pymarxan.multiobjective import (
    Hierarchical,
    MaxFeatures,
    MaxWtdSum,
    MinCost,
    MinShortfall,
    MultiObjectiveMIP,
    ObjectiveSpec,
    WeightedSum,
    nondominated,
    weights_matrix,
)
from pymarxan.solvers.base import Solution, SolverConfig
from tests.pymarxan.scripted_solver import Scripted, fail

COST = [4.0, 4.0, 1.0, 3.0, 5.0, 4.0, 2.0]
AMOUNTS = {1: [2.0, 1.0, 2.0, 1.0, 3.0, 0.0, 1.0], 2: [3.0, 0.0, 1.0, 0.0, 0.0, 3.0, 1.0]}
TARGETS = {1: 5.0, 2: 4.0}
SPF = {1: 1.0, 2: 2.0}
WTD = {1: 1.0, 2: 2.0}  # MaxWtdSum weights
ABS_TOL = 1e-9


def toy(status: Sequence[int] | None = None, **params: Any) -> ConservationProblem:
    ids = list(range(1, 8))
    pvf = [(f, pu, a) for f, row in AMOUNTS.items() for pu, a in zip(ids, row) if a > 0]
    return ConservationProblem(
        planning_units=pd.DataFrame({
            "id": ids, "cost": COST, "status": list(status or [0] * 7),
        }),
        features=pd.DataFrame({
            "id": [1, 2], "name": ["a", "b"], "target": [TARGETS[1], TARGETS[2]],
            "spf": [SPF[1], SPF[2]],
        }),
        pu_vs_features=pd.DataFrame(pvf, columns=["species", "pu", "amount"]),
        parameters=params,
    )


# --- the oracle ------------------------------------------------------------------


def value(spec: ObjectiveSpec, sel: tuple[int, ...], problem: ConservationProblem) -> float:
    held = {f: sum(a for a, s in zip(row, sel) if s) for f, row in AMOUNTS.items()}
    if isinstance(spec, MinCost):
        return sum(c for c, s in zip(COST, sel) if s)
    if isinstance(spec, MinShortfall):
        return sum(SPF[f] * max(0.0, TARGETS[f] - held[f]) / TARGETS[f] for f in held)
    if isinstance(spec, MaxWtdSum):
        weights = spec.weights or {f: 1.0 for f in held}
        return sum(weights[f] * held[f] for f in held)
    if isinstance(spec, MaxFeatures):
        return float(sum(held[f] >= TARGETS[f] for f in held))
    raise AssertionError(spec)


def points(
    problem: ConservationProblem, specs: Sequence[ObjectiveSpec],
    forbidden: Sequence[Sequence[int]] = (),
) -> list[tuple[tuple[int, ...], list[float]]]:
    """Every feasible selection with its objective values (locks, budget, no-good cuts)."""
    status = problem.planning_units["status"].tolist()
    budget = problem.parameters.get("COSTBUDGET")
    out = []
    for sel in itertools.product((0, 1), repeat=7):
        if any((st == 2 and not s) or (st == 3 and s) for st, s in zip(status, sel)):
            continue
        if budget is not None and sum(c for c, s in zip(COST, sel) if s) > budget:
            continue
        if any(list(sel) == list(f) for f in forbidden):
            continue
        out.append((sel, [value(spec, sel, problem) for spec in specs]))
    return out


def lexicographic(
    pts: list[tuple[tuple[int, ...], list[float]]], specs: Sequence[ObjectiveSpec],
    rel_tol: Sequence[float], abs_tol: float = ABS_TOL,
) -> tuple[list[tuple[tuple[int, ...], list[float]]], list[float]]:
    """Enumerated hierarchical result: the optimal set of the last stage, and each v."""
    vs = []
    for k, spec in enumerate(specs):
        vals = [p[1][k] for p in pts]
        v = min(vals) if spec.sense == "min" else max(vals)
        vs.append(v)
        if k == len(specs) - 1:
            return [p for p in pts if abs(p[1][k] - v) <= 1e-9], vs
        slack = rel_tol[k] * abs(v) + abs_tol
        pts = [p for p in pts if (
            p[1][k] <= v + slack if spec.sense == "min" else p[1][k] >= v - slack
        )]
    raise AssertionError


def run(specs: Sequence[ObjectiveSpec], approach: Any, problem: ConservationProblem,
        config: SolverConfig | None = None) -> list[Solution]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # PuLP 3.3.2
        return MultiObjectiveMIP(specs, approach, mip_backend="cbc").solve(problem, config)


def got(sol: Solution, specs: Sequence[ObjectiveSpec]) -> list[float]:
    return [sol.metadata["objective_values"][s.key] for s in specs]


def sel(sol: Solution) -> tuple[int, ...]:
    return tuple(int(v) for v in sol.selected)


# --- hierarchical, 2 objectives (§7.2) ----------------------------------------------


@pytest.mark.parametrize("budget", [None, 4.0])  # 4 binds: best shortfall 1.1
@pytest.mark.parametrize("order", ["cost_first", "shortfall_first"])
def test_hierarchical_rel_tol_0_is_the_lexicographic_optimum(
    order: str, budget: float | None,
) -> None:
    specs: list[ObjectiveSpec] = [MinCost(), MinShortfall()]
    if order == "shortfall_first":
        specs.reverse()
    p = toy() if budget is None else toy(COSTBUDGET=budget)
    best, _ = lexicographic(points(p, specs), specs, [0.0])
    sol = run(specs, Hierarchical(0.0), p)[0]
    assert got(sol, specs) == pytest.approx(best[0][1], abs=1e-9)
    assert sel(sol) in {b[0] for b in best}


@pytest.mark.parametrize("r", [0.25, 0.6])
def test_hierarchical_rel_tol_r_caps_stage_1(r: float) -> None:
    specs = [MinShortfall(), MinCost()]
    p = toy(COSTBUDGET=4.0)  # best shortfall 1.1 at cost 4; r = 0.6 allows cost 3
    best, (v1, _) = lexicographic(points(p, specs), specs, [r])
    sol = run(specs, Hierarchical(r), p)[0]
    f1, f2 = got(sol, specs)
    assert f2 == pytest.approx(best[0][1][1], abs=1e-9)  # stage 2 = enumerated min f2
    assert f1 <= v1 + r * abs(v1) + ABS_TOL + 1e-9  # the cap holds
    assert v1 > 0  # the fixture makes r matter


@pytest.mark.parametrize("r", [0.25, 1.0])
def test_hierarchical_rel_tol_r_caps_cost_first(r: float) -> None:
    """MinCost → MinShortfall with PU1 (cost 4) locked in, so v = 4 > 0 and r matters."""
    specs = [MinCost(), MinShortfall()]
    p = toy(status=[2, 0, 0, 0, 0, 0, 0])
    best, (v1, _) = lexicographic(points(p, specs), specs, [r])
    sol = run(specs, Hierarchical(r), p)[0]
    f1, f2 = got(sol, specs)
    assert f2 == pytest.approx(best[0][1][1], abs=1e-9)
    assert f1 <= v1 + r * abs(v1) + ABS_TOL + 1e-9
    # rel_tol 0 keeps cost 4 at shortfall 1.1; f1 itself can tie within the cap.
    assert v1 == 4.0 and f2 < 1.1


# --- 3 objectives: rel_tol indexed by solve order (§7.2) ----------------------------


def test_three_objectives_rel_tol_follows_stage_order() -> None:
    specs = [MaxWtdSum(weights=WTD), MinCost(), MinShortfall()]
    p = toy()
    pts = points(p, specs)
    r = 0.5
    results = {}
    for tols in ([0.0, r], [r, 0.0]):
        best, _ = lexicographic(pts, specs, tols)
        sol = run(specs, Hierarchical(tols), p)[0]
        assert got(sol, specs) == pytest.approx(best[0][1], abs=1e-9)
        results[tuple(tols)] = got(sol, specs)
    assert results[(0.0, r)] != results[(r, 0.0)]  # the fixture discriminates


# --- mixed senses (§7.2) ----------------------------------------------------------


@pytest.mark.parametrize("r", [0.0, 0.25])
def test_mixed_sense_hierarchy_max_then_min(r: float) -> None:
    specs = [MaxWtdSum(weights=WTD), MinCost()]
    p = toy(COSTBUDGET=10.0)
    best, (v1, _) = lexicographic(points(p, specs), specs, [r])
    sol = run(specs, Hierarchical(r), p)[0]
    f1, f2 = got(sol, specs)
    assert f2 == pytest.approx(best[0][1][1], abs=1e-9)
    assert f1 >= v1 - (r * abs(v1) + ABS_TOL) - 1e-9


@pytest.mark.parametrize("row", [[1, 1], [1, 3], [3, 1], [1, 0.2]])
def test_mixed_sense_weighted_sum(row: list[float]) -> None:
    specs = [MinCost(), MaxWtdSum(weights=WTD)]
    p = toy(COSTBUDGET=12.0)
    total = sum(row)
    score = [row[0] / total * v[0] - row[1] / total * v[1] for _, v in points(p, specs)]
    sol = run(specs, WeightedSum(row), p)[0]
    f = got(sol, specs)
    assert row[0] / total * f[0] - row[1] / total * f[1] == pytest.approx(min(score), abs=1e-9)


# --- locks and no-good cuts (§7.2) -----------------------------------------------


@pytest.mark.parametrize("r", [0.0, 0.5])
def test_hierarchical_respects_locks_and_no_good_cuts(r: float) -> None:
    specs = [MinShortfall(), MinCost()]
    # Budget 8 binds (best shortfall 0.9 > 0), so r = 0.5 lowers the cost from 8 to 7.
    p = toy(status=[0, 2, 0, 0, 3, 0, 0], COSTBUDGET=8.0)
    unforbidden = run(specs, Hierarchical(r), p)[0]
    forbidden = [list(sel(unforbidden))]
    config = SolverConfig(metadata={"forbidden_selections": forbidden})
    best, _ = lexicographic(points(p, specs, forbidden), specs, [r])
    sol = run(specs, Hierarchical(r), p, config)[0]
    assert got(sol, specs)[1] == pytest.approx(best[0][1][1], abs=1e-9)
    s = sel(sol)
    assert s[1] == 1 and s[4] == 0 and list(s) != forbidden[0]


# --- weighted-sum sweep reproduces the supported hull (§7.2) ------------------------


def _hull(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Lower-left convex hull vertices from (min cost, then min sf) to (min sf, then min cost)."""
    uniq = sorted(set(pts))
    lower: list[tuple[float, float]] = []
    for q in uniq:
        while len(lower) >= 2:
            (ax, ay), (bx, by) = lower[-2], lower[-1]
            if (bx - ax) * (q[1] - ay) - (by - ay) * (q[0] - ax) <= 1e-12:
                lower.pop()
            else:
                break
        lower.append(q)
    end = min(uniq, key=lambda q: (q[1], q[0]))
    return lower[: lower.index(end) + 1]


def _unit(v: tuple[float, float]) -> tuple[float, float]:
    n = (v[0] ** 2 + v[1] ** 2) ** 0.5
    return (v[0] / n, v[1] / n)


def _cone_rows(vertices: list[tuple[float, float]]) -> list[list[float]]:
    """One weight row strictly inside each vertex's normal cone."""
    normals = [
        _unit((a[1] - b[1], b[0] - a[0])) for a, b in zip(vertices, vertices[1:])
    ]  # edge a→b: cost rises, shortfall falls; both components > 0
    cones = [((1.0, 0.0), normals[0])]
    cones += [(normals[i - 1], normals[i]) for i in range(1, len(normals))]
    cones += [(normals[-1], (0.0, 1.0))]
    return [[(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] for a, b in cones]


SWEEP_SPECS = [MinCost(), MinShortfall()]


def _sweep_points() -> list[tuple[float, float]]:
    return [(v[0], v[1]) for _, v in points(toy(), SWEEP_SPECS)]


def test_weighted_sum_sweep_returns_each_hull_vertex() -> None:
    vertices = _hull(_sweep_points())
    assert len(vertices) >= 4  # several supported trade-offs
    lex_cost, _ = lexicographic(points(toy(), SWEEP_SPECS), SWEEP_SPECS, [0.0])
    lex_sf, _ = lexicographic(points(toy(), SWEEP_SPECS[::-1]), SWEEP_SPECS[::-1], [0.0])
    assert vertices[0] == tuple(lex_cost[0][1])
    assert vertices[-1] == tuple(lex_sf[0][1][::-1])
    sols = run(SWEEP_SPECS, WeightedSum(_cone_rows(vertices)), toy())
    for vertex, sol in zip(vertices, sols, strict=True):
        assert got(sol, SWEEP_SPECS) == pytest.approx(list(vertex), abs=1e-9)


def test_weighted_sum_grid_rows_reach_the_optimal_weighted_value() -> None:
    pts = _sweep_points()
    rows = weights_matrix(2, 5)
    for row, sol in zip(rows, run(SWEEP_SPECS, WeightedSum(rows), toy()), strict=True):
        w = [v / sum(row) for v in row]
        best = min(w[0] * c + w[1] * s for c, s in pts)
        f = got(sol, SWEEP_SPECS)
        assert w[0] * f[0] + w[1] * f[1] == pytest.approx(best, abs=1e-9)


def test_nondominated_sweep_equals_the_enumerated_pareto_points_found() -> None:
    pts = _sweep_points()
    pareto = {
        p for p in pts
        if not any(q[0] <= p[0] and q[1] <= p[1] and q != p for q in pts)
    }
    rows = weights_matrix(2, 5) + _cone_rows(_hull(pts))
    sols = run(SWEEP_SPECS, WeightedSum(rows), toy())
    found = {tuple(got(s, SWEEP_SPECS)) for s in sols}
    kept = [tuple(got(s, SWEEP_SPECS)) for s in nondominated(sols)]
    assert len(kept) == len(set(kept))  # identical vectors kept once
    assert set(kept) == {p for p in found if p in pareto}


# --- sweeps: rows are independent (§7.2, §9) ------------------------------------------


def test_hierarchical_sweep_rows_match_their_own_oracles() -> None:
    specs = [MinShortfall(), MinCost()]
    p = toy(COSTBUDGET=4.0)
    rows = [[0.6], [0.0]]  # the looser row first: its cap must not leak into row 2
    sols = run(specs, Hierarchical(rows), p)
    costs = []
    for row, sol in zip(rows, sols, strict=True):
        best, _ = lexicographic(points(p, specs), specs, row)
        assert got(sol, specs)[1] == pytest.approx(best[0][1][1], abs=1e-9)
        assert sol.metadata["rel_tol"] == row
        costs.append(got(sol, specs)[1])
    assert costs == [3.0, 4.0]


def test_failed_stage_leaves_no_cap_and_row_2_equals_a_fresh_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = [MinShortfall(), MinCost()]
    p = toy(COSTBUDGET=4.0)
    fresh = run(specs, Hierarchical(0.0), p)[0]
    solver = Scripted({2: fail})  # row 1, stage 2
    monkeypatch.setattr(ms, "_make_pulp_solver", lambda *a, **k: solver)
    with pytest.warns(UserWarning, match="stage 2"):
        sols = run(specs, Hierarchical([[0.0], [0.0]]), p)
    assert sels_equal(sols[1], fresh)
    assert "hier_cap_1" not in solver.models[2]  # row 2, stage 1


def sels_equal(a: Solution, b: Solution) -> bool:
    return sel(a) == sel(b) and a.metadata["objective_values"] == b.metadata[
        "objective_values"
    ]


# --- prioritizr parity (§7.3) ----------------------------------------------------------


def prioritizr_fixture() -> ConservationProblem:
    """prioritizr's 6-PU multi-objective test problem: cost 1 each, budget 2."""
    ft = {1: [5.0, 0.5, 0, 0, 0, 0], 2: [0, 0, 0, 0, 3.0, 2.0]}
    pvf = [(f, pu, a) for f, row in ft.items() for pu, a in enumerate(row, 1) if a > 0]
    return ConservationProblem(
        planning_units=pd.DataFrame({"id": range(1, 7), "cost": [1.0] * 6, "status": 0}),
        features=pd.DataFrame({"id": [1, 2], "name": ["ft1", "ft2"], "target": [0.0, 0.0]}),
        pu_vs_features=pd.DataFrame(pvf, columns=["species", "pu", "amount"]),
        parameters={"COSTBUDGET": 2.0},
    )


PARITY = [MaxWtdSum(weights={1: 1, 2: 0}, label="ft1"),
          MaxWtdSum(weights={1: 0, 2: 1}, label="ft2")]
PARITY_ROWS = [[1, 1], [1, 0], [0, 1], [1, 1], [1, 5]]
PARITY_EXPECTED = [
    ("100010", (5.0, 3.0)), ("110000", (5.5, 0.0)), ("000011", (0.0, 5.0)),
    ("100010", (5.0, 3.0)), ("000011", (0.0, 5.0)),
]


def test_prioritizr_parity_expectations_by_enumeration() -> None:
    ft = {1: [5.0, 0.5, 0, 0, 0, 0], 2: [0, 0, 0, 0, 3.0, 2.0]}
    for row, (x, vals) in zip(PARITY_ROWS, PARITY_EXPECTED, strict=True):
        scored = []
        for s in itertools.product((0, 1), repeat=6):
            if sum(s) > 2:
                continue
            v = tuple(sum(a for a, si in zip(ft[f], s) if si) for f in (1, 2))
            scored.append((row[0] * v[0] + row[1] * v[1], "".join(map(str, s)), v))
        best = max(sc[0] for sc in scored)
        winners = [sc for sc in scored if sc[0] == best]
        assert [(w[1], w[2]) for w in winners] == [(x, vals)]  # unique optimum


def test_prioritizr_parity_weighted_sum() -> None:
    sols = run(PARITY, WeightedSum(PARITY_ROWS), prioritizr_fixture())
    for sol, (x, vals) in zip(sols, PARITY_EXPECTED, strict=True):
        assert "".join(map(str, sel(sol)[:6])) == x
        assert got(sol, PARITY) == pytest.approx(list(vals))
```

- [ ] **Step 2: Write the regression and implied-evaluation tests:**

```python
"""MultiObjectiveMIP bug-fix regressions and implied evaluation (Stage 3 spec §7.4, §7.5)."""
from __future__ import annotations

import itertools
import warnings
from collections.abc import Sequence
from typing import Any

import pandas as pd
import pulp
import pytest

import pymarxan.multiobjective as mo
import pymarxan.solvers.mip_solver as ms
from pymarxan.models.problem import ConservationProblem
from pymarxan.multiobjective import (
    Hierarchical,
    MaxFeatures,
    MaxWtdSum,
    MinBoundary,
    MinCost,
    MinLargestShortfall,
    MinShortfall,
    MultiObjectiveMIP,
    ObjectiveSpec,
    WeightedSum,
)
from pymarxan.solvers.base import Solution
from tests.pymarxan.scripted_solver import Scripted


def P(pu: dict[str, list[Any]], feat: dict[str, list[Any]], pvf: dict[str, list[Any]],
      **kw: Any) -> ConservationProblem:
    params = kw.pop("params", {})
    return ConservationProblem(
        planning_units=pd.DataFrame(pu), features=pd.DataFrame(feat),
        pu_vs_features=pd.DataFrame(pvf), parameters=params, **kw,
    )


def run(specs: Sequence[ObjectiveSpec], approach: Any, problem: ConservationProblem,
        **kw: Any) -> list[Solution]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # PuLP 3.3.2
        return MultiObjectiveMIP(specs, approach, mip_backend="cbc", **kw).solve(problem)


def vals(sol: Solution) -> dict[str, float]:
    values: dict[str, float] = sol.metadata["objective_values"]
    return values


def sel(sol: Solution) -> list[int]:
    return [int(v) for v in sol.selected]


# --- caps (§7.4) -----------------------------------------------------------------


def test_negative_v_with_rel_tol_is_feasible() -> None:
    """Stage 1 maximises Σ −amount with PU1 locked in: v = −5. prioritizr's cap
    f >= v·(1 − r) = −2.5 is infeasible; ours, f >= v − (r·|v| + abs_tol), is not."""
    p = P({"id": [1, 2, 3], "cost": [1.0, 1.0, 1.0], "status": [2, 0, 0]},
          {"id": [1], "name": ["a"], "target": [0.0]},
          {"species": [1, 1], "pu": [1, 2], "amount": [5.0, 1.0]})
    sol = run([MaxWtdSum(weights={1: -1.0}), MinCost()], Hierarchical(0.5), p)[0]
    assert sol.metadata["stage_reached"] == 2
    assert vals(sol) == {"max_wtd_sum": -5.0, "min_cost": 1.0}
    # No plan reaches prioritizr's cap.
    assert all(-(5.0 + 1.0 * x2) < -2.5 for x2 in (0, 1))


@pytest.mark.parametrize(("abs_tol", "expected"), [(1e-9, [1, 1, 1]), (0.5, [1, 1, 0])])
def test_v_zero_allows_exactly_abs_tol(abs_tol: float, expected: list[int]) -> None:
    """Best shortfall is 0 (no budget), so rel_tol = 10 gives no slack; abs_tol 0.5 lets
    stage 2 drop PU3 (relative shortfall 0.4) and save cost 5."""
    p = P({"id": [1, 2, 3], "cost": [1.0, 1.0, 5.0], "status": [0, 0, 0]},
          {"id": [1], "name": ["a"], "target": [5.0]},
          {"species": [1, 1, 1], "pu": [1, 2, 3], "amount": [2.0, 1.0, 2.0]})
    sol = run([MinShortfall(), MinCost()], Hierarchical(10.0, abs_tol=abs_tol), p)[0]
    assert sel(sol) == expected


def test_tolerance_boundary_cap_returns_cost_2() -> None:
    """Round 4: 0.7 + 0.1 = 0.7999999999999999 < 0.8, but CBC calls the target met.
    Recompute-only would cap at 0 and return the empty plan."""
    p = P({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]},
          {"id": [1], "name": ["a"], "target": [0.8]},
          {"species": [1, 1], "pu": [1, 2], "amount": [0.7, 0.1]})
    sol = run([MaxFeatures(), MinCost()], Hierarchical(rel_tol=0.0), p)[0]
    assert sol.cost == 2.0 and sel(sol) == [1, 1]
    # The reported count follows check_targets (the documented tolerance caveat).
    assert vals(sol)["max_features"] == 0.0


def test_incumbent_cap_keeps_both_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Round 5: a stage-1 incumbent meets both targets but leaves y_f2 = 0. Model-only
    would cap at 1 and let stage 2 drop a target."""
    p = P({"id": [1, 2, 3], "cost": [1.0, 1.0, 5.0], "status": [0, 0, 0]},
          {"id": [1, 2], "name": ["a", "b"], "target": [1.0, 1.0]},
          {"species": [1, 2, 1, 2], "pu": [1, 2, 3, 3], "amount": [1.0] * 4},
          params={"COSTBUDGET": 5.0})

    def loose(lp: pulp.LpProblem) -> None:
        lp.sol_status = pulp.LpSolutionIntegerFeasible
        values = {"x_1": 1.0, "x_2": 1.0, "x_3": 0.0, "y_f1": 1.0, "y_f2": 0.0}
        for v in lp.variables():
            if v.name in values:
                v.varValue = values[v.name]

    solver = Scripted({1: loose})
    monkeypatch.setattr(ms, "_make_pulp_solver", lambda *a, **k: solver)
    sol = run([MaxFeatures(), MinCost()], Hierarchical(0.0), p)[0]
    assert vals(sol) == {"max_features": 2.0, "min_cost": 2.0}
    assert sol.metadata["sol_status"] == 1  # stage 2's (proven optimal) status


def test_zero_cost_pu_without_amounts() -> None:
    p = P({"id": [1, 2], "cost": [1.0, 0.0], "status": [0, 0]},
          {"id": [1], "name": ["a"], "target": [1.0]},
          {"species": [1], "pu": [1], "amount": [1.0]})
    sol = run([MinShortfall(), MinCost()], Hierarchical(0.0), p)[0]
    assert sel(sol) == [1, 0] and vals(sol) == {"min_shortfall": 0.0, "min_cost": 1.0}


def test_stage_reached_is_n_on_success() -> None:
    p = P({"id": [1, 2], "cost": [1.0, 2.0], "status": [0, 0]},
          {"id": [1], "name": ["a"], "target": [1.0]},
          {"species": [1, 1], "pu": [1, 2], "amount": [1.0, 1.0]})
    sol = run([MinShortfall(), MaxFeatures(), MinCost()], Hierarchical(0.0), p)[0]
    assert sol.metadata["stage_reached"] == 3


# --- reporting (§7.4) ------------------------------------------------------------


def test_objective_values_are_recomputed_not_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 0-weight MaxFeatures row leaves y free; a raw y_f1 = 0 is reported as met."""
    p = P({"id": [1, 2], "cost": [1.0, 1.0], "status": [2, 0]},
          {"id": [1], "name": ["a"], "target": [1.0]},
          {"species": [1], "pu": [1], "amount": [1.0]})
    raw: dict[str, float] = {}

    def zero_y(lp: pulp.LpProblem) -> None:
        for v in lp.variables():
            if v.name == "y_f1":
                v.varValue = 0.0
        raw["y_f1"] = 0.0

    monkeypatch.setattr(ms, "_make_pulp_solver", lambda *a, **k: Scripted({1: zero_y}))
    sol = run([MinCost(), MaxFeatures()], WeightedSum([1, 0]), p)[0]
    assert raw == {"y_f1": 0.0}
    assert vals(sol)["max_features"] == 1.0


def test_same_row_twice_gives_the_same_result() -> None:
    p = P({"id": [1, 2, 3, 4], "cost": [2.0, 3.0, 1.0, 4.0], "status": [0] * 4},
          {"id": [1, 2], "name": ["a", "b"], "target": [3.0, 2.0]},
          {"species": [1, 1, 2, 2], "pu": [1, 2, 3, 4], "amount": [2.0, 2.0, 1.0, 2.0]},
          params={"COSTBUDGET": 5.0})
    specs = [MinShortfall(), MinCost()]
    for approach in (WeightedSum([[1, 1], [1, 1]]), Hierarchical([[0.1], [0.1]])):
        a, b = run(specs, approach, p)
        assert sel(a) == sel(b) and vals(a) == vals(b)


# --- implied evaluation (§7.5) -----------------------------------------------------


def _fixture(kind: str) -> ConservationProblem:
    pu = {"id": [1, 2, 3, 4], "cost": [3.0, 3.0, 1.0, 2.0], "status": [0] * 4}
    feat = {"id": [1, 2], "name": ["a", "b"], "target": [2.0, 3.0]}
    pvf = {"species": [1, 1, 2, 2], "pu": [1, 3, 2, 4], "amount": [1.0, 1.5, 2.0, 2.0]}
    boundary = pd.DataFrame({
        "id1": [1, 1, 2, 3, 3, 4], "id2": [1, 2, 3, 3, 4, 4],
        "boundary": [1.0, 2.0, 1.0, 1.0, 3.0, 2.0],
    })
    params: dict[str, Any] = {"BLM": 0.0, "COSTBUDGET": 6.0}
    kw: dict[str, Any] = {"boundary": boundary}
    if kind == "blm":
        params["BLM"] = 0.5
    elif kind in ("conn_sym_dupes", "conn_asym"):
        kw["connectivity"] = pd.DataFrame({
            "id1": [1, 2, 3, 2], "id2": [2, 1, 4, 3], "value": [10.0, 4.0, 2.0, 1.5],
        })
        params["CONNECTIVITY_WEIGHT"] = 1.0
        params["ASYMMETRIC_CONNECTIVITY"] = int(kind == "conn_asym")
    else:  # probmode1_pos / probmode1_neg
        kw["probability"] = pd.DataFrame({"pu": [1, 2, 3, 4], "probability": [.5, .1, .3, 0]})
        params["PROBMODE"] = 1
        params["PROBABILITYWEIGHTING"] = 0.8 if kind == "probmode1_pos" else -1.0
    return P(pu, feat, pvf, params=params, **kw)


KINDS = ["blm", "conn_sym_dupes", "conn_asym", "probmode1_pos", "probmode1_neg"]
ALL = [MinCost(), MinShortfall(), MinLargestShortfall(), MaxFeatures(), MinBoundary()]


@pytest.mark.parametrize("kind", KINDS)
def test_objective_values_equal_the_model_at_an_optimal_row(
    kind: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw: list[list[float | None]] = []
    real = mo.evaluate_implied

    def spy(model: Any, x: Any, x_rounded: Any, implied: Any, terms: Any) -> Any:
        if len(terms) == len(ALL):
            raw.append([pulp.value(t.expr) for t in terms])
        return real(model, x, x_rounded, implied, terms)

    monkeypatch.setattr(mo, "evaluate_implied", spy)
    sol = run(ALL, WeightedSum([1, 2, 1, 1, 0.5]), _fixture(kind))[0]
    assert sol.metadata["sol_status"] == 1
    assert [vals(sol)[s.key] for s in ALL] == pytest.approx(raw[0], abs=1e-6)


@pytest.mark.parametrize("kind", KINDS)
def test_min_cost_then_min_shortfall_reaches_stage_2(kind: str) -> None:
    """Round 6: v_rec from the Marxan-style compute_* helpers is unattainable here."""
    pu_status = [2, 2, 0, 0]
    p = _fixture(kind)
    p = p.copy_with(planning_units=p.planning_units.assign(status=pu_status))
    sol = run([MinCost(), MinShortfall()], Hierarchical(0.0), p)[0]
    assert sol.metadata["stage_reached"] == 2


def test_asymmetric_connectivity_locked_endpoints_reach_stage_2() -> None:
    """Round 6 (§7.5): Hierarchical([MinCost(), MinShortfall()]) with asymmetric
    connectivity and two locked-in endpoints."""
    p = P({"id": [1, 2, 3], "cost": [1.0] * 3, "status": [2, 2, 0]},
          {"id": [1], "name": ["a"], "target": [5.0]},
          {"species": [1], "pu": [3], "amount": [5.0]},
          connectivity=pd.DataFrame({"id1": [1], "id2": [2], "value": [10.0]}),
          params={"CONNECTIVITY_WEIGHT": 1.0, "ASYMMETRIC_CONNECTIVITY": 1})
    sol = run([MinCost(), MinShortfall()], Hierarchical(0.0), p)[0]
    assert sol.metadata["stage_reached"] == 2
    assert vals(sol) == {"min_cost": 2.0, "min_shortfall": 1.0}


def test_implied_point_is_feasible_and_attains_the_reported_values() -> None:
    """§5 invariant on oracle data with connectivity and probability: enumerate every plan,
    and the solver's reported weighted value is the enumerated optimum of the model's own
    expressions evaluated at implied points."""
    p = _fixture("conn_sym_dupes")
    p = p.copy_with(
        probability=pd.DataFrame({"pu": [1, 2, 3, 4], "probability": [.5, .1, .3, 0]}),
        parameters={**p.parameters, "PROBMODE": 1, "PROBABILITYWEIGHTING": 0.8},
    )
    specs = [MinCost(), MinShortfall()]
    sol = run(specs, WeightedSum([1, 3]), p)[0]
    # Enumerate with the same machinery: rebuild terms and evaluate each plan.
    from pymarxan.solvers._mip_objectives import evaluate_implied
    from pymarxan.solvers.base import SolverConfig
    from pymarxan.solvers.mip_solver import MIPSolver

    base = MIPSolver()._build_base(p, SolverConfig())
    base.model += base.cost_expr <= 6.0, "cost_budget"
    ctx = base.context()
    terms = [s.build(ctx) for s in specs]
    best = None
    for plan in itertools.product((0.0, 1.0), repeat=4):
        values, feasible = evaluate_implied(
            base.model, base.x, dict(zip([1, 2, 3, 4], plan)), base.implied, terms,
        )
        if feasible:
            score = (values[0] + 3 * values[1]) / 4
            best = score if best is None else min(best, score)
    v = vals(sol)
    assert (v["min_cost"] + 3 * v["min_shortfall"]) / 4 == pytest.approx(best, abs=1e-9)
```

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/test_multiobjective_oracles.py tests/pymarxan/test_multiobjective_regressions.py -q -p no:cacheprovider`
Expected: `45 passed`.

- [ ] **Step 3: Mutation spot-checks (spec §7.7).** Save this script outside the repo (e.g. as `mutate.py` in the session scratchpad) and run it from the repository root as `/opt/micromamba/envs/shiny/bin/python <scratchpad>/mutate.py "$PWD"`. It applies each mutation, runs the Stage 3 tests, and restores the file with `git checkout`:

```python
"""Apply each spec §7.7 mutation to a clean, committed checkout (argv[1]), run the Stage 3
tests, and restore the file with git checkout. Fails if any mutation survives."""
import re, subprocess, sys
ROOT = sys.argv[1]
OBJ = "src/pymarxan/solvers/_mip_objectives.py"
MO = "src/pymarxan/multiobjective.py"
COMPUTE = '''        def evaluate(i: int) -> float | None:
            from pymarxan.solvers.utils import (
                compute_boundary, compute_connectivity_penalty, compute_probability_penalty,
            )
            chosen = np.array([bool(rounded()[pid]) for pid in base.pu_ids])
            idx = {pid: k for k, pid in enumerate(base.pu_ids)}
            spec = self.objectives[i]
            if isinstance(spec, MinCost):
                cost = float(problem.planning_units["cost"].values[chosen].sum())
                return (cost + base.blm * compute_boundary(problem, chosen, idx)
                        + compute_connectivity_penalty(problem, chosen, idx)
                        + compute_probability_penalty(problem, chosen, idx))
            if isinstance(spec, MinBoundary):
                return compute_boundary(problem, chosen, idx)
            values, feasible = evaluate_implied('''
MUTATIONS = [
    ("M1a flip min cap (<= to >=)", OBJ, "model += term.expr <= v + slack, name",
     "model += term.expr >= v + slack, name"),
    ("M1b flip max cap (>= to <=)", OBJ, "model += term.expr >= v - slack, name",
     "model += term.expr <= v - slack, name"),
    ("M2 drop |v|", OBJ, "float(rel_tols[k - 1]) * abs(v) + abs_tol",
     "float(rel_tols[k - 1]) * v + abs_tol"),
    ("M3a flip weighted-sum sign", MO, "sign = 1.0 if term.sense == pulp.LpMinimize else -1.0",
     "sign = -1.0 if term.sense == pulp.LpMinimize else 1.0"),
    ("M3b drop weighted-sum sign (s = +1)", MO,
     "sign = 1.0 if term.sense == pulp.LpMinimize else -1.0", "sign = 1.0"),
    ("M4 v_rec from compute_* utilities", MO,
     "        def evaluate(i: int) -> float | None:\n            values, feasible = evaluate_implied(",
     COMPUTE),
    ("M5a v = model value only", OBJ, "v_rec = evaluate(k - 1)", "v_rec = None"),
    ("M5b v = recomputed value only", OBJ,
     "            if v_rec is None:\n                v = v_model\n            elif term.sense == pulp.LpMinimize:",
     "            if v_rec is not None:\n                v = v_rec\n            elif v_rec is None:\n                v = v_model\n            elif term.sense == pulp.LpMinimize:"),
    ("M6 rel_tol indexed by objective from the other end", OBJ,
     "float(rel_tols[k - 1])", "float(rel_tols[-k])"),
    ("R6 no restore after a failed later stage", OBJ,
     "                for var in variables:\n                    var.varValue = None\n"
     "                for var, value in snapshot:\n                    var.varValue = value\n"
     "                return accepted",
     "                return accepted"),
    ("S1 hierarchical acceptance by status, not sol_status", OBJ,
     "            if model.sol_status not in ACCEPTED_SOL_STATUSES:",
     "            if model.status != pulp.LpStatusOptimal:"),
    ("S2 weighted-sum acceptance by status, not sol_status", MO,
     "                if model.sol_status not in ACCEPTED_SOL_STATUSES:",
     "                if model.status != pulp.LpStatusOptimal:"),
    ("R8 rounded via int() truncation", MO,
     "return {pid: float(round(v.varValue or 0.0)) for pid, v in x.items()}",
     "return {pid: float(int(v.varValue or 0.0)) for pid, v in x.items()}"),
    ("N5 nondominated identical-vector check with a 1e-6 tolerance", MO,
     "if any(vectors[j] == vec for j in range(i)):",
     "if any(\n            all(abs(p - q) <= 1e-6 for p, q in zip(vectors[j], vec)) for j in range(i)\n        ):"),
    ("N6 stage-1 failure raises only in the first sweep row", MO,
     "                if res.stage_reached == 0:",
     "                if res.stage_reached == 0 and r == 0:"),
    ("P9 MultiObjectiveMIP evaluate ignores feasible", MO,
     "return values[0] if feasible else None", "return values[0]"),
    ("M7 skip hier_cap_* in the feasibility check", OBJ,
     "feasible = all(c.valid(1e-6) for c in constraint_map(model).values())",
     "feasible = all(\n            c.valid(1e-6) for n, c in constraint_map(model).items()\n            if not n.startswith(\"hier_cap_\")\n        )"),
]
TESTS = ["tests/pymarxan/solvers/test_mip_hierarchical.py",
         "tests/pymarxan/test_multiobjective_core.py",
         "tests/pymarxan/test_multiobjective_solve.py",
         "tests/pymarxan/test_multiobjective_oracles.py",
         "tests/pymarxan/test_multiobjective_regressions.py"]
for label, path, old, new in MUTATIONS:
    full = f"{ROOT}/{path}"
    src = open(full).read()
    assert src.count(old) == 1, (label, src.count(old))
    open(full, "w").write(src.replace(old, new))
    try:
        out = subprocess.run(
            ["/opt/micromamba/envs/shiny/bin/pytest", *TESTS, "-q", "-p", "no:cacheprovider"],
            cwd=ROOT, env={"PYTHONPATH": "src:.", "PATH": "/usr/bin:/bin"},
            capture_output=True, text=True,
        ).stdout
    finally:
        subprocess.run(["git", "checkout", "--", path], cwd=ROOT, check=True)
    failed = re.findall(r"^FAILED (\S+)", out, re.M)
    summary = out.strip().splitlines()[-1]
    print(f"## {label}: {summary}")
    for f in failed:
        print("   ", f.split("::", 1)[1] if "::" in f else f)
    assert failed and "error" not in summary, f"{label} survived: {summary}"
```
Expected (prototype output): every mutation fails at least one test; the script exits 0.

```text
## M1a flip min cap (<= to >=): 20 failed, 107 passed, 150 warnings in 2.49s
    test_min_cap_is_v_plus_r_abs_v_plus_abs_tol[0.0]
    test_min_cap_is_v_plus_r_abs_v_plus_abs_tol[0.5]
    test_zero_cost_pu_in_no_row_does_not_break_v
    test_min_boundary_is_non_trivial_at_blm_zero
    test_hierarchical_rel_tol_0_is_the_lexicographic_optimum[cost_first-None]
    test_hierarchical_rel_tol_0_is_the_lexicographic_optimum[cost_first-4.0]
    test_hierarchical_rel_tol_0_is_the_lexicographic_optimum[shortfall_first-None]
    test_hierarchical_rel_tol_0_is_the_lexicographic_optimum[shortfall_first-4.0]
    test_hierarchical_rel_tol_r_caps_stage_1[0.25]
    test_hierarchical_rel_tol_r_caps_stage_1[0.6]
    test_hierarchical_rel_tol_r_caps_cost_first[0.25]
    test_hierarchical_rel_tol_r_caps_cost_first[1.0]
    test_three_objectives_rel_tol_follows_stage_order
    test_hierarchical_respects_locks_and_no_good_cuts[0.0]
    test_hierarchical_respects_locks_and_no_good_cuts[0.5]
    test_hierarchical_sweep_rows_match_their_own_oracles
    test_v_zero_allows_exactly_abs_tol[1e-09-expected0]
    test_v_zero_allows_exactly_abs_tol[0.5-expected1]
    test_zero_cost_pu_without_amounts
    test_asymmetric_connectivity_locked_endpoints_reach_stage_2
## M1b flip max cap (>= to <=): 10 failed, 117 passed, 150 warnings in 2.38s
    test_max_cap_is_v_minus_r_abs_v_minus_abs_tol
    test_tolerance_boundary_cap_keeps_the_model_target
    test_incumbent_cap_uses_the_recomputed_value
    test_ghost_feature_is_reported
    test_three_objectives_rel_tol_follows_stage_order
    test_mixed_sense_hierarchy_max_then_min[0.0]
    test_mixed_sense_hierarchy_max_then_min[0.25]
    test_negative_v_with_rel_tol_is_feasible
    test_tolerance_boundary_cap_returns_cost_2
    test_incumbent_cap_keeps_both_targets
## M2 drop |v|: 2 failed, 125 passed, 150 warnings in 2.17s
    test_min_cap_is_v_plus_r_abs_v_plus_abs_tol[0.5]
    test_negative_v_with_rel_tol_is_feasible
## M3a flip weighted-sum sign: 21 failed, 106 passed, 139 warnings in 2.40s
    test_weighted_sum_metadata_and_solution_fields
    test_two_min_largest_shortfall_labels_build
    test_presolve_runs_once_on_the_first_rows_objective
    test_weighted_sum_resets_values_before_each_row
    test_near_integral_x_is_rounded
    test_single_objective_equals_mip_solver
    test_mixed_sense_weighted_sum[row0]
    test_mixed_sense_weighted_sum[row1]
    test_mixed_sense_weighted_sum[row2]
    test_mixed_sense_weighted_sum[row3]
    test_weighted_sum_sweep_returns_each_hull_vertex
    test_weighted_sum_grid_rows_reach_the_optimal_weighted_value
    test_nondominated_sweep_equals_the_enumerated_pareto_points_found
    test_prioritizr_parity_weighted_sum
    test_same_row_twice_gives_the_same_result
    test_objective_values_equal_the_model_at_an_optimal_row[blm]
    test_objective_values_equal_the_model_at_an_optimal_row[conn_sym_dupes]
    test_objective_values_equal_the_model_at_an_optimal_row[conn_asym]
    test_objective_values_equal_the_model_at_an_optimal_row[probmode1_pos]
    test_objective_values_equal_the_model_at_an_optimal_row[probmode1_neg]
    test_implied_point_is_feasible_and_attains_the_reported_values
## M3b drop weighted-sum sign (s = +1): 5 failed, 122 passed, 149 warnings in 2.24s
    test_presolve_runs_once_on_the_first_rows_objective
    test_mixed_sense_weighted_sum[row0]
    test_mixed_sense_weighted_sum[row1]
    test_mixed_sense_weighted_sum[row2]
    test_prioritizr_parity_weighted_sum
## M4 v_rec from compute_* utilities: 4 failed, 123 passed, 153 warnings in 2.22s
    test_min_cost_then_min_shortfall_reaches_stage_2[conn_sym_dupes]
    test_min_cost_then_min_shortfall_reaches_stage_2[conn_asym]
    test_min_cost_then_min_shortfall_reaches_stage_2[probmode1_neg]
    test_asymmetric_connectivity_locked_endpoints_reach_stage_2
## M5a v = model value only: 2 failed, 125 passed, 149 warnings in 2.18s
    test_incumbent_cap_uses_the_recomputed_value
    test_incumbent_cap_keeps_both_targets
## M5b v = recomputed value only: 2 failed, 125 passed, 149 warnings in 2.14s
    test_tolerance_boundary_cap_keeps_the_model_target
    test_tolerance_boundary_cap_returns_cost_2
## M6 rel_tol indexed by objective from the other end: 2 failed, 125 passed, 149 warnings in 2.14s
    test_rel_tol_is_indexed_by_stage
    test_three_objectives_rel_tol_follows_stage_order
## R6 no restore after a failed later stage: 2 failed, 125 passed, 149 warnings in 2.19s
    test_stage_2_failure_restores_stage_1_and_its_status
    test_later_stage_failure_warns_and_returns_the_previous_stage
## S1 hierarchical acceptance by status, not sol_status: 2 failed, 125 passed, 148 warnings in 2.17s
    test_acceptance_follows_sol_status_not_status
    test_acceptance_follows_sol_status[approach0]
## S2 weighted-sum acceptance by status, not sol_status: 1 failed, 126 passed, 148 warnings in 2.20s
    test_acceptance_follows_sol_status[approach1]
## R8 rounded via int() truncation: 1 failed, 126 passed, 149 warnings in 2.20s
    test_near_integral_x_is_rounded
## N5 nondominated identical-vector check with a 1e-6 tolerance: 1 failed, 126 passed, 149 warnings in 2.20s
    test_nondominated_compares_exactly
## N6 stage-1 failure raises only in the first sweep row: 1 failed, 126 passed, 150 warnings in 2.19s
    test_stage_1_failure_in_a_later_row_names_the_row
## P9 MultiObjectiveMIP evaluate ignores feasible: 2 failed, 125 passed, 151 warnings in 2.19s
    test_near_integral_budget_falls_back_end_to_end
    test_earlier_caps_are_checked_end_to_end
## M7 skip hier_cap_* in the feasibility check: 2 failed, 125 passed, 150 warnings in 2.23s
    test_earlier_caps_are_checked_three_stages
    test_earlier_caps_are_checked_end_to_end
```
Afterwards `git status --short` must show only this task's two new test files.

- [ ] **Step 4: Lint and commit.**

```bash
~/.local/bin/ruff check src tests
/home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports
git add tests/pymarxan/test_multiobjective_oracles.py tests/pymarxan/test_multiobjective_regressions.py
git commit -m "test(multiobjective): enumeration oracles, prioritizr parity, regressions

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AZEiop9qqxVCrMHKqSDYP8"
```

---

### Task 8: Docs: tutorial section, CHANGELOG Added, README, subpackage table, roadmap

**Files:**
- Modify: `docs/TUTORIAL.md` (table of contents + section 10 before "River connectivity")
- Modify: `tests/test_tutorial_examples.py` (`test_section_multi_objective`, verbatim copy of the block)
- Modify: `CHANGELOG.md` (`[Unreleased]` Added, with the deviations from prioritizr)
- Modify: `README.md` (architecture tree, feature list)
- Modify: `.github/copilot-instructions.md` (subpackage table)
- Modify: `docs/plans/2026-10-11-prioritizr-parity-roadmap.md` (Stage 3 heading marked `(SHIPPED, v0.41)` as Stage 2 did in its docs commit, the "As built" note and the reference-point descope)

**Interfaces:** docs only; the tutorial block is pinned by its test.

Ruling: the tutorial and the CHANGELOG deviations also state the two spec §5 caveats, and the roadmap heading is marked `(SHIPPED, v0.41)` in this task, as Stage 2 did. Ruling: the tutorial test asserts backend-independent facts about the sweep (every point on the enumerated supported frontier, the three rows with unique optima present, shortfall decreasing with cost): enumeration over all 4,096 plans shows two weight rows tying between plans on a hull edge, and the tutorial does not force `mip_backend="cbc"`. The hierarchical pick (cost 5, shortfall 1.5667) is unique. The tutorial uses `rel_tol=0.1` with COSTBUDGET 6, as spec §8 says. The section still mentions `abs_tol` for v = 0. The DOIs were verified with scite during the spec's research (`stage3-prioritizr-multiobj.md` §7): Williams & Kendall 2017, Das & Dennis 1997, Schuster et al. 2023, Hanson et al. 2024.

- [ ] **Step 1: Write the tutorial test** (fails until the doc block exists only in the sense that the two must match; write both in this task):

```diff
diff --git a/tests/test_tutorial_examples.py b/tests/test_tutorial_examples.py
index 43b613d..bfc9adf 100644
--- a/tests/test_tutorial_examples.py
+++ b/tests/test_tutorial_examples.py
@@ -297,6 +297,65 @@ def test_section_target_methods():
     assert list(problem.features["target"]) == [5.0, 4.0, 3.0]  # original untouched
 
 
+# --------------------------------------------------------------------
+# Section: "10. Multi-objective planning (v0.41)"
+# --------------------------------------------------------------------
+
+
+def test_section_multi_objective():
+    problem = build_minimal_problem()
+
+    # --- tutorial block (verbatim) ---
+    from pymarxan.multiobjective import (
+        Hierarchical,
+        MinCost,
+        MinShortfall,
+        MultiObjectiveMIP,
+        WeightedSum,
+        nondominated,
+        weights_matrix,
+    )
+
+    # A budget that cannot meet every target, so the best shortfall is above 0.
+    budgeted = problem.copy_with(parameters={**problem.parameters, "COSTBUDGET": 6.0})
+
+    # Weighted sums are not scaled: divide each weight by a typical magnitude
+    # (total cost 20; relative shortfall of at most SPF per feature: 3 features at SPF 1).
+    scale = [1 / 20.0, 1 / 3.0]
+    rows = [[w * s for w, s in zip(row, scale)] for row in weights_matrix(2, 5)]
+    sweep = MultiObjectiveMIP([MinCost(), MinShortfall()], WeightedSum(rows)).solve(budgeted)
+    frontier = nondominated(sweep)
+    for sol in frontier:
+        v = sol.metadata["objective_values"]
+        print(f"cost {v['min_cost']:.0f}  shortfall {v['min_shortfall']:.3f}")
+
+    # The cheapest plan within 10% of the best achievable shortfall.
+    pick = MultiObjectiveMIP(
+        [MinShortfall(), MinCost()], Hierarchical(rel_tol=0.1),
+    ).solve(budgeted)[0]
+    # --- end tutorial block ---
+
+    points = sorted(
+        (s.metadata["objective_values"]["min_cost"],
+         round(s.metadata["objective_values"]["min_shortfall"], 4))
+        for s in frontier
+    )
+    # Two weight rows tie between plans on a hull edge (enumerated over all 4,096 plans),
+    # so the exact set depends on the backend's tie-breaking (CBC finds 0, 2, 3, 5, 6).
+    # Every row's optimum lies on this enumerated supported frontier:
+    supported = {(0.0, 3.0), (1.0, 2.55), (2.0, 2.2167), (3.0, 1.9667), (4.0, 1.7667),
+                 (5.0, 1.5667), (6.0, 1.4333)}
+    assert set(points) <= supported
+    assert {(0.0, 3.0), (2.0, 2.2167), (6.0, 1.4333)} <= set(points)  # unique row optima
+    assert [p[1] for p in points] == sorted((p[1] for p in points), reverse=True)
+    best = min(p[1] for p in points)
+    assert best > 0  # the budget makes rel_tol meaningful
+    v = pick.metadata["objective_values"]
+    assert v["min_cost"] == 5.0 and pick.cost == 5.0
+    assert v["min_shortfall"] == pytest.approx(1.5667, abs=1e-4)
+    assert v["min_shortfall"] <= 1.4333334 * 1.1
+
+
 # --------------------------------------------------------------------
 # Section: "River connectivity and barrier restoration"
 # --------------------------------------------------------------------
```

- [ ] **Step 2: Add the tutorial section:**

```diff
diff --git a/docs/TUTORIAL.md b/docs/TUTORIAL.md
index c6a82a2..38e13cd 100644
--- a/docs/TUTORIAL.md
+++ b/docs/TUTORIAL.md
@@ -22,6 +22,7 @@ covered by `docs/USER_MANUAL.md`; this file is for the Python API.
 7. [Solution portfolios via no-good cuts (Phase 25)](#7-solution-portfolios-via-no-good-cuts-phase-25)
 8. [BLM calibration with Pareto filter (Phase 25)](#8-blm-calibration-with-pareto-filter-phase-25)
 9. [Published target-setting methods (v0.40)](#9-published-target-setting-methods-v040)
+10. [Multi-objective planning (v0.41)](#10-multi-objective-planning-v041)
 
 ---
 
@@ -353,6 +354,64 @@ feature id. See each function's docstring for its source.
 `loglinear_targets` is deprecated: use
 `interp_absolute_targets(..., interp="log10")`.
 
+## 10. Multi-objective planning (v0.41)
+
+`pymarxan.multiobjective` trades objectives against each other exactly, on
+one problem. List the objectives (`MinCost`, `MinBoundary`, `MinShortfall`,
+`MinLargestShortfall`, `MaxFeatures`, `MaxWeightedFeatures`, `MaxWtdSum`)
+and pick an approach: `WeightedSum` (one solve per weight row) or
+`Hierarchical` (lexicographic, in list order). Targets are soft here, and
+`COSTBUDGET` is optional; when set, it binds every solve.
+
+```python
+from pymarxan.multiobjective import (
+    Hierarchical,
+    MinCost,
+    MinShortfall,
+    MultiObjectiveMIP,
+    WeightedSum,
+    nondominated,
+    weights_matrix,
+)
+
+# A budget that cannot meet every target, so the best shortfall is above 0.
+budgeted = problem.copy_with(parameters={**problem.parameters, "COSTBUDGET": 6.0})
+
+# Weighted sums are not scaled: divide each weight by a typical magnitude
+# (total cost 20; relative shortfall of at most SPF per feature: 3 features at SPF 1).
+scale = [1 / 20.0, 1 / 3.0]
+rows = [[w * s for w, s in zip(row, scale)] for row in weights_matrix(2, 5)]
+sweep = MultiObjectiveMIP([MinCost(), MinShortfall()], WeightedSum(rows)).solve(budgeted)
+frontier = nondominated(sweep)
+for sol in frontier:
+    v = sol.metadata["objective_values"]
+    print(f"cost {v['min_cost']:.0f}  shortfall {v['min_shortfall']:.3f}")
+
+# The cheapest plan within 10% of the best achievable shortfall.
+pick = MultiObjectiveMIP(
+    [MinShortfall(), MinCost()], Hierarchical(rel_tol=0.1),
+).solve(budgeted)[0]
+```
+
+The sweep finds the cost-vs-shortfall trade-offs, from the empty plan
+(cost 0, shortfall 3.0) to the best shortfall the budget allows (1.433 at
+cost 6); with CBC, five of them. Weight rows that tie between plans can
+return either plan, so the set can differ between backends.
+
+The hierarchical pick gives up 0.133 of shortfall to save one cost unit:
+stage 1 finds the best shortfall v, stage 2 minimises cost subject to
+`shortfall <= v + 0.1·|v| + abs_tol`. When the best value is 0 (for example
+with no budget), `rel_tol` gives no slack; raise `abs_tol` (default 1e-9)
+instead. `objective_values` are each objective's own value at the plan;
+`Solution.cost` and `Solution.objective` are filled as `MIPSolver` fills
+them. Met counts use the exact rule (amount >= target), so on data at a
+target's tolerance boundary (0.7 + 0.1 against 0.8) the reported count can
+sit below what a hierarchical cap kept, which the solver reached within its
+tolerance; and when a plan's implied point breaks a model row (such as an
+earlier cap), that stage's cap falls back to the solver's own value. A weighted sum only finds plans on the convex hull of the frontier
+(Das & Dennis 1997), so `Hierarchical` with a `rel_tol_matrix` sweep can
+reach trade-offs a weight grid misses.
+
 ---
 
 ## River connectivity and barrier restoration
```

- [ ] **Step 3: CHANGELOG, README, subpackage table, roadmap:**

```diff
diff --git a/.github/copilot-instructions.md b/.github/copilot-instructions.md
index e8ddb95..4d8a035 100644
--- a/.github/copilot-instructions.md
+++ b/.github/copilot-instructions.md
@@ -43,6 +43,7 @@ pymarxan is a three-layer monorepo:
 | `spatial/` | Grid generation, feature intersection, boundary generation, GeoPackage export |
 | `io/` | Marxan file readers/writers (binary & zone formats), output controller |
 | `calibration/` | BLM calibration & sensitivity analysis |
+| `multiobjective.py` | `MultiObjectiveMIP` (hierarchical / weighted-sum over one `ConservationProblem`), objective specs, `weights_matrix` / `rel_tol_matrix`, `nondominated`; builds on `MIPSolver._build_base` and `solvers/_mip_objectives.py` |
 | `analysis/` | Selection frequency, portfolio analysis |
 
 ### Data flow
diff --git a/CHANGELOG.md b/CHANGELOG.md
index 3564a67..140b5d5 100644
--- a/CHANGELOG.md
+++ b/CHANGELOG.md
@@ -7,6 +7,18 @@ and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0
 
 ## [Unreleased]
 
+### Added
+
+- **Exact multi-objective optimisation** in `pymarxan.multiobjective` (prioritizr 9.0.1 concepts, Stage 3 of the prioritizr roadmap): `MultiObjectiveMIP(objectives, approach)` over one `ConservationProblem`, with objective specs `MinCost`, `MinBoundary`, `MinShortfall`, `MinLargestShortfall`, `MaxFeatures`, `MaxWeightedFeatures` and `MaxWtdSum`, and approaches `Hierarchical(rel_tol, abs_tol=1e-9)` (lexicographic, in list order; Williams & Kendall 2017, https://doi.org/10.1016/j.ecolmodel.2016.10.010; applied in Schuster et al. 2023, https://doi.org/10.1111/cobi.14048) and `WeightedSum(weights)` (one solve per weight row; unscaled, so objectives in different units need scaled weights: Das & Dennis 1997, https://doi.org/10.1007/BF01197559). As prioritizr cautions for its weighted-sum objectives, `MaxWtdSum` ignores complementarity (more of a well-held feature scores as much as a missing one), so pair it with a shortfall objective when representation matters. Both take a 2-D sequence for a sweep and return one `Solution` per row, with `metadata["objective_values"]`, `["objective_senses"]`, `["status"]`, `["sol_status"]` and, for hierarchical runs, `["stage_reached"]`. Targets are soft and `COSTBUDGET` is optional. `weights_matrix` / `rel_tol_matrix` reproduce prioritizr's parameter matrices row for row, and `nondominated` filters a sweep's solutions (Hanson et al. 2024, prioritizr, https://doi.org/10.1111/cobi.14376).
+- MIP objective builders `build_min_cost`, `build_min_boundary` and `build_max_wtd_sum` in `solvers/_mip_objectives.py`, with an `owner=` name prefix on every builder and an implied-value registry for auxiliary variables.
+- **Deviations from prioritizr 9.0.1** (deliberate):
+  - Hierarchical caps are `v + r·|v| + abs_tol` (minimise) and `v − (r·|v| + abs_tol)` (maximise), not `v·(1 ± r)`: prioritizr's cap is infeasible when v < 0, and allows no slack at v = 0 (raise `abs_tol` for slack there).
+  - Priority is the order of the objectives list (no `priority` argument); `rel_tol` is indexed by solve order.
+  - Reported objective values are recomputed at the selected plan from the model's own expressions, not read from raw auxiliary variables (which misstate objectives not optimised at that stage). Met counts follow the exact `check_targets` rule, so on data at a target's tolerance boundary the reported count can sit below the value a hierarchical cap kept (the solver met the target within its tolerance); and when the implied point of a stage's plan violates a model row, including an earlier stage's cap, that cap falls back to the solver's own value.
+  - A hierarchical cap uses the better of the solver's value and the recomputed value, so an incumbent (time limit, `MIP_GAP` > 0) with loose auxiliaries cannot weaken later stages.
+  - An all-zero weights row raises `ValueError` (prioritizr gives NaN coefficients).
+  - The reference-point approach is omitted: in 9.0.1 its max rows are written `t − w·d ≤ 0` instead of `≥ 0` (`rcpp_convert_ref_point_method.cpp`), so it minimises a weighted sum of normalised shortfalls.
+
 ### Changed
 
 - `MIPSolver` internals are refactored onto the objective builders in `solvers/_mip_objectives.py`: `_build` is now a shared base model (`_build_base`: locks, cost, boundary, connectivity, probability, amounts, no-good cuts) plus one builder per objective. Solutions are unchanged (pinned by `tests/data/golden/stage3_v040.json`), except the ghost-feature fix below. LP variable and constraint names in exported models (`build_model`, `write_problem`) change for `max_features` / `max_weighted_features` / absolute `min_largest_shortfall`: `feat_met_{id}` → `y_f{id}`, `slack_{id}` → `s_f{id}`, the relaxed `target_{id}` → `met_f{id}`, `slack_{id}` rows → `short_f{id}`, `max_shortfall_{id}` → `maxs_f{id}`; the objective is no longer named `"objective"`. `min_set`'s hard `target_{id}` rows keep their names. The missing-`COSTBUDGET` error for `max_features`, `max_weighted_features` and absolute `min_largest_shortfall` now uses the shared `objective=... requires a COSTBUDGET parameter` text. Absolute `min_largest_shortfall` and `min_penalties` no longer add slack rows for a feature with target × MISSLEVEL ≤ 0, and `max_*` ghost features get a `met_f{id}` row instead of a fixed `upBound`.
diff --git a/README.md b/README.md
index d0da629..4fa9422 100644
--- a/README.md
+++ b/README.md
@@ -74,6 +74,7 @@ pymarxan (three-layer monorepo)
 │   ├── constraints/       Contiguity, feature contiguity, neighbor, linear, budget
 │   ├── objectives/        Deprecated (v0.39, removal v0.42): use MIPSolver(objective=...)
 │   ├── targets/           Target rules: relative, group, range-size, IUCN Red List, population, composition
+│   ├── multiobjective.py  Exact multi-objective MIP: hierarchical / weighted-sum sweeps, nondominated filter
 │   ├── spatial/           Grid generation, feature intersection, boundary, export
 │   ├── io/                Marxan file readers/writers (binary & zone formats)
 │   ├── calibration/       BLM calibration & sensitivity analysis
@@ -146,6 +147,7 @@ pymarxan (three-layer monorepo)
 - Distributional **equity** analysis (per-group totals/shares + Gini coefficient)
 - Published **target-setting** methods (Rodrigues, Watson, Polak, Sreekar, IUCN Red List species/ecosystems, Jung, Ward, population size, Wilson, Durán, rule-based, min/max) with explicit km² units
 - Multi-scenario **robustness** (minimax-regret no-regrets plan selection)
+- Exact **multi-objective** planning (cost vs representation vs boundary): hierarchical and weighted-sum approaches with prioritizr's parameter sweeps and a nondominated filter (`MaxWtdSum` ignores complementarity, as prioritizr cautions: pair it with a shortfall objective)
 
 ### Constraints
 - **Contiguity**: selected PUs must form a connected subgraph (MIP network flow)
diff --git a/docs/plans/2026-10-11-prioritizr-parity-roadmap.md b/docs/plans/2026-10-11-prioritizr-parity-roadmap.md
index fcc8353..6e0ec28 100644
--- a/docs/plans/2026-10-11-prioritizr-parity-roadmap.md
+++ b/docs/plans/2026-10-11-prioritizr-parity-roadmap.md
@@ -122,7 +122,9 @@ models.
 
 ---
 
-## Stage 3 — v0.41.0: Multi-objective optimisation
+## Stage 3 — v0.41.0: Multi-objective optimisation (SHIPPED, v0.41)
+
+**As built (v0.41.0):** see docs/plans/2026-10-15-stage3-multi-objective-design.md. The API is `pymarxan.multiobjective.MultiObjectiveMIP(objectives, approach)` over an ordinary `ConservationProblem` (not `MultiObjectiveProblem`); hierarchical and weighted-sum approaches, prioritizr's parameter matrices, and `nondominated` (new code: `calibration.pareto` filters a 2-objective `BLMResult`). **Descoped:** the reference-point approach and its acceptance item — prioritizr 9.0.1's implementation is broken (its max rows are written `t − w·d ≤ 0`), so there is no correct reference to match; a Chebyshev/achievement version can be built on the same builders later.
 
 **Goal:** trade off several objectives (e.g. cost vs representation vs connectivity) exactly,
 the way prioritizr 9's `multi_problem()` does.
```

- [ ] **Step 4: Run the tutorial tests.**

Run: `PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/test_tutorial_examples.py -q -p no:cacheprovider`

Expected: `11 passed`.

- [ ] **Step 5: Full gate.**

```bash
~/.local/bin/ruff check src/ tests/ examples/
/home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports
PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/pytest tests/ -q -m "not bench" -p no:cacheprovider --cov --cov-report=term
git diff --stat 00d01a3 HEAD -- src/pymarxan/zones/
```
Expected (prototype): ruff `All checks passed!`; mypy `Success: no issues found in 195 source files`; `3176 passed, 20 deselected`; total coverage 85% (`_mip_objectives.py` 99%, `mip_solver.py` 100%, `multiobjective.py` 99%); empty `zones/` diff. `tests/test_examples.py` (the 35.0 anchor harness) and `tests/pymarxan/zones/` are part of the run.

- [ ] **Step 6: Commit.**

```bash
~/.local/bin/ruff check src tests
/home/razinka/marxan/.venv/bin/mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports
git add docs/TUTORIAL.md tests/test_tutorial_examples.py CHANGELOG.md README.md .github/copilot-instructions.md docs/plans/2026-10-11-prioritizr-parity-roadmap.md
git commit -m "docs(multiobjective): tutorial section, CHANGELOG, README, subpackage table, roadmap

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AZEiop9qqxVCrMHKqSDYP8"
```

