# Stage 2 (v0.40.0) Target-Setting Methods Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the published target-setting methods prioritizr 9 ships (range-size interpolation, IUCN Red List, population-based, simple + composition) to `pymarxan.targets` as plain functions returning `{feature_id: target}` in native units, with explicit km² unit handling and prioritizr's bugs fixed.

**Architecture:** `src/pymarxan/targets.py` becomes a package. Private `_units.py` holds every shared rule (feature selection, id-keyed per-feature inputs, km² conversion, cap, clip). One module per method family (`_interp`, `_redlist`, `_population`, `_compose`) builds on it. `__init__.py` re-exports flat with an explicit `__all__`. The existing `relative_targets` / `group_targets` / `apply_targets` move unchanged to `_basic.py`, and `loglinear_targets` gains a `DeprecationWarning`.

**Tech Stack:** Python 3.11+, pandas, stdlib `math`; pytest; PuLP/CBC only in the end-to-end test.

**Spec:** `docs/plans/2026-10-15-stage2-target-methods-design.md` (approved 2026-10-15 after 3 review rounds). Executors read it. The spec is the authority, and this plan argues from it.

## Global Constraints

- Python 3.11+ floor; `from __future__ import annotations` as the first import in every new file; full type hints. mypy runs as `mypy src/pymarxan/ src/pymarxan_shiny/ --ignore-missing-imports`.
- ruff rules E, F, I, UP; line length 99. **Run `ruff check` on the test files too at every task handoff.** An E402 slipped through two reviews in v0.32.
- Tests run ONLY under `/opt/micromamba/envs/shiny/bin/pytest`, never `.venv`. Full gate: `PATH="/opt/micromamba/envs/shiny/bin:$HOME/.local/bin:$PWD/.venv/bin:$PATH" make check`.
- No `ConservationProblem` field changes, no Shiny changes, no top-level `pymarxan` export (spec §11).
- Every area-based method takes keyword-only `km2_per_unit` with **no default**, and `cap_km2: float | None = 1_000_000.0`. Exception: `duran_targets` defaults `cap_km2=None`. Every new method takes `features: Iterable[int] | None = None` (spec §3).
- Per-feature inputs are looked up **by feature id**, never by position. A missing key or an invalid value raises a `ValueError` that names the feature ids.
- Outputs are `dict[int, float]` in `problem.features` order; area-based results are clipped to the feature total; a zero total gives target 0.0. `absolute_targets` is excluded from both rules (spec §3.3–3.4).
- Test directory `tests/pymarxan/targets/` has **no `__init__.py`**, so every test file needs a repo-unique basename (`test_targets_*.py`). Precedent: `tests/pymarxan/rivers/` uses a `conftest.py` the same way.
- Pinned test values are hand-computed from the published formula, and each test comment names the formula (spec §1 provenance).
- The `tests/data/simple` MIP anchor (35.0) must stay green; nothing here touches solvers.
- Commit trailer on every commit:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01AZEiop9qqxVCrMHKqSDYP8
  ```

## Review Focus

These inputs are implied by the spec but not named by it, and are the ones most likely to bite a real user (Baltic/ROWER data arrives as DataFrames). The 2026-10-15 4-lens review found the first draft's tests for #1–#4 were vacuous or missing; each now has a test that fails if the guard is removed.
1. **Per-feature inputs as a `pandas.Series`** indexed by feature id, e.g. `df.set_index("id")["density"]`, possibly long-format with a **duplicated index**. Users expect it to work like a dict, and duplicates to raise rather than silently keep the last value. → `per_feature` converts the Series and rejects a duplicated index; the public type aliases `PerFeatureFloat`/`PerFeatureStr` include `pd.Series`, so typed callers pass mypy. Pinned in Task 2 (`test_per_feature_reads_series_by_index`, `test_per_feature_series_duplicate_index_raises`) and Task 6 (`test_rule_targets_traits_series`).
2. **Keys and `features=` that aren't feature ids**: dicts keyed by feature *name* (`{"cod": 1.0}`), float keys (`1.5`), or a boolean mask `features=df["threatened"]`. Users expect a `ValueError` naming the argument, not truncation to id 1 or a raw `int()` error. → `as_feature_id` + `select_features`. Pinned in Task 2 (`test_per_feature_rejects_non_id_keys`, `test_select_features_rejects_non_integral_ids`, `test_select_features_rejects_boolean_mask`).
3. **NaN or negative `pu_vs_features` amounts.** pandas' groupby-sum skips NaN, and `[100, -50]` sums to a plausible 50. Users expect an error naming the feature. → `feature_totals` checks rows. Pinned in Task 2 (`test_feature_totals_bad_row_raises_even_if_total_looks_fine`).
4. **Status strings with stray case or whitespace** (`" En "`, `"eN\t"`), as they come from spreadsheets. Users expect them to be accepted. → `_norm` strips and upper-cases. Pinned in Task 4 (`test_rl_species_status_case_and_whitespace`, `test_rl_ecosystem_a2b_any_case`).
5. **`features=[]`, or a dict that covers only the selected subset.** Users expect `{}` and a correct subset, with no "missing key" error for unselected features. → Pinned in Task 2 (`test_select_features_accepts_numpy_ids_and_empty`) and Task 4 (`test_rl_species_subset_with_status_only_for_subset`).

---

### Task 1: Convert `pymarxan.targets` to a package; deprecate `loglinear_targets`

**Files:**
- Move: `src/pymarxan/targets.py` → `src/pymarxan/targets/_basic.py` (with `git mv`)
- Create: `src/pymarxan/targets/__init__.py`
- Modify: `tests/pymarxan/test_targets.py` (lines 51–102: the three `loglinear_targets` tests, plus one new test)

**Interfaces:**
- Consumes: nothing new.
- Produces: package `pymarxan.targets` exporting `apply_targets`, `group_targets`, `loglinear_targets`, `relative_targets` with an explicit `__all__`. Later tasks add an import block and `__all__` entries to `__init__.py`. `loglinear_targets(...)` keeps exactly the v0.39 signature and behaviour, plus `DeprecationWarning`.

- [ ] **Step 1: Write the failing deprecation test.** In `tests/pymarxan/test_targets.py`, add after `test_loglinear_invalid_target_fraction_raises`:

```python
def test_loglinear_is_deprecated():
    problem = _problem()
    with pytest.warns(DeprecationWarning, match="interp_absolute_targets"):
        loglinear_targets(
            problem,
            lower_area=10.0,
            lower_target=1.0,
            upper_area=100.0,
            upper_target=0.1,
        )


def test_targets_package_all():
    import pymarxan.targets as targets_pkg

    assert {"apply_targets", "group_targets", "loglinear_targets", "relative_targets"} <= set(
        targets_pkg.__all__
    )
```

- [ ] **Step 2: Run to verify both fail.**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/test_targets.py -q`
Expected: 2 failed (`DID NOT WARN`; `AttributeError: module 'pymarxan.targets' has no attribute '__all__'`).

- [ ] **Step 3: Move the module and add the warning.**

```bash
mkdir -p src/pymarxan/targets
git mv src/pymarxan/targets.py src/pymarxan/targets/_basic.py
```

In `src/pymarxan/targets/_basic.py`:
- Replace the module docstring (lines 1–10) with:
```python
"""Relative, group and (deprecated) log-linear targets, and ``apply_targets``.

Moved unchanged from the v0.39 ``pymarxan/targets.py`` module, except that
:func:`loglinear_targets` now emits a ``DeprecationWarning``.
"""
```
- Change `import math` to `import math` followed by `import warnings` (keep `from collections.abc import Mapping` after them).
- In `loglinear_targets`, replace the end of the docstring and the first line of the body:
```python
    Returns:
        Mapping of feature id to target amount.
    """
    if lower_area <= 0 or upper_area <= 0:
```
with:
```python
    Returns:
        Mapping of feature id to target amount.

    .. deprecated:: 0.40.0
        Use :func:`interp_absolute_targets` with ``interp="log10"``; this
        function will be removed in a future release.
    """
    warnings.warn(
        "loglinear_targets is deprecated and will be removed in a future release; "
        "use interp_absolute_targets(..., interp='log10')",
        DeprecationWarning,
        stacklevel=2,
    )
    if lower_area <= 0 or upper_area <= 0:
```
Nothing else in `_basic.py` changes.

- [ ] **Step 4: Create `src/pymarxan/targets/__init__.py`** (later tasks add their import blocks and `__all__` names, keeping both alphabetical):

```python
"""Automatic target-setting rules.

Set each feature's representation target by a rule rather than by hand.
Each rule returns a ``{feature_id: target_amount}`` mapping in the feature's
native units; :func:`apply_targets` writes it onto a problem's features.
Features a rule does not return keep their existing target, so methods are
mixed across feature groups with ``features=`` plus repeated
:func:`apply_targets` calls.

Area-based methods take a required ``km2_per_unit`` (a factor or a
``{feature_id: factor}`` mapping): feature totals are converted to km², the
published rule is applied, and the target is converted back. There is no
default unit.

Mirrors prioritizr 9.0.1's target methods (Hanson et al. (2024),
*Conservation Biology*, https://doi.org/10.1111/cobi.14376); deviations
from prioritizr are listed in the CHANGELOG for v0.40.0.
"""
from __future__ import annotations

from pymarxan.targets._basic import (
    apply_targets,
    group_targets,
    loglinear_targets,
    relative_targets,
)

__all__ = [
    "apply_targets",
    "group_targets",
    "loglinear_targets",
    "relative_targets",
]
```

- [ ] **Step 5: Wrap the three existing loglinear tests.** In `tests/pymarxan/test_targets.py`, wrap each `loglinear_targets(...)` call in `test_loglinear_clamps_below_lower_and_above_upper`, `test_loglinear_interpolates_on_log_scale` and `test_loglinear_invalid_target_fraction_raises` in `with pytest.warns(DeprecationWarning):`. Don't change any assertion. For the third test, the warning fires before validation, so nest it like this:

```python
def test_loglinear_invalid_target_fraction_raises():
    problem = _problem()
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="target"):
        loglinear_targets(
            problem,
            lower_area=10.0,
            lower_target=5.0,
            upper_area=100.0,
            upper_target=0.1,
        )
```

and for the first two:

```python
    with pytest.warns(DeprecationWarning):
        targets = loglinear_targets(
            ...unchanged arguments...
        )
```

- [ ] **Step 6: Run the tests.**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/test_targets.py -q -W error::DeprecationWarning`
Expected: all pass. The `-W` flag proves no unwrapped deprecated call remains. Don't add a `:pymarxan` module filter: `stacklevel=2` attributes the warning to the *calling test module*, so a module-filtered `-W` would never fire (found by the 4-lens review). `test_targets.py` makes no PuLP calls, so PuLP's own deprecations can't trip it.

Run: `grep -rn "loglinear_targets\|pymarxan.targets\|from pymarxan import targets" src/ examples/ docs/*.md tests/ --include=*.py --include=*.md | grep -v "^tests/pymarxan/test_targets.py\|^src/pymarxan/targets/"`
Expected: no output (no other caller to update).

- [ ] **Step 7: Lint, type-check, commit.**

```bash
~/.local/bin/ruff check src/pymarxan/targets tests/pymarxan/test_targets.py
.venv/bin/mypy src/pymarxan/targets --ignore-missing-imports
git add -A src/pymarxan/targets tests/pymarxan/test_targets.py
git commit -m "refactor(targets): convert to package; deprecate loglinear_targets"
```
(Add the trailer from Global Constraints.)

---

### Task 2: Shared helpers `_units.py`

**Files:**
- Create: `src/pymarxan/targets/_units.py`
- Create: `tests/pymarxan/targets/conftest.py`
- Test: `tests/pymarxan/targets/test_targets_units.py`

**Interfaces:**
- Consumes: `ConservationProblem.feature_amounts() -> dict[int, float]`.
- Produces (all private, used by Tasks 3–6):
  - Type aliases `PerFeatureFloat = float | Mapping[int, float] | pd.Series` and `PerFeatureStr = str | Mapping[int, str] | pd.Series`. Every public per-feature argument uses them, and they are **exported** from `pymarxan.targets` so typed callers can import them.
  - `as_feature_id(name, key) -> int`: accepts int, numpy integers and integral floats; any other key raises `ValueError("{name} must be feature ids (int), ...")`.
  - `select_features(problem, features: Iterable[int] | None) -> list[int]`: ids in `problem.features` order; raises on an unknown id, a non-id value, or a boolean-mask Series.
  - `per_feature(name: str, value: Any, fids: list[int]) -> dict[int, Any]`: a scalar, a Mapping or a `pd.Series`. A Series with a duplicated index raises; keys go through `as_feature_id(f"{name} keys", k)`; extra keys are ignored; a missing key raises.
  - `numeric_per_feature(name, value, fids, *, lower=None, upper=None, lower_inclusive=True) -> dict[int, float]`
  - `check_scalar(name, value, *, lower=None, upper=None, lower_inclusive=True) -> float`: messages are `"{name} out of range: ..."` and `"{name} must be a number, ..."`.
  - `check_cap(name, cap: float | None) -> float | None`
  - `feature_totals(problem, fids) -> dict[int, float]`: 0.0 when a feature has no rows; totals include locked-out PUs. Raises when any `pu_vs_features` row of a selected feature has a NaN, infinite or negative amount.
  - `km2_factors(km2_per_unit, fids) -> dict[int, float]`: each factor > 0.
  - `finish_area(t_km2, f_km2, k, cap_km2) -> float`: `0.0` if `f_km2 == 0`, else `min(t, f, cap) / k`.
  - Test fixture `make_problem(totals: dict[int, float]) -> ConservationProblem`.

- [ ] **Step 1: Create the fixture** `tests/pymarxan/targets/conftest.py`:

```python
"""Shared fixture for the target-method tests."""
from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem


def build_problem(totals: dict[int, float]) -> ConservationProblem:
    """A problem whose feature ``fid`` has native total ``totals[fid]``.

    Each feature with a non-zero total sits on its own planning unit; a
    zero-total feature has no ``pu_vs_features`` rows at all.
    """
    fids = list(totals)
    nonzero = [f for f in fids if totals[f] != 0]
    planning_units = pd.DataFrame(
        {
            "id": list(range(1, len(nonzero) + 1)),
            "cost": [1.0] * len(nonzero),
            "status": [0] * len(nonzero),
        }
    )
    features = pd.DataFrame(
        {
            "id": fids,
            "name": [f"f{f}" for f in fids],
            "target": [0.0] * len(fids),
            "spf": [1.0] * len(fids),
        }
    )
    pu_vs_features = pd.DataFrame(
        {
            "species": nonzero,
            "pu": list(range(1, len(nonzero) + 1)),
            "amount": [float(totals[f]) for f in nonzero],
        }
    )
    return ConservationProblem(planning_units, features, pu_vs_features)


@pytest.fixture
def make_problem() -> Callable[[dict[int, float]], ConservationProblem]:
    return build_problem
```

- [ ] **Step 2: Write the failing tests** `tests/pymarxan/targets/test_targets_units.py`:

```python
"""Tests for the shared target-method helpers (pymarxan.targets._units)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pymarxan.targets._units import (
    check_cap,
    check_scalar,
    feature_totals,
    finish_area,
    km2_factors,
    numeric_per_feature,
    per_feature,
    select_features,
)


def test_select_features_defaults_to_all_in_problem_order(make_problem):
    p = make_problem({3: 1.0, 1: 2.0, 2: 3.0})
    assert select_features(p, None) == [3, 1, 2]


def test_select_features_subset_keeps_problem_order(make_problem):
    p = make_problem({3: 1.0, 1: 2.0, 2: 3.0})
    assert select_features(p, [2, 3]) == [3, 2]


def test_select_features_accepts_numpy_ids_and_empty(make_problem):
    p = make_problem({1: 1.0, 2: 2.0})
    assert select_features(p, np.array([2], dtype=np.int64)) == [2]
    assert select_features(p, [2.0]) == [2]
    assert select_features(p, []) == []


@pytest.mark.parametrize("bad", [[1.7], ["1"], [True]])
def test_select_features_rejects_non_integral_ids(make_problem, bad):
    p = make_problem({1: 1.0, 2: 2.0})
    with pytest.raises(ValueError, match="features must be feature ids"):
        select_features(p, bad)


def test_select_features_rejects_boolean_mask(make_problem):
    # features=df["threatened"] is a natural mistake; True would read as id 1
    p = make_problem({1: 1.0, 2: 2.0})
    with pytest.raises(ValueError, match="boolean mask"):
        select_features(p, pd.Series([True, True]))


def test_select_features_unknown_id_raises(make_problem):
    p = make_problem({1: 1.0})
    with pytest.raises(ValueError, match=r"not in problem: \[9\]"):
        select_features(p, [1, 9])


def test_per_feature_scalar_and_mapping():
    assert per_feature("x", 2.0, [1, 2]) == {1: 2.0, 2: 2.0}
    assert per_feature("x", {2: "b", 1: "a", 7: "z"}, [1, 2]) == {1: "a", 2: "b"}


def test_per_feature_reads_series_by_index():
    series = pd.Series({1: 0.5, 2: 0.7})
    assert per_feature("x", series, [2, 1]) == {2: 0.7, 1: 0.5}
    # a DataFrame column indexed by feature id, in any row order
    df = pd.DataFrame({"id": [2, 1], "density": [0.7, 0.5]})
    assert per_feature("x", df.set_index("id")["density"], [1, 2]) == {1: 0.5, 2: 0.7}


def test_per_feature_series_duplicate_index_raises():
    with pytest.raises(ValueError, match=r"duplicate entries for feature\(s\) \[1\]"):
        per_feature("x", pd.Series([1.0, 100.0, 1.0], index=[1, 1, 2]), [1, 2])


def test_per_feature_accepts_numpy_and_integral_float_keys():
    assert per_feature("x", {np.int64(1): 5.0, 2.0: 6.0}, [1, 2]) == {1: 5.0, 2: 6.0}


@pytest.mark.parametrize("key", ["f1", 1.5, True])
def test_per_feature_rejects_non_id_keys(key):
    # name-keyed dicts and float keys must not be misread or truncated
    with pytest.raises(ValueError, match="x keys must be feature ids"):
        per_feature("x", {key: 1.0}, [1])


def test_per_feature_missing_key_names_feature():
    with pytest.raises(ValueError, match=r"x has no value for feature\(s\) \[2\]"):
        per_feature("x", {1: 1.0}, [1, 2])


def test_numeric_per_feature_rejects_nan_negative_and_text():
    with pytest.raises(ValueError, match=r"invalid for feature\(s\) \[1, 2, 3\]"):
        numeric_per_feature("d", {1: math.nan, 2: -1.0, 3: "abc", 4: 1.0}, [1, 2, 3, 4],
                            lower=0.0)


def test_numeric_per_feature_exclusive_lower_bound():
    with pytest.raises(ValueError, match="> 0.0"):
        numeric_per_feature("d", 0.0, [1], lower=0.0, lower_inclusive=False)
    assert numeric_per_feature("d", 0.0, [1], lower=0.0) == {1: 0.0}


def test_numeric_per_feature_upper_bound():
    with pytest.raises(ValueError, match="<= 1.0"):
        numeric_per_feature("p", 1.5, [1], lower=0.0, upper=1.0)


def test_check_scalar_and_cap():
    assert check_scalar("a", 3, lower=0.0) == 3.0
    with pytest.raises(ValueError, match="a out of range"):
        check_scalar("a", -1.0, lower=0.0)
    with pytest.raises(ValueError, match="must be a number"):
        check_scalar("a", "x")
    assert check_cap("cap_km2", None) is None
    with pytest.raises(ValueError, match="cap_km2"):
        check_cap("cap_km2", math.inf)


def test_feature_totals_zero_for_features_without_rows(make_problem):
    p = make_problem({1: 4.0, 2: 0.0})
    assert feature_totals(p, [1, 2]) == {1: 4.0, 2: 0.0}


def test_feature_totals_negative_raises(make_problem):
    p = make_problem({1: -4.0})
    with pytest.raises(ValueError, match=r"invalid for feature\(s\) \[1\]"):
        feature_totals(p, [1])


@pytest.mark.parametrize("amounts", [[100.0, math.nan], [100.0, -50.0], [math.inf, 1.0]])
def test_feature_totals_bad_row_raises_even_if_total_looks_fine(make_problem, amounts):
    # pandas sums skip NaN, and [100, -50] sums to a plausible 50: check rows
    p = make_problem({1: 1.0, 2: 1.0})
    p.pu_vs_features = pd.DataFrame(
        {"species": [1, 1, 2], "pu": [1, 2, 2], "amount": [*amounts, 3.0]}
    )
    with pytest.raises(ValueError, match=r"amounts must be finite and >= 0.*\[1\]"):
        feature_totals(p, [1, 2])
    assert feature_totals(p, [2]) == {2: 3.0}  # other features unaffected


def test_feature_totals_include_locked_out_planning_units(make_problem):
    # as in prioritizr, totals count every PU, locked-out (status 3) ones too
    p = make_problem({1: 4.0, 2: 6.0})
    p.planning_units.loc[p.planning_units["id"] == 1, "status"] = 3
    assert feature_totals(p, [1, 2]) == {1: 4.0, 2: 6.0}


def test_km2_factors_requires_positive_factor_for_every_feature():
    assert km2_factors(0.5, [1, 2]) == {1: 0.5, 2: 0.5}
    with pytest.raises(ValueError, match="km2_per_unit"):
        km2_factors(0.0, [1])
    with pytest.raises(ValueError, match=r"no value for feature\(s\) \[2\]"):
        km2_factors({1: 1.0}, [1, 2])


def test_finish_area_clips_caps_converts_and_zero():
    # t 50 km² of a 40 km² range -> 40 km² -> /k=0.01 -> 4,000 native units
    assert finish_area(50.0, 40.0, 0.01, None) == pytest.approx(4000.0)
    assert finish_area(5e6, 2e7, 1.0, 1e6) == 1e6
    assert finish_area(math.inf, 300.0, 1.0, None) == 300.0
    assert finish_area(10.0, 0.0, 1.0, 1e6) == 0.0
```

- [ ] **Step 3: Run to verify they fail.**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/targets/test_targets_units.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'pymarxan.targets._units'`.

- [ ] **Step 4: Implement** `src/pymarxan/targets/_units.py`:

```python
"""Shared validation and unit helpers for the target-setting methods.

Private module: every public method in :mod:`pymarxan.targets` funnels its
inputs through these helpers so the contract (feature selection, per-feature
inputs keyed by feature id, explicit km² conversion, cap, clip to the
feature total) is enforced in one place.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any, TypeAlias

import numpy as np
import pandas as pd

from pymarxan.models.problem import ConservationProblem

# Input types accepted by the public methods (exported from pymarxan.targets):
# one value for every feature, a ``{feature_id: value}`` mapping, or a pandas
# Series indexed by feature id.
PerFeatureFloat: TypeAlias = float | Mapping[int, float] | pd.Series
PerFeatureStr: TypeAlias = str | Mapping[int, str] | pd.Series


def as_feature_id(name: str, key: object) -> int:
    """Return ``key`` as an int feature id; only integral numbers are accepted.

    Accepts ``int``, numpy integers and integral floats (``3.0``); rejects
    booleans, non-integral floats and strings with a ``ValueError`` naming
    ``name``, instead of silently truncating or misreading them.
    """
    if isinstance(key, (bool, np.bool_)):
        raise ValueError(f"{name} must be feature ids (int), got a boolean {key!r}")
    if isinstance(key, (int, np.integer)):
        return int(key)
    if isinstance(key, (float, np.floating)) and float(key).is_integer():
        return int(key)
    raise ValueError(f"{name} must be feature ids (int), got {key!r}")


def select_features(
    problem: ConservationProblem, features: Iterable[int] | None
) -> list[int]:
    """Feature ids to compute, in ``problem.features`` order.

    A boolean mask (e.g. ``df["threatened"]``) is rejected: pass the ids,
    e.g. ``df.loc[df["threatened"], "id"]``.
    """
    ids = [int(f) for f in problem.features["id"]]
    if features is None:
        return ids
    if isinstance(features, pd.Series) and features.dtype == bool:
        raise ValueError(
            "features must be feature ids, not a boolean mask; "
            "use e.g. df.loc[mask, 'id']"
        )
    wanted = {as_feature_id("features", f) for f in features}
    unknown = sorted(wanted.difference(ids))
    if unknown:
        raise ValueError(f"features not in problem: {unknown}")
    return [f for f in ids if f in wanted]


def per_feature(name: str, value: Any, fids: list[int]) -> dict[int, Any]:
    """Expand a scalar or a ``{feature_id: value}`` mapping to every id in ``fids``.

    A :class:`pandas.Series` is read as a mapping from its index (a duplicated
    index raises). Keys must be feature ids. Extra keys are ignored; a missing
    key raises ``ValueError`` naming the ids.
    """
    if isinstance(value, pd.Series):
        dup = value.index[value.index.duplicated()]
        if len(dup):
            raise ValueError(
                f"{name} has duplicate entries for feature(s) {sorted(set(dup.tolist()))}"
            )
        value = value.to_dict()
    if isinstance(value, Mapping):
        by_id = {as_feature_id(f"{name} keys", k): v for k, v in value.items()}
        missing = [f for f in fids if f not in by_id]
        if missing:
            raise ValueError(f"{name} has no value for feature(s) {missing}")
        return {f: by_id[f] for f in fids}
    return {f: value for f in fids}


def numeric_per_feature(
    name: str,
    value: Any,
    fids: list[int],
    *,
    lower: float | None = None,
    upper: float | None = None,
    lower_inclusive: bool = True,
) -> dict[int, float]:
    """Like :func:`per_feature`, but every value must be a finite float in range."""
    raw = per_feature(name, value, fids)
    out: dict[int, float] = {}
    bad: list[int] = []
    for f, v in raw.items():
        try:
            x = float(v)
        except (TypeError, ValueError):
            bad.append(f)
            continue
        if not math.isfinite(x):
            bad.append(f)
        elif lower is not None and (x < lower or (x == lower and not lower_inclusive)):
            bad.append(f)
        elif upper is not None and x > upper:
            bad.append(f)
        else:
            out[f] = x
    if bad:
        op = ">=" if lower_inclusive else ">"
        bounds = ", ".join(
            b
            for b in (
                None if lower is None else f"{op} {lower}",
                None if upper is None else f"<= {upper}",
            )
            if b
        )
        raise ValueError(
            f"{name} must be finite{' and ' + bounds if bounds else ''}; "
            f"invalid for feature(s) {bad}"
        )
    return out


def check_scalar(
    name: str,
    value: float,
    *,
    lower: float | None = None,
    upper: float | None = None,
    lower_inclusive: bool = True,
) -> float:
    """Validate one scalar argument: finite and (optionally) within bounds."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    ok = math.isfinite(x)
    if ok and lower is not None:
        ok = x > lower or (lower_inclusive and x == lower)
    if ok and upper is not None:
        ok = x <= upper
    if not ok:
        raise ValueError(f"{name} out of range: {value!r}")
    return x


def check_cap(name: str, cap: float | None) -> float | None:
    """``None`` (no cap) or a finite value >= 0."""
    if cap is None:
        return None
    return check_scalar(name, cap, lower=0.0)


def feature_totals(problem: ConservationProblem, fids: list[int]) -> dict[int, float]:
    """Native-unit total amount per feature (0.0 for features with no rows).

    Totals include every planning unit, locked-out (status 3) ones too, as
    prioritizr does. Raises ``ValueError`` naming the features when any of
    their ``pu_vs_features`` amounts is negative or non-finite (pandas would
    otherwise skip NaN rows silently).
    """
    pvf = problem.pu_vs_features
    rows = pvf[pvf["species"].isin(fids)]
    amounts = rows["amount"].astype(float)
    bad_rows = rows.loc[~np.isfinite(amounts) | (amounts < 0), "species"]
    if len(bad_rows):
        raise ValueError(
            "pu_vs_features amounts must be finite and >= 0; invalid for feature(s) "
            f"{sorted({int(f) for f in bad_rows})}"
        )
    totals = problem.feature_amounts()
    return {f: float(totals.get(f, 0.0)) for f in fids}


def km2_factors(km2_per_unit: Any, fids: list[int]) -> dict[int, float]:
    """Resolve ``km2_per_unit`` (scalar or mapping) to a factor > 0 per feature."""
    return numeric_per_feature(
        "km2_per_unit", km2_per_unit, fids, lower=0.0, lower_inclusive=False
    )


def finish_area(t_km2: float, f_km2: float, k: float, cap_km2: float | None) -> float:
    """Clip a km² target to the feature total and the cap, return native units."""
    if f_km2 == 0.0:
        return 0.0
    t = min(t_km2, f_km2)
    if cap_km2 is not None:
        t = min(t, cap_km2)
    return t / k
```

- [ ] **Step 4b: Export the type aliases.** In `src/pymarxan/targets/__init__.py`, add after the `_basic` import block:

```python
from pymarxan.targets._units import PerFeatureFloat, PerFeatureStr
```

Then add `"PerFeatureFloat"` and `"PerFeatureStr"` at the top of `__all__`; ruff/isort order puts upper-case names first. Later tasks insert their import blocks *above* this `_units` line, which must stay last because `_units` sorts after `_redlist`.

- [ ] **Step 5: Run the tests.**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/targets/test_targets_units.py -q`
Expected: all pass.

- [ ] **Step 6: Lint, type-check, commit.**

```bash
~/.local/bin/ruff check src/pymarxan/targets tests/pymarxan/targets
.venv/bin/mypy src/pymarxan/targets --ignore-missing-imports
git add src/pymarxan/targets/_units.py tests/pymarxan/targets/conftest.py tests/pymarxan/targets/test_targets_units.py
git commit -m "feat(targets): shared unit/validation helpers for target methods"
```

---

### Task 3: Interpolation family `_interp.py`

**Files:**
- Create: `src/pymarxan/targets/_interp.py`
- Modify: `src/pymarxan/targets/__init__.py` (add import block and 6 names to `__all__`)
- Test: `tests/pymarxan/targets/test_targets_interp.py`

**Interfaces:**
- Consumes: Task 2 helpers; `loglinear_targets` from Task 1 (equivalence test).
- Produces (public):
  - Type aliases in `_interp.py`: `Interp = Literal["log10", "linear"]`, `Method = Literal["min", "max"]`; `interp` and `*_method` parameters use them. The strings are case-sensitive, as `Literal` implies.
  - `interp_absolute_targets(problem, *, rare_absolute_threshold, rare_relative_target, common_absolute_threshold, common_relative_target, interp, rare_absolute_target=None, rare_method=None, common_absolute_target=None, common_method=None, cap=None, features=None) -> dict[int, float]`
  - `interp_area_targets(problem, *, km2_per_unit, rare_area_threshold, rare_relative_target, common_area_threshold, common_relative_target, interp, rare_area_target=None, rare_method=None, common_area_target=None, common_method=None, cap_km2=1_000_000.0, features=None)`
  - `rodrigues_targets`, `sreekar_targets`, `polak_targets` (`*, km2_per_unit, rare_area_threshold, rare_relative_target, common_area_threshold, common_relative_target, cap_km2, features`)
  - `watson_targets(*, km2_per_unit, rare_area_threshold=10_000.0, rare_relative_target=1.0, rare_area_target=1_000.0, common_relative_target=0.1, cap_km2, features)`

- [ ] **Step 1: Write the failing tests** `tests/pymarxan/targets/test_targets_interp.py`:

```python
"""Tests for the range-size interpolation target methods.

Pinned values are hand-computed from the published rule as implemented in
prioritizr 9.0.1 (spec §1 provenance); each test names the formula used.
With ``km2_per_unit=1`` the native total equals the range f in km².
"""
from __future__ import annotations

import warnings

import pytest

from pymarxan.targets import (
    interp_absolute_targets,
    interp_area_targets,
    loglinear_targets,
    polak_targets,
    rodrigues_targets,
    sreekar_targets,
    watson_targets,
)

REL = 1e-6


def _one(fn, make_problem, f, **kw):
    """Target for a single feature with range f (km2_per_unit=1)."""
    return fn(make_problem({1: f}), km2_per_unit=1.0, **kw)[1]


# --- interp_area_targets ---------------------------------------------------

_BOTH = dict(
    rare_area_threshold=1_000.0,
    rare_relative_target=0.5,
    rare_area_target=400.0,
    rare_method="max",
    common_area_threshold=100_000.0,
    common_relative_target=0.1,
    common_area_target=20_000.0,
    common_method="min",
    interp="log10",
)


@pytest.mark.parametrize(
    "f, expected",
    [
        (500.0, 400.0),  # f < a: max(400, 0.5*500=250)
        # a<=f<=e: f*(0.5 + (0.1-0.5)*(log10 2000 - 3)/(5 - 3)) = 2000*0.43979
        (2_000.0, 879.5880017),
        (500_000.0, 20_000.0),  # f > e: min(20000, 0.1*500000=50000)
    ],
)
def test_interp_area_with_area_targets(make_problem, f, expected):
    assert _one(interp_area_targets, make_problem, f, **_BOTH) == pytest.approx(expected, rel=REL)


def test_interp_area_linear_between_thresholds(make_problem):
    # linear: 5500 * (1 + (0.1-1)*(5500-1000)/(10000-1000)) = 5500*0.55
    t = _one(
        interp_area_targets, make_problem, 5_500.0,
        rare_area_threshold=1_000.0, rare_relative_target=1.0,
        common_area_threshold=10_000.0, common_relative_target=0.1, interp="linear",
    )
    assert t == pytest.approx(3_025.0, rel=REL)


def test_interp_area_converts_units(make_problem):
    # native unit = ha (0.01 km²): F = 1,000,000 ha -> f = 10,000 km²;
    # rodrigues-style rule gives 6,246.778 km² -> 624,677.8 ha
    p = make_problem({1: 1_000_000.0})
    t = interp_area_targets(
        p, km2_per_unit=0.01, rare_area_threshold=1_000.0, rare_relative_target=1.0,
        common_area_threshold=250_000.0, common_relative_target=0.1, interp="log10",
    )[1]
    assert t == pytest.approx(624_677.8498, rel=REL)


def test_interp_area_per_feature_factor_mapping(make_problem):
    p = make_problem({1: 500.0, 2: 500.0})
    out = interp_area_targets(
        p, km2_per_unit={1: 1.0, 2: 100.0}, rare_area_threshold=1_000.0,
        rare_relative_target=1.0, common_area_threshold=10_000.0,
        common_relative_target=0.1, interp="linear",
    )
    # f1 = 500 km² (rare, 100%) -> 500; f2 = 50,000 km² (common, 10%) -> 5,000 km² -> 50 native
    assert out == {1: pytest.approx(500.0), 2: pytest.approx(50.0)}


@pytest.mark.parametrize(
    "f, expected",
    [
        # f == a takes the interpolation branch: b*f = 500, NOT max(800, 500)
        (1_000.0, 500.0),
        (999.0, 800.0),  # f < a: max(800, 0.5*999)
        # f == e takes the interpolation branch: g*f = 10,000, NOT min(5,000, 10,000)
        (100_000.0, 10_000.0),
        (100_001.0, 5_000.0),  # f > e: min(5000, 0.1*100001)
    ],
)
def test_interp_area_threshold_boundaries(make_problem, f, expected):
    # prioritizr applies the rare/common area targets only for f < a and f > e
    t = _one(
        interp_area_targets, make_problem, f,
        rare_area_threshold=1_000.0, rare_relative_target=0.5, rare_area_target=800.0,
        rare_method="max", common_area_threshold=100_000.0, common_relative_target=0.1,
        common_area_target=5_000.0, common_method="min", interp="log10",
    )
    assert t == pytest.approx(expected, rel=REL)


def test_interp_area_zero_total_is_zero(make_problem):
    p = make_problem({1: 0.0, 2: 10.0})
    out = interp_area_targets(p, km2_per_unit=1.0, **_BOTH)
    assert out[1] == 0.0


def test_interp_area_features_subset(make_problem):
    p = make_problem({1: 500.0, 2: 500.0, 3: 500.0})
    out = interp_area_targets(p, km2_per_unit=1.0, features=[3, 1], **_BOTH)
    assert list(out) == [1, 3]


@pytest.mark.parametrize(
    "override, match",
    [
        ({"rare_area_threshold": 2e5}, "must not exceed"),
        ({"rare_area_threshold": 0.0}, "requires rare_area_threshold > 0"),
        ({"interp": "cubic"}, "interp must be one of"),
        ({"rare_relative_target": 1.5}, "rare_relative_target out of range"),
        ({"rare_method": None}, "must be given together"),
        ({"common_area_target": None}, "must be given together"),
        ({"rare_method": "mean"}, "rare_method must be 'min' or 'max'"),
        ({"cap_km2": -1.0}, "cap_km2 out of range"),
    ],
)
def test_interp_area_validation(make_problem, override, match):
    kw = {**_BOTH, **override}
    with pytest.raises(ValueError, match=match):
        interp_area_targets(make_problem({1: 10.0}), km2_per_unit=1.0, **kw)


def test_interp_area_requires_km2_per_unit(make_problem):
    with pytest.raises(TypeError):
        interp_area_targets(make_problem({1: 10.0}), **_BOTH)  # type: ignore[call-arg]


# --- interp_absolute_targets + loglinear equivalence ------------------------


def test_interp_absolute_native_units_and_cap(make_problem):
    p = make_problem({1: 50.0, 2: 5_000.0})
    out = interp_absolute_targets(
        p, rare_absolute_threshold=100.0, rare_relative_target=1.0,
        common_absolute_threshold=1_000.0, common_relative_target=0.2,
        interp="linear", cap=600.0,
    )
    # 50 < 100 -> 50; 5000 > 1000 -> 1000 -> capped at 600
    assert out == {1: pytest.approx(50.0), 2: pytest.approx(600.0)}


def test_interp_absolute_linear_allows_zero_threshold(make_problem):
    out = interp_absolute_targets(
        make_problem({1: 50.0}), rare_absolute_threshold=0.0, rare_relative_target=1.0,
        common_absolute_threshold=100.0, common_relative_target=0.0, interp="linear",
    )
    assert out[1] == pytest.approx(25.0)  # 50 * (1 - 50/100)


def test_interp_absolute_cap_validation(make_problem):
    with pytest.raises(ValueError, match="cap out of range"):
        interp_absolute_targets(
            make_problem({1: 50.0}), rare_absolute_threshold=10.0, rare_relative_target=1.0,
            common_absolute_threshold=100.0, common_relative_target=0.1, interp="log10",
            cap=-1.0,
        )


@pytest.mark.parametrize("total", [0.0, 5.0, 10.0, 31.6, 55.0, 100.0, 400.0])
def test_loglinear_equals_interp_absolute(make_problem, total):
    p = make_problem({1: total})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = loglinear_targets(
            p, lower_area=10.0, lower_target=1.0, upper_area=100.0, upper_target=0.1
        )
    new = interp_absolute_targets(
        p, rare_absolute_threshold=10.0, rare_relative_target=1.0,
        common_absolute_threshold=100.0, common_relative_target=0.1, interp="log10",
    )
    assert new[1] == pytest.approx(old[1], rel=1e-12, abs=0.0)


# --- presets ----------------------------------------------------------------


@pytest.mark.parametrize(
    "f, expected",
    [
        (500.0, 500.0),  # < 1,000: 100%
        # 1e4*(1 + (0.1-1)*(4-3)/(log10 250000 - 3))
        (10_000.0, 6_246.778498),
        (100_000.0, 24_935.569969),
        (5e6, 500_000.0),  # > 250,000: 10%
        (2e7, 1e6),  # 10% = 2e6 -> cap 1e6
    ],
)
def test_rodrigues(make_problem, f, expected):
    assert _one(rodrigues_targets, make_problem, f) == pytest.approx(expected, rel=REL)


def test_rodrigues_cap_onset_at_1e7(make_problem):
    assert _one(rodrigues_targets, make_problem, 1e7) == pytest.approx(1e6, rel=REL)
    assert _one(rodrigues_targets, make_problem, 9e6) == pytest.approx(9e5, rel=REL)
    assert _one(rodrigues_targets, make_problem, 2e7, cap_km2=None) == pytest.approx(2e6)


@pytest.mark.parametrize(
    "f, expected",
    [
        (1_000.0, 842.010584),  # 1e3*(1 + (0.3-1)*(3 - log10 200)/(log10 250000 - log10 200))
        (100_000.0, 38_994.707786),
        (250_000.0, 75_000.0),
        (5e6, 1e6),  # 30% = 1.5e6 -> cap
    ],
)
def test_sreekar(make_problem, f, expected):
    assert _one(sreekar_targets, make_problem, f) == pytest.approx(expected, rel=REL)


@pytest.mark.parametrize(
    "f, expected",
    [
        (500.0, 500.0),
        (5_500.0, 3_025.0),  # 5500*(1 - 0.9*4500/9000)
        (9_000.0, 1_800.0),  # 9000*0.2 — larger than at 10,000 (non-monotone)
        (10_000.0, 1_000.0),
        (50_000.0, 5_000.0),
    ],
)
def test_polak(make_problem, f, expected):
    assert _one(polak_targets, make_problem, f) == pytest.approx(expected, rel=REL)


@pytest.mark.parametrize(
    "f, expected",
    [
        (500.0, 500.0),  # min(500, 1000)
        (2_000.0, 1_000.0),
        (5_000.0, 1_000.0),
        # At the threshold both branches give 1,000; this only proves no crash
        # (prioritizr errors here: interpolation over two identical points).
        (10_000.0, 1_000.0),
        (50_000.0, 5_000.0),
        (2e7, 1e6),
    ],
)
def test_watson(make_problem, f, expected):
    assert _one(watson_targets, make_problem, f) == pytest.approx(expected, rel=REL)


def test_presets_reject_fixed_settings(make_problem):
    p = make_problem({1: 10.0})
    with pytest.raises(TypeError):
        rodrigues_targets(p, km2_per_unit=1.0, interp="linear")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        watson_targets(p, km2_per_unit=1.0, common_area_threshold=5.0)  # type: ignore[call-arg]


def test_preset_defaults_overridable(make_problem):
    # Rodrigues with a 30% common target: 5e6 km² -> 1.5e6 -> cap 1e6
    t = _one(rodrigues_targets, make_problem, 5e6, common_relative_target=0.3)
    assert t == pytest.approx(1e6)
```

- [ ] **Step 2: Run to verify they fail.**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/targets/test_targets_interp.py -q`
Expected: collection error `ImportError: cannot import name 'interp_absolute_targets' from 'pymarxan.targets'`.

- [ ] **Step 3: Implement** `src/pymarxan/targets/_interp.py`:

```python
"""Range-size interpolation target methods.

``interp_area_targets`` / ``interp_absolute_targets`` implement the general
rule from prioritizr 9.0.1 (``spec_interp_area_targets`` /
``spec_interp_absolute_targets``); the presets fix its parameters to the
published values. See ``docs/plans/2026-10-15-stage2-target-methods-design.md`` §4.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Literal

from pymarxan.models.problem import ConservationProblem
from pymarxan.targets._units import (
    PerFeatureFloat,
    check_cap,
    check_scalar,
    feature_totals,
    finish_area,
    km2_factors,
    select_features,
)

Interp = Literal["log10", "linear"]
Method = Literal["min", "max"]

_METHODS: dict[str, Callable[[float, float], float]] = {"min": min, "max": max}
_INTERPS = ("log10", "linear")


def _validate_rule(
    *,
    unit: str,
    a: float,
    b: float,
    c: float | None,
    rare_method: Method | None,
    e: float,
    g: float,
    h: float | None,
    common_method: Method | None,
    interp: Interp,
) -> None:
    check_scalar(f"rare_{unit}_threshold", a, lower=0.0)
    check_scalar(f"common_{unit}_threshold", e, lower=0.0)
    if a > e:
        raise ValueError(
            f"rare_{unit}_threshold ({a}) must not exceed common_{unit}_threshold ({e})"
        )
    if interp not in _INTERPS:
        raise ValueError(f"interp must be one of {_INTERPS}, got {interp!r}")
    if interp == "log10" and a <= 0:
        raise ValueError(f"interp='log10' requires rare_{unit}_threshold > 0, got {a}")
    check_scalar("rare_relative_target", b, lower=0.0, upper=1.0)
    check_scalar("common_relative_target", g, lower=0.0, upper=1.0)
    for side, target, method in (("rare", c, rare_method), ("common", h, common_method)):
        if (target is None) != (method is None):
            raise ValueError(
                f"{side}_{unit}_target and {side}_method must be given together"
            )
        if target is not None:
            check_scalar(f"{side}_{unit}_target", target, lower=0.0)
            if method not in _METHODS:
                raise ValueError(f"{side}_method must be 'min' or 'max', got {method!r}")


def _rule(
    f: float,
    *,
    a: float,
    b: float,
    c: float | None,
    rare_method: Method | None,
    e: float,
    g: float,
    h: float | None,
    common_method: Method | None,
    interp: Interp,
) -> float:
    """Un-capped, un-clipped target for total ``f`` (same unit as thresholds)."""
    if f < a or (a == e and f == a):
        t = b * f
        if c is not None and rare_method is not None:
            t = _METHODS[rare_method](c, t)
        return t
    if f > e:
        t = g * f
        if h is not None and common_method is not None:
            t = _METHODS[common_method](h, t)
        return t
    if interp == "log10":
        x, x1, x2 = math.log10(f), math.log10(a), math.log10(e)
    else:
        x, x1, x2 = f, a, e
    return f * (b + (g - b) * (x - x1) / (x2 - x1))


def interp_absolute_targets(
    problem: ConservationProblem,
    *,
    rare_absolute_threshold: float,
    rare_relative_target: float,
    common_absolute_threshold: float,
    common_relative_target: float,
    interp: Interp,
    rare_absolute_target: float | None = None,
    rare_method: Method | None = None,
    common_absolute_target: float | None = None,
    common_method: Method | None = None,
    cap: float | None = None,
    features: Iterable[int] | None = None,
) -> dict[int, float]:
    """Interpolated targets on feature totals in **native units**.

    Features whose total is below ``rare_absolute_threshold`` get
    ``rare_relative_target`` of their total (optionally combined with
    ``rare_absolute_target`` via ``rare_method``); above
    ``common_absolute_threshold`` they get ``common_relative_target``
    (optionally combined with ``common_absolute_target`` via
    ``common_method``); in between the proportion is interpolated on
    ``log10`` of the total (``interp="log10"``) or linearly
    (``interp="linear"``). The result is capped at ``cap`` (if given) and
    clipped to the feature total. When both thresholds are equal, a total
    exactly at the threshold takes the rare branch.

    Replaces the deprecated :func:`loglinear_targets`. Mirrors prioritizr
    9.0.1 ``spec_interp_absolute_targets``.

    Returns:
        Mapping of feature id to target amount (native units).
    """
    _validate_rule(
        unit="absolute",
        a=rare_absolute_threshold,
        b=rare_relative_target,
        c=rare_absolute_target,
        rare_method=rare_method,
        e=common_absolute_threshold,
        g=common_relative_target,
        h=common_absolute_target,
        common_method=common_method,
        interp=interp,
    )
    cap_v = check_cap("cap", cap)
    fids = select_features(problem, features)
    totals = feature_totals(problem, fids)
    out: dict[int, float] = {}
    for fid in fids:
        f = totals[fid]
        if f == 0.0:
            out[fid] = 0.0
            continue
        t = _rule(
            f,
            a=float(rare_absolute_threshold),
            b=float(rare_relative_target),
            c=rare_absolute_target,
            rare_method=rare_method,
            e=float(common_absolute_threshold),
            g=float(common_relative_target),
            h=common_absolute_target,
            common_method=common_method,
            interp=interp,
        )
        out[fid] = finish_area(t, f, 1.0, cap_v)
    return out


def interp_area_targets(
    problem: ConservationProblem,
    *,
    km2_per_unit: PerFeatureFloat,
    rare_area_threshold: float,
    rare_relative_target: float,
    common_area_threshold: float,
    common_relative_target: float,
    interp: Interp,
    rare_area_target: float | None = None,
    rare_method: Method | None = None,
    common_area_target: float | None = None,
    common_method: Method | None = None,
    cap_km2: float | None = 1_000_000.0,
    features: Iterable[int] | None = None,
) -> dict[int, float]:
    """Interpolated range-size targets, with thresholds and targets in km².

    Same rule as :func:`interp_absolute_targets`, applied to each feature's
    total converted to km² with ``km2_per_unit`` (a factor or a
    ``{feature_id: factor}`` mapping; required, no default). Targets are
    capped at ``cap_km2`` (default 1,000,000 km², Butchart et al. 2015,
    https://doi.org/10.1111/conl.12158; ``None`` disables), clipped to the
    feature total, and returned in native units. Mirrors prioritizr 9.0.1
    ``spec_interp_area_targets``.

    Returns:
        Mapping of feature id to target amount (native units).
    """
    _validate_rule(
        unit="area",
        a=rare_area_threshold,
        b=rare_relative_target,
        c=rare_area_target,
        rare_method=rare_method,
        e=common_area_threshold,
        g=common_relative_target,
        h=common_area_target,
        common_method=common_method,
        interp=interp,
    )
    cap_v = check_cap("cap_km2", cap_km2)
    fids = select_features(problem, features)
    totals = feature_totals(problem, fids)
    factors = km2_factors(km2_per_unit, fids)
    out: dict[int, float] = {}
    for fid in fids:
        k = factors[fid]
        f = totals[fid] * k
        if f == 0.0:
            out[fid] = 0.0
            continue
        t = _rule(
            f,
            a=float(rare_area_threshold),
            b=float(rare_relative_target),
            c=rare_area_target,
            rare_method=rare_method,
            e=float(common_area_threshold),
            g=float(common_relative_target),
            h=common_area_target,
            common_method=common_method,
            interp=interp,
        )
        out[fid] = finish_area(t, f, k, cap_v)
    return out


def rodrigues_targets(
    problem: ConservationProblem,
    *,
    km2_per_unit: PerFeatureFloat,
    rare_area_threshold: float = 1_000.0,
    rare_relative_target: float = 1.0,
    common_area_threshold: float = 250_000.0,
    common_relative_target: float = 0.1,
    cap_km2: float | None = 1_000_000.0,
    features: Iterable[int] | None = None,
) -> dict[int, float]:
    """Rodrigues et al. (2004) range-size targets.

    100% of range below 1,000 km², 10% above 250,000 km², interpolated on
    log10 of range size in between, capped at 1,000,000 km² (Butchart et al.
    2015). Rodrigues et al. 2004, *BioScience* 54:1092-1100,
    https://doi.org/10.1641/0006-3568(2004)054[1092:GGAPRF]2.0.CO;2 — rule and
    cap as restated in Butchart et al. 2015's methods. Mirrors prioritizr 9.0.1
    ``spec_rodrigues_targets``.
    """
    return interp_area_targets(
        problem,
        km2_per_unit=km2_per_unit,
        rare_area_threshold=rare_area_threshold,
        rare_relative_target=rare_relative_target,
        common_area_threshold=common_area_threshold,
        common_relative_target=common_relative_target,
        interp="log10",
        cap_km2=cap_km2,
        features=features,
    )


def sreekar_targets(
    problem: ConservationProblem,
    *,
    km2_per_unit: PerFeatureFloat,
    rare_area_threshold: float = 200.0,
    rare_relative_target: float = 1.0,
    common_area_threshold: float = 250_000.0,
    common_relative_target: float = 0.3,
    cap_km2: float | None = 1_000_000.0,
    features: Iterable[int] | None = None,
) -> dict[int, float]:
    """Sreekar & Watson (2026) updated range-size targets.

    100% below 200 km², 30% above 250,000 km², log10 interpolation, capped at
    1,000,000 km². Sreekar & Watson 2026, *Nature Reviews Biodiversity*
    2:419-420, https://doi.org/10.1038/s44358-026-00169-7. The numeric values
    follow prioritizr 9.0.1 ``spec_sreekar_targets``; they could not be
    confirmed from accessible paper text.
    """
    return interp_area_targets(
        problem,
        km2_per_unit=km2_per_unit,
        rare_area_threshold=rare_area_threshold,
        rare_relative_target=rare_relative_target,
        common_area_threshold=common_area_threshold,
        common_relative_target=common_relative_target,
        interp="log10",
        cap_km2=cap_km2,
        features=features,
    )


def polak_targets(
    problem: ConservationProblem,
    *,
    km2_per_unit: PerFeatureFloat,
    rare_area_threshold: float = 1_000.0,
    rare_relative_target: float = 1.0,
    common_area_threshold: float = 10_000.0,
    common_relative_target: float = 0.1,
    cap_km2: float | None = 1_000_000.0,
    features: Iterable[int] | None = None,
) -> dict[int, float]:
    """Polak et al. (2015) range-size targets.

    100% below 1,000 km², 10% above 10,000 km², **linear** interpolation in
    between, capped at 1,000,000 km². The absolute target is non-monotone in
    range size (9,000 km² → 1,800 km² but 10,000 km² → 1,000 km²); that is
    inherent to the rule. Polak et al. 2015, *R. Soc. Open Sci.* 2:150107,
    https://doi.org/10.1098/rsos.150107 — the breakpoints and linear
    interpolation are stated in its Methods. Polak et al. also set 100% for
    Critically Endangered species regardless of range; this preset (like
    prioritizr's) does not: target CR features separately, e.g.
    ``apply_targets(problem, group_targets(problem, {f: "CR" for f in cr_ids},
    {"CR": 1.0}))``.
    """
    return interp_area_targets(
        problem,
        km2_per_unit=km2_per_unit,
        rare_area_threshold=rare_area_threshold,
        rare_relative_target=rare_relative_target,
        common_area_threshold=common_area_threshold,
        common_relative_target=common_relative_target,
        interp="linear",
        cap_km2=cap_km2,
        features=features,
    )


def watson_targets(
    problem: ConservationProblem,
    *,
    km2_per_unit: PerFeatureFloat,
    rare_area_threshold: float = 10_000.0,
    rare_relative_target: float = 1.0,
    rare_area_target: float = 1_000.0,
    common_relative_target: float = 0.1,
    cap_km2: float | None = 1_000_000.0,
    features: Iterable[int] | None = None,
) -> dict[int, float]:
    """Watson et al. (2010) range-size targets.

    Below 10,000 km² the target is ``min(range, 1,000 km²)``; above it, 10% of
    range; capped at 1,000,000 km². The common threshold always equals
    ``rare_area_threshold``; a range exactly at it takes the rare branch
    (prioritizr errors there). Watson et al. 2010, *Conservation Biology*
    25:324-332, https://doi.org/10.1111/j.1523-1739.2010.01587.x. Values follow
    prioritizr 9.0.1 ``spec_watson_targets``; they could not be confirmed from
    the paper text. Polak et al. (2015), which says it follows Watson et al.,
    describes a different shape (linear 100% → 10% between 1,000 and
    10,000 km², see :func:`polak_targets`); check the Watson et al. paper
    before relying on this preset's exact shape.
    """
    return interp_area_targets(
        problem,
        km2_per_unit=km2_per_unit,
        rare_area_threshold=rare_area_threshold,
        rare_relative_target=rare_relative_target,
        rare_area_target=rare_area_target,
        rare_method="min",
        common_area_threshold=rare_area_threshold,
        common_relative_target=common_relative_target,
        interp="linear",
        cap_km2=cap_km2,
        features=features,
    )
```

- [ ] **Step 4: Re-export.** In `src/pymarxan/targets/__init__.py`, add after the `_basic` import block:

```python
from pymarxan.targets._interp import (
    interp_absolute_targets,
    interp_area_targets,
    polak_targets,
    rodrigues_targets,
    sreekar_targets,
    watson_targets,
)
```

and add `"interp_absolute_targets"`, `"interp_area_targets"`, `"polak_targets"`, `"rodrigues_targets"`, `"sreekar_targets"`, `"watson_targets"` to `__all__` (alphabetical). **Ordering rule for this and later tasks:** import blocks go in module-name order `_basic`, `_compose`, `_interp`, `_population`, `_redlist`, `_units` (ruff isort), and `__all__` stays sorted with the two aliases first.

- [ ] **Step 5: Run the tests.**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/targets/test_targets_interp.py tests/pymarxan/test_targets.py -q`
Expected: all pass.

- [ ] **Step 6: Lint, type-check, commit.**

```bash
~/.local/bin/ruff check src/pymarxan/targets tests/pymarxan/targets
.venv/bin/mypy src/pymarxan/targets --ignore-missing-imports
git add src/pymarxan/targets tests/pymarxan/targets/test_targets_interp.py
git commit -m "feat(targets): interpolated area/absolute targets + Rodrigues, Sreekar, Polak, Watson"
```

---

### Task 4: IUCN Red List family `_redlist.py`

**Files:**
- Create: `src/pymarxan/targets/_redlist.py`
- Modify: `src/pymarxan/targets/__init__.py`
- Test: `tests/pymarxan/targets/test_targets_redlist.py`

**Interfaces:**
- Consumes: Task 2 helpers.
- Produces (public):
  - `rl_species_targets(problem, *, km2_per_unit, status, criterion_a, criterion_b, prop_uplift=0.0, method="max", cap_km2=1_000_000.0, features=None)`
  - `rl_ecosystem_targets(...)`, with the same signature.
  - `jung_targets(problem, *, km2_per_unit, status="VU", prop_uplift=0.1, cap_km2, features)`
  - `ward_targets(problem, *, km2_per_unit, status="CR", prop_uplift=0.0, cap_km2, features)`
  - Private module constants `_SPECIES_A`, `_SPECIES_B`, `_ECOSYSTEM_A`, `_ECOSYSTEM_B` (dicts keyed by upper-case criterion, then status). They are not exported; spec §2 lists no public constants. `method` is typed `Literal["max", "min"]`.

- [ ] **Step 1: Write the failing tests** `tests/pymarxan/targets/test_targets_redlist.py`:

```python
"""Tests for the IUCN Red List target methods.

Pinned values are hand-computed from t = min(m(B(1+p), f(1-A)(1+p)), cap, f)
with the IUCN thresholds (spec §5); km2_per_unit=1 so the total is f in km².
"""
from __future__ import annotations

import pytest

from pymarxan.targets import (
    jung_targets,
    rl_ecosystem_targets,
    rl_species_targets,
    ward_targets,
)

REL = 1e-9


def _one(fn, make_problem, f, **kw):
    return fn(make_problem({1: f}), km2_per_unit=1.0, **kw)[1]


@pytest.mark.parametrize(
    "f, expected",
    [(3_000.0, 3_000.0), (10_000.0, 5_000.0), (50_000.0, 15_000.0)],
)
def test_rl_species_en_a1_b1(make_problem, f, expected):
    # max(5000, 0.3 f), clipped to f
    t = _one(rl_species_targets, make_problem, f, status="EN", criterion_a="A1",
             criterion_b="B1")
    assert t == pytest.approx(expected, rel=REL)


@pytest.mark.parametrize("f, expected", [(20.0, 4.8), (1_000.0, 12.0)])
def test_rl_species_cr_min_with_uplift(make_problem, f, expected):
    # min(10*1.2, f*0.2*1.2)
    t = _one(rl_species_targets, make_problem, f, status="CR", criterion_a="A2",
             criterion_b="B2", prop_uplift=0.2, method="min")
    assert t == pytest.approx(expected, rel=REL)


def test_rl_species_en_b2_is_500(make_problem):
    # prioritizr docs say 50 km²; code and IUCN say 500. max(500, 0.5*600=300)
    t = _one(rl_species_targets, make_problem, 600.0, status="EN", criterion_a="A2",
             criterion_b="B2")
    assert t == pytest.approx(500.0, rel=REL)


@pytest.mark.parametrize(
    "f, expected",
    [(4_000.0, 4_000.0), (8_000.0, 5_000.0), (40_000.0, 20_000.0)],
)
def test_rl_ecosystem_vu_a3_is_half(make_problem, f, expected):
    # max(5000, (1-0.5) f). prioritizr's VU-A3 = 0.3 would give 0.7*40,000 = 28,000.
    t = _one(rl_ecosystem_targets, make_problem, f, status="VU", criterion_a="A3",
             criterion_b="B2")
    assert t == pytest.approx(expected, rel=REL)


@pytest.mark.parametrize("crit", ["A2b", "a2b", "A2B", " a2B "])
def test_rl_ecosystem_a2b_any_case(make_problem, crit):
    # prioritizr can never match "A2b". max(2000, 0.5*10000)
    t = _one(rl_ecosystem_targets, make_problem, 10_000.0, status="en", criterion_a=crit,
             criterion_b="b2")
    assert t == pytest.approx(5_000.0, rel=REL)


@pytest.mark.parametrize("status", ["EN", "en", " En ", "eN\t"])
def test_rl_species_status_case_and_whitespace(make_problem, status):
    # spreadsheet values: max(5000, 0.3*50000) = 15,000 at EN/A1/B1
    t = _one(rl_species_targets, make_problem, 50_000.0, status=status, criterion_a="A1",
             criterion_b="B1")
    assert t == pytest.approx(15_000.0, rel=REL)


def test_rl_species_per_feature_status(make_problem):
    p = make_problem({1: 50_000.0, 2: 50_000.0})
    out = rl_species_targets(p, km2_per_unit=1.0, status={2: "VU", 1: "cr"},
                             criterion_a="A2", criterion_b="B1")
    # CR: max(100, 0.2*50000) = 10,000 ; VU: max(20000, 0.7*50000) = 35,000
    assert out == {1: pytest.approx(10_000.0), 2: pytest.approx(35_000.0)}


def test_rl_species_subset_with_status_only_for_subset(make_problem):
    p = make_problem({1: 50_000.0, 2: 50_000.0, 3: 50_000.0})
    out = rl_species_targets(p, km2_per_unit=1.0, status={3: "EN"}, criterion_a="A1",
                             criterion_b="B1", features=[3])
    assert out == {3: pytest.approx(15_000.0)}


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"status": "LC"}, r"status must be one of .*feature\(s\) \[1\]"),
        ({"status": {2: "VU"}}, r"status has no value for feature\(s\) \[1\]"),
        ({"criterion_a": "A5"}, "criterion_a must be one of"),
        ({"criterion_b": "B3"}, "criterion_b must be one of"),
        ({"prop_uplift": -0.1}, "prop_uplift out of range"),
        ({"method": "mean"}, "method must be 'max' or 'min'"),
    ],
)
def test_rl_species_validation(make_problem, kw, match):
    args = {"status": "VU", "criterion_a": "A2", "criterion_b": "B2", **kw}
    with pytest.raises(ValueError, match=match):
        rl_species_targets(make_problem({1: 10.0}), km2_per_unit=1.0, **args)


def test_rl_ecosystem_rejects_species_only_criterion(make_problem):
    with pytest.raises(ValueError, match="criterion_a"):
        rl_ecosystem_targets(make_problem({1: 10.0}), km2_per_unit=1.0, status="VU",
                             criterion_a="A4", criterion_b="B1")


@pytest.mark.parametrize(
    "f, expected",
    [
        (2_000.0, 2_000.0),  # max(2200, 1540) = 2200 -> clipped to f
        (2_500.0, 2_200.0),  # plateau
        (2_857.142857142857, 2_200.0),  # 2200/0.77: plateau ends
        (5_000.0, 3_850.0),  # 0.77 f
        (50_000.0, 38_500.0),
        (2e7, 1e6),
    ],
)
def test_jung(make_problem, f, expected):
    assert _one(jung_targets, make_problem, f) == pytest.approx(expected, rel=REL)


@pytest.mark.parametrize(
    "f, expected",
    [(50.0, 50.0), (100.0, 100.0), (500.0, 100.0), (50_000.0, 10_000.0), (2e7, 1e6)],
)
def test_ward(make_problem, f, expected):
    # max(100, 0.2 f)
    assert _one(ward_targets, make_problem, f) == pytest.approx(expected, rel=REL)


def test_presets_reject_fixed_settings(make_problem):
    p = make_problem({1: 10.0})
    with pytest.raises(TypeError):
        jung_targets(p, km2_per_unit=1.0, criterion_a="A1")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        ward_targets(p, km2_per_unit=1.0, method="min")  # type: ignore[call-arg]


def test_zero_total_is_zero(make_problem):
    p = make_problem({1: 0.0, 2: 5.0})
    assert jung_targets(p, km2_per_unit=1.0)[1] == 0.0
```

- [ ] **Step 2: Run to verify they fail.**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/targets/test_targets_redlist.py -q`
Expected: collection error `ImportError: cannot import name 'jung_targets'`.

- [ ] **Step 3: Implement** `src/pymarxan/targets/_redlist.py`:

```python
"""IUCN Red List target methods (species and ecosystems).

The target is the area that would leave a feature qualifying at no worse
than ``status`` if only the protected part survived:
``t = min(m(B·(1+p), f·(1−A)·(1+p)), cap, f)`` with ``A`` the criterion-A
reduction threshold and ``B`` the criterion-B range threshold (km²).
Mirrors prioritizr 9.0.1 ``spec_rl_species_targets`` /
``spec_rl_ecosystem_targets``. See the Stage 2 design spec §5.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pymarxan.models.problem import ConservationProblem
from pymarxan.targets._units import (
    PerFeatureFloat,
    PerFeatureStr,
    check_cap,
    check_scalar,
    feature_totals,
    finish_area,
    km2_factors,
    per_feature,
    select_features,
)

_Table = dict[str, dict[str, float]]

_STATUSES = ("CR", "EN", "VU")
_A2_SPECIES = {"CR": 0.8, "EN": 0.5, "VU": 0.3}

# IUCN (2012) Red List Categories and Criteria v3.1: criterion A reduction thresholds and
# criterion B range thresholds (km²; B1 = extent of occurrence, B2 = area of
# occupancy). Species EN-B2 is 500 km² (prioritizr's docs say 50; its code
# and IUCN say 500).
_SPECIES_A: _Table = {
    "A1": {"CR": 0.9, "EN": 0.7, "VU": 0.5},
    "A2": _A2_SPECIES,
    "A3": _A2_SPECIES,
    "A4": _A2_SPECIES,
}
_SPECIES_B: _Table = {
    "B1": {"CR": 100.0, "EN": 5_000.0, "VU": 20_000.0},
    "B2": {"CR": 10.0, "EN": 500.0, "VU": 2_000.0},
}

_A1_ECO = {"CR": 0.8, "EN": 0.5, "VU": 0.3}
# IUCN Red List of Ecosystems guidelines v2.0. A3 VU is 0.5 (IUCN thresholds
# 90/70/50%); prioritizr 9.0.1 has 0.3 — a deliberate deviation.
_ECOSYSTEM_A: _Table = {
    "A1": _A1_ECO,
    "A2A": _A1_ECO,
    "A2B": _A1_ECO,
    "A3": {"CR": 0.9, "EN": 0.7, "VU": 0.5},
}
_ECOSYSTEM_B: _Table = {
    "B1": {"CR": 2_000.0, "EN": 20_000.0, "VU": 50_000.0},
    "B2": {"CR": 200.0, "EN": 2_000.0, "VU": 5_000.0},
}


def _norm(value: object) -> str:
    return str(value).strip().upper()


def _rl_targets(
    problem: ConservationProblem,
    *,
    km2_per_unit: PerFeatureFloat,
    status: PerFeatureStr,
    criterion_a: str,
    criterion_b: str,
    prop_uplift: float,
    method: Literal["max", "min"],
    cap_km2: float | None,
    features: Iterable[int] | None,
    a_table: _Table,
    b_table: _Table,
) -> dict[int, float]:
    crit_a = _norm(criterion_a)
    crit_b = _norm(criterion_b)
    if crit_a not in a_table:
        raise ValueError(f"criterion_a must be one of {sorted(a_table)}, got {criterion_a!r}")
    if crit_b not in b_table:
        raise ValueError(f"criterion_b must be one of {sorted(b_table)}, got {criterion_b!r}")
    p = check_scalar("prop_uplift", prop_uplift, lower=0.0)
    if method not in ("max", "min"):
        raise ValueError(f"method must be 'max' or 'min', got {method!r}")
    combine = max if method == "max" else min
    cap_v = check_cap("cap_km2", cap_km2)
    fids = select_features(problem, features)
    statuses = {f: _norm(s) for f, s in per_feature("status", status, fids).items()}
    bad = [f for f, s in statuses.items() if s not in _STATUSES]
    if bad:
        raise ValueError(
            f"status must be one of {list(_STATUSES)}; invalid for feature(s) {bad}"
        )
    totals = feature_totals(problem, fids)
    factors = km2_factors(km2_per_unit, fids)
    out: dict[int, float] = {}
    for fid in fids:
        k = factors[fid]
        f = totals[fid] * k
        reduction = a_table[crit_a][statuses[fid]]
        range_km2 = b_table[crit_b][statuses[fid]]
        t = combine(range_km2 * (1.0 + p), f * (1.0 - reduction) * (1.0 + p))
        out[fid] = finish_area(t, f, k, cap_v)
    return out


def rl_species_targets(
    problem: ConservationProblem,
    *,
    km2_per_unit: PerFeatureFloat,
    status: PerFeatureStr,
    criterion_a: str,
    criterion_b: str,
    prop_uplift: float = 0.0,
    method: Literal["max", "min"] = "max",
    cap_km2: float | None = 1_000_000.0,
    features: Iterable[int] | None = None,
) -> dict[int, float]:
    """IUCN Red List of Threatened Species targets.

    ``t = min(m(B·(1+p), f·(1−A)·(1+p)), cap_km2, f)`` where ``A`` is the
    ``criterion_a`` reduction threshold (A1-A4) and ``B`` the ``criterion_b``
    range threshold (B1 EOO / B2 AOO, km²) for each feature's ``status``.

    Args:
        status: ``"CR"``, ``"EN"`` or ``"VU"`` for every feature, or a
            ``{feature_id: status}`` mapping (case-insensitive). Other
            categories raise ``ValueError``; select threatened features with
            ``features=`` and target the rest with another method.
        criterion_a: ``"A1"``-``"A4"`` (case-insensitive).
        criterion_b: ``"B1"`` or ``"B2"`` (case-insensitive).
        prop_uplift: Proportional uplift ``p >= 0``.
        method: ``"max"`` (default) or ``"min"`` combination of the two terms.

    Thresholds from IUCN (2012) Red List Categories and Criteria, Version 3.1,
    2nd edition (Gland & Cambridge: IUCN); cap from Butchart et al. 2015.
    Mirrors prioritizr 9.0.1 ``spec_rl_species_targets`` but accepts a
    per-feature status.
    """
    return _rl_targets(
        problem,
        km2_per_unit=km2_per_unit,
        status=status,
        criterion_a=criterion_a,
        criterion_b=criterion_b,
        prop_uplift=prop_uplift,
        method=method,
        cap_km2=cap_km2,
        features=features,
        a_table=_SPECIES_A,
        b_table=_SPECIES_B,
    )


def rl_ecosystem_targets(
    problem: ConservationProblem,
    *,
    km2_per_unit: PerFeatureFloat,
    status: PerFeatureStr,
    criterion_a: str,
    criterion_b: str,
    prop_uplift: float = 0.0,
    method: Literal["max", "min"] = "max",
    cap_km2: float | None = 1_000_000.0,
    features: Iterable[int] | None = None,
) -> dict[int, float]:
    """IUCN Red List of Ecosystems targets.

    Same formula as :func:`rl_species_targets` with the ecosystem thresholds:
    ``criterion_a`` in A1, A2a, A2b, A3 and ``criterion_b`` in B1, B2
    (case-insensitive). A3 VU uses 0.5 per the IUCN (2024) Red List of
    Ecosystems guidelines v2.0 (thresholds 90/70/50%; criteria defined in
    Keith et al. 2013, *PLoS ONE* 8:e62111,
    https://doi.org/10.1371/journal.pone.0062111); prioritizr 9.0.1 uses 0.3.
    """
    return _rl_targets(
        problem,
        km2_per_unit=km2_per_unit,
        status=status,
        criterion_a=criterion_a,
        criterion_b=criterion_b,
        prop_uplift=prop_uplift,
        method=method,
        cap_km2=cap_km2,
        features=features,
        a_table=_ECOSYSTEM_A,
        b_table=_ECOSYSTEM_B,
    )


def jung_targets(
    problem: ConservationProblem,
    *,
    km2_per_unit: PerFeatureFloat,
    status: PerFeatureStr = "VU",
    prop_uplift: float = 0.1,
    cap_km2: float | None = 1_000_000.0,
    features: Iterable[int] | None = None,
) -> dict[int, float]:
    """Jung et al. (2021) targets: Red List species rule at VU, A2/B2, +10%.

    With the defaults: ``min(max(2,200 km², 0.77·range), 1,000,000 km², range)``
    (prioritizr's docs say 80%; its code and tests give 77%). Jung et al.
    2021, *Nat. Ecol. Evol.* 5:1499-1509,
    https://doi.org/10.1038/s41559-021-01528-7 (correction:
    https://doi.org/10.1038/s41559-021-01569-y). The 2,200 km² floor and the
    100% clip for small ranges are stated in its Methods ("Target setting");
    the 77% proportional term follows prioritizr 9.0.1 ``spec_jung_targets``
    (the equation is not recoverable from accessible text).
    """
    return rl_species_targets(
        problem,
        km2_per_unit=km2_per_unit,
        status=status,
        criterion_a="A2",
        criterion_b="B2",
        prop_uplift=prop_uplift,
        method="max",
        cap_km2=cap_km2,
        features=features,
    )


def ward_targets(
    problem: ConservationProblem,
    *,
    km2_per_unit: PerFeatureFloat,
    status: PerFeatureStr = "CR",
    prop_uplift: float = 0.0,
    cap_km2: float | None = 1_000_000.0,
    features: Iterable[int] | None = None,
) -> dict[int, float]:
    """Ward et al. (2025) targets: Red List species rule at CR, A2/B1.

    With the defaults: ``min(max(100 km², 0.2·range), 1,000,000 km², range)``
    (prioritizr's docs say 100,000 km²; its code uses the CR-B1 threshold
    100 km²). Ward et al. 2025, *PNAS* 122:e2414985122,
    https://doi.org/10.1073/pnas.2414985122. Parameters follow prioritizr
    9.0.1 ``spec_ward_targets``; not confirmable from accessible paper text.
    """
    return rl_species_targets(
        problem,
        km2_per_unit=km2_per_unit,
        status=status,
        criterion_a="A2",
        criterion_b="B1",
        prop_uplift=prop_uplift,
        method="max",
        cap_km2=cap_km2,
        features=features,
    )
```

- [ ] **Step 4: Re-export.** In `__init__.py` add:

```python
from pymarxan.targets._redlist import (
    jung_targets,
    rl_ecosystem_targets,
    rl_species_targets,
    ward_targets,
)
```

and `"jung_targets"`, `"rl_ecosystem_targets"`, `"rl_species_targets"`, `"ward_targets"` to `__all__`. Keep the import blocks in module-name order, as ruff's isort requires: `_basic`, `_compose`, `_interp`, `_population`, `_redlist`, then `_units` last. This block goes directly above the `_units` line.

- [ ] **Step 5: Run the tests.**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/targets/test_targets_redlist.py -q`
Expected: all pass.

- [ ] **Step 6: Lint, type-check, commit.**

```bash
~/.local/bin/ruff check src/pymarxan/targets tests/pymarxan/targets
.venv/bin/mypy src/pymarxan/targets --ignore-missing-imports
git add src/pymarxan/targets tests/pymarxan/targets/test_targets_redlist.py
git commit -m "feat(targets): IUCN Red List species/ecosystem targets + Jung, Ward"
```

---

### Task 5: Population family `_population.py`

**Files:**
- Create: `src/pymarxan/targets/_population.py`
- Modify: `src/pymarxan/targets/__init__.py`
- Test: `tests/pymarxan/targets/test_targets_population.py`

**Interfaces:**
- Consumes: Task 2 helpers.
- Produces (public):
  - `pop_size_targets(problem, *, km2_per_unit, pop_size, density_per_km2, cap_km2=1_000_000.0, features=None)`
  - `wilson_targets(problem, *, km2_per_unit, mean_growth_rate, var_growth_rate, density_per_km2, cap_km2=1_000_000.0, features=None)`. Returns prioritizr's large-population approximation. Raises `ValueError` naming the features when `β <= 0`, or when that target differs by more than 1% (after cap and clip) from the target given by Lande's exact model (spec §6 as revised by the review rounds).
  - Private `_lande_exact_x(base) -> float`, which solves `expm1(x) − x = base` by Newton's method.
  - `duran_targets(problem, *, km2_per_unit, probability_target, historical_area_km2, cap_km2=None, features=None)`

- [ ] **Step 1: Write the failing tests** `tests/pymarxan/targets/test_targets_population.py`:

```python
"""Tests for the population-based target methods (spec §6).

Pinned values are hand-computed from the formulas in each test comment;
km2_per_unit=1 unless stated.
"""
from __future__ import annotations

import math

import pytest

from pymarxan.targets import duran_targets, pop_size_targets, wilson_targets

REL = 1e-6


def test_pop_size_divides_by_density(make_problem):
    # n / d = 500 / 2 = 250 km² (prioritizr's docs say n*d; code divides)
    p = make_problem({1: 10_000.0})
    out = pop_size_targets(p, km2_per_unit=1.0, pop_size=500.0, density_per_km2=2.0)
    assert out[1] == pytest.approx(250.0, rel=REL)


def test_pop_size_clips_to_range(make_problem):
    # 10,000 / 0.5 = 20,000 km²: fits a 50,000 range, clipped on a 5,000 range
    p = make_problem({1: 50_000.0, 2: 5_000.0})
    out = pop_size_targets(p, km2_per_unit=1.0, pop_size=10_000.0, density_per_km2=0.5)
    assert out == {1: pytest.approx(20_000.0), 2: pytest.approx(5_000.0)}


def test_pop_size_per_feature_out_of_order_subset(make_problem):
    # id-keyed lookups: a positional bug would swap the two features' inputs
    p = make_problem({1: 1e5, 2: 1e5, 3: 1e5})
    out = pop_size_targets(
        p, km2_per_unit=1.0, pop_size={3: 300.0, 2: 200.0},
        density_per_km2={2: 1.0, 3: 3.0}, features=[3, 2],
    )
    assert out == {2: pytest.approx(200.0), 3: pytest.approx(100.0)}


def test_pop_size_native_units(make_problem):
    # native unit = ha: 1e6 ha = 10,000 km²; 250 km² -> 25,000 ha
    p = make_problem({1: 1e6})
    out = pop_size_targets(p, km2_per_unit=0.01, pop_size=500.0, density_per_km2=2.0)
    assert out[1] == pytest.approx(25_000.0, rel=REL)


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"pop_size": 0.0}, "pop_size"),
        ({"density_per_km2": -1.0}, "density_per_km2"),
        ({"pop_size": {2: 1.0}}, r"pop_size has no value for feature\(s\) \[1\]"),
    ],
)
def test_pop_size_validation(make_problem, kw, match):
    args = {"pop_size": 500.0, "density_per_km2": 2.0, **kw}
    with pytest.raises(ValueError, match=match):
        pop_size_targets(make_problem({1: 10.0}), km2_per_unit=1.0, **args)


@pytest.mark.parametrize(
    "r, s2, k_pop",
    [
        (0.1, 0.04, 13.374806099),  # beta = 4, k = 32000 ** 0.25
        (0.3, 1e-4, 1.003173576),  # beta = 5999 (prioritizr's example)
    ],
)
def test_wilson_population(make_problem, r, s2, k_pop):
    # density 1/km² so the km² target equals k
    p = make_problem({1: 1e6})
    out = wilson_targets(p, km2_per_unit=1.0, mean_growth_rate=r, var_growth_rate=s2,
                         density_per_km2=1.0)
    assert out[1] == pytest.approx(k_pop, rel=REL)


def test_wilson_overflow_maps_to_cap_or_range(make_problem):
    # r = 500.5, s2 = 1000 -> beta = 1e-3; 50 ** 1000 overflows -> k = inf
    p = make_problem({1: 2e7})
    capped = wilson_targets(p, km2_per_unit=1.0, mean_growth_rate=500.5,
                            var_growth_rate=1_000.0, density_per_km2=1.0)
    uncapped = wilson_targets(p, km2_per_unit=1.0, mean_growth_rate=500.5,
                              var_growth_rate=1_000.0, density_per_km2=1.0, cap_km2=None)
    assert capped[1] == 1e6
    assert uncapped[1] == 2e7


def test_wilson_divides_by_density(make_problem):
    # k / d = 13.374806099 / 0.5 (prioritizr's docs write d*k; its code divides)
    p = make_problem({1: 1e6, 2: 1e6})
    out = wilson_targets(p, km2_per_unit=1.0, mean_growth_rate=0.1, var_growth_rate=0.04,
                         density_per_km2={2: 2.0, 1: 0.5})
    assert out == {1: pytest.approx(26.749612199, rel=REL),
                   2: pytest.approx(6.687403050, rel=REL)}


@pytest.mark.parametrize(
    "r, d",
    [
        # beta = 5e-6, 1e5*s2*beta**2/2 = 5e-8: the approximation gives k ~ 0 (no
        # protection) while Lande's exact model needs ~e^63 individuals
        (0.0200001, 1.0),
        # 1e5*s2*beta**2/2 = 1.0125: approximation k ~ 1.737 (173.7 km² target at
        # d = 0.01) vs exact ~1.7e22 individuals (target = cap)
        (0.02045, 0.01),
    ],
)
def test_wilson_inaccurate_approximation_raises(make_problem, r, d):
    p = make_problem({1: 1e6, 2: 1e6})
    with pytest.raises(ValueError, match=r"differs from Lande's exact.*feature\(s\) \[1\]"):
        wilson_targets(p, km2_per_unit=1.0, mean_growth_rate={1: r, 2: 0.1},
                       var_growth_rate=0.04, density_per_km2=d)


def test_wilson_lande_exact_solver():
    # expm1(x) - x = base; large base -> x ~ ln(base); small base -> x ~ sqrt(2 base)
    from pymarxan.targets._population import _lande_exact_x

    for base in (0.5, 1.0, 32_000.0, 1e300):
        x = _lande_exact_x(base)
        assert math.expm1(x) - x == pytest.approx(base, rel=1e-9)
    # tiny base: compare with the series inverse x = s - s²/6 + s³/36, s = sqrt(2 base)
    # (expm1(x) - x cancels catastrophically here, so it is not a usable check)
    for base in (1e-8, 1e-30, 1e-300):
        s = math.sqrt(2.0 * base)
        assert _lande_exact_x(base) == pytest.approx(s - s * s / 6 + s**3 / 36, rel=1e-9)


@pytest.mark.parametrize(
    "r, expected",
    [
        # beta = 0.7, base = 980: approx k = 18,757.7 vs exact 18,974.0 (1.14% apart)
        (0.034, None),
        # beta = 0.75, base = 1125: approx k = 11,700.47 vs exact 11,811.99 (0.94%)
        (0.035, 11_700.4715047),
    ],
)
def test_wilson_tolerance_is_one_percent(make_problem, r, expected):
    # approx k = (1e5*s2*beta**2/2)**(1/beta); exact x solves expm1(x) - x = base,
    # k = e**(x/beta). Returned values are prioritizr's approximation.
    p = make_problem({1: 2e7})
    kw = dict(km2_per_unit=1.0, mean_growth_rate=r, var_growth_rate=0.04, density_per_km2=1.0)
    if expected is None:
        with pytest.raises(ValueError, match="differs from Lande's exact"):
            wilson_targets(p, **kw)
    else:
        assert wilson_targets(p, **kw)[1] == pytest.approx(expected, rel=1e-6)


def test_wilson_compares_after_cap(make_problem):
    # beta = 0.05, base = 12.5: raw k differ ~200x (8.7e21 vs 1.7e24 individuals) but
    # both targets exceed the 1e6 km² cap, so they agree and no error is raised
    p = make_problem({1: 2e7})
    out = wilson_targets(p, km2_per_unit=1.0, mean_growth_rate=0.0525, var_growth_rate=0.1,
                         density_per_km2=1.0)
    assert out[1] == 1e6


def test_wilson_small_variance_route_raises(make_problem):
    # r = s2 = 0.01: beta = 1, base = 50 -> approx k = 500 vs exact 507.2 (1.43% apart);
    # small variance, not r near s2/2, makes the approximation inaccurate here
    p = make_problem({1: 2e7})
    with pytest.raises(ValueError, match=r"r near σ²/2, or σ² small"):
        wilson_targets(p, km2_per_unit=1.0, mean_growth_rate=0.01, var_growth_rate=0.01,
                       density_per_km2=1.0)


def test_wilson_base_underflow_names_feature(make_problem):
    # s2 = 1e-300 and beta ~ 4e-16 make 1e5*s2*beta**2/2 underflow to 0.0; the error
    # must still name the feature rather than be a bare "math domain error"
    p = make_problem({1: 1e4})
    with pytest.raises(ValueError, match=r"Lande's exact.*\[1\]"):
        wilson_targets(p, km2_per_unit=1.0, mean_growth_rate=1e-300 / 2 * (1 + 4e-16),
                       var_growth_rate=1e-300, density_per_km2=1.0)


def test_wilson_nonpositive_beta_raises(make_problem):
    p = make_problem({1: 1e4, 2: 1e4})
    with pytest.raises(ValueError, match=r"beta > 0.*feature\(s\) \[2\]"):
        wilson_targets(p, km2_per_unit=1.0, mean_growth_rate={1: 0.1, 2: 0.02},
                       var_growth_rate=0.04, density_per_km2=1.0)


@pytest.mark.parametrize(
    "p_target, expected", [(0.9, 656.1), (0.95, 814.50625), (1.0, 1_000.0), (0.0, 0.0)]
)
def test_duran(make_problem, p_target, expected):
    # h * p**4 with h = 1,000 km²
    pr = make_problem({1: 5_000.0})
    out = duran_targets(pr, km2_per_unit=1.0, probability_target=p_target,
                        historical_area_km2=1_000.0)
    assert out[1] == pytest.approx(expected, rel=REL)


def test_duran_clips_to_current_range_and_has_no_default_cap(make_problem):
    # historical 3e6 km², p = 1 -> 3e6, current range 2e6 -> 2e6 (no 1e6 cap by default)
    pr = make_problem({1: 2e6})
    out = duran_targets(pr, km2_per_unit=1.0, probability_target=1.0,
                        historical_area_km2=3e6)
    assert out[1] == 2e6
    capped = duran_targets(pr, km2_per_unit=1.0, probability_target=1.0,
                           historical_area_km2=3e6, cap_km2=1e6)
    assert capped[1] == 1e6


def test_duran_validation(make_problem):
    pr = make_problem({1: 10.0})
    with pytest.raises(ValueError, match="probability_target"):
        duran_targets(pr, km2_per_unit=1.0, probability_target=1.2, historical_area_km2=5.0)
    with pytest.raises(ValueError, match="historical_area_km2"):
        duran_targets(pr, km2_per_unit=1.0, probability_target=0.5, historical_area_km2=0.0)
```

- [ ] **Step 2: Run to verify they fail.**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/targets/test_targets_population.py -q`
Expected: collection error `ImportError: cannot import name 'duran_targets'`.

- [ ] **Step 3: Implement** `src/pymarxan/targets/_population.py`:

```python
"""Population-based target methods (population size, Wilson, Durán).

See the Stage 2 design spec §6. Per-feature inputs are scalars or
``{feature_id: value}`` mappings; densities are individuals per km².
"""
from __future__ import annotations

import math
import sys
from collections.abc import Iterable

from pymarxan.models.problem import ConservationProblem
from pymarxan.targets._units import (
    PerFeatureFloat,
    check_cap,
    feature_totals,
    finish_area,
    km2_factors,
    numeric_per_feature,
    select_features,
)

_LN_FLOAT_MAX = math.log(sys.float_info.max)
_WILSON_RTOL = 0.01


def _exp_or_inf(ln_x: float) -> float:
    return math.inf if ln_x > _LN_FLOAT_MAX else math.exp(ln_x)


def _lande_exact_x(base: float) -> float:
    """Solve ``expm1(x) - x = base`` for ``x > 0`` (Newton; convex, increasing).

    Lande's (1993) exact mean time to extinction, ``T·σ²·β²/2 = e^x − 1 − x``
    with ``x = β·ln K``; the large-population approximation drops ``− 1 − x``.
    """
    if not math.isfinite(base):
        return math.inf
    if base <= 0.0:  # reachable only via float underflow of 1e5·σ²·β²/2
        return 0.0
    if base < 1e-200:  # x*x would underflow; expm1(x) - x = x²/2 to double precision
        return math.sqrt(2.0 * base)
    x = max(math.log(base), 1.0) if base > 1.0 else math.sqrt(2.0 * base)
    for _ in range(200):
        if x < 1e-4:  # series for expm1(x) - x avoids cancellation
            g = x * x / 2.0 * (1.0 + x / 3.0 + x * x / 12.0)
            gp = x * (1.0 + x / 2.0 + x * x / 6.0)
        else:
            g = math.expm1(x) - x
            gp = math.expm1(x)
        step = (g - base) / gp
        x -= step
        if abs(step) <= 1e-15 * x:
            break
    return x


def pop_size_targets(
    problem: ConservationProblem,
    *,
    km2_per_unit: PerFeatureFloat,
    pop_size: PerFeatureFloat,
    density_per_km2: PerFeatureFloat,
    cap_km2: float | None = 1_000_000.0,
    features: Iterable[int] | None = None,
) -> dict[int, float]:
    """Area needed to hold a target population: ``t = min(n / d, cap, range)``.

    Args:
        pop_size: Target number of individuals ``n > 0`` (scalar or per feature).
        density_per_km2: Individuals per km² ``d > 0`` (scalar or per feature).
            Convert other densities first (per ha × 100 = per km²).

    prioritizr's docs write ``n·d``; its code and tests divide, as here.
    Mirrors prioritizr 9.0.1 ``spec_pop_size_targets``.
    """
    cap_v = check_cap("cap_km2", cap_km2)
    fids = select_features(problem, features)
    n = numeric_per_feature("pop_size", pop_size, fids, lower=0.0, lower_inclusive=False)
    d = numeric_per_feature(
        "density_per_km2", density_per_km2, fids, lower=0.0, lower_inclusive=False
    )
    totals = feature_totals(problem, fids)
    factors = km2_factors(km2_per_unit, fids)
    return {
        fid: finish_area(n[fid] / d[fid], totals[fid] * factors[fid], factors[fid], cap_v)
        for fid in fids
    }


def wilson_targets(
    problem: ConservationProblem,
    *,
    km2_per_unit: PerFeatureFloat,
    mean_growth_rate: PerFeatureFloat,
    var_growth_rate: PerFeatureFloat,
    density_per_km2: PerFeatureFloat,
    cap_km2: float | None = 1_000_000.0,
    features: Iterable[int] | None = None,
) -> dict[int, float]:
    """Wilson et al. (2010) persistence-based targets.

    ``β = 2r/σ² − 1``; the population needed to persist 100,000 years is
    ``k = (1e5·σ²·β²/2)^(1/β)``; ``t = min(k / d, cap, range)``. This is the
    large-population approximation of Lande's (1993) mean time to extinction
    under environmental stochasticity, solved for ``k``
    (https://doi.org/10.1086/285580).

    That approximation is accurate only when ``1e5·σ²·β²/2 ≫ 1``; near and
    below 1 it can give a near-zero target where the exact model needs an
    enormous population. So the target is also computed from Lande's exact
    model, and ``ValueError`` (naming the features) is raised wherever the two
    targets differ by more than 1% after the cap and clip; ``β <= 0``
    (``r <= σ²/2``) also raises. Where no error is raised the result is
    prioritizr's. For refused features, set a minimum viable population and
    use :func:`pop_size_targets`. ``k`` is computed in log space; an
    overflowing ``k`` is treated as infinite (target ``min(cap, range)``).

    Wilson et al. 2010, *Ecol. Appl.* 20:1721-1732,
    https://doi.org/10.1890/09-1051.1. The formula follows prioritizr 9.0.1
    ``spec_wilson_targets``; it could not be checked against the (closed
    access) paper text.
    """
    cap_v = check_cap("cap_km2", cap_km2)
    fids = select_features(problem, features)
    r = numeric_per_feature(
        "mean_growth_rate", mean_growth_rate, fids, lower=0.0, lower_inclusive=False
    )
    s2 = numeric_per_feature(
        "var_growth_rate", var_growth_rate, fids, lower=0.0, lower_inclusive=False
    )
    d = numeric_per_feature(
        "density_per_km2", density_per_km2, fids, lower=0.0, lower_inclusive=False
    )
    beta = {fid: 2.0 * r[fid] / s2[fid] - 1.0 for fid in fids}
    bad = [fid for fid in fids if beta[fid] <= 0.0]
    if bad:
        raise ValueError(
            "wilson_targets needs mean_growth_rate > var_growth_rate / 2 "
            f"(beta > 0); invalid for feature(s) {bad}"
        )
    base = {fid: 1e5 * s2[fid] * beta[fid] ** 2 / 2.0 for fid in fids}
    totals = feature_totals(problem, fids)
    factors = km2_factors(km2_per_unit, fids)
    out: dict[int, float] = {}
    inaccurate: list[int] = []
    for fid in fids:
        k_f = factors[fid]
        f = totals[fid] * k_f
        ln_base = math.log(base[fid]) if base[fid] > 0.0 else -math.inf
        k_approx = _exp_or_inf(ln_base / beta[fid])
        k_exact = _exp_or_inf(_lande_exact_x(base[fid]) / beta[fid])
        t_approx = finish_area(k_approx / d[fid], f, k_f, cap_v)
        t_exact = finish_area(k_exact / d[fid], f, k_f, cap_v)
        if abs(t_approx - t_exact) > _WILSON_RTOL * t_exact:
            inaccurate.append(fid)
        out[fid] = t_approx
    if inaccurate:
        raise ValueError(
            "wilson_targets' large-population approximation differs from Lande's exact "
            f"model by more than {_WILSON_RTOL:.0%} for feature(s) {inaccurate} "
            "(1e5·σ²·β²/2 is not much larger than 1: r near σ²/2, or σ² small); set a "
            "minimum viable population and use pop_size_targets instead"
        )
    return out


def duran_targets(
    problem: ConservationProblem,
    *,
    km2_per_unit: PerFeatureFloat,
    probability_target: PerFeatureFloat,
    historical_area_km2: PerFeatureFloat,
    cap_km2: float | None = None,
    features: Iterable[int] | None = None,
) -> dict[int, float]:
    """Durán et al. (2020) persistence-probability targets.

    Persistence ``P = (A / A_hist)^0.25`` solved for area: ``t = min(h·p⁴, range)``
    (and ``cap_km2`` if given; no cap by default, as in prioritizr).

    Args:
        probability_target: ``p`` in [0, 1] (scalar or per feature). Published
            per-category values: Davis et al. 2018 (LC 0.9983, NT 0.9859,
            VU 0.9, EN 0.3277, CR 0.001); Gumbs et al. 2023 (LC 0.939375,
            NT 0.87875, VU 0.7575, EN 0.515, CR 0.03).
        historical_area_km2: Historical range ``h > 0`` in km².

    Durán et al. 2020, *Methods Ecol. Evol.* 11:910-921,
    https://doi.org/10.1111/2041-210X.13427 (z = 0.25).
    """
    cap_v = check_cap("cap_km2", cap_km2)
    fids = select_features(problem, features)
    p = numeric_per_feature("probability_target", probability_target, fids, lower=0.0, upper=1.0)
    h = numeric_per_feature(
        "historical_area_km2", historical_area_km2, fids, lower=0.0, lower_inclusive=False
    )
    totals = feature_totals(problem, fids)
    factors = km2_factors(km2_per_unit, fids)
    return {
        fid: finish_area(h[fid] * p[fid] ** 4, totals[fid] * factors[fid], factors[fid], cap_v)
        for fid in fids
    }
```

- [ ] **Step 4: Re-export.** In `__init__.py` add this block directly above the `_redlist` block (and so above the `_units` line):

```python
from pymarxan.targets._population import (
    duran_targets,
    pop_size_targets,
    wilson_targets,
)
```

and `"duran_targets"`, `"pop_size_targets"`, `"wilson_targets"` to `__all__`.

- [ ] **Step 5: Run the tests.**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/targets/test_targets_population.py -q`
Expected: all pass.

- [ ] **Step 6: Lint, type-check, commit.**

```bash
~/.local/bin/ruff check src/pymarxan/targets tests/pymarxan/targets
.venv/bin/mypy src/pymarxan/targets --ignore-missing-imports
git add src/pymarxan/targets tests/pymarxan/targets/test_targets_population.py
git commit -m "feat(targets): population-size, Wilson and Duran targets"
```

---

### Task 6: Simple methods and combinators `_compose.py`

**Files:**
- Create: `src/pymarxan/targets/_compose.py`
- Modify: `src/pymarxan/targets/__init__.py`
- Test: `tests/pymarxan/targets/test_targets_compose.py`

**Interfaces:**
- Consumes: Task 2 helpers; `relative_targets` (Task 1) in one test.
- Produces (public):
  - `absolute_targets(problem, *, amount, features=None)`: not clipped.
  - `area_targets(problem, *, km2_per_unit, area_km2, cap_km2=1_000_000.0, features=None)`
  - `rule_targets(problem, *, km2_per_unit, baseline, rules: Mapping[str, float], traits: Mapping[int, Collection[str]], cap_km2=1_000_000.0, features=None)`
  - `min_targets(*results)` and `max_targets(*results)`: need at least 2 inputs with identical key sets; output follows the first input's order.

- [ ] **Step 1: Write the failing tests** `tests/pymarxan/targets/test_targets_compose.py`:

```python
"""Tests for absolute/area/rule targets and the min/max combinators (spec §7)."""
from __future__ import annotations

import pandas as pd
import pytest

from pymarxan.targets import (
    absolute_targets,
    area_targets,
    max_targets,
    min_targets,
    relative_targets,
    rule_targets,
)


def test_absolute_scalar_not_clipped_even_at_zero_total(make_problem):
    p = make_problem({1: 0.0, 2: 3.0})
    assert absolute_targets(p, amount=5.0) == {1: 5.0, 2: 5.0}


def test_absolute_mapping_and_subset(make_problem):
    p = make_problem({1: 10.0, 2: 10.0, 3: 10.0})
    assert absolute_targets(p, amount={1: 2.5, 2: 7, 3: 1.0}, features=[2, 1]) == {
        1: 2.5,
        2: 7.0,
    }


def test_absolute_rejects_negative_and_positional_use(make_problem):
    p = make_problem({1: 10.0})
    with pytest.raises(ValueError, match="amount"):
        absolute_targets(p, amount=-1.0)
    with pytest.raises(TypeError):
        absolute_targets(p, 5.0)  # type: ignore[misc]


def test_area_targets_clip_and_units(make_problem):
    p = make_problem({1: 600.0, 2: 5_000.0})
    assert area_targets(p, km2_per_unit=1.0, area_km2=1_000.0) == {
        1: pytest.approx(600.0),
        2: pytest.approx(1_000.0),
    }
    # native unit = ha: 500,000 ha = 5,000 km²; 1,000 km² -> 100,000 ha
    ha = make_problem({1: 500_000.0})
    assert area_targets(ha, km2_per_unit=0.01, area_km2=1_000.0)[1] == pytest.approx(100_000.0)


def test_area_targets_cap(make_problem):
    p = make_problem({1: 5e6})
    assert area_targets(p, km2_per_unit=1.0, area_km2=2e6)[1] == 1e6
    assert area_targets(p, km2_per_unit=1.0, area_km2=2e6, cap_km2=None)[1] == 2e6


_RULES = {"endemic": 0.2, "harvested": -0.1}


@pytest.mark.parametrize(
    "traits, baseline, expected",
    [
        ({"endemic"}, 0.3, 5_000.0),  # 0.3 + 0.2
        ({"harvested"}, 0.3, 2_000.0),  # 0.3 - 0.1
        ({"endemic", "harvested"}, 0.3, 4_000.0),
        (set(), 0.3, 3_000.0),
        ({"endemic"}, 0.9, 10_000.0),  # 1.1 clamped to 1.0
        ({"harvested"}, 0.05, 0.0),  # -0.05 clamped to 0.0
    ],
)
def test_rule_targets(make_problem, traits, baseline, expected):
    p = make_problem({1: 10_000.0})
    out = rule_targets(p, km2_per_unit=1.0, baseline=baseline, rules=_RULES,
                       traits={1: traits})
    assert out[1] == pytest.approx(expected)


def test_rule_targets_joined_by_id_not_position(make_problem):
    p = make_problem({1: 10_000.0, 2: 10_000.0})
    out = rule_targets(p, km2_per_unit=1.0, baseline=0.3, rules=_RULES,
                       traits={2: {"endemic"}, 1: set()})
    assert out == {1: pytest.approx(3_000.0), 2: pytest.approx(5_000.0)}


def test_rule_targets_traits_series(make_problem):
    p = make_problem({1: 10_000.0, 2: 10_000.0})
    traits = pd.Series({2: {"endemic"}, 1: {"harvested"}})
    out = rule_targets(p, km2_per_unit=1.0, baseline=0.3, rules=_RULES, traits=traits)
    assert out == {1: pytest.approx(2_000.0), 2: pytest.approx(5_000.0)}


def test_rule_targets_rejects_non_mapping_traits(make_problem):
    # a plain list would otherwise apply to every feature positionally
    with pytest.raises(ValueError, match="traits must map feature id"):
        rule_targets(make_problem({1: 10.0, 2: 10.0}), km2_per_unit=1.0, baseline=0.3,
                     rules={"endemic": 0.2}, traits=["endemic"])  # type: ignore[arg-type]


def test_rule_targets_rejects_bare_string_trait(make_problem):
    # "ex" would otherwise be read as traits {"e", "x"}
    p = make_problem({1: 10_000.0})
    with pytest.raises(ValueError, match=r"feature\(s\) \[1\] must be a collection"):
        rule_targets(p, km2_per_unit=1.0, baseline=0.3, rules={"e": 0.1, "x": 0.1},
                     traits={1: "ex"})


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"traits": {1: {"migratory"}}}, r"traits not in rules: \['migratory'\]"),
        ({"traits": {}}, r"traits has no value for feature\(s\) \[1\]"),
        ({"baseline": 1.2}, "baseline out of range"),
        ({"rules": {"endemic": 1.5}}, r"rules\['endemic'\] out of range"),
    ],
)
def test_rule_targets_validation(make_problem, kw, match):
    args = {"baseline": 0.3, "rules": _RULES, "traits": {1: {"endemic"}}, **kw}
    with pytest.raises(ValueError, match=match):
        rule_targets(make_problem({1: 10.0}), km2_per_unit=1.0, **args)


def test_max_and_min_targets():
    a = {1: 100.0, 2: 50.0}
    b = {1: 80.0, 2: 90.0}
    assert max_targets(a, b) == {1: 100.0, 2: 90.0}
    assert min_targets(a, b) == {1: 80.0, 2: 50.0}
    assert list(max_targets({2: 1.0, 1: 1.0}, {1: 0.0, 2: 0.0})) == [2, 1]


def test_max_targets_three_inputs():
    assert max_targets({1: 1.0}, {1: 3.0}, {1: 2.0}) == {1: 3.0}


def test_combinators_reject_mismatched_keys_and_too_few_inputs():
    with pytest.raises(ValueError, match=r"only in input 0: \[2\]"):
        max_targets({1: 1.0, 2: 1.0}, {1: 2.0})
    with pytest.raises(ValueError, match="at least 2"):
        min_targets({1: 1.0})


def test_max_of_area_and_relative(make_problem):
    # the docstring example: at least 1,000 km² or 30%, whichever is larger
    p = make_problem({1: 2_000.0, 2: 10_000.0})
    out = max_targets(
        area_targets(p, km2_per_unit=1.0, area_km2=1_000.0), relative_targets(p, 0.3)
    )
    assert out == {1: pytest.approx(1_000.0), 2: pytest.approx(3_000.0)}
```

- [ ] **Step 2: Run to verify they fail.**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/targets/test_targets_compose.py -q`
Expected: collection error `ImportError: cannot import name 'absolute_targets'`.

- [ ] **Step 3: Implement** `src/pymarxan/targets/_compose.py`:

```python
"""Simple target methods and combinators.

See the Stage 2 design spec §7.
"""
from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping

import pandas as pd

from pymarxan.models.problem import ConservationProblem
from pymarxan.targets._units import (
    PerFeatureFloat,
    check_cap,
    check_scalar,
    feature_totals,
    finish_area,
    km2_factors,
    numeric_per_feature,
    per_feature,
    select_features,
)


def absolute_targets(
    problem: ConservationProblem,
    *,
    amount: PerFeatureFloat,
    features: Iterable[int] | None = None,
) -> dict[int, float]:
    """Explicit targets in native units (scalar or ``{feature_id: amount}``).

    Not clipped to the feature total: an explicit target is the caller's
    statement (``presolve_check`` flags targets above totals). Mirrors
    prioritizr 9.0.1 ``spec_absolute_targets``.
    """
    fids = select_features(problem, features)
    return numeric_per_feature("amount", amount, fids, lower=0.0)


def area_targets(
    problem: ConservationProblem,
    *,
    km2_per_unit: PerFeatureFloat,
    area_km2: PerFeatureFloat,
    cap_km2: float | None = 1_000_000.0,
    features: Iterable[int] | None = None,
) -> dict[int, float]:
    """Fixed-area targets: ``t = min(area_km2, cap_km2, range)``, in native units.

    Mirrors prioritizr 9.0.1 ``spec_area_targets``.
    """
    cap_v = check_cap("cap_km2", cap_km2)
    fids = select_features(problem, features)
    a = numeric_per_feature("area_km2", area_km2, fids, lower=0.0)
    totals = feature_totals(problem, fids)
    factors = km2_factors(km2_per_unit, fids)
    return {
        fid: finish_area(a[fid], totals[fid] * factors[fid], factors[fid], cap_v)
        for fid in fids
    }


def rule_targets(
    problem: ConservationProblem,
    *,
    km2_per_unit: PerFeatureFloat,
    baseline: float,
    rules: Mapping[str, float],
    traits: Mapping[int, Collection[str]] | pd.Series,
    cap_km2: float | None = 1_000_000.0,
    features: Iterable[int] | None = None,
) -> dict[int, float]:
    """Rule-based targets (Harris & Holness 2023).

    ``proportion = clamp(baseline + Σ rules[τ] for τ in traits[feature], 0, 1)``;
    ``t = min(proportion·range, cap_km2, range)``.

    Args:
        baseline: Baseline proportion in [0, 1].
        rules: Per-trait adjustment, each in [-1, 1], e.g.
            ``{"endemic": 0.2, "harvested": -0.1}``.
        traits: ``{feature_id: {"endemic", ...}}`` (or a Series indexed by
            feature id) for every selected feature; an empty collection is
            allowed. A bare string is rejected (it would be read as one trait
            per character): wrap it, ``{"endemic"}``. A trait missing from
            ``rules`` raises ``ValueError``.

    Harris & Holness 2023, *Biol. Conserv.* 285:110218,
    https://doi.org/10.1016/j.biocon.2023.110218. Mirrors prioritizr 9.0.1
    ``spec_rule_targets``, joined by feature id.
    """
    base = check_scalar("baseline", baseline, lower=0.0, upper=1.0)
    for name, adj in rules.items():
        check_scalar(f"rules[{name!r}]", adj, lower=-1.0, upper=1.0)
    cap_v = check_cap("cap_km2", cap_km2)
    fids = select_features(problem, features)
    if not isinstance(traits, (Mapping, pd.Series)):
        raise ValueError("traits must map feature id to a collection of trait names")
    feature_traits = per_feature("traits", traits, fids)
    bare = [fid for fid, ts in feature_traits.items() if isinstance(ts, str)]
    if bare:
        raise ValueError(
            f"traits for feature(s) {bare} must be a collection of trait names, "
            'not a string; wrap it, e.g. {"endemic"}'
        )
    unknown = sorted(
        {str(t) for ts in feature_traits.values() for t in ts}.difference(rules)
    )
    if unknown:
        raise ValueError(f"traits not in rules: {unknown}")
    totals = feature_totals(problem, fids)
    factors = km2_factors(km2_per_unit, fids)
    out: dict[int, float] = {}
    for fid in fids:
        prop = base + sum(float(rules[t]) for t in feature_traits[fid])
        prop = min(max(prop, 0.0), 1.0)
        f = totals[fid] * factors[fid]
        out[fid] = finish_area(prop * f, f, factors[fid], cap_v)
    return out


def _combine(
    name: str,
    results: tuple[Mapping[int, float], ...],
    pick: Callable[[float, float], float],
) -> dict[int, float]:
    if len(results) < 2:
        raise ValueError(f"{name} needs at least 2 target mappings, got {len(results)}")
    first = [int(k) for k in results[0]]
    keys = set(first)
    for i, res in enumerate(results[1:], start=1):
        other = {int(k) for k in res}
        if other != keys:
            raise ValueError(
                f"{name}: input {i} has different features from input 0 "
                f"(only in input 0: {sorted(keys - other)}, "
                f"only in input {i}: {sorted(other - keys)})"
            )
    norm = [{int(k): float(v) for k, v in res.items()} for res in results]
    out: dict[int, float] = {}
    for fid in first:
        value = norm[0][fid]
        for res in norm[1:]:
            value = pick(value, res[fid])
        out[fid] = value
    return out


def min_targets(*results: Mapping[int, float]) -> dict[int, float]:
    """Feature-wise minimum of two or more target mappings (same feature ids)."""
    return _combine("min_targets", results, min)


def max_targets(*results: Mapping[int, float]) -> dict[int, float]:
    """Feature-wise maximum of two or more target mappings (same feature ids).

    Example: ``max_targets(area_targets(p, km2_per_unit=k, area_km2=1000),
    relative_targets(p, 0.3))``.
    """
    return _combine("max_targets", results, max)
```

- [ ] **Step 4: Re-export.** In `__init__.py` add (directly after the `_basic` block, for isort order):

```python
from pymarxan.targets._compose import (
    absolute_targets,
    area_targets,
    max_targets,
    min_targets,
    rule_targets,
)
```

and `"absolute_targets"`, `"area_targets"`, `"max_targets"`, `"min_targets"`, `"rule_targets"` to `__all__`. After this task `__init__.py` must be exactly:

```python
"""Automatic target-setting rules.

Set each feature's representation target by a rule rather than by hand.
Each rule returns a ``{feature_id: target_amount}`` mapping in the feature's
native units; :func:`apply_targets` writes it onto a problem's features.
Features a rule does not return keep their existing target, so methods are
mixed across feature groups with ``features=`` plus repeated
:func:`apply_targets` calls.

Area-based methods take a required ``km2_per_unit`` (a factor or a
``{feature_id: factor}`` mapping): feature totals are converted to km², the
published rule is applied, and the target is converted back. There is no
default unit.

Mirrors prioritizr 9.0.1's target methods (Hanson et al. (2024),
*Conservation Biology*, https://doi.org/10.1111/cobi.14376); deviations
from prioritizr are listed in the CHANGELOG for v0.40.0.
"""
from __future__ import annotations

from pymarxan.targets._basic import (
    apply_targets,
    group_targets,
    loglinear_targets,
    relative_targets,
)
from pymarxan.targets._compose import (
    absolute_targets,
    area_targets,
    max_targets,
    min_targets,
    rule_targets,
)
from pymarxan.targets._interp import (
    interp_absolute_targets,
    interp_area_targets,
    polak_targets,
    rodrigues_targets,
    sreekar_targets,
    watson_targets,
)
from pymarxan.targets._population import (
    duran_targets,
    pop_size_targets,
    wilson_targets,
)
from pymarxan.targets._redlist import (
    jung_targets,
    rl_ecosystem_targets,
    rl_species_targets,
    ward_targets,
)
from pymarxan.targets._units import PerFeatureFloat, PerFeatureStr

__all__ = [
    "PerFeatureFloat",
    "PerFeatureStr",
    "absolute_targets",
    "apply_targets",
    "area_targets",
    "duran_targets",
    "group_targets",
    "interp_absolute_targets",
    "interp_area_targets",
    "jung_targets",
    "loglinear_targets",
    "max_targets",
    "min_targets",
    "polak_targets",
    "pop_size_targets",
    "relative_targets",
    "rl_ecosystem_targets",
    "rl_species_targets",
    "rodrigues_targets",
    "rule_targets",
    "sreekar_targets",
    "ward_targets",
    "watson_targets",
    "wilson_targets",
]
```

- [ ] **Step 5: Run the tests.**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/targets tests/pymarxan/test_targets.py -q`
Expected: all pass.

- [ ] **Step 6: Lint, type-check, commit.**

```bash
~/.local/bin/ruff check src/pymarxan/targets tests/pymarxan/targets
.venv/bin/mypy src/pymarxan/targets --ignore-missing-imports
git add src/pymarxan/targets tests/pymarxan/targets/test_targets_compose.py
git commit -m "feat(targets): absolute/area/rule targets and min/max combinators"
```

---

### Task 7: Cross-method invariants, public API test, end-to-end MIP

**Files:**
- Test: `tests/pymarxan/targets/test_targets_api.py`
- Modify: `tests/pymarxan/test_targets.py` (remove `test_targets_package_all` from Task 1, which this task's full-set test supersedes)

**Interfaces:**
- Consumes: every public function from Tasks 1–6; fixture `tiny_problem` (`tests/conftest.py`, loads `tests/data/simple`, totals 60/45/33); `MIPSolver`, `SolverConfig`; `ConservationProblem.copy_with()`.
- Produces: none (tests only). If a test here fails, the fix belongs in the owning module. Don't weaken the test.

- [ ] **Step 1: Write the tests** `tests/pymarxan/targets/test_targets_api.py`:

```python
"""Cross-method invariants, public API and end-to-end tests (spec §9.3, §9.7)."""
from __future__ import annotations

import pytest

import pymarxan.targets as targets_pkg
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver
from pymarxan.targets import (
    apply_targets,
    area_targets,
    duran_targets,
    interp_area_targets,
    jung_targets,
    polak_targets,
    pop_size_targets,
    rl_ecosystem_targets,
    rl_species_targets,
    rodrigues_targets,
    rule_targets,
    sreekar_targets,
    ward_targets,
    watson_targets,
    wilson_targets,
)

EXPECTED_ALIASES = {"PerFeatureFloat", "PerFeatureStr"}
EXPECTED_PUBLIC = {
    "absolute_targets",
    "apply_targets",
    "area_targets",
    "duran_targets",
    "group_targets",
    "interp_absolute_targets",
    "interp_area_targets",
    "jung_targets",
    "loglinear_targets",
    "max_targets",
    "min_targets",
    "polak_targets",
    "pop_size_targets",
    "relative_targets",
    "rl_ecosystem_targets",
    "rl_species_targets",
    "rodrigues_targets",
    "rule_targets",
    "sreekar_targets",
    "ward_targets",
    "watson_targets",
    "wilson_targets",
}


def test_public_api_all_matches_expected():
    # pdoc renders an empty page for a re-exporting package without __all__
    assert set(targets_pkg.__all__) == EXPECTED_PUBLIC | EXPECTED_ALIASES
    for name in EXPECTED_PUBLIC:
        assert callable(getattr(targets_pkg, name))
    for name in EXPECTED_ALIASES:  # type aliases used in public signatures
        assert getattr(targets_pkg, name) is not None


# Every area-based method, as (name, callable taking problem + km2_per_unit).
AREA_METHODS = {
    "interp_area": lambda p, k: interp_area_targets(
        p, km2_per_unit=k, rare_area_threshold=1_000.0, rare_relative_target=0.5,
        rare_area_target=400.0, rare_method="max", common_area_threshold=100_000.0,
        common_relative_target=0.1, common_area_target=20_000.0, common_method="min",
        interp="log10",
    ),
    "rodrigues": lambda p, k: rodrigues_targets(p, km2_per_unit=k),
    "sreekar": lambda p, k: sreekar_targets(p, km2_per_unit=k),
    "polak": lambda p, k: polak_targets(p, km2_per_unit=k),
    "watson": lambda p, k: watson_targets(p, km2_per_unit=k),
    "rl_species": lambda p, k: rl_species_targets(
        p, km2_per_unit=k, status="EN", criterion_a="A1", criterion_b="B1", prop_uplift=0.1
    ),
    "rl_ecosystem": lambda p, k: rl_ecosystem_targets(
        p, km2_per_unit=k, status="VU", criterion_a="A3", criterion_b="B2"
    ),
    "jung": lambda p, k: jung_targets(p, km2_per_unit=k),
    "ward": lambda p, k: ward_targets(p, km2_per_unit=k),
    "pop_size": lambda p, k: pop_size_targets(
        p, km2_per_unit=k, pop_size=5_000.0, density_per_km2=0.5
    ),
    "wilson": lambda p, k: wilson_targets(
        p, km2_per_unit=k, mean_growth_rate=0.1, var_growth_rate=0.04, density_per_km2=0.01
    ),
    "duran": lambda p, k: duran_targets(
        p, km2_per_unit=k, probability_target=0.9, historical_area_km2=50_000.0, cap_km2=1e6
    ),
    "area": lambda p, k: area_targets(p, km2_per_unit=k, area_km2=3_000.0),
    "rule": lambda p, k: rule_targets(
        p, km2_per_unit=k, baseline=0.3, rules={"x": 0.2},
        traits={f: ({"x"} if f % 2 else set()) for f in range(1, 10)},
    ),
}

TOTALS_KM2 = {1: 0.0, 2: 50.0, 3: 900.0, 4: 2_500.0, 5: 10_000.0, 6: 60_000.0,
              7: 300_000.0, 8: 5e6, 9: 3e7}


@pytest.mark.parametrize("name", sorted(AREA_METHODS))
def test_targets_within_total_and_cap(make_problem, name):
    p = make_problem(TOTALS_KM2)
    out = AREA_METHODS[name](p, 1.0)
    assert list(out) == list(TOTALS_KM2)
    for fid, t in out.items():
        assert 0.0 <= t <= TOTALS_KM2[fid] * (1 + 1e-12), (fid, t)
        assert t <= 1e6 * (1 + 1e-12), (fid, t)
    assert out[1] == 0.0  # zero total -> zero target


@pytest.mark.parametrize("name", sorted(AREA_METHODS))
def test_rescaled_native_unit_scales_targets(make_problem, name):
    # Same ranges expressed in a unit 100x smaller: amounts x100, factor /100.
    km2 = AREA_METHODS[name](make_problem(TOTALS_KM2), 1.0)
    small = make_problem({f: v * 100.0 for f, v in TOTALS_KM2.items()})
    native = AREA_METHODS[name](small, 0.01)
    for fid in TOTALS_KM2:
        assert native[fid] == pytest.approx(km2[fid] * 100.0, rel=1e-12, abs=1e-9)


def test_end_to_end_rodrigues_mip(tiny_problem):
    # tests/data/simple totals: species 1 = 60, 2 = 45, 3 = 33 (native units).
    # Treat one unit as 1,000 km²: ranges 60,000 / 45,000 / 33,000 km² (log10 band).
    problem = tiny_problem.copy_with(features=tiny_problem.features.copy())
    t = rodrigues_targets(problem, km2_per_unit=1_000.0)
    totals = problem.feature_amounts()
    for fid, value in t.items():
        assert 0.0 < value < totals[fid]
    apply_targets(problem, t)
    sol = MIPSolver().solve(problem, SolverConfig(num_solutions=1))[0]
    assert all(sol.targets_met.values())
```

- [ ] **Step 2: Delete `test_targets_package_all`** from `tests/pymarxan/test_targets.py`.

- [ ] **Step 3: Run.**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/targets tests/pymarxan/test_targets.py -q`
Expected: all pass. These tests pass immediately because they check the invariants of Tasks 2–6. If any fails, a contract rule is broken in an owning module: fix it there and record it in the ledger.

- [ ] **Step 4: Prove the invariant tests can fail.** Temporarily change `finish_area` in `_units.py` to skip the clip (`t = t_km2`). Run `test_targets_api.py` and confirm `test_targets_within_total_and_cap` fails for several methods. Revert, and confirm `git diff src/` is empty.

- [ ] **Step 5: Lint, commit.**

```bash
~/.local/bin/ruff check tests/pymarxan/targets tests/pymarxan/test_targets.py
git add tests/pymarxan/targets/test_targets_api.py tests/pymarxan/test_targets.py
git commit -m "test(targets): cross-method invariants, public API, end-to-end MIP"
```

---

### Task 8: Tutorial, CHANGELOG, README, roadmap

**Files:**
- Modify: `docs/TUTORIAL.md` (TOC entry 9 + new section before `## River connectivity and barrier restoration`)
- Modify: `tests/test_tutorial_examples.py` (new `test_section_target_methods` before the rivers section)
- Modify: `CHANGELOG.md` (`## [Unreleased]`)
- Modify: `README.md` (line 76 architecture comment; line 147 feature bullet)
- Modify: `docs/plans/2026-10-11-prioritizr-parity-roadmap.md` (Stage 2 heading note)

**Interfaces:**
- Consumes: `build_minimal_problem()` in `tests/test_tutorial_examples.py` (12 PUs; feature totals coral 6, seagrass 4, mangrove 3); `jung_targets`, `rodrigues_targets`, `apply_targets`, `MIPSolver`, `SolverConfig`.

- [ ] **Step 1: Write the failing tutorial test.** In `tests/test_tutorial_examples.py`, before the `# Section: "River connectivity…"` banner, add:

```python
# --------------------------------------------------------------------
# Section: "9. Published target-setting methods (v0.40)"
# --------------------------------------------------------------------


def test_section_target_methods():
    problem = build_minimal_problem()

    # --- tutorial block (verbatim) ---
    from pymarxan.solvers.base import SolverConfig
    from pymarxan.solvers.mip_solver import MIPSolver
    from pymarxan.targets import apply_targets, jung_targets, rodrigues_targets

    # apply_targets edits features in place, so work on a copy.
    targeted = problem.copy_with(features=problem.features.copy())

    # One amount unit = 1,000 km² of habitat. There is no default unit.
    KM2 = 1_000.0
    threatened = [3]  # mangrove: IUCN VU
    others = [1, 2]

    apply_targets(targeted, jung_targets(targeted, km2_per_unit=KM2, features=threatened))
    apply_targets(targeted, rodrigues_targets(targeted, km2_per_unit=KM2, features=others))

    best = MIPSolver().solve(targeted, SolverConfig(num_solutions=1))[0]
    # --- end tutorial block ---

    targets = dict(zip(targeted.features["id"], targeted.features["target"]))
    # coral 6,000 km² -> log10 band; mangrove 3,000 km² -> max(2,200, 0.77*3,000) = 2,310
    assert targets[1] == pytest.approx(4.24772, rel=1e-4)
    assert targets[2] == pytest.approx(3.09612, rel=1e-4)
    assert targets[3] == pytest.approx(2.31, rel=1e-9)
    assert all(best.targets_met.values())
    assert list(problem.features["target"]) == [5.0, 4.0, 3.0]  # original untouched
```

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/test_tutorial_examples.py::test_section_target_methods -q`
Expected: PASS. The code exists after Tasks 3–4; this test pins the tutorial text written next. If it fails, recheck the expected values against the formulas before changing anything.

- [ ] **Step 2: Add the tutorial section.** In `docs/TUTORIAL.md`, add TOC line `9. [Published target-setting methods (v0.40)](#9-published-target-setting-methods-v040)` after item 8. Insert this section immediately before `## River connectivity and barrier restoration`. The code block is exactly the lines between `# --- tutorial block (verbatim) ---` and `# --- end tutorial block ---` in the test, dedented. Like every other section, it reuses the section-1 `problem`; `build_minimal_problem()` exists only in the test file:

````markdown
## 9. Published target-setting methods (v0.40)

`pymarxan.targets` implements the target rules conservation assessments cite:
range-size interpolation (Rodrigues, Sreekar, Polak, Watson), IUCN Red List
rules (species, ecosystems, Jung, Ward), population-based targets
(population size, Wilson, Durán), and fixed-area, rule-based and min/max
combinations. Every method returns `{feature_id: target}` in the feature's
own units; `apply_targets` writes it onto the problem and leaves features
it isn't given untouched, so methods mix across feature groups.

Area-based methods need to know what one unit of your feature data is in
km². There is no default: pass `km2_per_unit` (one factor, or a
`{feature_id: factor}` dict).

```python
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver
from pymarxan.targets import apply_targets, jung_targets, rodrigues_targets

# apply_targets edits features in place, so work on a copy.
targeted = problem.copy_with(features=problem.features.copy())

# One amount unit = 1,000 km² of habitat. There is no default unit.
KM2 = 1_000.0
threatened = [3]  # mangrove: IUCN VU
others = [1, 2]

apply_targets(targeted, jung_targets(targeted, km2_per_unit=KM2, features=threatened))
apply_targets(targeted, rodrigues_targets(targeted, km2_per_unit=KM2, features=others))

best = MIPSolver().solve(targeted, SolverConfig(num_solutions=1))[0]
```

Mangrove (3,000 km²) gets Jung et al.'s `max(2,200 km², 77%)` = 2,310 km²
(2.31 units); coral and seagrass get Rodrigues et al.'s log-interpolated
share between 100% (< 1,000 km²) and 10% (> 250,000 km²). Targets never
exceed a feature's total, and most methods cap at 1,000,000 km²
(`cap_km2=None` disables). Totals include locked-out planning units (as in
prioritizr), so a target can still be unattainable; `presolve_check`
flags that. Per-feature inputs accept a dict or a pandas Series indexed by
feature id. See each function's docstring for its source.
`loglinear_targets` is deprecated: use
`interp_absolute_targets(..., interp="log10")`.

---
````

- [ ] **Step 3: CHANGELOG.** Under `## [Unreleased]`, *above* the existing `### Fixed`, add:

```markdown
### Added

- **Published target-setting methods** in `pymarxan.targets` (prioritizr 9.0.1 parity, Stage 2 of the prioritizr roadmap). Each returns `{feature_id: target}` in native units; area-based methods take a required `km2_per_unit` (no silent unit assumption), cap at `cap_km2=1e6` km² (Butchart et al. 2015, https://doi.org/10.1111/conl.12158) and clip to the feature total. Mix methods across feature groups with `features=` and `apply_targets`.
  - `interp_area_targets`, `interp_absolute_targets`: general rare/common threshold interpolation (log10 or linear).
  - `rodrigues_targets`: Rodrigues et al. 2004, https://doi.org/10.1641/0006-3568(2004)054[1092:GGAPRF]2.0.CO;2.
  - `sreekar_targets`: Sreekar & Watson 2026, https://doi.org/10.1038/s44358-026-00169-7.
  - `polak_targets`: Polak et al. 2015, https://doi.org/10.1098/rsos.150107.
  - `watson_targets`: Watson et al. 2010, https://doi.org/10.1111/j.1523-1739.2010.01587.x.
  - `rl_species_targets`: IUCN (2012) Red List Categories and Criteria v3.1, 2nd ed.
  - `rl_ecosystem_targets`: IUCN (2024) Red List of Ecosystems guidelines v2.0; Keith et al. 2013, https://doi.org/10.1371/journal.pone.0062111.
  - `jung_targets`: Jung et al. 2021, https://doi.org/10.1038/s41559-021-01528-7 (correction https://doi.org/10.1038/s41559-021-01569-y).
  - `ward_targets`: Ward et al. 2025, https://doi.org/10.1073/pnas.2414985122.
  - `pop_size_targets`: minimum viable population ÷ density.
  - `wilson_targets`: Wilson et al. 2010, https://doi.org/10.1890/09-1051.1.
  - `duran_targets`: Durán et al. 2020, https://doi.org/10.1111/2041-210X.13427.
  - `rule_targets`: Harris & Holness 2023, https://doi.org/10.1016/j.biocon.2023.110218.
  - `absolute_targets`, `area_targets`, `min_targets`, `max_targets`.
- **Deviations from prioritizr 9.0.1** (deliberate):
  - RLE criterion A3 VU threshold is 0.5 (IUCN guidelines; prioritizr has 0.3).
  - Watson at exactly the 10,000 km² threshold returns 1,000 km² (prioritizr errors).
  - Zero-total features get target 0 (prioritizr returns NaN).
  - `wilson_targets` returns prioritizr's large-population approximation (Lande 1993, https://doi.org/10.1086/285580), but raises where it is unreliable: when `r <= σ²/2`, and when its target differs by more than 1% from the target given by Lande's exact mean-time-to-extinction model. Near `1e5·σ²·β²/2 ≈ 1` the approximation gives near-zero targets for the most extinction-prone species, while the exact model needs enormous populations; prioritizr returns those values, or NaN/Inf. Refused features can use `pop_size_targets` with a chosen minimum viable population. The calculation runs in log space, so it can't overflow.
  - Per-feature inputs are keyed by feature id (prioritizr misaligns them on feature subsets and in `rule` ordering).
  - Red List criteria are case-insensitive (prioritizr cannot match `"A2b"`).
  - Red List `status` can vary per feature.
  - `min_targets`/`max_targets` refuse inputs with different features.
  - Where prioritizr's documentation and code disagree, pymarxan follows the code: Jung 77% (not 80%), Ward 100 km² (not 100,000), species EN-B2 500 km² (not 50), and population targets divide by density.
- Pinned worked examples are hand-computed from each published rule as prioritizr implements it. Paper text confirms Rodrigues/Butchart, Polak's breakpoints, Jung's 2,200 km² floor and Durán's z = 0.25. Watson, Sreekar, Ward and Wilson parameters, and Jung's 77% term, could not be confirmed from accessible paper text; the docstrings say so. `polak_targets` does not apply Polak et al.'s 100%-for-Critically-Endangered rule (target CR features separately).
- Feature totals include locked-out planning units, as in prioritizr. Per-feature inputs accept dicts or pandas Series keyed by feature id; non-id keys, duplicated Series indexes, boolean masks and NaN/negative amounts raise `ValueError`.

### Changed

- `pymarxan.targets` is now a package; `from pymarxan.targets import ...` is unchanged.

### Deprecated

- `loglinear_targets`: use `interp_absolute_targets(..., interp="log10")`, which reproduces it exactly. It still works and now emits `DeprecationWarning`.
```

- [ ] **Step 4: README.**
  - Line 76: change `│   ├── targets/           Automatic target rules (relative, log-linear, group)` to `│   ├── targets/           Target rules: relative, group, range-size, IUCN Red List, population, composition`.
  - Line 147: change `- Automatic **target-setting** rules (relative, IUCN-style log-linear, per-group)` to `- Published **target-setting** methods (Rodrigues, Watson, Polak, Sreekar, IUCN Red List species/ecosystems, Jung, Ward, population size, Wilson, Durán, rule-based, min/max) with explicit km² units`.

- [ ] **Step 5: Roadmap.** In `docs/plans/2026-10-11-prioritizr-parity-roadmap.md`, follow the Stage 1 convention. Change the heading `## Stage 2 — v0.40.0: Published target-setting methods` to `## Stage 2 — v0.40.0: Published target-setting methods (SHIPPED, v0.40)`, and add under it: `**As shipped (v0.40.0):** see docs/plans/2026-10-15-stage2-target-methods-design.md. Worked examples are hand-computed from each rule as prioritizr implements it (no paper publishes numeric pairs); wilson_targets refuses inputs where its large-population approximation disagrees with Lande's exact model by more than 1%.` The release follows this branch directly (spec §10). If it doesn't, the release session must re-check this line.

- [ ] **Step 6: Full gate.**

Run: `PATH="/opt/micromamba/envs/shiny/bin:$HOME/.local/bin:$PWD/.venv/bin:$PATH" make check`
Expected: lint and types clean; the full suite passes (≈2690 + ~190 new); coverage ≥ 75%.
Run: `rm -rf /tmp/pdoc-targets && .venv/bin/pdoc src/pymarxan/targets -o /tmp/pdoc-targets && grep -c "rodrigues_targets" /tmp/pdoc-targets/pymarxan/targets.html`
Expected: a count ≥ 1, which proves the pdoc page is not empty. Don't use `make docs`: pdoc is installed only in `.venv`, which lacks rasterio, so documenting all of `pymarxan` fails at `pymarxan.spatial` (found by the 4-lens review). Documenting just the targets package avoids that and writes nothing into the repo.

- [ ] **Step 7: Commit.**

```bash
git add docs/TUTORIAL.md tests/test_tutorial_examples.py CHANGELOG.md README.md docs/plans/2026-10-11-prioritizr-parity-roadmap.md
git commit -m "docs(targets): tutorial section, CHANGELOG with sources, README, roadmap"
```
