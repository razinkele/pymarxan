# Tier C — raptr space/adequacy targets — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add raptr-style **space/adequacy targets** — a feature must be spatially spread across its attribute-space distribution, not just meet an amount target — evaluated by `compute_space_held` (`1 − WSS/TSS`) and enforced as an SA/greedy penalty.

**Architecture:** A new `pymarxan.adequacy` subpackage: the model (`SpaceSpec`, attribute space, demand points) + `compute_space_held`. A `SpaceState` (mirroring Phase-20 `SepState`) carries the incremental space penalty into the SA loop; the greedy solver gains a marginal-space term. `has_geometry`-style: absent `space_target` column → zero cost, behaviour unchanged.

**Tech Stack:** Python 3.12+, NumPy. (No new dep.)

**Design spec:** `docs/plans/2026-07-15-tierc-raptr-space-targets-design.md`. Science verified vs raptr source: `...-review.md`. Precedents: `solvers/separation.py::SepState` (geographic SA-delta), `solvers/separation.py::get_pu_coordinates` (attribute-space default), `solvers/heuristic.py::_score_pu` (greedy scoring).

## Global Constraints

- Python 3.12+, `from __future__ import annotations`, full type hints.
- No new third-party dependency.
- Tests **must** run under the `shiny` micromamba env: `/opt/micromamba/envs/shiny/bin/pytest`.
- Lint: ruff (E, F, I, UP; line length 99). Types: mypy clean. Coverage ≥ 75%.
- The bar before done: `make check` green.
- **Verified science (do not re-derive):** `space_held_f = 1 − WSS_f/TSS_f`, `WSS = Σ_d w_d·min_{selected}‖pos−p_d‖²`, `TSS = Σ_d w_d·‖p_d − c‖²`, `c` = **unweighted** mean of demand-point coords. Empty selection → 0; `TSS==0` → 1.
- **Honest deviations (must be in docstrings):** soft penalty ≠ raptr's hard IP constraint; occupied-PU demand points ≠ raptr's KDE-sample; `space_spf` is a pymarxan choice.
- **Parity:** no `space_target` column / all-zero → the whole path is skipped; `validate_marxan_parity.py` (35.0) untouched.

## File Structure

- Create: `src/pymarxan/adequacy/__init__.py`, `.../model.py` (`SpaceSpec`, demand points), `.../space.py` (`compute_space_held`).
- Modify: `src/pymarxan/solvers/space_state.py` (new `SpaceState`) or fold into `adequacy/`.
- Modify: `src/pymarxan/solvers/simulated_annealing.py` (wire `SpaceState`), `src/pymarxan/solvers/heuristic.py` (marginal-space term + stopping).
- Test: `tests/pymarxan/adequacy/test_space.py`, `tests/pymarxan/solvers/test_space_state.py`.
- Modify: `CHANGELOG.md`.

---

### Task 1: The model + `compute_space_held` (verified core)

**Files:**
- Create: `src/pymarxan/adequacy/__init__.py`, `.../model.py`, `.../space.py`
- Test: `tests/pymarxan/adequacy/test_space.py`

**Interfaces:**
- Produces: `SpaceSpec` (attribute-space config); `build_demand_points(problem, spec) -> dict[int, DemandPoints]`; `compute_space_held(problem, selected, spec) -> dict[int, float]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/pymarxan/adequacy/test_space.py`:

```python
"""Tests for raptr-style space/adequacy targets (1 - WSS/TSS)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pymarxan.adequacy import SpaceSpec, compute_space_held
from pymarxan.models.problem import ConservationProblem


def _line_problem(n=5, feature_at=None):
    # n PUs on a line at x=0..n-1 (y=0), all holding feature 1 unless feature_at given.
    ids = np.arange(1, n + 1)
    pu = pd.DataFrame({"id": ids, "cost": 1.0, "status": 0,
                       "xloc": np.arange(n, dtype=float), "yloc": 0.0})
    feats = pd.DataFrame({"id": [1], "name": ["a"], "target": [1.0], "spf": [1.0]})
    occ = ids if feature_at is None else np.asarray(feature_at)
    pvf = pd.DataFrame({"species": 1, "pu": occ, "amount": 1.0})
    return ConservationProblem(pu, feats, pvf)


def test_space_held_full_selection_is_one():
    p = _line_problem(5)
    held = compute_space_held(p, np.ones(5, bool), SpaceSpec())
    assert held[1] == 1.0  # all occupied PUs selected -> WSS 0 -> held 1


def test_space_held_empty_selection_is_zero():
    p = _line_problem(5)
    held = compute_space_held(p, np.zeros(5, bool), SpaceSpec())
    assert held[1] == 0.0


def test_space_held_spread_beats_clustered():
    # 5 PUs on a line; select 2. Spread {ends} covers the range better than clustered {adjacent}.
    p = _line_problem(5)
    spread = np.array([True, False, False, False, True])   # x=0 and x=4
    clustered = np.array([True, True, False, False, False])  # x=0 and x=1
    hs = compute_space_held(p, spread, SpaceSpec())[1]
    hc = compute_space_held(p, clustered, SpaceSpec())[1]
    assert hs > hc


def test_space_held_monotone_in_added_pu():
    p = _line_problem(6)
    base = np.array([True, False, False, False, False, False])
    more = base.copy(); more[3] = True
    assert compute_space_held(p, more, SpaceSpec())[1] >= compute_space_held(p, base, SpaceSpec())[1]


def test_space_held_matches_hand_computed():
    # 3 PUs on a line x=0,1,2, feature everywhere (amount 1). demand pts = the 3 PUs, w=1.
    # centroid c = mean(0,1,2)=1. TSS = 1*(0-1)^2 + 1*(1-1)^2 + 1*(2-1)^2 = 2.
    # Select {x=0}: WSS = min-dist^2 of each demand pt to x=0 = 0 + 1 + 4 = 5. held = 1 - 5/2 = -1.5 -> clip 0.
    # Select {x=1} (centre): WSS = 1 + 0 + 1 = 2. held = 1 - 2/2 = 0.0.
    # Select all: WSS 0 -> held 1.
    p = _line_problem(3)
    assert compute_space_held(p, np.array([False, True, False]), SpaceSpec())[1] == 0.0
    assert compute_space_held(p, np.array([True, True, True]), SpaceSpec())[1] == 1.0


def test_space_held_zscored_attribute_columns():
    # A large-unit attribute column must not dominate after z-scoring.
    ids = np.arange(1, 4)
    pu = pd.DataFrame({"id": ids, "cost": 1.0, "status": 0,
                       "env": [0.0, 1000.0, 2000.0]})  # big-unit column
    feats = pd.DataFrame({"id": [1], "name": ["a"], "target": [1.0], "spf": [1.0]})
    pvf = pd.DataFrame({"species": 1, "pu": ids, "amount": 1.0})
    p = ConservationProblem(pu, feats, pvf)
    # Review BUG-A: env-only space (no coords on this problem) -> include_geographic=False.
    spec = SpaceSpec(attribute_columns=["env"], include_geographic=False)
    # centre PU alone -> held 0.0 (same structure as the line, post z-score)
    assert compute_space_held(p, np.array([False, True, False]), spec)[1] == 0.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/adequacy/test_space.py -q`
Expected: FAIL — `pymarxan.adequacy` does not exist.

- [ ] **Step 3: Implement the model + `compute_space_held`**

Create `src/pymarxan/adequacy/model.py`:

```python
"""raptr-style space/adequacy: attribute space + demand points (Hanson et al. 2018)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.separation import get_pu_coordinates


@dataclass
class SpaceSpec:
    """Attribute-space configuration for space/adequacy targets.

    Default: geographic PU centroids (via ``get_pu_coordinates``), z-scored per dimension.
    ``attribute_columns`` uses those ``planning_units`` columns instead of / in addition to
    geography. Demand points default to the occupied PUs (amount>0), weighted by amount — a
    documented deviation from raptr's KDE-sampled, density-weighted default.
    """
    attribute_columns: list[str] | None = None
    include_geographic: bool = True
    zscore: bool = True


def pu_attribute_space(problem: ConservationProblem, spec: SpaceSpec) -> np.ndarray:
    """(n_pu, n_dim) attribute positions, z-scored per dimension when ``spec.zscore``.

    Geographic centroids (when ``include_geographic``) and/or ``attribute_columns``, in that
    order; raises if neither is configured. (Review BUG-A: a single clean dim-collection loop —
    no duplicate geographic append.)
    """
    dims = []
    if spec.include_geographic:
        dims.append(np.asarray(get_pu_coordinates(problem), dtype=float))  # (n_pu, 2)
    if spec.attribute_columns:
        dims.append(problem.planning_units[spec.attribute_columns].to_numpy(dtype=float))
    if not dims:
        raise ValueError(
            "SpaceSpec has no attribute dimensions "
            "(set include_geographic=True or provide attribute_columns)."
        )
    pos = np.column_stack(dims) if len(dims) > 1 else dims[0]
    if spec.zscore:
        mu = pos.mean(axis=0)
        sd = pos.std(axis=0)
        sd[sd == 0] = 1.0
        pos = (pos - mu) / sd
    return pos
```

Create `src/pymarxan/adequacy/space.py`:

```python
"""compute_space_held — raptr's 1 - WSS/TSS proportion of attribute-space variation captured."""
from __future__ import annotations

import numpy as np

from pymarxan.adequacy.model import SpaceSpec, pu_attribute_space
from pymarxan.models.problem import ConservationProblem


def compute_space_held(
    problem: ConservationProblem, selected: np.ndarray, spec: SpaceSpec | None = None,
) -> dict[int, float]:
    """Per-feature space held = ``1 - WSS/TSS`` (raptr, Hanson et al. 2018,
    doi:10.1111/2041-210x.12862).

    Demand points = the occupied PUs (amount>0), positioned in ``spec``'s (z-scored) attribute
    space, weight = feature amount. ``WSS = Σ_d w_d·min_{selected occupied}‖p_d − pos‖²``;
    ``TSS = Σ_d w_d·‖p_d − c‖²``, ``c`` = unweighted mean of demand-point coords. Empty
    selection → 0; ``TSS == 0`` → 1. (Occupied-PU demand points are a documented deviation from
    raptr's KDE-sample.)
    """
    spec = spec or SpaceSpec()
    pos = pu_attribute_space(problem, spec)  # (n_pu, n_dim)
    idx = problem.pu_id_to_index
    pv = problem.pu_vs_features
    held: dict[int, float] = {}
    for fid, grp in pv.groupby("species"):
        rows = grp[grp["amount"] > 0]
        # Review BUG-B: keep-mask so weights stay aligned to demand points even when
        # pu_vs_features references PU ids absent from planning_units (defensive, per
        # separation.py:274-278). A positional `[:len(occ)]` slice misaligns.
        pu_ids = rows["pu"].to_numpy()
        keep = np.array([int(p) in idx for p in pu_ids], dtype=bool)
        occ = np.array([idx[int(p)] for p in pu_ids[keep]], dtype=int)
        w = np.asarray(rows["amount"].to_numpy(), dtype=float)[keep]
        if len(occ) == 0:
            held[int(fid)] = 0.0
            continue
        p_d = pos[occ]                                   # (n_dp, n_dim) demand-point coords
        c = p_d.mean(axis=0)                             # unweighted centroid
        tss = float(np.sum(w * np.sum((p_d - c) ** 2, axis=1)))
        sel_occ = occ[selected[occ]]
        if len(sel_occ) == 0:
            held[int(fid)] = 0.0
            continue
        if tss == 0.0:
            held[int(fid)] = 1.0
            continue
        d2 = ((p_d[:, None, :] - pos[sel_occ][None, :, :]) ** 2).sum(axis=2)  # (n_dp, n_sel)
        wss = float(np.sum(w * d2.min(axis=1)))
        held[int(fid)] = float(np.clip(1.0 - wss / tss, 0.0, 1.0))
    return held
```

Create `src/pymarxan/adequacy/__init__.py`:

```python
from pymarxan.adequacy.model import SpaceSpec, pu_attribute_space
from pymarxan.adequacy.space import compute_space_held

__all__ = ["SpaceSpec", "compute_space_held", "pu_attribute_space"]
```

- [ ] **Step 4: Run to verify they pass**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/adequacy/test_space.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/pymarxan/adequacy/ tests/pymarxan/adequacy/
git commit -m "feat(adequacy): compute_space_held — raptr 1-WSS/TSS space/adequacy measure"
```

---

### Task 2: `SpaceState` + SA integration

> **⚠ REDESIGNED — Phase B.** The engineering design review
> (`...-engineering-review.md`) revised this task. Build it as **Phase B** with these findings
> folded in (superseding the code sketch below where they conflict):
> - `SpaceState` lives in **`solvers/space_state.py`** (a `ProblemCache` companion, like
>   `SepState`), NOT `adequacy/state.py`. Its stateless `selected`-passing design is valid but
>   *not* SepState's `cache`-passing/`apply_flip` design — don't "mirror SepState exactly".
> - **`delta_penalty` MUST use a precomputed numpy kernel** over the affected features' demand
>   points (per-feature demand positions/weights/TSS + a `pu_to_space_feats` inverse index
>   precomputed once) — **never** call the Task-1 `compute_space_held` per flip (it re-groups the
>   DataFrame + re-z-scores every call → blows the per-flip budget the `bench` marker guards).
> - Space is **ADDITIVE**: a feature has both an amount and a space target; both penalties apply.
>   Do **NOT** exclude space features from `_det_spf` (unlike clump/sep, which replace it).
> - Add `Solution.space_held`/`space_penalty` + `build_solution` population + a `supports_space()`
>   gate (every prior constraint type did — MIP/zone report the gap).
> - `space_target` / `space_spf` are **`features` columns** (like `spf`/`target2`/`sepnum`);
>   `SpaceSpec` is a **solver constructor arg**, NOT `SolverConfig` and NOT `problem.parameters`.
> - SA wiring: init penalty (`:172`), the **temp-estimation** delta site (`:183-193`), AND the main
>   trial delta (`:249-252`) — but **no** `apply_flip`/accept-site work (stateless → nothing to
>   commit).

**Files:**
- Create: `src/pymarxan/solvers/space_state.py` (`SpaceState`)
- Modify: `src/pymarxan/solvers/simulated_annealing.py`, `src/pymarxan/solvers/base.py`
  (`Solution.space_held`/`space_penalty`, `supports_space()`)
- Test: `tests/pymarxan/adequacy/test_space_state.py`

**Interfaces:**
- Consumes: `compute_space_held` (Task 1) — for the reference/oracle test only, NOT the hot loop.
- Produces: `SpaceState.from_problem(problem, spec)` (reads `space_target`/`space_spf` feature
  columns) / `.penalty_total(selected)` / `.delta_penalty(selected, idx, adding)` via a precomputed
  kernel; `active` gate.

- [ ] **Step 1: Write the failing tests**

Create `tests/pymarxan/adequacy/test_space_state.py` — assert `penalty_total` = `Σ space_spf·max(0, target − space_held)`; `delta_penalty` equals the difference of `penalty_total` before/after a flip (recompute-based, so exact); a problem with no `space_target` → the state is inactive (zero penalty). (Full test code: build a `_line_problem` with a `space_target` column on `features`, a `SpaceState`, and compare `delta_penalty(sel, idx, adding)` to `penalty_total(flipped) − penalty_total(sel)` for several flips.)

- [ ] **Step 2: Run to verify fail** — `SpaceState` not importable.

- [ ] **Step 3: Implement `SpaceState`**

`SpaceState.from_problem` reads the `space_target` column (default 0 = inactive) + `space_spf`
(defaults to the feature `spf`), stores the `SpaceSpec` + attribute positions + per-feature demand
points/weights/TSS (precomputed once). `active = any(space_target > 0)`.
- `penalty_total(selected)` → `Σ_f space_spf_f · max(0, space_target_f − space_held_f(selected))`
  using `compute_space_held` (or the cached per-feature machinery).
- `delta_penalty(selected, idx, adding)` → **recompute** `space_held` for the features that PU `idx`
  carries (only those change) before/after the flip, return the penalty difference. Correct
  (recompute, not an incremental facility-location delta — the true incremental `WSS` update via
  nearest/second-nearest tracking is a documented Phase-B optimisation). Inactive state → returns 0.

- [ ] **Step 4: Wire into the SA loop** (mirror `sep_state`, `simulated_annealing.py`)

- Build it near the ClumpState/SepState spin-up:
  ```python
  space_state = None
  if <problem has an active space_target column>:
      from pymarxan.adequacy.state import SpaceState
      space_state = SpaceState.from_problem(problem, spec, space_spf)
      current_obj += space_state.penalty_total(selected)
  ```
- Add its delta at BOTH delta sites (the trial and the accepted-move recompute), exactly like
  `sep_state`:
  ```python
  if space_state is not None:
      delta += space_state.delta_penalty(selected, idx, adding)
  ```
- The `spec`/`space_spf` reach the solver via `SolverConfig` (add optional `space_spec` /
  `space_spf` fields) or `problem.parameters` — pick per the codebase convention (verify in the
  plan/grounding).

- [ ] **Step 5: Run tests + a space-active SA smoke** — a small problem where a clustered optimum
  violates a space target: SA with the space penalty returns a more-spread reserve (higher
  space_held) than without. And a **no-space** problem's SA is unchanged (parity).

- [ ] **Step 6: Commit** — `feat(adequacy): SpaceState + SA-loop integration for space targets`.

---

### Task 3: greedy integration + parity + CHANGELOG

> **⚠ REDESIGNED — two-phase greedy (Phase B).** The engineering review found that `_score_pu`
> returns `None` once **amount** targets are met (`if not unmet: return None`), so the loop breaks
> *before* adding any space-only PU — "extend the stopping criterion" cannot work, and mixing a
> space term into the incommensurable HEURTYPE scales distorts rankings. **Redesign:** Phase 1 =
> existing HEURTYPE greedy to meet amount targets (unchanged); **Phase 2** (gated on
> `space_active`) = keep adding the PU with the largest marginal space-penalty reduction until
> `space_held_f ≥ space_target_f ∀f` or no candidate improves. Non-space greedy is bit-identical.

**Files:**
- Modify: `src/pymarxan/solvers/heuristic.py`
- Modify: `CHANGELOG.md`
- Test: `tests/pymarxan/adequacy/test_space.py` (append greedy test)

- [ ] **Step 1: Two-phase greedy.** Phase 1 unchanged. Phase 2 (only when `space_active`): after
  amount targets are met, repeatedly pick the remaining candidate maximising the space-penalty
  reduction (`−SpaceState.delta_penalty(selected, idx, adding=True)`) and add it, stopping when all
  space targets are met or no candidate reduces the penalty. Do NOT thread a space term into
  `_score_pu` (scale-distortion). Inactive → the loop never enters Phase 2.

- [ ] **Step 2: Test** — greedy on a space-target problem yields a reserve whose `space_held ≥
  target` (or improves toward it) vs the no-space greedy; a no-space problem's greedy is unchanged.

- [ ] **Step 3: Parity harness** — `/opt/micromamba/envs/shiny/bin/python examples/validate_marxan_parity.py` (35.0 intact — no space targets on the simple project).

- [ ] **Step 4: CHANGELOG + full check.**
  ```markdown
  - **Space/adequacy targets (raptr-style, `pymarxan.adequacy`).** Features can carry a
    ``space_target`` (proportion) requiring their selected sites to be spatially spread across the
    feature's attribute-space distribution (geographic or environmental), measured by
    ``compute_space_held`` (raptr's ``1 − WSS/TSS``, Hanson et al. 2018) and enforced as an SA /
    greedy penalty (``SpaceState``). Attribute space defaults to z-scored PU centroids. Deviations
    from raptr (documented): a soft penalty rather than raptr's hard IP constraint; occupied-PU
    demand points rather than a KDE-sample. No change without a ``space_target`` column.
  ```
  Run: `PATH="/opt/micromamba/envs/shiny/bin:$HOME/.local/bin:$PWD/.venv/bin:$PATH" make check`
  (+ `make bench` — SpaceState is off for non-space problems, so per-flip cost is unchanged there).

- [ ] **Step 5: Commit** — `feat(solvers): greedy space-target integration + docs (raptr adequacy)`.

---

## Post-plan notes

- **Design review — DONE.** Scientific lens (`...-review.md`, `1 − WSS/TSS` verified vs raptr
  source) + engineering lenses (architect + grounding; `...-engineering-review.md`). The grounding
  agent RAN the Task-1 code and confirmed the maths, and found 2 fixable Task-1 bugs (folded above:
  BUG-A `include_geographic` default, BUG-B `w` alignment) + the Tasks 2-3 redesign (folded into the
  Task 2/3 banners). **Sequencing:** ship Task 1 (`compute_space_held`) as the clean verified first
  piece; build Tasks 2-3 (SA `SpaceState` + two-phase greedy + `Solution` reporting) as Phase B.
- **Parity:** UI-free solver feature; the 35.0 anchor is untouched (no `space_target` on the simple project → the whole path is skipped).
- **Deferred (own specs):** the exact p-median MILP (hard-constraint) formulation; raptr's KDE-sampled demand points + the "reliable"/probabilistic variant; the true incremental `WSS` facility-location delta.
