# raptr space targets — Phase B implementation plan (SA + greedy enforcement)

> **For agentic workers:** TDD, one bite-sized step at a time, commit per sub-phase.

**Goal:** Enforce per-feature space/adequacy targets in the SA and greedy solvers via a soft
space-penalty, using the Task-1 `compute_space_held` measure — with the engineering-review redesign
(`...-engineering-review.md`) baked in.

**Architecture:** A stateful `SpaceState` companion (in `solvers/`, self-contained on a precomputed
numpy kernel — NOT the DataFrame `compute_space_held` per flip) carries an incremental space penalty
into the SA loop, mirroring `SepState`'s lifecycle. Greedy gets a **two-phase** treatment (amount
targets via HEURTYPE unchanged, then a space-penalty-reduction phase). `Solution` reports
`space_held`/`space_penalty` for all solver paths; `supports_space()` gates the ABC.

## Global constraints (verbatim from the reviews)

- **Precomputed kernel, never the DataFrame in the hot loop.** `SpaceState` precomputes attribute
  positions, per-feature demand positions/weights/TSS, and a `pu_to_space_feats` inverse index once;
  `delta_penalty`/`apply_flip` recompute space_held only for `pu_to_space_feats[idx]` (features whose
  occupied set contains `idx`) via numpy — no `groupby`, no z-score, no `compute_space_held` per flip.
- **Space is ADDITIVE.** A feature can carry both an amount target and a space target; both penalties
  apply. Do **NOT** touch `_det_spf` (unlike clump/sep, which replace the amount penalty).
- **Config as columns.** `space_target` (proportion, absent/0 = inactive) and `space_spf` (defaults
  to the feature `spf`) are **`features` columns**. `SpaceSpec` is a **solver constructor arg**.
- **Parity.** No `space_target` column → every branch is inert; the 35.0 anchor is untouched.
- Python 3.12+, `from __future__ import annotations`, full type hints; run tests under the `shiny`
  micromamba env.

---

## SpaceState interface (the crux)

`src/pymarxan/solvers/space_state.py` — stateful, self-contained (holds its own precompute; unlike
`SepState` it does NOT take `cache`, because the attribute space may be non-geographic columns the
cache doesn't hold — a deliberate, documented departure):

```python
"""SpaceState — incremental space/adequacy penalty companion for the SA / greedy loops.

Stateful, self-contained: precomputes the attribute-space kernel (positions, per-feature demand
points/weights/TSS, PU->feature inverse index) once, then recomputes space_held only for the
features whose occupied set contains a flipped PU (the documented v1 recompute delta, mirroring
SepState). NOT thread-safe; lives within one solver call frame.
"""
from __future__ import annotations

import numpy as np

from pymarxan.adequacy.model import SpaceSpec, pu_attribute_space
from pymarxan.models.problem import ConservationProblem


class SpaceState:
    def __init__(self, pos, feats, selected):
        # feats: list of per-active-feature dicts {fid, occ (idx array), w, tss, target, spf}
        self.pos = pos
        self.feats = feats
        self.pu_to_space_feats = self._build_inverse(pos.shape[0], feats)
        self.selected = selected.copy()
        self.held = np.array([self._held_for(f, self.selected) for f in feats], dtype=float)
        self.space_penalty_total = float(self._penalty(self.held))

    @property
    def active(self) -> bool:
        return len(self.feats) > 0

    @staticmethod
    def _build_inverse(n_pu, feats):
        inv: list[list[int]] = [[] for _ in range(n_pu)]
        for fi, f in enumerate(feats):
            for idx in f["occ"]:
                inv[int(idx)].append(fi)
        return [np.array(x, dtype=int) for x in inv]

    def _held_for(self, f, selected) -> float:
        occ, w, tss = f["occ"], f["w"], f["tss"]
        sel_occ = occ[selected[occ]]
        if len(sel_occ) == 0:
            return 0.0
        if tss == 0.0:
            return 1.0
        d2 = ((self.pos[occ][:, None, :] - self.pos[sel_occ][None, :, :]) ** 2).sum(axis=2)
        wss = float(np.sum(w * d2.min(axis=1)))
        return float(np.clip(1.0 - wss / tss, 0.0, 1.0))

    def _penalty(self, held) -> float:
        tot = 0.0
        for fi, f in enumerate(self.feats):
            tot += f["spf"] * max(0.0, f["target"] - held[fi])
        return tot

    @classmethod
    def from_problem(cls, problem: ConservationProblem, spec: SpaceSpec, selected) -> "SpaceState":
        feats_df = problem.features
        pos = pu_attribute_space(problem, spec)
        idx_map = problem.pu_id_to_index
        pv = problem.pu_vs_features
        species = np.asarray(pv["species"].to_numpy(), dtype=np.int64)
        pu_all = np.asarray(pv["pu"].to_numpy(), dtype=np.int64)
        amt_all = np.asarray(pv["amount"].to_numpy(), dtype=float)
        spf_col = "space_spf" if "space_spf" in feats_df.columns else "spf"
        feats = []
        for _, row in feats_df.iterrows():
            tgt = float(row.get("space_target", 0.0) or 0.0)
            if tgt <= 0.0:
                continue
            fid = int(row["id"])
            fmask = (species == fid) & (amt_all > 0)
            pu_ids = pu_all[fmask]
            keep = np.array([int(p) in idx_map for p in pu_ids], dtype=bool)
            occ = np.array([idx_map[int(p)] for p in pu_ids[keep]], dtype=int)
            w = amt_all[fmask][keep]
            if len(occ) == 0:
                # occupied set empty -> held always 0 -> constant penalty spf*tgt; keep with
                # empty occ so penalty is reported but no PU affects it.
                c = np.zeros(pos.shape[1]); tss = 0.0
            else:
                p_d = pos[occ]; c = p_d.mean(axis=0)
                tss = float(np.sum(w * np.sum((p_d - c) ** 2, axis=1)))
            feats.append({"fid": fid, "occ": occ, "w": w, "tss": tss,
                          "target": tgt, "spf": float(row.get(spf_col, 1.0))})
        return cls(pos, feats, selected)

    def penalty_total(self) -> float:
        return self.space_penalty_total

    def delta_penalty(self, idx: int, adding: bool) -> float:
        affected = self.pu_to_space_feats[idx]
        if len(affected) == 0:
            return 0.0
        sel_after = self.selected.copy(); sel_after[idx] = adding
        delta = 0.0
        for fi in affected:
            f = self.feats[fi]
            new_h = self._held_for(f, sel_after)
            delta += f["spf"] * (max(0.0, f["target"] - new_h) - max(0.0, f["target"] - self.held[fi]))
        return delta

    def apply_flip(self, idx: int, adding: bool) -> None:
        affected = self.pu_to_space_feats[idx]
        self.selected[idx] = adding
        for fi in affected:
            new_h = self._held_for(self.feats[fi], self.selected)
            self.space_penalty_total += self.feats[fi]["spf"] * (
                max(0.0, self.feats[fi]["target"] - new_h)
                - max(0.0, self.feats[fi]["target"] - self.held[fi])
            )
            self.held[fi] = new_h

    def all_targets_met(self) -> bool:
        return all(self.held[fi] >= self.feats[fi]["target"] for fi in range(len(self.feats)))

    def held_by_id(self) -> dict[int, float]:
        return {f["fid"]: float(self.held[fi]) for fi, f in enumerate(self.feats)}
```

---

### Task B1: `SpaceState` + unit tests

**Files:** Create `src/pymarxan/solvers/space_state.py`; Test
`tests/pymarxan/adequacy/test_space_state.py`.

**Interfaces produced:** `SpaceState.from_problem(problem, spec, selected)` / `.active` /
`.penalty_total()` / `.delta_penalty(idx, adding)` / `.apply_flip(idx, adding)` /
`.all_targets_met()` / `.held_by_id()`.

- [ ] **Step 1 — failing tests.** In `test_space_state.py`, using a `_line_problem` helper that adds
  a `space_target` column on `features`:
  - `penalty_total` equals `Σ spf·max(0, target − compute_space_held(...))` (oracle vs Task-1 fn).
  - **delta == recompute:** for several `idx`, `delta_penalty(idx, not selected[idx])` equals
    `penalty_total_after − penalty_total_before` where the "after" is a fresh `SpaceState` built on
    the flipped selection (proves the affected-feature recompute is exact).
  - `apply_flip` keeps `penalty_total()` consistent with a fresh rebuild after the same flips.
  - `held_by_id()` matches `compute_space_held` for the current selection.
  - no `space_target` column → `active is False`, `penalty_total()==0`, `delta_penalty(...)==0`.
- [ ] **Step 2** — run, verify fail (`SpaceState` not importable).
- [ ] **Step 3** — implement `space_state.py` (code above).
- [ ] **Step 4** — run, verify pass. ruff + mypy clean.
- [ ] **Step 5** — commit: `feat(adequacy): SpaceState — precomputed-kernel incremental space penalty`.

### Task B2: SA integration + `Solution` reporting + `supports_space()`

**Files:** Modify `src/pymarxan/solvers/simulated_annealing.py` (ctor `space_spec` + 4 wiring sites),
`src/pymarxan/solvers/base.py` (`Solution.space_held`/`space_penalty`, `Solver.supports_space()`),
`src/pymarxan/solvers/utils.py` (`build_solution` populates the two fields),
`src/pymarxan/adequacy/space.py` (add `evaluate_solution_space`); Test append to
`tests/pymarxan/adequacy/test_space_state.py` + a SA smoke.

- [ ] **Step 1 — `evaluate_solution_space(problem, selected, spec=None)`** in `adequacy/space.py`:
  returns `(held: dict[int,float], penalty: float)` over features with `space_target>0`, using
  `space_spf` (default `spf`). DataFrame-level (post-hoc, once per solution — fine to be slow).
- [ ] **Step 2 — `Solution` fields + `supports_space()`.** Add `space_held: dict[int,float]|None =
  None` and `space_penalty: float|None = None` to `Solution`. Add `Solver.supports_space()`
  (default False — space is opt-in; SA/greedy override True). Failing test: SA `supports_space()`.
- [ ] **Step 3 — `build_solution` populates** `space_held`/`space_penalty` when a `space_target`
  column with any `>0` exists, via `evaluate_solution_space`, wrapped in a
  `PUCoordinatesUnavailableError` try/except (mirror the sep block). Optional `space_spec` param
  threaded through (default `SpaceSpec()`).
- [ ] **Step 4 — SA wiring** (mirror `sep_state`): add `space_spec: SpaceSpec | None = None` to
  `SimulatedAnnealingSolver.__init__`. After the SepState spin-up:
  ```python
  space_state = None
  if "space_target" in problem.features.columns and (problem.features["space_target"] > 0).any():
      from pymarxan.solvers.space_state import SpaceState
      from pymarxan.adequacy.model import SpaceSpec
      space_state = SpaceState.from_problem(problem, self._space_spec or SpaceSpec(), selected)
      current_obj += space_state.penalty_total()
  ```
  Add `if space_state is not None: delta += space_state.delta_penalty(idx, adding=not selected[idx])`
  at the **temp-estimation** site and `delta += space_state.delta_penalty(idx, adding)` at the main
  trial; `space_state.apply_flip(idx, adding)` at the accept site. `supports_space()` → True.
  Thread `self._space_spec` into `build_solution(..., space_spec=self._space_spec)`.
- [ ] **Step 5 — tests.** A clustered-optimum-violates-space problem: SA with `space_spec` returns a
  reserve whose `space_held` exceeds the clustered baseline (spread wins); a no-space problem's SA is
  unchanged (parity smoke). `Solution.space_penalty` is reported.
- [ ] **Step 6** — ruff + mypy + commit: `feat(solvers): SA space-target penalty + Solution reporting`.

### Task B3: two-phase greedy + parity + CHANGELOG

**Files:** Modify `src/pymarxan/solvers/heuristic.py` (ctor `space_spec` + Phase 2 + `supports_space`);
`CHANGELOG.md`; Test append.

- [ ] **Step 1 — two-phase greedy.** Add `space_spec` to `HeuristicSolver.__init__`. After the
  existing Phase-1 loop in `_solve_once`, when a `space_target` column is active:
  ```python
  if "space_target" in problem.features.columns and (problem.features["space_target"] > 0).any():
      from pymarxan.solvers.space_state import SpaceState
      from pymarxan.adequacy.model import SpaceSpec
      ss = SpaceState.from_problem(problem, self._space_spec or SpaceSpec(), selected)
      while not ss.all_targets_met() and len(available) > 0:
          best_idx, best_gain = -1, 0.0
          for idx in available:
              gain = -ss.delta_penalty(int(idx), True)  # penalty reduction
              if gain > best_gain:
                  best_gain, best_idx = gain, int(idx)
          if best_idx < 0:
              break
          selected[best_idx] = True
          ss.apply_flip(best_idx, True)
          available = available[available != best_idx]
  ```
  Do NOT thread a space term into `_score_pu` (scale-distortion). `supports_space()` → True. Thread
  `space_spec` into `build_solution`.
- [ ] **Step 2 — tests.** Greedy on a space-target problem yields `space_held ≥ target` (or strictly
  improved vs the amount-only greedy); a no-space greedy is unchanged.
- [ ] **Step 3 — parity.** `/opt/micromamba/envs/shiny/bin/python examples/validate_marxan_parity.py`
  → 35.0 intact (no space targets on the simple project).
- [ ] **Step 4 — CHANGELOG** (append to the existing `[Unreleased]` adequacy entry): note SA/greedy
  enforcement (`SpaceState`), soft space penalty, two-phase greedy, `Solution.space_held/space_penalty`.
  Run `PATH="/opt/micromamba/envs/shiny/bin:$HOME/.local/bin:$PWD/.venv/bin:$PATH" make check` +
  `make bench` (SpaceState off for non-space → per-flip cost unchanged).
- [ ] **Step 5** — commit: `feat(solvers): two-phase greedy space enforcement + docs`.

## Self-review notes

- delta/apply_flip only touch `pu_to_space_feats[idx]` — correct because held_f changes iff the
  selected∩occ_f set changes, i.e. iff idx ∈ occ_f. Verified in B1's delta==recompute test.
- Empty-occ features keep a constant `spf·target` penalty (held always 0) and appear in no inverse
  list, so no flip ever changes them — matches `compute_space_held` returning 0 for absent features.
- `space_spf` defaults to `spf` consistently in both `SpaceState` and `evaluate_solution_space`, so
  the SA objective's penalty and the reported `space_penalty` agree.
- MIP/zone keep `supports_space()` == False (default) — they don't optimize space but still report
  the gap via `build_solution`, matching the "report even when not optimized" contract.
