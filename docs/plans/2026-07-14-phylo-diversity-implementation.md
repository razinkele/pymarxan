# Phylogenetic-diversity objectives — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `pymarxan.phylo` subpackage that scores a solution's Faith phylogenetic diversity (PD) and lets the existing solvers maximize PD via a branch-as-feature decomposition.

**Architecture:** A pure-Python `PhylogeneticTree` (edge table + Newick parser, zero new deps) drives two functions — `compute_phylogenetic_diversity` (scoring) and `phylogenetic_branch_problem` (turns a phylogeny into a standard `ConservationProblem` whose branch-features carry length as `spf`). One small additive MIP objective mode, `max_weighted_features`, lets the budget path maximize PD instead of feature count. `max_features` is left untouched.

**Tech Stack:** Python 3.12+, NumPy, pandas, PuLP/CBC (existing). No new dependencies.

**Design spec:** `docs/plans/2026-07-14-phylo-diversity-design.md` (read it first).

## Global Constraints

- Python 3.12+, `from __future__ import annotations` at the top of every new file, full type hints.
- Zero new third-party dependencies (pure Python / NumPy / pandas only).
- Tests **must** run under the `shiny` micromamba env: `/opt/micromamba/envs/shiny/bin/pytest` (the `.venv` lacks rasterio/ipyleaflet). See the `marxan-testing` skill.
- New optional `ConservationProblem` fields must be `kw_only=True` — but this plan adds none; it uses `copy_with(features=..., pu_vs_features=...)`.
- Domain models are dataclasses; test layout mirrors `src/` under `tests/pymarxan/`.
- Lint: ruff (E, F, I, UP; line length 99). Types: mypy clean. Coverage threshold 75%.
- The bar before done: `make check` green (lint + types + full suite).
- Commit after each task (TDD: failing test → implementation → passing test → commit).

## File Structure

- `src/pymarxan/phylo/__init__.py` — package exports (grows per task).
- `src/pymarxan/phylo/tree.py` — `PhylogeneticTree` (structure, Newick parser, descendant caching, `total_pd`, `validate`).
- `src/pymarxan/phylo/diversity.py` — `compute_phylogenetic_diversity` + `PDResult` + the shared `_resolve_tips` helper.
- `src/pymarxan/phylo/decomposition.py` — `phylogenetic_branch_problem`.
- `src/pymarxan/solvers/mip_solver.py` — **modify**: add `max_weighted_features` objective mode (3 sites: validation tuple L116, objective build after L344, relaxed-target constraint L433).
- `tests/pymarxan/phylo/__init__.py` — empty (test package marker).
- `tests/pymarxan/phylo/test_tree.py`, `test_diversity.py`, `test_decomposition.py`.
- `tests/pymarxan/solvers/test_mip_objectives.py` — **modify**: add `max_weighted_features` parity/behavior tests.

**Reference values** used throughout: the tree `((A:1,B:1):2,C:3);` has branches A:1, B:1, internal:2, C:3 → `total_pd == 7`. A reserve representing tips `{A, C}` captures A:1 + internal:2 + C:3 = **6** (B's branch excluded).

---

### Task 1: `PhylogeneticTree` core (edge table)

**Files:**
- Create: `src/pymarxan/phylo/__init__.py`
- Create: `src/pymarxan/phylo/tree.py`
- Create: `tests/pymarxan/phylo/__init__.py`
- Test: `tests/pymarxan/phylo/test_tree.py`

**Interfaces:**
- Produces:
  - `PhylogeneticTree.from_edges(edges: Iterable[tuple[NodeId, NodeId, float]]) -> PhylogeneticTree` where `NodeId = int | str`.
  - `tree.tips -> list[NodeId]` (sorted by `str`), `tree.n_tips -> int`.
  - `tree.branches -> list[tuple[NodeId, float, frozenset[NodeId]]]` — `(child_node, length, descendant_tips)`, sorted by `str(child_node)`.
  - `tree.total_pd -> float`.
  - `tree.validate() -> list[str]`.
  - `from_edges` raises `ValueError` on a node with two parents or a cycle.

- [ ] **Step 1: Write the failing tests**

Create `tests/pymarxan/phylo/__init__.py` (empty file) and `tests/pymarxan/phylo/test_tree.py`:

```python
"""Tests for the pure-Python PhylogeneticTree structure."""
from __future__ import annotations

import pytest

from pymarxan.phylo.tree import PhylogeneticTree

# ((A:1,B:1):2,C:3); with the internal node named "I".
REFERENCE_EDGES = [
    ("root", "I", 2.0),
    ("I", "A", 1.0),
    ("I", "B", 1.0),
    ("root", "C", 3.0),
]


def test_from_edges_tips_and_total_pd():
    tree = PhylogeneticTree.from_edges(REFERENCE_EDGES)
    assert tree.tips == ["A", "B", "C"]  # sorted by str
    assert tree.n_tips == 3
    assert tree.total_pd == pytest.approx(7.0)


def test_branches_carry_descendant_tips():
    tree = PhylogeneticTree.from_edges(REFERENCE_EDGES)
    desc = {child: (length, tips) for child, length, tips in tree.branches}
    assert desc["A"] == (1.0, frozenset({"A"}))
    assert desc["C"] == (3.0, frozenset({"C"}))
    assert desc["I"] == (2.0, frozenset({"A", "B"}))


def test_validate_accepts_well_formed_tree():
    assert PhylogeneticTree.from_edges(REFERENCE_EDGES).validate() == []


def test_validate_flags_two_roots():
    # X and Y are both never children → two roots.
    edges = [("X", "A", 1.0), ("Y", "B", 1.0)]
    errors = PhylogeneticTree.from_edges(edges).validate()
    assert any("root" in e for e in errors)


def test_validate_flags_negative_length():
    edges = [("root", "A", -1.0), ("root", "B", 1.0)]
    errors = PhylogeneticTree.from_edges(edges).validate()
    assert any("negative" in e for e in errors)


def test_validate_flags_empty_tree():
    errors = PhylogeneticTree.from_edges([]).validate()
    assert any("tip" in e for e in errors)


def test_from_edges_raises_on_two_parents():
    edges = [("root", "A", 1.0), ("other", "A", 1.0)]
    with pytest.raises(ValueError, match="more than one parent"):
        PhylogeneticTree.from_edges(edges)


def test_from_edges_raises_on_cycle():
    edges = [("A", "B", 1.0), ("B", "C", 1.0), ("C", "A", 1.0)]
    with pytest.raises(ValueError, match="cycle"):
        PhylogeneticTree.from_edges(edges)


def test_single_tip_and_zero_length():
    tree = PhylogeneticTree.from_edges([("root", "A", 0.0)])
    assert tree.tips == ["A"]
    assert tree.total_pd == pytest.approx(0.0)
    assert tree.validate() == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/phylo/test_tree.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pymarxan.phylo'`.

- [ ] **Step 3: Create the package and implement the tree**

Create `src/pymarxan/phylo/__init__.py`:

```python
"""Phylogenetic-diversity objectives for pymarxan.

Faith (1992) phylogenetic diversity via a branch-as-feature decomposition
(Rodrigues & Gaston 2002): each branch of a phylogeny becomes a synthetic
conservation feature weighted by its length, so the existing solvers maximize
PD directly. See ``docs/plans/2026-07-14-phylo-diversity-design.md``.
"""
from __future__ import annotations

from pymarxan.phylo.tree import PhylogeneticTree

__all__ = ["PhylogeneticTree"]
```

Create `src/pymarxan/phylo/tree.py`:

```python
"""Pure-Python rooted phylogenetic tree (no networkx / dendropy dependency)."""
from __future__ import annotations

from collections.abc import Iterable

NodeId = int | str


class PhylogeneticTree:
    """A rooted phylogeny: branch lengths keyed by child node, with cached
    descendant-tip sets. Structural impossibilities (a node with two parents,
    a cycle) raise at construction; softer issues (wrong root count, negative
    lengths, no tips) are reported by :meth:`validate`, mirroring
    ``ConservationProblem.validate``.
    """

    def __init__(
        self,
        length: dict[NodeId, float],
        children: dict[NodeId, list[NodeId]],
        parent: dict[NodeId, NodeId],
    ) -> None:
        self._length = length
        self._children = children
        self._parent = parent
        self._nodes: set[NodeId] = set(parent) | set(children)
        self._roots = [n for n in self._nodes if n not in parent]
        self._tips = [n for n in self._nodes if n not in children]
        self._descendant_tips = self._compute_descendant_tips()

    @classmethod
    def from_edges(
        cls, edges: Iterable[tuple[NodeId, NodeId, float]]
    ) -> PhylogeneticTree:
        length: dict[NodeId, float] = {}
        children: dict[NodeId, list[NodeId]] = {}
        parent: dict[NodeId, NodeId] = {}
        for p, c, ln in edges:
            if c in parent:
                raise ValueError(
                    f"node {c!r} has more than one parent "
                    f"({parent[c]!r} and {p!r})"
                )
            parent[c] = p
            length[c] = float(ln)
            children.setdefault(p, []).append(c)
        return cls(length=length, children=children, parent=parent)

    def _compute_descendant_tips(self) -> dict[NodeId, frozenset[NodeId]]:
        result: dict[NodeId, frozenset[NodeId]] = {}
        state: dict[NodeId, int] = {}  # 1 = in-progress, 2 = done

        def visit(n: NodeId) -> frozenset[NodeId]:
            if state.get(n) == 2:
                return result[n]
            if state.get(n) == 1:
                raise ValueError(f"cycle detected in phylogeny at node {n!r}")
            state[n] = 1
            kids = self._children.get(n)
            if not kids:
                tips = frozenset({n})
            else:
                acc: set[NodeId] = set()
                for k in kids:
                    acc |= visit(k)
                tips = frozenset(acc)
            result[n] = tips
            state[n] = 2
            return tips

        for node in self._nodes:
            visit(node)
        return result

    @property
    def tips(self) -> list[NodeId]:
        return sorted(self._tips, key=str)

    @property
    def n_tips(self) -> int:
        return len(self._tips)

    @property
    def branches(self) -> list[tuple[NodeId, float, frozenset[NodeId]]]:
        return [
            (child, self._length[child], self._descendant_tips[child])
            for child in sorted(self._length, key=str)
        ]

    @property
    def total_pd(self) -> float:
        return float(sum(self._length.values()))

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self._tips:
            errors.append("tree has no tips")
        if len(self._roots) == 0:
            errors.append("tree has no root (every node has a parent)")
        elif len(self._roots) > 1:
            roots = sorted(map(str, self._roots))
            errors.append(f"tree has {len(self._roots)} roots, expected 1: {roots}")
        for child, ln in self._length.items():
            if ln < 0:
                errors.append(f"branch to {child!r} has negative length {ln}")
        return errors
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/phylo/test_tree.py -v`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add src/pymarxan/phylo/__init__.py src/pymarxan/phylo/tree.py tests/pymarxan/phylo/
git commit -m "feat(phylo): PhylogeneticTree core (edge table, descendant caching, validate)"
```

---

### Task 2: Newick parser (`from_newick`)

**Files:**
- Modify: `src/pymarxan/phylo/tree.py` (add `from_newick` classmethod + a small parse guard)
- Test: `tests/pymarxan/phylo/test_tree.py` (append)

**Interfaces:**
- Consumes: `PhylogeneticTree.from_edges` (Task 1).
- Produces: `PhylogeneticTree.from_newick(text: str) -> PhylogeneticTree`. Accepts nested `(...)` clades, named tips, optional `:length` on any node, optional internal-node names (numeric bootstrap labels accepted as ids); ignores a root `:length`; raises `ValueError` on a missing `;`, quoted labels (`'`), or NHX/comment brackets (`[`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/pymarxan/phylo/test_tree.py`:

```python
def test_from_newick_matches_reference_structure():
    tree = PhylogeneticTree.from_newick("((A:1,B:1):2,C:3);")
    assert tree.tips == ["A", "B", "C"]
    assert tree.total_pd == pytest.approx(7.0)
    desc = {child: (length, tips) for child, length, tips in tree.branches}
    # A and B share an internal parent; C hangs off the root.
    assert desc["A"] == (1.0, frozenset({"A"}))
    assert desc["B"] == (1.0, frozenset({"B"}))
    assert desc["C"] == (3.0, frozenset({"C"}))
    # exactly one internal branch of length 2 spanning {A, B}
    internal = [(ln, tips) for ch, ln, tips in tree.branches if tips == frozenset({"A", "B"})]
    assert internal == [(2.0, frozenset({"A", "B"}))]


def test_from_newick_requires_semicolon():
    with pytest.raises(ValueError, match="';'"):
        PhylogeneticTree.from_newick("(A:1,B:1)")


def test_from_newick_rejects_quoted_or_comment():
    with pytest.raises(ValueError, match="not supported"):
        PhylogeneticTree.from_newick("('A':1,B:1);")


def test_from_newick_named_internal_node():
    tree = PhylogeneticTree.from_newick("((A:1,B:1)clade:2,C:3);")
    # the named internal node "clade" spans {A, B}
    assert any(
        ch == "clade" and tips == frozenset({"A", "B"})
        for ch, ln, tips in tree.branches
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/phylo/test_tree.py -k newick -v`
Expected: FAIL with `AttributeError: type object 'PhylogeneticTree' has no attribute 'from_newick'`.

- [ ] **Step 3: Add the parser**

In `src/pymarxan/phylo/tree.py`, add this classmethod to `PhylogeneticTree` (below `from_edges`):

```python
    @classmethod
    def from_newick(cls, text: str) -> PhylogeneticTree:
        s = text.strip()
        if "'" in s or "[" in s:
            raise ValueError(
                "quoted labels and NHX/comment brackets are not supported"
            )
        if not s.endswith(";"):
            raise ValueError("Newick string must end with ';'")
        s = s[:-1].strip()

        pos = 0
        internal_counter = 0
        edges: list[tuple[NodeId, NodeId, float]] = []

        def parse_name() -> str:
            nonlocal pos
            start = pos
            while pos < len(s) and s[pos] not in "(),:;":
                pos += 1
            return s[start:pos].strip()

        def parse_length() -> float:
            nonlocal pos
            if pos < len(s) and s[pos] == ":":
                pos += 1
                start = pos
                while pos < len(s) and s[pos] not in "(),:;":
                    pos += 1
                return float(s[start:pos])
            return 0.0

        def parse_clade() -> tuple[str, float]:
            nonlocal pos, internal_counter
            if pos < len(s) and s[pos] == "(":
                pos += 1  # consume '('
                kids: list[tuple[str, float]] = []
                while True:
                    kids.append(parse_clade())
                    if pos < len(s) and s[pos] == ",":
                        pos += 1
                        continue
                    break
                if pos >= len(s) or s[pos] != ")":
                    raise ValueError(f"expected ')' at position {pos} in Newick")
                pos += 1  # consume ')'
                name = parse_name()
                if not name:
                    internal_counter += 1
                    name = f"__node{internal_counter}"
                length = parse_length()
                for child_name, child_len in kids:
                    edges.append((name, child_name, child_len))
                return name, length
            name = parse_name()
            if not name:
                raise ValueError(f"empty tip name at position {pos} in Newick")
            length = parse_length()
            return name, length

        parse_clade()  # root; its own :length (if any) is ignored
        if pos != len(s):
            raise ValueError(f"trailing characters in Newick at position {pos}")
        return cls.from_edges(edges)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/phylo/test_tree.py -v`
Expected: PASS (all tree tests, including the 4 new Newick ones).

- [ ] **Step 5: Commit**

```bash
git add src/pymarxan/phylo/tree.py tests/pymarxan/phylo/test_tree.py
git commit -m "feat(phylo): pure-Python Newick parser (from_newick)"
```

---

### Task 3: `max_weighted_features` MIP objective mode

**Files:**
- Modify: `src/pymarxan/solvers/mip_solver.py` (3 sites)
- Test: `tests/pymarxan/solvers/test_mip_objectives.py` (append)

**Interfaces:**
- Produces: `MIPSolver(objective="max_weighted_features")` — maximizes `Σ spf_j·z_j` subject to `Σ cost·x ≤ COSTBUDGET`; `z_j` binary = feature `j`'s target met. Requires `parameters["COSTBUDGET"]`. `max_features` semantics unchanged.

**Parity note:** This adds solver objective math. After it passes, run the `marxan-parity-check` skill. The tests below are the parity guards.

- [ ] **Step 1: Write the failing tests**

The file already defines `_two_feature_problem(cost_budget: float | None = None)` (2 features, 4 PUs, both `spf == 1`) and already imports `pandas as pd`, `ConservationProblem`, `SolverConfig`, `MIPSolver`, and `pytest`. Append these tests, reusing that helper:

```python
def test_max_weighted_features_stores_objective():
    solver = MIPSolver(objective="max_weighted_features")
    assert solver.objective == "max_weighted_features"


def test_max_weighted_features_rejects_when_no_cost_budget():
    p = _two_feature_problem()
    with pytest.raises(ValueError, match="COSTBUDGET"):
        MIPSolver(objective="max_weighted_features").solve(
            p, SolverConfig(num_solutions=1)
        )


def test_max_weighted_features_uniform_spf_matches_max_features_count():
    # With every spf == 1, weighted == unweighted: same number of targets met.
    wf = MIPSolver(objective="max_weighted_features").solve(
        _two_feature_problem(cost_budget=1.0), SolverConfig(num_solutions=1)
    )
    mf = MIPSolver(objective="max_features").solve(
        _two_feature_problem(cost_budget=1.0), SolverConfig(num_solutions=1)
    )
    n_wf = sum(bool(v) for v in wf[0].targets_met.values())
    n_mf = sum(bool(v) for v in mf[0].targets_met.values())
    assert n_wf == n_mf


def test_max_weighted_features_prefers_high_spf_under_tight_budget():
    # Two features, budget pays for only one. Feature 2 has spf 10 vs 1;
    # the weighted objective must pick the target that includes feature 2.
    planning_units = pd.DataFrame(
        {"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]}
    )
    features = pd.DataFrame(
        {"id": [1, 2], "name": ["a", "b"], "target": [1.0, 1.0], "spf": [1.0, 10.0]}
    )
    # feature 1 only in PU1; feature 2 only in PU2.
    puvspr = pd.DataFrame(
        {"species": [1, 2], "pu": [1, 2], "amount": [1.0, 1.0]}
    )
    p = ConservationProblem(
        planning_units, features, puvspr, parameters={"COSTBUDGET": 1.0}
    )
    sol = MIPSolver(objective="max_weighted_features").solve(
        p, SolverConfig(num_solutions=1)
    )[0]
    # PU2 (carrying the spf-10 feature) is chosen.
    assert bool(sol.selected[1]) is True
    assert bool(sol.selected[0]) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_mip_objectives.py -k max_weighted -v`
Expected: FAIL — `MIPSolver(objective="max_weighted_features")` raises `ValueError` from the objective validator (mode not in the allowed tuple).

- [ ] **Step 3: Add the objective mode (3 sites in `mip_solver.py`)**

Site 1 — extend the validated-objective tuple (currently line ~116):

```python
        _validate_mip_strategy(
            "objective", objective,
            (
                "min_set",
                "max_features",
                "min_largest_shortfall",
                "min_penalties",
                "max_weighted_features",
            ),
        )
```

Site 2 — in the objective-build block, add this `elif` immediately **after** the `max_features` block (after its `model += cost_expr <= cost_budget, "cost_budget"` line, before `elif self.objective == "min_largest_shortfall":`):

```python
        elif self.objective == "max_weighted_features":
            # Binary z_j per feature; relaxed target ≥ target · z_j; cost cap
            # from COSTBUDGET. Maximise Σ spf_j·z_j (LpMinimize → negate) — the
            # spf-weighted count. With spf = branch length this maximises PD.
            cost_budget = problem.parameters.get("COSTBUDGET")
            if cost_budget is None:
                raise ValueError(
                    "objective='max_weighted_features' requires a COSTBUDGET "
                    "parameter on the problem "
                    "(problem.parameters['COSTBUDGET'] = ...). Without a budget "
                    "the formulation is degenerate — every target is trivially "
                    "met by selecting all PUs."
                )
            cost_budget = float(cost_budget)
            feat_spf_arr = problem.features["spf"].values.astype(float)
            spf_by_id = {
                int(feat_ids_init[fi]): float(feat_spf_arr[fi])
                for fi in range(len(feat_ids_init))
            }
            for fi in range(len(feat_ids_init)):
                fid = int(feat_ids_init[fi])
                feat_met[fid] = pulp.LpVariable(f"feat_met_{fid}", cat="Binary")
            model += (
                -pulp.lpSum(spf_by_id[fid] * feat_met[fid] for fid in feat_met),
                "objective",
            )
            model += cost_expr <= cost_budget, "cost_budget"
```

Site 3 — the relaxed-target constraint (currently line ~433): change

```python
                if self.objective == "max_features":
```

to

```python
                if self.objective in ("max_features", "max_weighted_features"):
```

- [ ] **Step 4: Run the new tests + the full objectives suite to verify pass and no regression**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/solvers/test_mip_objectives.py -v`
Expected: PASS — the 4 new tests plus every pre-existing `max_features` / `min_*` test still green (parity guard: `max_features` behavior unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/pymarxan/solvers/mip_solver.py tests/pymarxan/solvers/test_mip_objectives.py
git commit -m "feat(solvers): additive max_weighted_features MIP objective (spf-weighted)"
```

---

### Task 4: PD scoring (`compute_phylogenetic_diversity` + `PDResult`)

**Files:**
- Create: `src/pymarxan/phylo/diversity.py`
- Modify: `src/pymarxan/phylo/__init__.py` (export the new names)
- Test: `tests/pymarxan/phylo/test_diversity.py`

**Interfaces:**
- Consumes: `PhylogeneticTree` (Task 1); `ConservationProblem`, `Solution`.
- Produces:
  - `PDResult` dataclass: `pd_represented`, `pd_total`, `pd_representable`, `fraction_pd_total`, `fraction_pd_representable`, `n_tips`, `n_tips_represented`, `branch_child: list`, `branch_length: list[float]`, `branch_represented: list[bool]`; method `to_dataframe() -> pd.DataFrame` with columns `child_node`, `length`, `represented`.
  - `compute_phylogenetic_diversity(problem, solution, tree, *, tip_feature_map: dict[str, int] | None = None) -> PDResult`.
  - `_resolve_tips(problem, tree, tip_feature_map) -> dict[NodeId, int | None]` (module-level helper reused by Task 5).

- [ ] **Step 1: Write the failing tests**

Create `tests/pymarxan/phylo/test_diversity.py`:

```python
"""Tests for Faith PD scoring of a solution."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.phylo.diversity import PDResult, compute_phylogenetic_diversity
from pymarxan.phylo.tree import PhylogeneticTree
from pymarxan.solvers.base import Solution

TREE = PhylogeneticTree.from_newick("((A:1,B:1):2,C:3);")


def _problem() -> ConservationProblem:
    # 3 PUs; feature A only in PU1, B only in PU2, C only in PU3.
    planning_units = pd.DataFrame(
        {"id": [1, 2, 3], "cost": [1.0, 1.0, 1.0], "status": [0, 0, 0]}
    )
    features = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "target": [1.0, 1.0, 1.0],
            "spf": [1.0, 1.0, 1.0],
        }
    )
    pu_vs_features = pd.DataFrame(
        {"species": [1, 2, 3], "pu": [1, 2, 3], "amount": [1.0, 1.0, 1.0]}
    )
    return ConservationProblem(planning_units, features, pu_vs_features)


def _solution(selected: list[bool]) -> Solution:
    return Solution(
        selected=np.array(selected, dtype=bool),
        cost=0.0,
        boundary=0.0,
        objective=0.0,
        targets_met={},
    )


def test_pd_of_A_and_C_reserve():
    # Select PU1 (A) and PU3 (C): PD = A:1 + internal:2 + C:3 = 6; B excluded.
    res = compute_phylogenetic_diversity(_problem(), _solution([True, False, True]), TREE)
    assert res.pd_represented == pytest.approx(6.0)
    assert res.pd_total == pytest.approx(7.0)
    assert res.pd_representable == pytest.approx(7.0)  # all tips occur somewhere
    assert res.fraction_pd_total == pytest.approx(6.0 / 7.0)
    assert res.fraction_pd_representable == pytest.approx(6.0 / 7.0)
    assert res.n_tips == 3
    assert res.n_tips_represented == 2


def test_empty_reserve_has_zero_pd():
    res = compute_phylogenetic_diversity(
        _problem(), _solution([False, False, False]), TREE
    )
    assert res.pd_represented == pytest.approx(0.0)
    assert res.fraction_pd_total == pytest.approx(0.0)
    assert res.fraction_pd_representable == pytest.approx(0.0)


def test_unrepresentable_tip_shrinks_representable_ceiling():
    # Drop C from the data entirely: C's tip can never be represented.
    p = _problem()
    p = p.copy_with(
        pu_vs_features=p.pu_vs_features[p.pu_vs_features["species"] != 3].reset_index(
            drop=True
        )
    )
    # Reserve {A, B}: PD = A:1 + B:1 + internal:2 = 4. Representable excludes
    # C's branch (3) → pd_representable = 4, so representable fraction == 1.0,
    # while total fraction = 4/7 < 1.
    res = compute_phylogenetic_diversity(p, _solution([True, True, False]), TREE)
    assert res.pd_represented == pytest.approx(4.0)
    assert res.pd_representable == pytest.approx(4.0)
    assert res.fraction_pd_representable == pytest.approx(1.0)
    assert res.fraction_pd_total == pytest.approx(4.0 / 7.0)


def test_explicit_tip_feature_map_matches_name_default():
    p = _problem()
    sol = _solution([True, False, True])
    default = compute_phylogenetic_diversity(p, sol, TREE)
    explicit = compute_phylogenetic_diversity(
        p, sol, TREE, tip_feature_map={"A": 1, "B": 2, "C": 3}
    )
    assert explicit.pd_represented == pytest.approx(default.pd_represented)


def test_to_dataframe_columns():
    res = compute_phylogenetic_diversity(_problem(), _solution([True, False, True]), TREE)
    df = res.to_dataframe()
    assert list(df.columns) == ["child_node", "length", "represented"]
    assert isinstance(res, PDResult)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/phylo/test_diversity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pymarxan.phylo.diversity'`.

- [ ] **Step 3: Implement the scorer**

Create `src/pymarxan/phylo/diversity.py`:

```python
"""Faith (1992) phylogenetic-diversity scoring of a solution."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pymarxan.models.problem import ConservationProblem
from pymarxan.phylo.tree import NodeId, PhylogeneticTree
from pymarxan.solvers.base import Solution


@dataclass
class PDResult:
    """Phylogenetic diversity a solution represents, with per-branch detail."""

    pd_represented: float
    pd_total: float
    pd_representable: float
    fraction_pd_total: float
    fraction_pd_representable: float
    n_tips: int
    n_tips_represented: int
    branch_child: list[NodeId]
    branch_length: list[float]
    branch_represented: list[bool]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "child_node": self.branch_child,
                "length": self.branch_length,
                "represented": self.branch_represented,
            }
        )


def _resolve_tips(
    problem: ConservationProblem,
    tree: PhylogeneticTree,
    tip_feature_map: dict[str, int] | None,
) -> dict[NodeId, int | None]:
    """Map each tree tip to a feature id (or None if it matches nothing)."""
    name_to_id = {
        str(n): int(i)
        for n, i in zip(problem.features["name"], problem.features["id"])
    }
    id_to_id = {str(i): int(i) for i in problem.features["id"]}
    resolved: dict[NodeId, int | None] = {}
    for tip in tree.tips:
        key = str(tip)
        if tip_feature_map is not None:
            resolved[tip] = tip_feature_map.get(key)
        elif key in name_to_id:
            resolved[tip] = name_to_id[key]
        elif key in id_to_id:
            resolved[tip] = id_to_id[key]
        else:
            resolved[tip] = None
    return resolved


def compute_phylogenetic_diversity(
    problem: ConservationProblem,
    solution: Solution,
    tree: PhylogeneticTree,
    *,
    tip_feature_map: dict[str, int] | None = None,
) -> PDResult:
    """Report the Faith PD a ``solution`` represents against ``tree``.

    A tip is *represented* if its feature occurs (amount > 0) in at least one
    selected PU. ``pd_represented`` sums branch lengths whose descendant tips
    intersect the represented set (rooted PD). ``pd_representable`` is the
    ceiling reachable given which tips occur in any PU at all.
    """
    resolved = _resolve_tips(problem, tree, tip_feature_map)

    selected_ids = {
        int(pid)
        for pid, sel in zip(problem.planning_units["id"], solution.selected)
        if sel
    }
    puvspr = problem.pu_vs_features
    present = puvspr[puvspr["amount"] > 0]
    features_in_reserve = {
        int(s) for s in present.loc[present["pu"].isin(selected_ids), "species"]
    }
    features_anywhere = {int(s) for s in present["species"]}

    represented_tips = {
        tip
        for tip, fid in resolved.items()
        if fid is not None and fid in features_in_reserve
    }
    representable_tips = {
        tip
        for tip, fid in resolved.items()
        if fid is not None and fid in features_anywhere
    }

    branch_child: list[NodeId] = []
    branch_length: list[float] = []
    branch_represented: list[bool] = []
    pd_represented = 0.0
    pd_representable = 0.0
    for child, length, desc in tree.branches:
        is_repr = bool(desc & represented_tips)
        branch_child.append(child)
        branch_length.append(length)
        branch_represented.append(is_repr)
        if is_repr:
            pd_represented += length
        if desc & representable_tips:
            pd_representable += length

    pd_total = tree.total_pd
    return PDResult(
        pd_represented=pd_represented,
        pd_total=pd_total,
        pd_representable=pd_representable,
        fraction_pd_total=(pd_represented / pd_total if pd_total > 0 else 0.0),
        fraction_pd_representable=(
            pd_represented / pd_representable if pd_representable > 0 else 0.0
        ),
        n_tips=tree.n_tips,
        n_tips_represented=len(represented_tips),
        branch_child=branch_child,
        branch_length=branch_length,
        branch_represented=branch_represented,
    )
```

Update `src/pymarxan/phylo/__init__.py`:

```python
"""Phylogenetic-diversity objectives for pymarxan.

Faith (1992) phylogenetic diversity via a branch-as-feature decomposition
(Rodrigues & Gaston 2002): each branch of a phylogeny becomes a synthetic
conservation feature weighted by its length, so the existing solvers maximize
PD directly. See ``docs/plans/2026-07-14-phylo-diversity-design.md``.
"""
from __future__ import annotations

from pymarxan.phylo.diversity import PDResult, compute_phylogenetic_diversity
from pymarxan.phylo.tree import PhylogeneticTree

__all__ = [
    "PhylogeneticTree",
    "PDResult",
    "compute_phylogenetic_diversity",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/phylo/test_diversity.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/pymarxan/phylo/diversity.py src/pymarxan/phylo/__init__.py tests/pymarxan/phylo/test_diversity.py
git commit -m "feat(phylo): compute_phylogenetic_diversity scoring + PDResult"
```

---

### Task 5: Branch decomposition (`phylogenetic_branch_problem`)

**Files:**
- Create: `src/pymarxan/phylo/decomposition.py`
- Modify: `src/pymarxan/phylo/__init__.py` (export)
- Test: `tests/pymarxan/phylo/test_decomposition.py`

**Interfaces:**
- Consumes: `PhylogeneticTree` (Task 1), `_resolve_tips` + `compute_phylogenetic_diversity` (Task 4), `MIPSolver(objective="max_weighted_features")` (Task 3), `ConservationProblem.copy_with`.
- Produces: `phylogenetic_branch_problem(problem, tree, *, target: float = 1.0, tip_feature_map: dict[str, int] | None = None) -> ConservationProblem`. Returned problem's features are entirely branch features (`id` = 0..k-1, `name` = `"branch:<child_node>"`, `target` = `target`, `spf` = branch length); `pu_vs_features` rows are presence (`amount == 1.0`) for PUs where any descendant tip's feature occurs. Branches with zero occurrence are dropped. PU columns, boundary, and `parameters` carry through. Branch provenance is recoverable from the `name` column (no separate return value).

- [ ] **Step 1: Write the failing tests**

Create `tests/pymarxan/phylo/test_decomposition.py`:

```python
"""Tests for the branch-as-feature decomposition (PD as an objective)."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.phylo.decomposition import phylogenetic_branch_problem
from pymarxan.phylo.diversity import compute_phylogenetic_diversity
from pymarxan.phylo.tree import PhylogeneticTree
from pymarxan.solvers.base import Solution, SolverConfig
from pymarxan.solvers.mip_solver import MIPSolver

TREE = PhylogeneticTree.from_newick("((A:1,B:1):2,C:3);")


def _problem(cost=(1.0, 1.0, 1.0)) -> ConservationProblem:
    planning_units = pd.DataFrame(
        {"id": [1, 2, 3], "cost": list(cost), "status": [0, 0, 0]}
    )
    features = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "target": [1.0, 1.0, 1.0],
            "spf": [1.0, 1.0, 1.0],
        }
    )
    pu_vs_features = pd.DataFrame(
        {"species": [1, 2, 3], "pu": [1, 2, 3], "amount": [1.0, 1.0, 1.0]}
    )
    return ConservationProblem(planning_units, features, pu_vs_features)


def _solution_from_pus(problem, selected_pu_ids) -> Solution:
    sel = np.array(
        [pid in selected_pu_ids for pid in problem.planning_units["id"]], dtype=bool
    )
    return Solution(selected=sel, cost=0.0, boundary=0.0, objective=0.0, targets_met={})


def test_branch_features_carry_length_as_spf():
    bp = phylogenetic_branch_problem(_problem(), TREE)
    # 4 branches, all representable (A, B, C each occur; internal via A/B).
    assert len(bp.features) == 4
    spf_by_name = dict(zip(bp.features["name"], bp.features["spf"]))
    assert spf_by_name["branch:A"] == pytest.approx(1.0)
    assert spf_by_name["branch:C"] == pytest.approx(3.0)
    # amounts are presence (1.0)
    assert set(bp.pu_vs_features["amount"].unique()) == {1.0}
    # branch feature ids are unique
    assert bp.features["id"].is_unique


def test_unrepresentable_branch_is_dropped_and_min_set_feasible():
    # Remove C from the data → C's pendant branch (length 3) is unrepresentable.
    p = _problem()
    p = p.copy_with(
        pu_vs_features=p.pu_vs_features[p.pu_vs_features["species"] != 3].reset_index(
            drop=True
        )
    )
    bp = phylogenetic_branch_problem(p, TREE)
    names = set(bp.features["name"])
    assert "branch:C" not in names  # dropped
    assert "branch:A" in names and "branch:B" in names
    # min_set stays feasible: cheapest reserve capturing the retained branches.
    sols = MIPSolver(objective="min_set").solve(bp, SolverConfig(num_solutions=1))
    assert len(sols) == 1
    assert sols[0].all_targets_met


def test_min_set_captures_full_representable_pd_and_matches_brute_force():
    p = _problem(cost=(1.0, 1.0, 5.0))  # C's PU is expensive
    bp = phylogenetic_branch_problem(p, TREE)
    sols = MIPSolver(objective="min_set").solve(bp, SolverConfig(num_solutions=1))
    sol = sols[0]
    # realized PD is 100% of representable
    pd_res = compute_phylogenetic_diversity(p, sol, TREE)
    assert pd_res.fraction_pd_representable == pytest.approx(1.0)
    # brute-force min-cost reserve covering every branch
    pu_ids = list(p.planning_units["id"])
    costs = dict(zip(p.planning_units["id"], p.planning_units["cost"]))
    best = None
    for r in range(1, len(pu_ids) + 1):
        for combo in itertools.combinations(pu_ids, r):
            s = _solution_from_pus(p, set(combo))
            if compute_phylogenetic_diversity(p, s, TREE).fraction_pd_representable == 1.0:
                c = sum(costs[i] for i in combo)
                if best is None or c < best:
                    best = c
    assert sol.cost == pytest.approx(best)


def test_max_weighted_features_under_budget_maximizes_pd_vs_brute_force():
    p = _problem()
    bp = phylogenetic_branch_problem(p, TREE)
    bp = bp.copy_with(parameters={**bp.parameters, "COSTBUDGET": 2.0})
    sol = MIPSolver(objective="max_weighted_features").solve(
        bp, SolverConfig(num_solutions=1)
    )[0]
    got_pd = compute_phylogenetic_diversity(p, sol, TREE).pd_represented
    # brute-force max PD achievable for total cost <= 2.0
    pu_ids = list(p.planning_units["id"])
    costs = dict(zip(p.planning_units["id"], p.planning_units["cost"]))
    best_pd = 0.0
    for r in range(0, len(pu_ids) + 1):
        for combo in itertools.combinations(pu_ids, r):
            if sum(costs[i] for i in combo) <= 2.0:
                s = _solution_from_pus(p, set(combo))
                best_pd = max(best_pd, compute_phylogenetic_diversity(p, s, TREE).pd_represented)
    assert got_pd == pytest.approx(best_pd)


def test_all_unrepresentable_yields_empty_feature_set():
    # A tree whose single tip matches no feature → no branches retained.
    tree = PhylogeneticTree.from_newick("(Z:1);")
    bp = phylogenetic_branch_problem(_problem(), tree)
    assert len(bp.features) == 0
    assert len(bp.pu_vs_features) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/phylo/test_decomposition.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pymarxan.phylo.decomposition'`.

- [ ] **Step 3: Implement the decomposition**

Create `src/pymarxan/phylo/decomposition.py`:

```python
"""Branch-as-feature decomposition: PD as a standard ConservationProblem."""
from __future__ import annotations

import pandas as pd

from pymarxan.models.problem import ConservationProblem
from pymarxan.phylo.diversity import _resolve_tips
from pymarxan.phylo.tree import PhylogeneticTree


def phylogenetic_branch_problem(
    problem: ConservationProblem,
    tree: PhylogeneticTree,
    *,
    target: float = 1.0,
    tip_feature_map: dict[str, int] | None = None,
) -> ConservationProblem:
    """Return a ConservationProblem whose features are the tree's branches.

    Each branch becomes a synthetic feature: presence amount (1.0) in every PU
    where any descendant tip's feature occurs, ``target`` occurrences required,
    and ``spf`` = branch length. Solving ``min_set`` on the result yields the
    cheapest reserve capturing 100% of representable PD; ``max_weighted_features``
    with a ``COSTBUDGET`` maximizes PD under budget. Branches whose descendant
    tips occur in no PU are dropped (they would make ``min_set`` infeasible).
    The original species features are entirely replaced. A branch's provenance
    is recoverable from its feature ``name`` (``"branch:<child_node>"``).
    """
    resolved = _resolve_tips(problem, tree, tip_feature_map)

    present = problem.pu_vs_features[problem.pu_vs_features["amount"] > 0]
    pus_by_feature: dict[int, set[int]] = {}
    for fid, pu in zip(present["species"], present["pu"]):
        pus_by_feature.setdefault(int(fid), set()).add(int(pu))

    feature_rows: list[dict] = []
    puvspr_rows: list[dict] = []
    next_id = 0
    for child, length, desc in tree.branches:
        pus: set[int] = set()
        for tip in desc:
            fid = resolved.get(tip)
            if fid is not None:
                pus |= pus_by_feature.get(fid, set())
        if not pus:
            continue  # unrepresentable branch — drop (keeps min_set feasible)
        bfid = next_id
        next_id += 1
        feature_rows.append(
            {
                "id": bfid,
                "name": f"branch:{child}",
                "target": float(target),
                "spf": float(length),
            }
        )
        for pu in sorted(pus):
            puvspr_rows.append({"species": bfid, "pu": pu, "amount": 1.0})

    branch_features = pd.DataFrame(
        feature_rows, columns=["id", "name", "target", "spf"]
    )
    branch_puvspr = pd.DataFrame(
        puvspr_rows, columns=["species", "pu", "amount"]
    )
    return problem.copy_with(
        features=branch_features, pu_vs_features=branch_puvspr
    )
```

Update `src/pymarxan/phylo/__init__.py` to export it:

```python
"""Phylogenetic-diversity objectives for pymarxan.

Faith (1992) phylogenetic diversity via a branch-as-feature decomposition
(Rodrigues & Gaston 2002): each branch of a phylogeny becomes a synthetic
conservation feature weighted by its length, so the existing solvers maximize
PD directly. See ``docs/plans/2026-07-14-phylo-diversity-design.md``.
"""
from __future__ import annotations

from pymarxan.phylo.decomposition import phylogenetic_branch_problem
from pymarxan.phylo.diversity import PDResult, compute_phylogenetic_diversity
from pymarxan.phylo.tree import PhylogeneticTree

__all__ = [
    "PhylogeneticTree",
    "PDResult",
    "compute_phylogenetic_diversity",
    "phylogenetic_branch_problem",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/phylo/test_decomposition.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/pymarxan/phylo/decomposition.py src/pymarxan/phylo/__init__.py tests/pymarxan/phylo/test_decomposition.py
git commit -m "feat(phylo): phylogenetic_branch_problem — PD via branch-as-feature decomposition"
```

---

### Task 6: CHANGELOG + full-suite green

**Files:**
- Modify: `CHANGELOG.md` (`## [Unreleased]` → `### Added`)

**Interfaces:** none (documentation + verification task).

- [ ] **Step 1: Add the CHANGELOG entry**

Under `## [Unreleased]` → `### Added` in `CHANGELOG.md` (create the `## [Unreleased]` / `### Added` headers if the section is currently empty):

```markdown
- **Phylogenetic-diversity objectives (`pymarxan.phylo`).** Faith (1992) PD via a
  branch-as-feature decomposition (Rodrigues & Gaston 2002): a pure-Python
  ``PhylogeneticTree`` (edge table or Newick, zero new deps),
  ``compute_phylogenetic_diversity`` scoring (rooted PD, total & representable
  fractions), and ``phylogenetic_branch_problem`` — each branch becomes a
  synthetic feature weighted by length so the existing solvers maximize PD
  (``min_set`` for the cheapest full-PD reserve; the new ``max_weighted_features``
  MIP objective for max PD under a ``COSTBUDGET``). ``max_features`` is
  unchanged. +27 tests.
```

(The +27 = 13 tree, 5 diversity, 5 decomposition, 4 objective. Confirm with
`/opt/micromamba/envs/shiny/bin/pytest tests/pymarxan/phylo tests/pymarxan/solvers/test_mip_objectives.py -q` and adjust if a test was split.)

- [ ] **Step 2: Run the full check**

Run: `source /opt/micromamba/etc/profile.d/micromamba.sh && micromamba activate shiny && make check`
Expected: `make check` green — 0 ruff, 0 mypy, full suite passes (previous count + ~24 new). If `test_solutions_are_different` fails, rerun once (known SA flake; see the `marxan-testing` skill).

- [ ] **Step 3: Run the marxan-parity-check skill on the solver edit**

Invoke the `marxan-parity-check` skill and confirm the `max_weighted_features` addition left `max_features` / `min_set` behavior and the known-exact ground truth (`tests/data/simple` optimum 35.0) intact.

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(phylo): CHANGELOG entry for phylogenetic-diversity objectives"
```

---

## Post-plan notes

- **Multi-agent design review:** Per CLAUDE.md, run the `multi-agent-design-review` skill on the design spec before executing this plan — the parity-sensitive `max_weighted_features` edit and the PD math are exactly what that review is for.
- **Not in this plan (deferred to a release task):** version bump / tagging (see the `release-pymarxan` skill), README feature blurb, and any Shiny panel (Python API only, like the rivers first cut).
- **Scientific citations:** Faith 1992 (`10.1016/0006-3207(92)91201-3`) and Rodrigues & Gaston 2002 (`10.1016/S0006-3207(01)00208-7`) verified via scite during the design phase.
