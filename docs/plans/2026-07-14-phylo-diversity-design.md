# Phylogenetic-diversity objectives — design

**Date:** 2026-07-14
**Status:** Approved (brainstorm), pending implementation plan + multi-agent review
**Subpackage:** new top-level `pymarxan.phylo`

## Motivation

Phylogenetic diversity (PD; Faith 1992, *Biol. Conserv.*
[doi:10.1016/0006-3207(92)91201-3](https://doi.org/10.1016/0006-3207(92)91201-3))
measures the total evolutionary history captured by a set of taxa: the sum of
branch lengths of the minimal sub-tree of a phylogeny that spans them. Reserve
selection that maximizes PD protects more of the "tree of life" than
species-count targets alone. This is a standing gap identified in the
2026-06-12 ecosystem survey (prioritizr has `add_max_phylo_div_objective`;
pymarxan had no phylogenetic support) and is self-contained: it rides entirely
on the existing solver stack via a **branch-as-feature decomposition**
(Rodrigues & Gaston 2002, *Biol. Conserv.*
[doi:10.1016/S0006-3207(01)00208-7](https://doi.org/10.1016/S0006-3207(01)00208-7)).

## Key insight — PD is optimizable with the solvers we already have

Decompose the phylogeny so each **branch** becomes a synthetic conservation
feature whose distribution across planning units (PUs) is the union of its
descendant tips' occurrences, weighted by branch length. Maximizing weighted
branch representation then equals maximizing PD. No new solver is required —
the decomposition produces a standard `ConservationProblem` that any existing
objective (`min_set`, `max_features`) consumes.

## Scope (approved)

Both halves of "PD objectives":

1. **Scoring** — `compute_phylogenetic_diversity(problem, solution, tree)` reports
   the PD a given solution represents.
2. **Optimization** — `phylogenetic_branch_problem(problem, tree)` produces a
   branch-decomposed `ConservationProblem` so existing solvers maximize PD or
   find the cheapest full-PD reserve.

Tree input: pure-Python `PhylogeneticTree`, built from an edge table **or** a
hand-written Newick parser. Zero new dependencies (matches the repo's
pure-NumPy / lean-deps ethos; networkx was deliberately avoided for rivers).

## Module layout

```
src/pymarxan/phylo/
  __init__.py          # exports PhylogeneticTree, phylogenetic_branch_problem,
                       # compute_phylogenetic_diversity, PDResult
  tree.py              # PhylogeneticTree
  decomposition.py     # phylogenetic_branch_problem(...)
  diversity.py         # compute_phylogenetic_diversity(...) + PDResult
tests/pymarxan/phylo/
  test_tree.py
  test_decomposition.py
  test_diversity.py
```

Plus one small edit to `src/pymarxan/solvers/mip_solver.py` (weighted
`max_features`, §5).

## Component 1 — `PhylogeneticTree` (`tree.py`)

Pure-Python rooted tree, no dependency.

**Constructors**
- `PhylogeneticTree.from_edges(edges)` — `edges` is an iterable of
  `(parent, child, length)` triples. Node ids may be `int` or `str`.
- `PhylogeneticTree.from_newick(text)` — hand-written recursive-descent parser
  for standard Newick with branch lengths, e.g. `"((A:1,B:1):2,C:3);"`.
  Named tips keep their labels; internal/anonymous nodes get generated ids.

**Internal representation**
- List of nodes; per-branch length keyed by the child node (each non-root node
  has exactly one parent branch).
- **Cached** `descendant_tips[node] -> frozenset[tip_label]` computed once by
  post-order traversal (plain sets; no networkx).
- Tips = nodes that are never a parent (leaves).

**API**
- `tips -> list[label]`, `n_tips -> int`
- `branches -> list[(child_node, length, frozenset_of_descendant_tips)]`
- `total_pd -> float` = Σ all branch lengths (rooted Faith PD of the whole tree)
- `validate() -> list[str]` — returns errors, does **not** raise (mirrors
  `ConservationProblem.validate()`): exactly one root (a node that is never a
  child), fully connected, acyclic, all branch lengths ≥ 0, ≥ 1 tip.

**Rooting convention:** rooted Faith PD — a branch is counted whenever its
descendant-tip set intersects the taxa of interest, which includes the deep
branches on the path to the root. This is Faith's original definition and what
prioritizr uses.

## Component 2 — scoring (`diversity.py`)

```python
compute_phylogenetic_diversity(
    problem: ConservationProblem,
    solution: Solution,
    tree: PhylogeneticTree,
    *,
    tip_feature_map: dict[str, int] | None = None,
) -> PDResult
```

**Tip → feature resolution.** `tip_feature_map` maps a tip label to a feature
id. If `None`: match tip labels against `features['name']`, then fall back to
`str(features['id'])`. Tip labels are compared as strings (node ids may be
`int` or `str`). Tips with no matching feature are reported (they can never be
represented) but do not raise.

**Algorithm.**
1. Determine the set of **represented tips**: a tip is represented iff its
   feature occurs (`amount > 0`) in at least one selected PU. (Reuse the same
   selected-PU → `pu_vs_features` filter as `compute_representation`.)
2. `pd_represented` = Σ branch lengths over branches whose `descendant_tips`
   intersect the represented-tip set.
3. `pd_representable` = Σ branch lengths over branches whose descendant tips
   intersect the set of tips whose feature occurs in **any** PU of the problem
   (the ceiling a perfect reserve could reach).
4. `pd_total` = `tree.total_pd`.
5. `fraction_pd = pd_represented / pd_representable` (0.0 if
   `pd_representable == 0`).

**`PDResult` dataclass** (with `to_dataframe()`, like `RepresentationResult`):
`pd_represented`, `pd_total`, `pd_representable`, `fraction_pd`,
`n_tips`, `n_tips_represented`, and a per-branch table
(`child_node`, `length`, `represented: bool`).

## Component 3 — the objective / decomposition (`decomposition.py`)

```python
phylogenetic_branch_problem(
    problem: ConservationProblem,
    tree: PhylogeneticTree,
    *,
    target: float = 1.0,
    tip_feature_map: dict[str, int] | None = None,
) -> ConservationProblem
```

Returns a **new** `ConservationProblem` (via `problem.copy_with(...)`, preserving
subclass type) whose feature set is **entirely replaced** by branch features —
the original species features and their `pu_vs_features` rows are dropped, so a
solve optimizes PD directly rather than double-counting species alongside
branches. The PU set, cost, status, boundary, and `parameters` carry through
unchanged. Each branch is a synthetic feature:

- **Feature id namespace.** Branch feature ids are assigned in a non-colliding
  range (e.g. `max(existing feature id) + 1 + branch_index`), recorded so the
  scoring/reporting can map back. Original PU set, cost, status, and boundary
  are carried through unchanged.
- **`pu_vs_features` amount** per (branch, PU) = **presence**: `1.0` if any
  descendant tip's feature occurs in that PU, else the row is omitted (sparse).
  Faith's unit — a branch is captured when a descendant lives in the PU.
- **`target`** = the `target` argument (default `1.0` → "represent every branch
  at least once", i.e. capture 100% of representable PD).
- **`spf`** = branch length (evolutionary distinctness drives the
  under-representation penalty; matters for SA / heuristic / `min_penalties`).
- **`name`** = a stable branch label (e.g. `"branch:<child_node>"`).

**Usage**
- `MIPSolver(objective='min_set').solve(branch_problem, cfg)` → cheapest reserve
  capturing 100% of representable PD.
- Set `branch_problem.parameters['COSTBUDGET']` and
  `MIPSolver(objective='max_features')` → maximize PD captured under budget.
- SA / greedy also consume it (spf-weighted penalties approximate PD).

**Semantics note (why `target` occurrences, not "% of total PD").** Every
captured branch contributes its full length to PD, so `min_set` with per-branch
targets inherently drives toward ~100% of representable PD — the well-posed
Marxan-native framing. A "cheapest reserve for ≥ X% of *total* PD" would require
a single aggregate knapsack constraint (Σ Lᵦ·yᵦ ≥ X·PD_total), which is the
budget / `max_features` path, not per-feature targets. Hence the API exposes an
absolute per-branch `target` (default 1), consistent with how Marxan targets
work, rather than a total-PD fraction.

## Component 4 — weighted `max_features` (parity-sensitive edit)

Currently `mip_solver.py` builds the `max_features` objective as `-Σ z_j`
(unweighted count of features whose target is met). Change it to `-Σ spf_j·z_j`.

- **Backward compatible:** `spf` defaults to `1.0` for every feature, so at the
  default the objective is byte-identical to today's `-Σ z_j`. Existing
  `max_features` tests and users are unaffected.
- **Effect:** with branch features carrying `spf = branch length`, the budget
  path maximizes captured **PD** (Σ length) rather than branch **count** — the
  correct max-PD-under-budget objective.
- **Parity:** because this touches solver objective math, run the
  `marxan-parity-check` skill and add a test asserting default-`spf` equivalence
  to the prior formulation.

## Testing strategy (TDD, hand-computed + brute-force oracles)

Follows the rivers pattern (brute-force oracle for optimization; hand-computed
values for metrics). Reference tree `((A:1,B:1):2,C:3);` (total PD = 7).

**`test_tree.py`**
- Newick parse → edges, per-branch descendant-tip sets, `total_pd == 7`.
- `from_edges` round-trips to the same structure.
- `validate()` flags: two roots, a cycle, a negative length, an empty tree.
- Zero-length branch and single-tip tree.

**`test_diversity.py`**
- Reserve covering `{A, C}` → `pd_represented == 6` (A:1 + internal:2 + C:3),
  B's branch (1) excluded; `fraction_pd == 6/7` when all tips are representable.
- Empty reserve → `pd_represented == 0`.
- A tip whose feature occurs in no PU → excluded from `pd_representable`;
  `fraction_pd` uses the representable ceiling.
- `tip_feature_map` explicit vs. name-match default agree.

**`test_decomposition.py`**
- Branch features have presence amounts and `spf == length`; ids don't collide
  with original feature ids.
- `min_set` MIP on a tiny tree selects a reserve whose realized PD (via
  `compute_phylogenetic_diversity`) is 100% of representable; MIP cost == brute
  force over all PU subsets.
- Weighted `max_features` + `COSTBUDGET` on a small instance: selected PD ==
  brute-force max PD achievable within budget.
- `max_features` with all `spf == 1` still equals the old `-Σ z_j` optimum
  (parity guard).

**Target:** ~20–25 new tests, `make check` green (0 ruff / 0 mypy), coverage
≥ 75%.

## Out of scope (YAGNI)

- Newick metadata beyond topology + branch lengths (bootstrap values, comments).
- Abundance-weighted PD, phylogenetic endemism, ED/EDGE scores — future adds.
- A Shiny panel — Python API only for this build (UI can follow, like rivers).
- Ultrametric / re-rooting utilities (accept the tree as rooted).

## References (to verify via scite during implementation)

- Faith, D. P. (1992). Conservation evaluation and phylogenetic diversity.
  *Biological Conservation*, 61(1), 1–10.
  https://doi.org/10.1016/0006-3207(92)91201-3
- Rodrigues, A. S. L., & Gaston, K. J. (2002). Maximising phylogenetic diversity
  in the selection of networks of conservation areas. *Biological Conservation*,
  105(1), 103–111. https://doi.org/10.1016/S0006-3207(01)00208-7
