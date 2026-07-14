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
the decomposition produces a standard `ConservationProblem` that the existing
`min_set` objective consumes directly, plus one small additive
`max_weighted_features` mode (Component 4) for the budget path.

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

Plus one small additive edit to `src/pymarxan/solvers/mip_solver.py` (new
`max_weighted_features` objective mode, Component 4).

## Component 1 — `PhylogeneticTree` (`tree.py`)

Pure-Python rooted tree, no dependency.

**Constructors**
- `PhylogeneticTree.from_edges(edges)` — `edges` is an iterable of
  `(parent, child, length)` triples. Node ids may be `int` or `str`.
- `PhylogeneticTree.from_newick(text)` — hand-written recursive-descent parser
  for the Newick subset we need: nested `(...)` clades, named tips, optional
  `:length` on any node, optional names on internal nodes, terminating `;`.
  Named tips keep their labels; anonymous internal nodes get generated ids; a
  root `:length` (if present) is ignored. Bootstrap values, quoted labels, and
  NHX comments are out of scope (documented, raise a clear error if encountered).

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

**Rooting convention:** the standard rooted Faith PD — a branch is counted
whenever its descendant-tip set intersects the taxa of interest, which includes
the deep branches on the path to the root. This is Faith's original (1992)
definition; the tree is accepted as already rooted (no re-rooting, per YAGNI).

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
5. Report **two** fractions, so neither reading is ambiguous:
   - `fraction_pd_total = pd_represented / pd_total` — share of the whole tree of
     life captured (0.0 if `pd_total == 0`).
   - `fraction_pd_representable = pd_represented / pd_representable` — share of
     what this PU layer *could* capture (0.0 if `pd_representable == 0`). A
     `min_set` solution reaches `1.0` here even when some tips are absent from
     every PU — which is exactly why the total-based fraction is reported too.

**`PDResult` dataclass** (with `to_dataframe()`, like `RepresentationResult`):
`pd_represented`, `pd_total`, `pd_representable`, `fraction_pd_total`,
`fraction_pd_representable`, `n_tips`, `n_tips_represented`, and a per-branch
table (`child_node`, `length`, `represented: bool`).

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
branches. The PU set (ids, cost, status), boundary, and `parameters` carry
through unchanged. Each retained branch becomes a synthetic feature:

- **Feature id.** Branch features get fresh ids (branch index → stable id); since
  the original features are fully replaced, ids need only be unique among
  branches. A `branch_feature_id → child_node` map is returned/recorded so
  reporting can trace a feature back to its branch.
- **`pu_vs_features` amount** per (branch, PU) = **presence**: `1.0` if any
  descendant tip's feature occurs in that PU, else the row is omitted (sparse).
  Faith's unit — a branch is captured when a descendant lives in the PU.
- **`target`** = the `target` argument (default `1.0` → "represent every branch
  at least once", i.e. capture 100% of representable PD). Values `> 1.0` can make
  a sparse branch (occurring in fewer than `target` PUs) infeasible for
  `min_set`; the default 1.0 never triggers this, and the caveat is documented
  for callers who raise it.
- **`spf`** = branch length (drives the under-representation penalty in
  SA / heuristic / `min_penalties`, and is the weight for
  `max_weighted_features`, §4).
- **`name`** = a stable branch label (e.g. `"branch:<child_node>"`).

**Unrepresentable branches are excluded.** A branch whose descendant tips occur
in **no** PU has zero total occurrence and can never meet a positive `target` —
including it would make `min_set` infeasible. Such branches are dropped from the
decomposition; they contribute nothing to representable PD anyway, so the
retained branch set is exactly the `pd_representable` support from the scoring
(§2). The dropped count is recorded for reporting. (Zero-length but *occurring*
branches are kept: they're feasible and simply add 0 to PD / 0 weight.)

**Usage**
- `MIPSolver(objective='min_set').solve(branch_problem, cfg)` → cheapest reserve
  capturing 100% of representable PD.
- Set `branch_problem.parameters['COSTBUDGET']` and
  `MIPSolver(objective='max_weighted_features')` (§4) → maximize PD captured
  under budget.
- SA / greedy also consume it (spf-weighted penalties approximate PD).

**Semantics note (why `target` occurrences, not "% of total PD").** Every
captured branch contributes its full length to PD, so `min_set` with per-branch
targets inherently drives toward ~100% of representable PD — the well-posed
Marxan-native framing. A "cheapest reserve for ≥ X% of *total* PD" would require
a single aggregate knapsack constraint (Σ Lᵦ·yᵦ ≥ X·PD_total), which is the
budget / `max_weighted_features` path, not per-feature targets. Hence the API
exposes an
absolute per-branch `target` (default 1), consistent with how Marxan targets
work, rather than a total-PD fraction.

## Component 4 — new `max_weighted_features` objective mode (additive)

The budget path must maximize **PD** = Σ lengthᵦ·zᵦ (zᵦ = branch target met),
i.e. a *weighted* count of features whose target is met. Today's `max_features`
is the *unweighted* count `-Σ z_j` and does **not** read `spf`.

**Do not overload `max_features`.** Real Marxan problems routinely set
non-uniform, non-1.0 `spf` (it's the species penalty factor), and those users
rely on `max_features` being a pure count. Silently changing it to `-Σ spf_j·z_j`
would alter their results — a parity break disguised as "backward compatible"
(the claim only holds when every `spf == 1.0`, which real problems violate).

Instead **add a new objective mode** `max_weighted_features` to `mip_solver.py`,
alongside the existing four (extend the validated-objective tuple
`("min_set", "max_features", "min_largest_shortfall", "min_penalties")`):

- Objective `-Σ spf_j·z_j`, subject to `Σ costᵢ·xᵢ ≤ COSTBUDGET` (requires a
  `COSTBUDGET`, same as `max_features` / `min_largest_shortfall`).
- `max_features` is left **exactly as-is** — zero behavior change, zero parity
  risk to existing users.
- Branch problems use `objective='max_weighted_features'`; with `spf = branch
  length` this maximizes captured PD under budget, matching prioritizr's
  `add_max_phylo_div_objective`. (Note this weights the *target-met* indicator,
  distinct from prioritizr's `add_max_utility_objective`, which sums held
  amounts — hence the name `max_weighted_features`, not `max_utility`.)
- **Parity:** because this adds solver objective math, run the
  `marxan-parity-check` skill; add tests that (a) `max_weighted_features` with
  uniform `spf == 1` yields the same optimal *number* of features met as
  `max_features`, and (b) `max_features` output is byte-identical before/after
  (guard against accidental mutation).

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
  B's branch (1) excluded; with all tips representable both
  `fraction_pd_total` and `fraction_pd_representable == 6/7`.
- Empty reserve → `pd_represented == 0`, both fractions `0.0`.
- A tip whose feature occurs in no PU → excluded from `pd_representable`, so
  `fraction_pd_representable` can reach 1.0 while `fraction_pd_total < 1.0`
  (the two fractions diverge — the reason both are reported).
- `tip_feature_map` explicit vs. name-match default agree.

**`test_decomposition.py`**
- Branch features have presence amounts and `spf == length`; branch ids are
  unique; the returned `branch_feature_id → child_node` map is correct.
- A tree with an unrepresentable tip (feature in no PU) → its pendant branch is
  dropped; `min_set` stays **feasible** and the retained branches equal the
  `pd_representable` support.
- `min_set` MIP on a tiny tree selects a reserve whose realized PD (via
  `compute_phylogenetic_diversity`) is 100% of representable; MIP cost == brute
  force over all PU subsets.
- `max_weighted_features` + `COSTBUDGET` on a small instance: selected PD ==
  brute-force max PD achievable within budget.

**`test_mip.py` (Component 4, parity guards)**
- `max_weighted_features` with all `spf == 1` yields the same *number* of
  features met as `max_features` on the same instance/budget.
- `max_features` optimum is unchanged before/after the edit (no mutation).
- `max_weighted_features` without `COSTBUDGET` raises the same clear error as the
  other budget-requiring objectives.

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
