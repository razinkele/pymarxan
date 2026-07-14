# Phylo-diversity design review — synthesis

**Date:** 2026-07-14
**Reviewed:** `2026-07-14-phylo-diversity-design.md` + `...-implementation.md`
**Method:** four parallel perspectives (architect, codebase-grounding, scientific-accuracy via scite, independent re-design).

## Verdict

**Green-light. No CRITICAL or HIGH findings from any lens.** The independent
re-design (built blind from requirements) converged on the plan's design
line-for-line, including its non-obvious choices — strong evidence they're
right:
- additive `max_weighted_features` (leave `max_features` untouched) — the
  parity-safe seam; overloading `spf` in `max_features` was correctly rejected.
- presence + `target=1` branch encoding (not abundance) — Faith-correct, and
  makes the MIP objective value equal the scored PD at the optimum.
- rooted Faith PD, dual fractions (total vs representable), drop unrepresentable
  branches (keeps `min_set` feasible).

Codebase-grounding confirmed every factual claim, including the exhaustive
`max_features` site inventory (exactly 3 sites, no hidden 4th) and exact line
numbers. Scientific-accuracy confirmed both DOIs resolve to the right,
non-retracted papers, the rooted-PD convention, the worked example (7 / 6), and
presence semantics.

## Findings absorbed into the plan

### MEDIUM — silent tip↔feature mapping failure (independent re-design)
`_resolve_tips` returns `None` for unmatched tips with no signal; a whole-tree
name mismatch (or accidentally scoring against the branch problem, whose tips
are all named `branch:*`) yields PD 0 — indistinguishable from an empty reserve.
**Fix:** add `n_tips_unresolved` to `PDResult`; `_resolve_tips` emits
`warnings.warn` when *every* tip is unresolved. (The convenience wrapper
`optimize_phylo_diversity` the re-designer also suggested is deferred — YAGNI;
the warning + field closes the silent-failure root cause.)

### MEDIUM — inherited `parameters`/`probability` carry-through (architect + independent)
`copy_with` carries `parameters` and the `probability` frame into the branch
problem. `BLM`/`boundary` is intentional (compact PD reserves). But an inherited
`PROBMODE`/`probability` frame references now-deleted feature ids and would run
wasted chance-constraint work (verified: no crash — guarded — but can populate a
misleading `Solution.prob_shortfalls`). **Fix:** pass `probability=None` in the
decomposition's `copy_with`, and document which inherited parameters affect the
PD solve.

### MEDIUM — `max_weighted_features` credits zero-occurrence features (architect + independent, LOW)
Inherited from `max_features`: `feat_met[fid]` is created for every feature but
only constrained when the feature has rows, so a zero-occurrence feature gets a
free binary the maximiser sets to 1. **Not triggered by the phylo path** (the
decomposition drops unrepresentable branches). **Fix:** document as an inherited
limitation in the new mode's docstring (callers must prune zero-occurrence
features); do NOT change `max_features` (parity risk the plan deliberately
avoids).

### MEDIUM — scientific wording (scientific-accuracy)
- "matching prioritizr's `add_max_phylo_div_objective`" overstates it —
  prioritizr adds a probabilistic persistence layer; ours is the deterministic
  special case. **Fix:** "analogous to the deterministic case of…".
- The PD-maximization identity (weighted objective == PD) holds **only at
  `target=1`**. **Fix:** state this in the design.

### MEDIUM — design doc names wrong test file (codebase-grounding)
Design §Testing says `test_mip.py`; the real file (and the implementation plan)
is `test_mip_objectives.py`. **Fix:** correct the design doc.

### LOW — absorbed
- Name the rooted-vs-unrooted PD distinction (rooted = picante/prioritizr
  default) in Component 1.
- Newick parser: collapse internal whitespace so multi-line Newick from real
  tools parses. Add a test.
- Construct the (possibly empty) branch DataFrames with explicit dtypes.
- One-line comment in `mip_solver.py` explaining why the mode is
  `max_weighted_features`, not the `objectives/` `MaxUtility` class; note
  `Solution.objective` is not the PD value for this mode (as for the other
  budgeted modes).

## Findings NOT absorbed (with reason)
- **`optimize_phylo_diversity` wrapper** — deferred (YAGNI); the primitives
  compose and the unresolved-tip warning removes the footgun's silence.
- **General zero-occurrence-feature guard in `max_features`** — parity risk;
  the plan's whole point is to not touch `max_features`. Documented instead.
- **Redundant internal-branch constraints in `min_set`** — informational; the
  unified single decomposition is more maintainable (a marginally larger MIP).
