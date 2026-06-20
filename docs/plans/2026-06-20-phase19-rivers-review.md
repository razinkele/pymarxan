# Phase 19 (rivers) — multi-agent design review

**Date:** 2026-06-20
**Reviews:** `docs/plans/2026-06-20-phase19-rivers-aquatic-restoration-design.md`
**Method:** four parallel review lenses (architect, codebase-grounding,
scientific-accuracy via scite, independent re-design), per the
`multi-agent-design-review` skill. DCI fixture arithmetic independently
re-verified by the synthesiser.

## Verdict

**The design is fundamentally sound — approve to implement after the HIGH
fixes below.** Strong convergence across lenses: the independent re-design
arrived at essentially the same architecture (downstream-pointer tree,
root-products + LCA DCI, AND-linearised binary-diadromous MIP, defer
potamodromous-exact, brute-force exactness check, rpy2-gated R validation).
Scientific accuracy: **every cited paper is real, not retracted, correctly
attributed** (Côté 2009; O'Hanley 2011; Kuby 2005; King & O'Hanley 2014;
Neeson 2015; Baldan 2022 / riverconn; the `dci` R package). The DCI formulas,
the `p^eff = p + (1−p)y` removal rule, and the MIP linearisation are all
mathematically correct, and the §5 hand fixtures (DCId 58.33, DCIp 61.11,
binary-MIP "pick B1→66.67 over B2→33.33") are arithmetically verified.

No CRITICAL findings.

## HIGH

- **H1 — `networkx` is an *optional* dependency, not core.** It lives only under
  `[project.optional-dependencies] shiny` (`pyproject.toml:37`); core modules
  that use it lazy-import inside functions (`connectivity/metrics.py:29`,
  `resistance.py:43`). §2's "already a dependency" is misleading. Decide: lazy-
  import in `rivers/network.py` (matches `connectivity/`) **or** promote
  `networkx` to a core dependency. *(blocks Phase A)*

- **H2 — "asymmetric (current-driven) connectivity" mischaracterises the repo.**
  `connectivity/circuit.py::current_flow_to_matrix` is **symmetric** (resistance
  distance / Circuitscape). The asymmetric facility is separate
  (`connectivity/io.py::connectivity_to_matrix(symmetric=False)` + degree/donor
  metrics). §1's out-of-scope justification for marine restoration leans on a
  capability described inaccurately — restate in correct terms.

- **H3 — "reuse `selection_freq.py` / `portfolio.py`" is pattern-reuse, not API
  reuse.** Both hard-couple to `Solution.selected` (length-`n_pu` bool) and
  `.objective` (`analysis/selection_freq.py:20`, `portfolio.py:15`).
  `BarrierSolution.removed` is a `set[int]`. Reword §2/§10 to "mirror the
  pattern"; the barrier-frequency count is ~10 trivial lines in `rivers/`.

- **H4 — barrier optimizers cannot join `SolverRegistry` / the Shiny picker.**
  `Solver.solve` is hard-typed to `ConservationProblem` (`base.py`), `registry`
  takes `type[Solver]`. Modelling barriers as a standalone
  `BarrierProblem`/optimizer family is the *right* call (different problem
  shape), but the doc must state the dual-solver-universe explicitly so nobody
  later files "river solvers don't show in the picker."

- **H5 — no confluence (Y-tree) test fixture.** The only hand-computed fixture
  is a linear chain where every LCA is a path endpoint, so the potamodromous
  `c_ij = c_i·c_j / c_lca²` is never exercised with a non-trivial LCA — an LCA
  bug would pass §5 silently. Add a ≥3-leaf Y network with a hand-computed DCIp
  whose pair LCA is an interior node. Single most important test addition.

## MEDIUM

- **M1 — extract the PuLP backend factory before Phase C.** `_available_backends()`
  / `_make_*` are module-private in `mip_solver.py:48,73`. Promote to a shared
  importable home (no behaviour change, keeps tests green) and have both
  `mip_solver` and `rivers` import it — otherwise it gets copy-pasted. Make this
  an explicit Phase C prerequisite, not a same-PR afterthought.

- **M2 — pin `dci_before` semantics.** `BarrierSolution` reports
  `dci_before/after/gain` but "before" is undefined (current passabilities vs
  fully-barriered) and it's unclear whether locked-in removals sit in the
  baseline or count as gain. Specify, or `gain` will differ across optimizers.

- **M3 — specify + test the *potamodromous* incremental SA delta.** §5 only
  details the diadromous incremental evaluator; the potamodromous delta
  (straddling-pairs = mass-above × mass-below of a flipped barrier) is the
  error-prone part. Add an SA-delta-vs-full-recompute equality test for it.

- **M4 — adopt `ProblemCache`'s precompute discipline, and label the evaluator a
  deliberate parallel.** Build `_root_products`, per-barrier→affected-segments
  index, and LCA tables as frozen fields once (cf. `cache.py:7-23`); never walk
  paths in the SA hot loop. There is no clean shared abstraction with
  `ProblemCache` (different domain) — a parallel impl is correct, just say so.

- **M5 — `SolverConfig` won't transfer wholesale.** It carries
  `num_solutions/seed/verbose/metadata` (`base.py:53`); only `seed`/`verbose`
  are meaningful. Reuse `cooling.CoolingSchedule` (cleanly reusable) + the
  `seed`/`verbose` convention; write a `rivers`-specific config/kwargs.

## LOW

- `selection_frequency()` (ndarray) is in `portfolio.py:15`;
  `compute_selection_frequency()`/`SelectionFrequency` is in `selection_freq.py`
  — the doc's module/function mapping is slightly imprecise (not fabricated).
- Match the NumPy-docstring majority; **don't** copy `connectivity/circuit.py`'s
  Google-style docstrings.
- **YAGNI:** consider shipping **symmetric-only** DCI in v0.5 and deferring the
  directional / round-trip math (open decision #2) until there's a validation
  target — keep the `pass_up/pass_down/pass_if_mitigated` columns, defer the
  directional *formula* as an unvalidated correctness liability.
- Optional extra exact engine: an O(n·budget) **tree-knapsack DP** gives an
  exact diadromous solver with zero MIP-backend dependency and a clean
  cross-check against the MIP.
- Specify cost units (integer vs float) for the budget knapsack / DP / brute
  force; add infeasible guards (budget too small → return baseline, all
  locked-out, zero-length segments) per this repo's review history.
- Scientific (all minor): cite Baldan 2022 (riverconn) for the directional /
  round-trip passability convention in §5; note in §7.3 that the probability-
  chain technique (King & O'Hanley 2014; Neeson 2015) is what generalises the
  binary AND-linearisation to the deferred partial-passability case; use "Côté"
  and pin one year for King & O'Hanley (2014 online / 2016 print) in the JOSS
  BibTeX.

## Non-issue flagged in review (recorded so it isn't "fixed" wrongly)

- The scientific lens claimed the §5 cross-term expansion should read
  `2(0.25)` in the middle. **It is already correct as written**
  (`3 + 2(0.5) + 2(0.25) + 2(0.5) = 5.5`): the 1-2 and 2-3 pairs are 0.5, the
  1-3 pair is 0.25. No change needed.

## Recommended sequence

1. Apply HIGH wording/scope fixes (H1–H4) + add the Y-tree fixture (H5) to the
   design doc.
2. Pin `dci_before` (M2) and the potamodromous-delta test (M3); fold M1 backend
   extraction into Phase C as a prerequisite.
3. Then Phase A (`network.py` + `dci.py` + tests) is directly implementable —
   TDD-first, validate DCI against the `dci` R package, and use the
   `marxan-parity-check` mindset (hand-verified fixtures as ground truth).
