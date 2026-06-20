---
name: marxan-parity-check
description: >
  Validate that a pymarxan solver or I/O change still matches Marxan-classic
  behavior and the known-exact ground truth. Use whenever you touch a solver
  (MIP, simulated annealing, heuristic, zone solvers), the objective/penalty
  math, target resolution, or the Marxan file readers/writers — and before
  claiming "parity", "matches Marxan", or "correct". Also use when the user
  asks to compare against the Marxan C++ binary. Catches the classic failure
  mode of a heuristic landing *below* the exact optimum (an infeasible plan
  reported as a win).
---

# Marxan parity & correctness checks

Correctness for the minimum-set problem is anchored on a fact, not a vibe: on
the bundled six-unit project the exact optimum is reserve `{2, 4, 6}` at cost
**35.0**, hand-verified in `docs/VALIDATION.md`. Every check here flows from
that anchor.

## The fast loop: run the harness

`examples/validate_marxan_parity.py` is the runnable companion to
`docs/VALIDATION.md`. It loads `tests/data/simple` (native Marxan format),
solves it with every engine, and asserts three independent properties. It is
deterministic and needs no network or external binary:

```bash
/opt/micromamba/envs/shiny/bin/python examples/validate_marxan_parity.py
```

It is also pinned by `tests/test_examples.py`, so it runs on every CI build —
if you broke parity, that test goes red.

## The three axes it checks (and what a failure means)

1. **Exact optimality as ground truth.** The MIP solver must return cost
   **35.0** on the simple project. If MIP drifts off 35.0, the objective,
   constraint build, or target resolution changed — investigate before
   anything else, because the other axes are measured against it.

2. **Feasibility across every engine.** MIP, simulated annealing (Kirkpatrick
   et al., 1983), and the greedy heuristic must each return a reserve meeting
   *all* representation targets. The expected pattern: heuristics land **at or
   above** the exact cost (SA ≈ 43, greedy ≈ 45). A heuristic reporting a cost
   **below 35.0 is a bug, not a better solver** — it means a target was
   silently violated. Treat any sub-optimum cost as a red flag.

3. **Format round-trip.** Writing the problem back to Marxan files and
   re-reading reproduces the same problem. Touch `pymarxan.io` readers/writers
   → this is the axis that guards existing Marxan projects.

## When you changed solver or objective math

After the change, in order:

1. Run the harness above — it's the cheapest signal.
2. Run the solver test files: `... pytest tests/ -q -k "solver or objective or
   penalty"` (and the parity/validation tests).
3. Reason about *why* the numbers moved. If the optimum or a heuristic cost
   changed, either the change is wrong, or the ground truth in
   `docs/VALIDATION.md` needs a deliberate, documented update — never silently
   edit the expected 35.0 to make a test pass.

Past parity bugs in this codebase came from subtle Marxan-source details
(hyperbolic vs linear penalty curve; PU-id vs amount-sorted greedy ordering;
MISSLEVEL application). When a cost shifts, suspect that class of detail first.

## Opt-in: compare against the Marxan C++ binary

A direct numerical comparison against the real Marxan binary is **intentionally
out of scope** for the self-contained harness. pymarxan ships
`MarxanBinarySolver`, which shells out to a `marxan` executable when one is on
`PATH`. Only go here when the user explicitly asks to compare against the C++
binary, and state clearly that it requires a compiled `marxan` on `PATH`.

## Don't claim parity without evidence

"Matches Marxan" / "still correct" is a verifiable claim — back it with the
harness output and the relevant test result, not assertion (see
`superpowers:verification-before-completion`). If you're editing the
citation-bearing prose in `docs/VALIDATION.md`, also run
`scientific-validation` on the claims.
