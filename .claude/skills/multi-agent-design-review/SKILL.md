---
name: multi-agent-design-review
description: >
  Run this project's proven four-perspective design review before executing any
  non-trivial feature, phase, solver change, or refactor in pymarxan. Use when
  you (or the user) have a draft plan/spec and are about to start coding
  something substantive — a new solver, objective, connectivity method,
  Marxan-parity-sensitive change, or any multi-day phase — and want to catch
  parity bugs and design flaws before they cost an execute-fail-revise cycle.
  Also fire on "review this plan", "design review", "is this plan sound",
  "check before I build this". Skip for trivial one-file mechanical edits.
---

# Multi-agent design review

This pattern has repeatedly paid for itself in pymarxan: on Phase 18 alone,
~10 minutes of parallel agent time caught 3 Marxan-parity bugs and 4 mechanical
plan errors that would otherwise have burned 1–2 days of execute-fail-revise.
It's been reused for clumping (CLUMPTYPE bugs), separation distance (the
hyperbolic-vs-linear penalty bug), and the Tier-A/B features. Use it for any
non-trivial change before writing implementation code.

## When it's worth it

Worth it: a new solver or objective, penalty/target math, connectivity or
spatial methods, anything touching Marxan parity, or a phase spanning more than
a file or two. Not worth it: a rename, a docstring fix, a one-line guard.

## Inputs

You need a written draft first — a design and/or implementation plan (this repo
keeps them in `docs/plans/`, see `superpowers:writing-plans`). The review
critiques *that artifact*; don't run it on a vague idea.

## The four perspectives

Dispatch four subagents **in parallel, in one message** (see
`superpowers:dispatching-parallel-agents`). Each gets the draft plan plus
pointers to the relevant source. The four lenses, each deliberately distinct so
they don't collapse into the same critique:

1. **Architect** — does the design fit the existing three-layer architecture
   (`pymarxan` core / `pymarxan_shiny` / `pymarxan_app`), the solver
   abstractions, the `ProblemCache`/delta-computation performance model, and
   the established patterns? Where will it create friction or duplication?

2. **Codebase grounding** — verify every claim the plan makes about what
   exists. This repo's biggest waste is *re-proposing things already built*
   (e.g. `min_largest_shortfall`, `LinearConstraint`, `MinNeighborConstraint`,
   Ferrier importance were all found already present). The agent greps the tree
   and reports what's real, what's missing, and what the plan misremembers.
   Give it read tools and have it cite `file:line`.

3. **Scientific accuracy** — check the conservation-science claims and any
   citations against real literature using the `scientific-validation` skill /
   scite MCP. Marxan-parity math (penalty curves, Z-score chance constraints,
   target rules) must match the source method, and any cited paper must exist,
   not be retracted, and actually say what it's used to support.

4. **Independent re-design** — *without reading the plan's solution*, design the
   feature from the requirements alone, then diff against the plan. Divergences
   are where the plan made a non-obvious (possibly wrong) choice. This is the
   lens that catches "there was a simpler/correct way."

Use the `Explore`/`general-purpose` agent types as appropriate; the grounding
and scientific lenses especially benefit from real tool access.

## Synthesis

Collect the four reports into one synthesis doc in `docs/plans/`
(`YYYY-MM-DD-<phase>-review.md`), classifying findings by severity
(CRITICAL / HIGH / MEDIUM) and noting which are parity-critical. Then **patch
the design and implementation plans** to absorb the accepted findings — the
review is only valuable if the plan changes. Apply
`superpowers:receiving-code-review` judgment: verify each finding technically
rather than accepting or dismissing it reflexively.

## Then execute

Once the plan is patched, hand off to execution (TDD-first; see
`superpowers:subagent-driven-development` / `test-driven-development`) and
validate solver-touching work with the `marxan-parity-check` skill.
