# Negative-weight features — multi-agent design review synthesis

**Date:** 2026-08-08 · **Run:** Workflow `wf_255edcca-6b7`, 4 lenses, ~536k tokens, 19 min.
**Verdicts: all four lenses `revise`.** Two science CRITICALs invalidate design premises; two
architect/grounding CRITICAL+HIGH defects sit in the code the revision deletes.
**Net effect: the phase gets SMALLER and more defensible** — ABF-only, kernel untouched.

## The decisive findings

| # | Sev | Lens | Finding | Resolution |
|---|-----|------|---------|-----------|
| 1 | **CRITICAL** | science | **CAZ + negative weights is structurally degenerate.** `max` is not a trade-off operator: a cell holding `q_benefit=0.1` and `q_threat=1e6` scores `+0.0100` — bitwise identical to a cell carrying no threat (verified by execution). The held-and-extant mask does not fix §1.1's silent inertness, it *relocates* it onto mixed cells — precisely where an opportunity-cost layer bites (Moilanen 2011's own case study uses near-ubiquitous agriculture/urban layers). Moilanen 2011 places CAZ in its multi-criterion Eq. 3 but every worked negative-feature mechanism is ABF, "which aggregates value by direct summation" — summation is *what makes the trade-off expressible*; CAZ has no `v()` to invert. | **Take the spec's own §9 fallback: `rule="caz"` + any negative weight raises `ValueError`, pointing at `rule="abf"`. Ship ABF-only.** Strictly cheaper: deletes the kernel split, the held-mask, the `-inf` floor, the extra `qd` chunk buffer, both CAZ tests, and design §3 entirely. Positive-weight CAZ becomes *untouched*, a stronger guarantee than bitwise-equivalence. |
| 2 | **CRITICAL** | science | **ABF's `1/R` marginal is directionally OPPOSITE to the paper cited as warrant.** Moilanen 2011 inverts the benefit function for negative features (`z_k = 1/0.25 = 4`) so they become *decreasingly* important to exclude once mostly excluded. Ours ∝ `1/R` grows unbounded: as `R` falls 1.0→0.01, ours runs 1→100, theirs 4.0→4e-6; removal orders differ materially on the same fixture. The existing "we already deviate from z≈0.25" defense fails because for *positive* features both forms increase as `R` falls (same direction, different steepness) — the deviation **changes kind at the sign boundary**. The grid-smoothing precedent (ship a documented variant) also fails: that was licensed by proven ranking-*inertness*; this is ranking-*active*. | **Honest-variant (science's option b) + roadmap entry for the faithful form.** Design §2's "ABF already correct" non-goal is deleted. Docs cite Moilanen 2011 for the *concept* only and state plainly that our proportional/remaining-sum marginal makes a negative feature *increasingly* urgent to exclude as it nears elimination, opposite to the paper's `z_k=4`, and that orders differ. The faithful fix — a per-feature benefit exponent — attaches to the docstring's pre-existing "concave power-benefit generalization is a future extension" deferral. |
| 3 | HIGH | science | **Cost division inverts at the sign boundary, and the plan pinned it as a contract.** `delta=-2` at costs 1/2/10 → `-2.0/-1.0/-0.2`: the costlier threat cell is removed *later*. The conservation reading is unambiguous — a cell that both carries a threat and is expensive to protect is the one to concede first. Moilanen 2011's stated purpose for negative features is to *replace* the single-cost divisor ("different costs can be accounted for in our framework"), not compose with it; the composition is unattested. `test_negative_weight_cost_division` enshrined the inversion. | **`use_cost=True` + any negative weight raises `ValueError`**, directing the user to express costs as negative features (the paper's own practice) or pass `use_cost=False`. Conservative and relaxable later; tightening after release would not be. The pinning test is replaced by a guard test. |
| 4 | **CRITICAL** | architect (grounding + re-design concur independently) | **The `-inf` sentinel collides with a legal `-inf` score.** `fac = w/Q_safe` overflows to `-inf` for `w<0` and tiny `Q`, making a genuine `-inf` term — the most urgent removal — which the floor `out[out == -inf] = 0.0` rewrites to *neutral*, inverting the order (proved: plan kernel `[2,1,3]` vs correct `[1,2,3]`; reachable at `w=-1e300`, no subnormal needed). Same class as the grid-smoothing FFT-dust finding: a sentinel stops being distinguishable from data once signs are admitted. | **Moot under #1** — the CAZ kernel branch is deleted. Recorded because it is the reason a value-sentinel must never be used for masking in this engine; if the faithful-ABF phase ever adds a masked reduction, use an explicit participation mask (`part = qd != 0; part[:, dead] = False; r[~part] = -inf; out[~part.any(axis=1)] = 0.0`). |
| 5 | HIGH | architect | **The masked write silently disabled the NaN no-progress guard** for CAZ+negative (`r[qd == 0] = -inf` overwrites exactly the entries that produce `0.0*inf → NaN`), making the guard rule- and sign-dependent while the docstring claims it flat. | **Moot under #1.** ABF keeps the guard: verified firing on the pinned subnormal fixture with negative weights. |
| 6 | HIGH | architect + grounding | **An existing test asserts the raise being deleted** — `test_negative_or_nan_weight_raises` (`test_rank_removal_scale.py:421-425`) expects `match="weights must be >= 0"`. The plan claimed the suite passes unmodified; it would not. | Plan now edits that test explicitly: the NaN half stays, the negative half becomes the new CAZ/`use_cost` guard assertions. |
| 7 | MEDIUM | science | **Performance curves invert their reading for negatively weighted features** — `Q/T` means "fraction retained" (higher better) for a biodiversity feature and "fraction of threat still inside" (lower better) for a negative one, with nothing in the output distinguishing them. A genuinely new failure mode: before this change the uniform reading was always valid. | Document in `rank_removal`'s docstring and on `ZonationResult.performance_curves`; record negatively-weighted feature ids in `ZonationSolver`'s metadata, following the `smoothed`/`smoothing_alpha` provenance precedent. |
| 8 | LOW | science | Negative terms are unbounded *below* as `Q → 0` (`Q=1e-9` → `-1e9`) — a magnitude regime the `RuntimeError` guard has only ever met from the positive side. | Noted in the spec; the guard test now covers ABF+negative on the near-extinction fixture. |

## Verified correct — do not re-litigate

- **Bitwise equivalence for `w >= 0`: CONFIRMED by execution** — 388 comparisons (removal_order +
  priority_rank + raw curve float bytes) across every fixture family, plus a swap-in run of the real
  suite (137 passed). *Now moot in the stronger direction: the revision doesn't touch the kernel at all.*
- **All four hand-computed fixtures were arithmetically correct** (`caz [4,3,2,1]`, `abf [2,3,4,1]`,
  zero-weight `[4,2,3,1]`, cost-division `[2,3,1]`), and the discrimination claim held — the shipped
  CAZ formula really gives `[2,3,4,1]`. The ABF fixture survives into the revision; the CAZ ones go.
- **Heap bypass is right** (science + grounding): with `w<0` the term is strictly *decreasing* under
  removal (`-0.2,-0.4,-1.0,-2.0,-4.0`), cached keys become upper bounds, and Minoux-1978 exactness
  genuinely fails; batch selection is unaffected because its dirty set tracks changed *inputs*.
  The `heapq.heapify` probe works (0 calls with negative weights; positive weights reach it).
- **Direction is right**: lower delta = earlier removal matches Moilanen 2011's "rapid removal
  outside conservation"; the "Zonation v3+" attribution is substantiated by the paper's own text.
- **Nothing downstream assumes `delta >= 0`** — it never escapes `rank_removal`'s local scope.
- **Warning granularity is a non-issue**: Python's default filter shows it once per call site per
  process (measured: 5 solves → 1 warning), so `ZonationSolver`'s `warp=1` default does not mean
  every solve warns, and a module-level once-flag would be strictly worse.

## Revised shape

Two tasks, no kernel edit: (1) three validation guards — drop the negative-weight raise, add the
CAZ raise and the `use_cost` raise — plus heap routing, the warning helper, and ABF semantic tests;
(2) honest-variant docs, the curve-reading note, the solver metadata marker, CHANGELOG, gate.

## Outcome

Spec and plan rewritten in place (same commit as this doc). References (scite-verified,
unretracted): Moilanen et al. 2011, doi:10.1890/10-1865.1; Moilanen et al. 2005,
doi:10.1098/rspb.2005.3164; Moilanen 2007, doi:10.1016/j.biocon.2006.09.008.
