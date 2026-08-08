# Grid-convolution smoothing — multi-agent design review synthesis

**Date:** 2026-08-08 · **Run:** Workflow `wf_f313e9e2-f7d`, 4 lenses, ~437k tokens, 10 min.
**Verdicts:** architect approve-with-fixes · grounding **approve** · science approve-with-fixes ·
re-design approve-with-fixes. No CRITICAL.
**Empirical core:** grounding executed the Task-1 math verbatim — 560-case grid-vs-vector sweep
(20 seeds × 5 alphas × 3 aspect ratios × hole masks), worst relative error **1.55e-15** vs the
rtol=1e-10 contract (~65,000× headroom); conservation to 3.7e-16 at truncate=1e-2; reach-threshold
margin ~11 orders of magnitude; 1M-cell column 0.35–0.61 s; Task-4 bench emulation 18.5 s vs the
120 s budget; Task-3 sparse assembly and the self-consistency oracle premise verified **bitwise**.
The re-design lens independently derived the identical architecture (same two-convolution math,
same normalization, same reach mask, same assembly) — and both it and the science lens
independently found the one HIGH.

## Accepted findings and resolutions

| # | Sev | Lens | Finding | Resolution |
|---|-----|------|---------|-----------|
| 1 | **HIGH*** | science + re-design (independent) | **§5 "nonnegative by construction" falsified under FFT**: oaconvolve emits negative dust INSIDE the reach mask — science reproduced at truncate ≤1e-13 (legal), re-design at the DEFAULT 1e-9 with high-dynamic-range inputs (1e12-amplitude blob + remote unit source → 3,570 negatives). Raw-amount validation runs pre-smoothing, so negatives enter the engine silently; a negative q_ij inverts the heap's monotonicity (removal then DECREASES scores) → silent heap≠batch divergence; the planned suite could never catch it (tiny grids produce no dust). | `np.maximum(smoothed, 0.0, out=smoothed)` after the reach masking (clipped mass ~FFT-noise scale, far inside the conservation rtol — verified by both lenses); §5 reworded "nonnegative by construction and clamped against FFT roundoff"; regression test with the high-dynamic-range construction asserting `(out >= 0).all()`. |
| 2 | MED | science | **Provenance: Zonation smooths UNNORMALIZED** (Moilanen et al. 2005 I_ij = Σ exp(−αd)·A, no normalizer; Zonation 5 likewise — Moilanen et al. 2022). pymarxan's mass-conserving per-source normalization (chosen in Phase C) is a deliberate deviation — ranking-inert wherever a source's truncated window fits inside the valid mask (CAZ/ABF scores are invariant to per-feature constant scaling), differing only near edges/holes. | Docs only: docstrings + spec state (a) Zonation's transform is raw accumulation, (b) ours is an edge-corrected mass-conserving variant, (c) the exact equivalence condition. No math change. Bonus: Moilanen et al. 2022 explicitly endorses kernel truncation for tiled FFT — cite it as legitimizing our window truncation. |
| 3 | MED | architect + grounding | **Bench sparsity claim fails at its own parameters**: alpha=0.5/truncate=1e-9 gives a radius-42 window → smoothed nnz = 36% of dense, not "≪"; §7.7's promised nnz assert was silently dropped from Task 4. | Bench re-parameterized alpha=2.0 (radius ~11) + assert `nnz < 0.25 * n_pu * n_feat`; §7.7 amended. |
| 4 | MED | architect (+re-design LOW) | Design/plan contradiction: §4/§8 promise "kernel+Z once per apply" but the dispatch calls apply() per column (recomputing kernel/Z — actual overhead ~1.5× in convolutions, mislabeled "2×"). The per-column call is RIGHT (memory flatness); the docs disagreed with the code. | §4/§8 + plan note amended: deliberate per-column recompute, ~1.5×, kernel-cache deferred until profiling says otherwise; multi-column form documented as materializing dense (loop columns at raster scale). |
| 5 | LOW | architect | Negated-isinstance cap guard is a trap for future spec types (a third spec silently inherits the dense cap + dense apply). | Positive isinstance dispatch for both specs + final `else: raise TypeError(...)`; rejected spec-polymorphic alternative recorded in §4. |
| 6 | LOW | architect | Planned `test_vector_smoothing_cap_unchanged` duplicates the existing `test_smoothing_capped_at_vector_scale`. | Dropped; the existing test gains one assertion pinning the new message's "GridSmoothingSpec" redirect. |
| 7 | LOW | science + grounding (independent) | Pre-existing `smooth_distribution` docstring states the normalization direction BACKWARDS ("destination incoming"; the code is source-outgoing — verified numerically: per-destination normalization differs by 0.19 and violates conservation). | One-line drive-by fix in Task 1 (file already touched); grid docstring keeps "source-normalised". |
| 8 | LOW | science | Alpha convention verified: α = 2/mean-dispersal-distance is the 2-D kernel's mean-dispersal identity and real Zonation practice (Westwood et al. 2020 uses exactly 2/d), though Moilanen 2005 calibrated differently — document as convention, not definition, with the inverse-CRS-units caveat. | Docstring sentence added with DOIs; §1's "science review to verify" resolved. |
| 9 | LOW | architect (polish) + grounding | Cap-message redirect dead-ends gridless users; TYPE_CHECKING hedge moot (plain import, no cycle — precedent connectivity/features.py); `eq=False` needs a rationale word or removal; §9 export phrasing invites inventing a top-level export (none exists for siblings); stale line anchors (real: rank_removal.py:170-184); Task-1 "9 tests" collects 10 items. | All folded: message gains "(problems constructed with grid=GridGeometry(...))"; plain import; `eq=False` dropped for GridSmoothingSpec (two float fields — a working __eq__ is a small win); §9 reworded; plan anchors/counts corrected. |
| 10 | LOW | science (optional) | The truncated-kernel explicit-oracle comparison (two-convolution result vs an independently built window-bounded column-normalized kernel matrix) is a stronger pin than full-window-only. | Added to §7/Task-1 tests (both verifying lenses already built it; worst error 6.3e-16). |

## Key confirmations (verified by execution — do not re-litigate)

- Math: two-convolution ≡ explicit truncated column-normalized kernel (6.3e-16); full-window ≡
  vector oracle incl. holes + anisotropy (≤3.9e-16); conservation exact at any truncation; masked
  division NECESSARY (unmasked 0/0 spreads non-finite values — observed); window-bounded corner
  semantics make full-window equivalence exact-in-values; reach>0.5 threshold has ~11 orders margin.
- Normalization direction settled NUMERICALLY: the oracle is per-SOURCE outgoing (design §10's open
  question closed); symmetric kernels make the normalizer values coincide, which is why the sibling
  docstring bug was prose-only.
- Integration: ZonationSolver kwargs/metadata confirmed (`smoothed`/`smoothing_alpha`, `.alpha`
  duck-types); bench `_grid_problem` lacks `grid=` (plan's with_grid note required); scipy 1.17.1
  has oaconvolve; exports live in `zonation/__init__` only; no models↔connectivity cycle; pandas
  pvf round-trip preserves float64 bitwise (self-consistency oracle sound); CSC assembly ==
  dense reference bitwise.
- Perf: 1M-cell column 0.35–0.61 s; 20 columns 6.3 s; bench emulation 18.5 s vs 120 s budget;
  the 60k no-cap test ~3 s.
- References (scite-verified, unretracted): Moilanen et al. 2005 (10.1098/rspb.2005.3164);
  Moilanen et al. 2022 MEE Zonation 5 (10.1111/2041-210X.13819); Lehtomäki & Moilanen 2013;
  Westwood et al. 2020 (10.3390/d12020061); Jung et al. 2024 (10.3897/arphapreprints.e138574).

## Outcome

Spec + plan patched in place (same commit as this doc): the FFT-dust clamp (#1), provenance +
alpha-convention docs (#2/#8), bench re-parameterization + nnz assert (#3), per-column framing
(#4), positive dispatch (#5), test dedup (#6), sibling docstring fix (#7), polish (#9), extra
oracle test (#10). Execution may proceed.
