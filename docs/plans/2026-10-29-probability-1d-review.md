# Probability 1D (v0.47.0 + v0.48.0) — spec review synthesis

**Date:** 2026-10-29 · **Reviewed:** spec `2026-10-18-probability-1d-design.md` (four review-loop
rounds and a fresh-eyes final review) · Reference: Marxan 3 (`marxan3` branch) `probability.c`,
`input.c`, `marxan.c` · Raw findings and probes: session scratchpad `p1d/review/`.

## Spec review loop

| Round | Critical | Important | Minor | Headline |
|---|---|---|---|---|
| 1 | 3 | 11 | 8 | Float zero-variance threshold misclassifies; in-memory `ptarget` rename breaks code-built problems; Shiny slider rewrites PROBMODE; mode-3 SA step 200–330× mode 0; Z = 4 rule makes risky PUs preferable to safe ones |
| 2 | 0 | 6 | 8 | Running E/V not exact (NaN penalty when V ≤ 0 with nonzero > 0); cache has no selection state → `ChanceState` sidecar; modes 3–5 drop the PROBMODE 1 premium; `_probmode` cannot warn (31 calls per solve) |
| 3 | 0 | 6 | 4 | "Large cancellation" resync never fires on chains (error 5e-2 at amounts 1e8) → per-feature float error bound; commit must reuse the preview; diagnostics in `validate()` disable presolve; no common emit hook (12 sites) |
| 4 | 0 | 2 | 7 | Recompute must use the post-flip selection (pre-flip: errors 1–3.8); `run_sweep_parallel` drops `problem.probability` |
| Final (fresh eyes) | 0 | 8 | 14 | E-trigger fires on every touch for sparse-risk layouts → `64·eps·|E|` floor; `Solution` 1D fields; requested vs active terms; `PROBZEROVAR` in public signatures; dense matrices removed; `write_puvspr` must keep an all-1 `prob`; RUNMODE 1 message; Shiny via pure helpers |

## User decisions (2026-10-29)

- Zero variance: Marxan 3's Z = 4 rule by default, `PROBZEROVAR 1` opt-out.
- Fix the mode 3/4 SA speed now (incremental E/V).
- Run 1D and 2D together (PROBMODE 5), as Marxan does.
- Two releases: v0.47.0 = 2D chance-term core (A), v0.48.0 = Probability 1D (B).

## Marxan 3 facts established

- Score: E/V over selected PUs; total = cost + connection + penalty + probability1D +
  probability2D; zero variance → Z = 4; step-function penalty normalised by ptarget; no SPF.
- Not reproduced: `ChangeProbability1D/2D` index `spec[i]` instead of `spec[isp]`; `doChange`
  skips E/V updates for clumping species; running-sum zero-variance test.
- `probZUT`: Ibbetson polynomial, `Z_MAX = 5`, ≤ 1.05e-9 from `norm.sf` for |z| < 5.
- Activation from file columns alone (pu.dat `prob`, puvspr `prob` header); spec columns
  `ptarget1d` / `ptarget2d` only.
