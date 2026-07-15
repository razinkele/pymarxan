# Tier C — raptr space targets — scientific-accuracy review (front-loaded)

**Date:** 2026-07-15
**Lens:** Scientific-accuracy (scite + WebFetch of raptr source), run **before** writing-plans
because the space-held formula is the load-bearing crux of this research feature.

## What was verified

The reviewer characterized raptr's spatial-adequacy model directly from **raptr's source code**
(jeffreyhanson/raptr, commit `f67fa73` — `rcpp_generate_model_object.cpp`, `functions.cpp`,
`rcpp_proportion_held.cpp`, `DemandPoints.R`, `RapData.R`) — the authoritative implementation of
Hanson, Rhodes, Fuller & Possingham (2018), *Methods Ecol. Evol.* doi:10.1111/2041-210x.12862
(paper paywalled; source code is ground truth). Cross-checked against the package prose /
`urap.proportion.held` docs. Both DOIs (12862 + the PNAS surrogate 10.1073/pnas.1711009114) verified
real.

## Findings (folded into the design spec)

- **CRITICAL correction — normalization.** raptr's space held is **`1 − WSS/TSS`** (proportion of
  attribute-space variation captured), where `WSS = Σ_d w_d·min_{selected}‖pos−p_d‖²` and
  **`TSS = Σ_d w_d·‖p_d − c‖²`** with `c` = the **unweighted mean of the demand-point coordinates**.
  The spec's original `cost_worst = max_i cost({i})` denominator was **WRONG** — it inflates
  space_held so a numeric target is materially easier to meet than raptr's, and loses the
  variance-captured meaning. *Folded:* rewrote `compute_space_held` to `1 − WSS/TSS`.
- **`WSS` (numerator) — FAITHFUL.** The weighted-squared-distance-to-nearest-selected-PU is exactly
  raptr's WSS. No change.
- **Mechanism — deviation to document.** raptr enforces `space_held ≥ target` as a **hard constraint
  in an exact IP (Gurobi), no SPF/penalty**. pymarxan's SA/greedy re-cast as a soft shortfall
  penalty is a legitimate heuristic adaptation but **not raptr-exact**; `space_spf` is a pymarxan
  choice with no raptr analogue. *Folded:* labeled the mechanism note; exact hard-constraint form is
  the deferred MILP.
- **Demand points — deviation to document.** raptr's default is a **KDE-sample** (n=100,
  density-weighted); the spec's occupied-PU/amount-weighted discretization is a legitimate special
  case (raptr's vignette uses PU centroids) but shifts results where occupancy is uneven. *Folded:*
  documented as a deviation; KDE-sampled demand points noted as a future option.
- **Attribute space — FAITHFUL.** Geographic centroids default + per-dimension z-scoring (geographic
  axes included) matches raptr's `include.geographic.space=TRUE` + `scale=TRUE` defaults. No change.
- **Optimization** — raptr is exact IP (facility-location / p-median; a "reliable" probabilistic
  variant adds R-level backups). No heuristic in raptr. pymarxan's SA/greedy penalty is the
  adaptation; the exact p-median MILP is the deferred path.

## Consequence for the pipeline

The science is now settled, so the architect / grounding / independent-redesign lenses of the
design review focus on the **engineering** (the SA delta feasibility, greedy scoring, attribute-space
plumbing) rather than re-deriving the maths. The spec's `1 − WSS/TSS` is the formula the plan builds.
