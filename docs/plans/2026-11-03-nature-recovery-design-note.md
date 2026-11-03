# Nature recovery in pymarxan — design note (2026-11-03)

Status: **idea / roadmap input**, not a spec. Source reviewed: Baker, D. J., Gaston, K. J.,
Metcalfe, K., & Maclean, I. M. D. (2025). Systematic conservation planning for nature recovery.
*BioScience*, 75(6), 472–489. https://doi.org/10.1093/biosci/biaf030 (open access, CC-BY; no
editorial notices; verified through scite on 2026-11-03 — read via term-matched excerpts of Box 1,
Figures 2–3, the prioritisation sections and the discussion; Table 2's case studies were not read
in full).

## What the paper argues

A perspective, not a methods paper. Systematic conservation planning has mostly chosen *existing*
habitat to protect; the authors argue it should target *recovery actions* and reframe Pressey &
Bottrill's (2009) 11-stage framework (their Box 1) for that purpose. The differences that matter
for software:

1. **Several actions per planning unit.** "It may be possible to convert the same arable field to
   species rich grassland, woodland, or wetland, with each potential action benefiting a very
   different set of species … Marxan with Zones … can be used to prioritize management actions
   (or zones)." Exact methods (prioritizr) "can find optimal solutions and incorporate different
   management actions or zones."
2. **Restoration gaps** (Fig. 2c): restored habitat lags natural habitat in quality, so targets
   "require either more time or a larger area".
3. **Likelihood of success and funding risk.** "Unreliable funding effectively expands the
   restoration gap … Spatial prioritizations must evaluate the consequence of such uncertainty
   … exploring outcomes across a range of long-term funding scenarios."
4. **Conditional spatial effects** (Fig. 3): an action's contribution at unit A can depend on
   actions at unit B (dispersal), and actions can make neighbouring units occupiable (patch size,
   connectivity).
5. **Scenarios:** land-use, climate and socioeconomic scenarios should feed the prioritisation.
6. **Existing protected areas:** good-condition areas are locked in; poorly managed ones "might be
   considered among the opportunities for delivering nature recovery".
7. **Outputs are not prescriptive**; the last stage is stakeholder evaluation; training and access
   to the methods matter.

## Mapping to pymarxan

| Paper | pymarxan now | Opportunity |
|---|---|---|
| 1. Actions per unit | Zones: core + Shiny, exact (zone MIP) and heuristic | Present zones as recovery actions (action-specific cost and contribution); a template / wizard. |
| 2. Restoration gaps | `pymarxan.restoration` (`RestorationProblem`, MESH, optimiser) and `pymarxan.temporal` (`dynamic_reserve_greedy`, `two_stage_reserve_mip`) exist in the core — **not exposed in the Shiny app** | Expose both; discount a restoration action's contribution vs. existing habitat (contribution multiplier per action), optionally by time to maturity. |
| 3. Success / funding risk | Probability 1D (v0.48.0, in progress): per-unit loss probability, chance constraint | Interpret `prob` as P(action fails or loses funding) = 1 − P(action succeeds and stays funded). **Implemented on the local branch `proto/v0.48-b` (B6 help text, B7 manual paragraph citing this paper); ships with v0.48.0.** |
| 4. Conditional effects | Connectivity penalties, clumping (`target2`, minimum patch), separation, MESH | Partial coverage. True conditional contributions are nonlinear → heuristic solvers only. Research item. |
| 5. Scenarios | Scenario comparison, sweeps; the core already has multi-scenario robustness — `pymarxan.analysis.robustness` (`evaluate_plans_across_scenarios`, `minimax_regret`) — and `analysis.selection_freq.compute_selection_frequency` (used by the frequency map). **Robustness is not exposed in the Shiny app.** | Scenario robustness panel that wires the existing core API: plans × scenarios matrix, minimax-regret choice, selection frequency across scenarios, targets missed per scenario. No new backend logic. |
| 6. Existing areas | Status 2 (locked in) | "Existing, degraded → restorable" option: locked as a site, but eligible for a restoration action (zones). |
| 7. Stakeholders | Solution pools, frequency maps, help | Frame results as options (near-optimal alternatives, trade-offs); guided help per stage. |
| Box 1 framework | App tabs follow tools (Data, Configure, Probability, …) | Optional workflow view mapping tabs to stages 7 (targets), 8 (review existing areas), 9 (assessment), 10 (scheduling). |

## Recommended order

1. **"Nature recovery" workflow in the app** — zones as actions + expose `restoration` and
   `temporal`. Core already built; the sibling ROWER_pymarxan_DST project uses pymarxan as its
   restoration engine and is the natural first user. Needs a spec (brainstorm → spec → review).
2. **Probability 1D framing** — built on `proto/v0.48-b` (B6 help text, B7 manual paragraph citing the paper); ships with v0.48.0.
3. **Scenario robustness panel** — UI only, over `pymarxan.analysis.robustness` and `compute_selection_frequency`.
4. Research: conditional spatial contributions (Fig. 3) in SA / iterative improvement.

Caveat for any write-up: the paper motivates these features; it does not propose pymarxan's
models (e.g. the chance constraint is Marxan 3's).
