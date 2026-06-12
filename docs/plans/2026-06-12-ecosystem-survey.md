# Conservation-Planning Ecosystem Survey & Next-Roadmap Candidates

**Date:** 2026-06-12
**Trigger:** The 2026-05-16 realignment roadmap (Phases 18–25) is fully shipped through v0.4.1. That plan's gap list is exhausted, so a fresh survey is needed before picking the next direction.
**Method:** Five parallel research agents — four surveying slices of the ecosystem (the prioritizr/R world; the Marxan family + Zonation + other tools; the Python-native landscape; the 2024–2026 methods literature via scite) and one *grounding* agent that produced a file-cited inventory of what pymarxan actually implements. Every "gap" below was reconciled against that inventory, and the load-bearing literature citations were re-verified directly against scite.

## TL;DR

pymarxan has now **matched or exceeded the entire dormant Marxan family** (classic, MarZone, MarxanConnect, Zonae Cogito — all unmaintained) and reached **broad parity with `prioritizr`** on objectives, constraints, and exact solvers. Two findings reshape the next roadmap:

1. **The Python niche is genuinely uncontested.** No maintained Python package reimplements the Marxan family; the only credible alternatives (`prioritizr`, Zonation 5, `restoptr`) are R/C++/Julia. pymarxan's positioning claim holds — but the field's centre of gravity is R, so **adoption work (PyPI, conda-forge, the JOSS paper) matters as much as new features.**
2. **The one live competitor doing something pymarxan structurally cannot is Zonation 5** (actively maintained, v2.4 released 2025-09-16). Its *priority-rank-removal* paradigm (continuous priority maps) and its richer connectivity-kernel family are the highest-value genuinely-new capabilities. The broader methods frontier (circuit-theory connectivity, climate-robust "no-regrets" planning, equity objectives, 30×30 reporting) is well-evidenced in the 2024–2026 literature and mostly tractable.

The recommended next release is a **v0.5 "modern conservation planning" line**: a batch of low-effort, high-visibility additions (Tier A) shipped quickly, followed by one or two flagship capabilities (Tier B) that differentiate pymarxan from both the dormant Marxan family and the R incumbents.

## 1. Grounding — what is ALREADY present (do NOT re-implement)

The previous survey overstated three gaps because it didn't check the code. The grounding agent re-verified against the source tree; the following are **confirmed present** and were removed from the candidate list:

| Claimed gap | Actually present | Evidence |
|---|---|---|
| Linear side-constraints | `LinearConstraint` | `constraints/linear.py` |
| Neighbour constraints/penalties | `MinNeighborConstraint` | `constraints/neighbor.py` |
| Ferrier importance scores | `ferrier_importance` | `analysis/` |
| Raster *input* (zonal stats) | `apply_cost_from_raster`, `intersect_raster_features` (rasterio.mask) | `spatial/cost_surface.py`, `spatial/feature_intersection.py` |
| min-shortfall objective | `MinShortfall` (sum-shortfall) | `objectives/` |
| Probability (PROBMODE 1/2/3), eigenvector centrality, in/out-degree | all present (the three the *2026-05-16* survey wrongly flagged) | `solvers/probability.py`, `connectivity/metrics.py` |

Important nuance for the raster discussion below: pymarxan **can already ingest raster cost/feature layers as per-PU zonal statistics**, but planning units themselves are always *vector polygons* (`spatial/grid.py` returns a GeoDataFrame). The gap is the raster-grid-**as-planning-units** model and a streamlined extraction path — not raster ingestion per se.

## 2. Ecosystem snapshot (June 2026)

| Tool | Last meaningful activity | Status | Relevance |
|---|---|---|---|
| Marxan classic / MarZone | v4.0.6 (2021–22) | **Dormant** | Parity baseline — already exceeded |
| MarxanConnect / marxanconpy | 2020 | **Abandoned** | Connectivity algorithms worth mining |
| Zonae Cogito (Marxan GUI) | v853, 2021 | **Frozen** | Superseded |
| Vizzuality marxan-cloud (MaPP) | commits to 2026-04 | Maintained, slowing | SaaS wrapper, **not** an algorithm source |
| **prioritizr** (R) | v8.1.0 (2025-11-10); v9-dev | **Very active, dominant** | The competitor; parity target |
| **Zonation 5** | **v2.4 (2025-09-16)** | **Actively maintained** | Different paradigm — biggest gap source |
| `oppr` / `restoptr` / `raptr` (R) | 2025 | Active (Hanson lab) | Distinct problem types |
| Python packages (marxanconpy, spopt, pyspatialopt) | 2019 / active-but-generic / dead | — | **No direct Python competitor** |

**Implication:** the Marxan family won't compete. prioritizr and Zonation 5 are the living references. Within Python, pymarxan is alone — which is both the opportunity (own the niche) and the risk (a small field that defaults to R won't adopt unless pymarxan is frictionless and credible).

## 3. Candidate directions, ranked by impact-per-effort

### Tier A — quick wins (low effort, high visibility) → a v0.5 minor

| # | Capability | Source | Effort | Why |
|---|---|---|---|---|
| A1 | **30×30 / GBF target encoding + gap-vs-existing-PA reporting** | policy frontier [@robinson2024; @schloss2024] | Very low | Optimization already exists; this is a target-rule + reporting layer. Highest visibility ("Target 3" is everywhere). |
| A2 | **`min_largest_shortfall` (minimax) objective** | prioritizr | Low | Closes an objective-parity gap; `MinShortfall` (sum) exists, minimax does not. Small MILP add. |
| A3 | **Auto / group target-setting rules** (IUCN, area-based, log-linear interpolation) | prioritizr 8.1 `add_auto_targets`/`add_group_targets` | Low–med | Convenience layer over existing target handling. |
| A4 | **Multi-scenario "no-regrets" / min-regret robustness** | climate frontier [@liczner2024] (and the field's preference for overlap over info-gap) | Low | A thin wrapper over the existing MIP + `ScenarioSet` + selection-frequency overlap. Hot topic, cheap. |
| A5 | **Equity constraints + distributional (Gini) reporting** | equity frontier [@gopalakrishna2024] | Low | Min-benefit-per-group constraint + post-hoc distributional metric on existing solutions. Growing, cheap. |
| A6 | **Distribution smoothing** (dispersal-kernel convolution preprocessor) | Zonation 5 | Low–med | scipy convolution that turns feature layers into connectivity-aware inputs. High value per line. |

### Tier B — flagship new capabilities (medium effort, the differentiators)

| # | Capability | Source | Effort | Why |
|---|---|---|---|---|
| B1 | **Raster-grid-as-planning-units + streamlined `exactextract` ingestion** | prioritizr / Zonation | Medium | The single biggest functional disadvantage vs the incumbents. Raster zonal-stat input *partially exists* — this extends it to a raster PU model + a fast extraction path. |
| B2 | **Circuit-theory / current-flow connectivity** (Circuitscape-style sparse-linear solve) | connectivity frontier [@liczner2024] | Medium | The connectivity method "becoming standard"; pymarxan has graph metrics + least-cost-path but no current flow. Feasible with `scipy.sparse.linalg`. |
| B3 | **Phylogenetic-diversity objectives** (`max_phylo_div`, `max_phylo_end`) | prioritizr | Medium | Well-cited objective-parity gap; needs tree input + branch-length features. |
| B4 | **Zonation-style priority-rank-removal solver → continuous priority map** (CAZ / ABF) | Zonation 5 | High | The only *paradigm* pymarxan structurally lacks, and the differentiator vs the one live competitor. A flagship, not a quick win. |

### Tier C — large / strategic, scope carefully before committing

- **C1 Multi-action / project prioritization** (oppr-style: actions with cost + success probability, expected-persistence objectives). Strong trend [@gopalakrishna2024; @hanson2024], but a genuinely new decision model (action × site variables) — a major addition, not a feature.
- **C2 Restoration via landscape-index objectives** (effective mesh size, IIC; restoptr uses constraint programming). Different optimization paradigm.
- **C3 raptr-style space/representativeness targets** (securing environmental/genetic variation, not just amount). Distinctive but niche.
- **C4 Multi-period / dynamic scheduling.** Recognised gap but *low current demand* in the 2024–26 literature — defer.
- **C5 prioritizr v9 multi-objective framework** (hierarchical / reference-point / weighted-sum). Still unreleased and partly overlaps the existing Cohon Pareto filter — **watch, don't build yet.**

## 4. Explicitly out of scope

- **Cloud / multi-tenant deployment** (marxan-cloud's territory) — pymarxan is a library; deploy behind FastAPI if needed.
- **Reimplementing Circuitscape/Omniscape wholesale** — implement the *current-flow connectivity objective* (B2), not a general-purpose connectivity package.
- **eDNA-driven targets** — a data-input concern, not optimization.
- **Numba/Cython/OpenMP performance work** — the project's pure-NumPy decision stands; raster scale (B1) is the real performance lever.

## 5. Strategic note — features are necessary but not sufficient

Three of the five agents independently concluded the same thing: pymarxan's niche is real but **narrow because the field defaults to R**. Leadership is won by being frictionless enough that Python-first users don't fall back to `reticulate` + prioritizr. That means the feature roadmap must run *in parallel* with the adoption thread already underway this session:

1. **Get on PyPI** (then conda-forge) — currently the most urgent credibility gap; nothing is installable yet.
2. **Ship the JOSS paper** — both prioritizr and restoptr have papers; this is table stakes for citation.
3. **prioritizr / raster interoperability** — read prioritizr-style and raster inputs natively so pymarxan is a drop-in for existing workflows.
4. **Market the exact-MILP path**, not just SA fidelity — the headline modern result is that exact solvers beat Marxan's SA on cost and speed [@schuster2020], and pymarxan already has HiGHS/Gurobi/CBC.

## 6. Suggested sequencing

```
Ship the adoption thread  ──► PyPI + conda-forge + JOSS (in flight)
        │
Tier A batch (A1–A6)      ──► v0.5.0 ("modern conservation planning")
        │
Tier B picks: B1 (raster) + B2 (circuit connectivity)  ──► v0.6.0
        │
Flagship: B4 (Zonation-style priority-rank solver)     ──► v0.7.0 (headline differentiator)
        │
Re-survey before Tier C — multi-action/restoration are new problem types, not features
```

A defensible first move is **Tier A as a single v0.5 minor** (all low-effort, all extend existing machinery), shipped *after* PyPI/JOSS land, so the first installable release the wider community sees already speaks the modern vocabulary (30×30, equity, climate-robustness).

## 7. Acceptance criteria (unchanged from the realignment plan)

A direction is "done" when: new code has TDD-per-task tests; `make check` stays green (0 ruff/mypy, no skipped tests); any user-facing knob is exposed in the Shiny app; `CHANGELOG.md` gets an `[Unreleased]` entry; test count grows and coverage stays ≥ 90 %.

## References

Load-bearing literature was retrieved and editorial-checked via scite (June 2026); none retracted.

- Gopalakrishna, T., Visconti, P., & Lomax, G. (2024). Optimizing restoration: A holistic spatial approach to deliver Nature's Contributions to People with minimal tradeoffs and maximal equity. *PNAS, 121*(34), e2402970121. https://doi.org/10.1073/pnas.2402970121
- Hanson, J. O., Schuster, R., Strimas-Mackey, M., et al. (2024). Systematic conservation prioritization with the prioritizr R package. *Conservation Biology, 39*(1), e14376. https://doi.org/10.1111/cobi.14376
- Liczner, A., Pither, R., & Bennett, J. (2024). Advances and challenges in ecological connectivity science. *Ecology and Evolution, 14*(9), e70231. https://doi.org/10.1002/ece3.70231
- Robinson, J. G., LaBruna, D., & O'Brien, T. (2024). Scaling up area-based conservation to implement the Global Biodiversity Framework's 30×30 target. *PLoS Biology, 22*(5), e3002613. https://doi.org/10.1371/journal.pbio.3002613
- Schloss, C. A., Cameron, D. R., & Franklin, B. (2024). An approach to designing efficient implementation of 30×30 terrestrial conservation commitments. *Conservation Science and Practice, 6*(10), e13232. https://doi.org/10.1111/csp2.13232
- Schuster, R., Hanson, J. O., Strimas-Mackey, M., et al. (2020). Exact integer linear programming solvers outperform simulated annealing for solving conservation planning problems. *PeerJ, 8*, e9258. https://doi.org/10.7717/peerj.9258

### Survey sources (software)
prioritizr <https://prioritizr.net/> (v8.1.0, 2025-11-10) · oppr / restoptr / raptr (prioritizr org) · Zonation 5 <https://zonationteam.github.io/Zonation5/> (v2.4, 2025-09-16) · Marxan <https://github.com/Marxan-source-code/marxan> (v4.0.6, dormant) · marxan-cloud <https://github.com/Vizzuality/marxan-cloud> · spopt <https://pysal.org/spopt/> · exactextract <https://github.com/isciences/exactextract> · highspy <https://pypi.org/project/highspy/>
