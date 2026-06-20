# Survey — AI-based & frontier conservation-planning methods for pymarxan

**Date:** 2026-06-20
**Question:** what else could pymarxan implement — e.g. CAPTAIN (RL for prioritization) and other newer methods?
**Method:** literature search + per-paper verification via the scite MCP (citation
tallies are scite Smart Citations; editorial-notice/retraction fields checked — **none of
the cited papers are retracted or carry concerns**). Complements the
2026-06-12 ecosystem survey (Zonation rank-removal, phylo, raster-grid PUs).

## TL;DR

The honest, high-value next features are **not** the AI frontier — they're methods that
reuse machinery pymarxan already owns (MIP backend, circuit-theory solve, greedy/Ferrier,
minimax-regret). CAPTAIN and deep-RL are real and citable but **experimental, single-group,
and heavy**; their headline "beats Marxan" claims are self-selected to dynamic objectives
Marxan never modelled. Recommend: cite the AI work in the ecosystem survey; build the
established gaps below.

## Already in pymarxan (do not re-propose)

Marxan-classic parity (penalty/target/HEURTYPE/ITIMPTYPE/RUNMODE), Marxan-with-Zones,
connectivity incl. **circuit-theory current-flow (grounded Laplacian)** + LCP resistance,
PROBMODE 1/2/3 (incl. Z-score chance constraints), clumping/TARGET2, separation distance,
importance/irreplaceability + selection frequency, MIP (CBC/HiGHS/Gurobi), equity/Gini,
30×30 representation, auto-targets, distribution smoothing, multi-scenario **minimax-regret**
robustness, portfolios (no-good cuts) + Pareto BLM, and the rivers/DCI + barrier-removal feature.

---

## Theme 1 — CAPTAIN (reinforcement learning)

**Silvestro, Goria, Sterner & Antonelli (2022), *Nature Sustainability*** — doi:10.1038/s41893-022-00851-6
(OA, 172 citing pubs, no retraction). RL where *state* = a spatially explicit, time-evolving
individual-based biodiversity simulator (revealed via a "monitor" action), *action* =
monitor+protect under budget (one-shot or staged), *reward* = the conservation objective
(e.g. minimised species loss); a small net maps state→action, trained gradient-free.
Open impl: `captain-project` (GitHub + PyPI), Zenodo 10.5281/zenodo.5643665.

**Skeptic's read:** the "beats Marxan" result is structurally self-selected — CAPTAIN
optimises a *dynamic, simulation-based extinction* outcome Marxan doesn't model, wins biggest
in the temporal setting, disables Marxan's BLM "for comparability", and scores reward on the
same simulator that trained it. It does **not** show RL beating ILP/SA on the classic
minimum-set problem pymarxan solves. The irreducible cost is the biodiversity simulator (no
analogue in pymarxan's static-DataFrame `ConservationProblem`). **Verdict: cite as a paradigm
import; do not port.** A thin "sequential greedy-under-budget" is cheap but won't reproduce
the claimed gains.

## Theme 2 — Other AI/ML

- **Equihua, Beckmann & Seppelt (2024)** deep-RL for connectivity (IIC/PC) — doi:10.1111/2041-210x.14300.
  Most on-target; notable for *decoupling optimiser from index calc* (mirrors pymarxan's
  Solver/objective split). High effort (DRL/GPU/28 h). Experimental.
- **Lapeyrolerie et al. (2022)** deep-RL for conservation *decisions* — doi:10.1111/2041-210x.13954
  (foundational framing; about dynamic management, not site selection).
- **No credible GNN for the reserve-*selection* step exists.** GNN-SDM (Wu 2025,
  doi:10.1111/geb.70162) is at the upstream SDM stage. "Active learning for site selection"
  doesn't exist as a method. ML-SDM→prioritization is plumbing pymarxan already consumes,
  not an AI innovation.

## Theme 3 — Dynamic / multi-period reserve design (uncontested Python gap)

Pick *which sites and when* under irreversible loss + staged budgets. Exact SDP is
intractable; the literature always substitutes a heuristic/approximation.
- **Costello & Polasky (2004)** — doi:10.1016/j.reseneeco.2003.11.005 (224 cites). Their
  **informed-myopic heuristic** (loss-probability-weighted greedy) captures most of the value
  and is **very close to pymarxan's existing greedy/Ferrier code — lowest-effort build here.**
- **Snyder, Haight & ReVelle (2005)** — doi:10.1007/s10666-005-3799-1. Two-period 0–1
  **scenario MIP** (expected coverage under per-period budgets) — directly on the existing MIP
  backend; robust variant reuses `minimax_regret`.
- Lineage (cite, don't build): Tóth et al. 2011 (doi:10.1287/opre.1110.0961), Sabbadin 2007
  (doi:10.1016/j.ecolmodel.2006.07.036), Sheldon 2015 (doi:10.1613/jair.4679).
  **Hype flag:** Lin et al. 2022 "Multiperiod DP" (doi:10.3390/su14063266, 1 cite) is thin.

## Theme 4 — Climate-adaptive connectivity (best fit for code we own)

| Feature | Maturity | Effort | Verdict |
|---|---|---|---|
| **Climate velocity** (gradient + forward/backward) | Established | Low (raster grad + kdtree) | Build first — real Python gap; feeds Marxan as cost/feature/BLM |
| **Omniscape** (omnidirectional climate-flow) | Established | Medium (tile+sum the *existing* grounded-Laplacian solve, subtract flat-resistance null) | **Flagship — best leverage of existing circuit code** |
| **Refugia scoring → feature** | Established | Low (raster algebra; reuses targets/30×30) | Build (composes velocity + Omniscape) |

Verified: Burrows 2014 (doi:10.1038/nature12976), Hamann 2014 fwd/back velocity
(doi:10.1111/gcb.12736), Brito-Morales 2018 (doi:10.1016/j.tree.2018.03.009), Keppel 2015
refugia (doi:10.1890/140055), Anderson 2023 resilient-and-connected (doi:10.1073/pnas.2204434119),
Arafeh-Dalmau 2021 velocity-into-Marxan (doi:10.1111/2041-210x.13675), Schloss 2021
climate-Omniscape (doi:10.1002/eap.2468). Impls: `VoCC` (R), **Omniscape.jl / Circuitscape.jl**.
**Hype flags:** "AdvConnect"/"advective connectivity" is not a standard named method (just
Circuitscape advanced-mode used directionally — don't brand a feature with it);
"climate-smart framework" (Buenafe 2023) is a recipe, not a solver.

## Theme 5 — Robust/stochastic optimization beyond minimax-regret

Only three families have a real conservation foothold; **CVaR and Bertsimas–Sim are absent
from spatial conservation** despite sounding established.
1. **Chance constraints** — *largely already present* via PROBMODE (Polasky 2004
   doi:10.1111/j.0021-8901.2004.00905.x; McCarthy 2011 doi:10.1111/j.1461-0248.2011.01608.x).
   Extending to joint/scenario form = moderate, low hype.
2. **Two-stage stochastic programming + SAA** — best new value; reuses MIP + the minimax-regret
   scenario generator (Meir 2004 doi:10.1111/j.1461-0248.2004.00624.x). Mature; watch scenario blow-up.
3. **CVaR-of-shortfall** (Rockafellar–Uryasev LP form) — a *trivial linear add* and natural
   sibling of minimax-regret, **but zero conservation precedent → label experimental.**
4. **Bertsimas–Sim interval-robust targets** (doi:10.1287/opre.1030.0065) — MIP-clean, one Γ
   dial, **zero conservation precedent → experimental.**
5. **Info-gap** (Regan 2005 doi:10.1890/03-5419) is **contested** (Sniedovich 2012
   doi:10.1890/12-0262.1; Hayes 2013 doi:10.1111/2041-210x.12046) — at most a post-hoc
   robustness curve, with the critiques cited.

## Theme 6 — Multi-action / project prioritization (highest-value clean add)

Two distinct classes — keep separate:
- **Project prioritization (PPP/oppr)** — spatially *implicit* knapsack-with-shared-actions.
  **Joseph, Maloney & Possingham (2009)** doi:10.1111/j.1523-1739.2008.01124.x (598 cites,
  canonical cost-effectiveness ranking); **Hanson et al. (2019) `oppr`** doi:10.1111/2041-210X.13264
  — exact IP. **Highest-value / lowest-risk: a new objective + data model on the existing
  PuLP/HiGHS backend** (the max-over-projects + shared-action linking are the same AND-
  linearisation already done for rivers). Building greedy + backward-heuristic baselines
  reproduces oppr's "exact beats heuristic" story.
- **Multi-action spatial (prioriactions)** — spatially *explicit* ≈ Zones-with-one-zone-per-action
  (Salgado-Rojas/Álvarez-Miranda lineage: doi:10.1016/j.ecolmodel.2019.108901,
  doi:10.1111/2041-210X.14220). Moderate effort, in our wheelhouse.
- **ROI ranking is suboptimal under complementarity** (Carwardine 2014 doi:10.1111/cobi.12413)
  — implement only diminishing-returns benefit curves + a cost-effectiveness *diagnostic*;
  **do not market ranking as optimal** (mirrors our own "heuristic-below-exact is a bug" anchor).

---

## Recommended roadmap (effort × maturity × honesty)

**Tier 1 — build (low effort, established, real Python gap, reuses our code):**
1. **Project prioritization (oppr-equivalent)** — cleanest new MIP objective; ~1,200 citations
   of backing; no Python competitor.
2. **Climate velocity + refugia layers** — raster math feeding existing Marxan targets.
3. **Omniscape omnidirectional connectivity** — wraps our existing grounded-Laplacian solve.
4. **Costello–Polasky informed-myopic heuristic** + **Snyder–Haight–ReVelle two-stage MIP** —
   reuse greedy + MIP stacks; the temporal paradigm is an uncontested gap.

**Tier 2 — build, label experimental:** CVaR and Bertsimas–Sim robust targets (trivial MIP
adds, no conservation precedent).

**Cite/contextualize, don't port now:** CAPTAIN + deep-RL (experimental, single-group, heavy),
info-gap (contested), exact multi-period SDP (intractable).

**Avoid:** GNN-for-site-selection (doesn't exist), "AdvConnect" branding, "ML-SDM as AI
innovation" (plumbing), predatory-venue agentic-AI papers.

*(Also still open from the 2026-06-12 survey: Zonation-style rank-removal solver, phylogenetic-
diversity objectives, raster-grid PUs + exactextract.)*
