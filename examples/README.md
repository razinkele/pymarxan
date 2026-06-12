# Examples

Runnable, self-contained scripts that carry a realistic scenario all the
way through pymarxan. Unlike [`docs/TUTORIAL.md`](../docs/TUTORIAL.md),
which introduces one API feature at a time on a toy problem, each example
here is an end-to-end workflow you can adapt to your own region.

Every example is deterministic (fixed RNG seed, analytic data) and needs
no network access, so they double as regression fixtures —
[`tests/test_examples.py`](../tests/test_examples.py) runs them on every CI
build.

## `baltic_marine_planning.py`

A south-eastern Baltic marine spatial planning scenario (the Curonian
Lagoon and the Klaipėda coastal shelf). It:

1. lays a 16×16 square planning grid over the region;
2. builds a cost surface (human-use pressure peaking at the port) and four
   conservation features (reedbeds, submerged macrophytes, a pikeperch
   spawning ground, migratory waterbirds);
3. locks in the two cells that are already protected;
4. targets 30 % of each feature's regional total (the "30×30" framing);
5. solves the minimum-set problem **exactly** (MIP) and **heuristically**
   (SA) and compares the two reserves;
6. runs a before/after **gap analysis** and ranks cells by
   **irreplaceability**.

Run it:

```bash
python examples/baltic_marine_planning.py
```

Things it demonstrates that the tutorial does not:

- **Exact vs. heuristic trade-off** — the SA reserve lands a little above
  the MIP optimum; on larger problems SA scales where the MIP does not.
- **SPF calibration** — the species penalty factor must be high enough for
  SA to bother meeting targets (the MIP enforces them as hard constraints
  regardless).
- **Irreplaceability is about scarcity** — widespread habitats leave the
  map flexible, but the single concentrated spawning ground is critical at
  every target level.
