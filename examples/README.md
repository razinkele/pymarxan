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

## `curonian_spit_reedbeds.py`

Reed beds on the lagoon shore of the Curonian Spit: a narrow, **linear**
habitat whose biodiversity depends on staying **fragmented**, a mosaic of
cut and uncut stretches rather than a few long uniform runs. Classic Marxan
rewards compact reserves, so this example inverts it. It:

1. splits the shore from Smiltynė to the border south of Nida into 52
   one-kilometre segments, each touching only its neighbours;
2. gives each segment a reed area, a cutting cost (higher far from the
   villages that give road access) and two habitats: old-reed nesting
   habitat (counts only where the reed is left) and cut-reed mosaic habitat
   (counts only where it is cut);
3. models one cutting season as Marxan with Zones, with zones `cut` and
   `leave`, every segment in one of them, and a zone target per habitat;
4. rewards fragmentation with a **negative zone boundary cost** between
   `cut` and `leave`, so every cut/uncut edge earns a credit;
5. solves exactly (zone MIP) and sweeps the reward.

```bash
python examples/curonian_spit_reedbeds.py
```

Things it demonstrates that the tutorial does not:

- **Fragmentation as an objective** — a plain BLM can only penalise edges;
  a negative zone boundary cost rewards them. On the bundled data, without it
  the cheapest plan cuts four solid blocks (7 edges, a 15 km uncut stretch); with a reward of
  1 k€ per edge it becomes an interleaved mosaic (31 edges, longest uncut
  stretch 9 km) for about 13 k€ more cutting.
- **The reward is a calibration knob** — like BLM, the sweep shows what each
  extra edge costs, and the exact optimum makes edges and cost rise
  monotonically with the reward.
- **BLM 1 and unit boundary lengths** — zone boundary costs are multiplied by
  BLM and by the shared boundary length, as in Marxan with Zones. With BLM 1,
  every length 1 and every segment assigned, each cut/uncut edge earns exactly
  the reward.

## `validate_marxan_parity.py`

The runnable companion to [`docs/VALIDATION.md`](../docs/VALIDATION.md).
Loads the bundled six-unit project in **native Marxan format**, solves it
with every engine (exact MIP, simulated annealing, greedy), and checks
that the heuristics meet all targets and never beat the exact optimum
(35.0, verifiable by hand), then confirms the problem round-trips through
the Marxan file format unchanged.

```bash
python examples/validate_marxan_parity.py
```
