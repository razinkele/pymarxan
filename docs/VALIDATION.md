# Validation

How do you trust a reimplementation of a 20-year-old conservation-planning
algorithm? This document records how `pymarxan` is checked for
correctness, what is demonstrated, and — just as importantly — what is
*not*.

The runnable harness behind everything below is
[`examples/validate_marxan_parity.py`](../examples/validate_marxan_parity.py),
pinned by `tests/test_examples.py` so it runs on every CI build:

```bash
python examples/validate_marxan_parity.py
```

## What "correct" means here

`pymarxan` solves the **minimum-set** conservation planning problem: pick
the set of planning units of least total cost whose pooled features all
meet their targets. Correctness is checked along three independent axes.

### 1. Exact optimality as ground truth

For the minimum-set objective, the mixed-integer-programming (MIP) solver
finds the **provably cost-optimal** reserve — on a small problem the exact
optimum is a hard fact, not an estimate. Any heuristic must land *at or
above* that cost and never below it; landing below would mean it had
violated a target.

The bundled six-unit project ([`tests/data/simple`](../tests/data/simple),
in native Marxan format) makes this concrete. Its optimum is the reserve
`{2, 4, 6}` at cost **35.0** — verifiable by hand:

| Feature   | Target | From PU 2 | PU 4 | PU 6 | Total |
|-----------|-------:|----------:|-----:|-----:|------:|
| species_a |     30 |        15 |   12 |   10 |    37 |
| species_b |     20 |        10 |    3 |    7 |    20 |
| species_c |     15 |         6 |    4 |    8 |    18 |

No cheaper combination meets all three targets. The harness reproduces
this and reports each engine against it:

| Solver               | Cost | Cells | Targets met | Gap vs exact |
|----------------------|-----:|------:|:-----------:|-------------:|
| MIP (exact)          | 35.0 |     3 |     yes     |       +0.0 % |
| Simulated annealing  | 43.0 |     3 |     yes     |      +22.9 % |
| Greedy heuristic     | 45.0 |     4 |     yes     |      +28.6 % |

This is exactly the pattern the conservation-planning literature reports:
exact integer-programming solvers produce strictly cheaper plans than
simulated annealing, which only approximates the optimum (Schuster et
al., 2020). `pymarxan` offers both, so a user can prototype with the
heuristic and finish with the exact solver.

### 2. Feasibility across every engine

Every solver pymarxan ships — exact MIP, native simulated annealing
(after Kirkpatrick et al., 1983, as used in Marxan), and the greedy
heuristic — must return a reserve that meets all representation targets.
The harness asserts this for each.

### 3. Native Marxan-format round-trip

`pymarxan` reads and writes the native Marxan files (`input.dat`,
`pu.dat`, `spec.dat`, `puvspr.dat`, `bound.dat`). The harness writes the
loaded problem back out, re-reads it, and confirms the planning units,
features, and total feature supply are reproduced — so an existing Marxan
project survives a `pymarxan` round-trip unchanged. This is what lets the
large existing body of Marxan projects and teaching material carry forward
into the Python toolchain.

## Comparing against the Marxan C++ binary

The strongest possible check is a numerical comparison against the
original Marxan executable. `pymarxan` is built for it: `MarxanBinarySolver`
shells out to a `marxan` binary when one is on `PATH`, reading and writing
the same files, so the two can be run on an identical project and their
outputs compared directly.

That comparison is **opt-in and out of scope for this self-contained
harness**, because it requires a compiled Marxan binary that CI cannot
assume. To run it yourself, put a `marxan` executable on `PATH` and solve
the same project with `MarxanBinarySolver`; because the heuristics are
stochastic, compare *objective values and target satisfaction* rather than
expecting an identical planning-unit set.

## Scope and honesty

This harness validates internal consistency (heuristics bounded by the
exact optimum), feasibility, and file-format compatibility. It does **not**
claim bit-identical reproduction of Marxan binary output — simulated
annealing is stochastic, and the two implementations differ in
random-number streams and cooling schedules. The parity-critical
algorithmic details (the hyperbolic penalty curve, the amount-sorted
greedy ordering, the separation-distance and clumping rules) are checked
by the unit-test suite, which encodes them directly from the Marxan source.

## References

- Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by
  simulated annealing. *Science, 220*(4598), 671–680.
  https://doi.org/10.1126/science.220.4598.671
- Schuster, R., Hanson, J. O., Strimas-Mackey, M., et al. (2020). Exact
  integer linear programming solvers outperform simulated annealing for
  solving conservation planning problems. *PeerJ, 8*, e9258.
  https://doi.org/10.7717/peerj.9258
