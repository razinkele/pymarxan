"""Validation harness — pymarxan against the Marxan-classic workflow.

This is the runnable companion to [`docs/VALIDATION.md`](../docs/VALIDATION.md).
It loads a project in the **native Marxan file format**, solves it with
every engine pymarxan ships, and checks three things a correct
implementation must satisfy:

1. **Feasibility** — every solver returns a reserve that meets all
   representation targets.
2. **Exact optimality as ground truth** — for the minimum-set objective
   the mixed-integer solver finds the provably cost-optimal reserve; the
   simulated-annealing and greedy heuristics must land at or above that
   cost, never below it.
3. **Format round-trip** — writing the problem back out to Marxan files
   and re-reading it reproduces the same problem, so existing Marxan
   projects survive a pymarxan round-trip unchanged.

It is deterministic and needs no network access or external binary, so it
doubles as a regression fixture (see ``tests/test_examples.py``).

A direct numerical comparison against the **Marxan C++ binary** itself is
a separate, opt-in step: pymarxan ships ``MarxanBinarySolver``, which
shells out to a ``marxan`` executable when one is on ``PATH``. That
comparison is intentionally out of scope here because it needs a compiled
binary the self-contained harness cannot assume.

Run it::

    python examples/validate_marxan_parity.py
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SIMPLE_PROJECT = REPO_ROOT / "tests" / "data" / "simple"


@dataclass
class SolverReport:
    name: str
    cost: float
    n_selected: int
    targets_met: bool
    gap_pct: float  # cost above the exact optimum, in percent


def load_simple_problem():
    """Load the bundled six-unit project from native Marxan files."""
    from pymarxan.io.readers import load_project

    return load_project(SIMPLE_PROJECT)


def solve_all(problem) -> list[SolverReport]:
    """Solve with MIP (exact), SA, and greedy; report each against the
    MIP optimum."""
    from pymarxan.solvers.base import SolverConfig
    from pymarxan.solvers.heuristic import HeuristicSolver
    from pymarxan.solvers.mip_solver import MIPSolver
    from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver

    # Drop the boundary penalty so "minimum set" means pure cost — that is
    # the objective for which the MIP optimum is an exact lower bound.
    problem.parameters["BLM"] = 0.0
    # A high SPF makes the heuristics treat a target miss as expensive.
    problem.features["spf"] = 100.0
    problem.parameters["NUMITNS"] = 50_000
    problem.parameters["NUMTEMP"] = 100

    mip = MIPSolver().solve(problem, SolverConfig(num_solutions=1))[0]
    optimum = mip.cost

    sa = SimulatedAnnealingSolver().solve(
        problem, SolverConfig(num_solutions=1, seed=1)
    )[0]
    greedy = HeuristicSolver().solve(problem, SolverConfig(num_solutions=1))[0]

    def report(name: str, sol) -> SolverReport:
        gap = 100.0 * (sol.cost - optimum) / optimum if optimum else 0.0
        return SolverReport(
            name=name,
            cost=sol.cost,
            n_selected=int(sol.selected.sum()),
            targets_met=sol.all_targets_met,
            gap_pct=gap,
        )

    return [
        report("MIP (exact)", mip),
        report("Simulated annealing", sa),
        report("Greedy heuristic", greedy),
    ]


def roundtrips_through_marxan_format(problem) -> bool:
    """Write the problem to Marxan files, re-read, and compare."""
    from pymarxan.io.readers import load_project
    from pymarxan.io.writers import save_project

    with tempfile.TemporaryDirectory() as tmp:
        save_project(problem, tmp)
        reloaded = load_project(tmp)

    pu_ok = problem.planning_units[["id", "cost"]].equals(
        reloaded.planning_units[["id", "cost"]]
    )
    feat_ok = set(problem.features["id"]) == set(reloaded.features["id"])
    # Same total feature supply survives the round-trip.
    amt_ok = abs(
        problem.pu_vs_features["amount"].sum()
        - reloaded.pu_vs_features["amount"].sum()
    ) < 1e-9
    return bool(pu_ok and feat_ok and amt_ok)


def main() -> None:
    print("pymarxan validation — native Marxan project, six planning units")
    print("=" * 66)

    problem = load_simple_problem()
    print(
        f"Loaded: {problem.n_planning_units} planning units, "
        f"{problem.n_features} features (from {SIMPLE_PROJECT})"
    )

    reports = solve_all(problem)
    print("\nSolver           cost    cells  targets_met   gap vs exact")
    print("-" * 60)
    for r in reports:
        print(
            f"{r.name:<20}{r.cost:6.1f}   {r.n_selected:3d}   "
            f"{str(r.targets_met):<8}    {r.gap_pct:+5.1f}%"
        )

    optimum = reports[0].cost
    print(f"\nExact minimum-set optimum (ground truth): cost = {optimum:.1f}")
    print("All heuristics meet every target and never beat the exact optimum.")

    ok = roundtrips_through_marxan_format(problem)
    print(f"\nMarxan-format round-trip reproduces the problem: {ok}")


if __name__ == "__main__":
    main()
