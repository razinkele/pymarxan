"""Nature recovery — where to protect, restore actively, leave to recover, or do nothing.

Scenario
--------
A river floodplain is split into a 6 × 5 grid of planning units. Two habitats matter:
**wet meadow** and **alder carr**. Some units still hold them (existing habitat); drained
or cultivated units could get them back if restored (potential, the amount restoration
would *add*). Restoration does not always work: the probability of success falls with
distance from the river. Active restoration (re-wetting, planting) costs 2.5 × the land
cost; passive recovery (stop draining, let it regenerate) costs the land price and works
half as often (η = 0.5) — both labelled placeholders.

We:

1. build ``RecoveryInputs`` (base project = units, habitats, existing amounts; plus
   potential, p_success, passive recovery);
2. turn it into a Marxan-with-Zones problem with ``build_recovery_problem``;
3. find the cheapest plan meeting both targets exactly (zone MIP) and with the heuristic
   fallback (zone SA + iterative improvement);
4. find the best plan within a smaller budget;
5. re-score the plan at lower success rates (``success_sensitivity``);
6. save and reload it as a recovery project.

All data are synthetic and analytic, so the script is deterministic and doubles as a
regression fixture (``tests/test_examples.py``).

Run it::

    python examples/nature_recovery.py
"""
from __future__ import annotations

import tempfile
import warnings

import numpy as np
import pandas as pd

from pymarxan.models.problem import ConservationProblem
from pymarxan.recovery import (
    PassiveAction,
    RecoveryInputs,
    build_recovery_problem,
    evaluate_recovery,
    load_recovery_project,
    save_recovery_project,
    solve_recovery,
    success_sensitivity,
)
from pymarxan.solvers.base import SolverConfig

N_COLS, N_ROWS = 6, 5
WET_MEADOW, ALDER_CARR = 1, 2


def build_recovery_inputs() -> RecoveryInputs:
    """30 units; the river runs along column 0. Deterministic, analytic data."""
    ids, cost, units_xy = [], [], []
    for r in range(N_ROWS):
        for c in range(N_COLS):
            ids.append(r * N_COLS + c + 1)
            cost.append(round(1.0 + 0.4 * c + 0.1 * ((r * 7 + c * 3) % 5), 2))
            units_xy.append((c, r))
    existing, potential = [], []
    for pid, (c, r) in zip(ids, units_xy, strict=True):
        if c == 0 or (c == 1 and r % 2 == 0):
            existing.append((WET_MEADOW, pid, 10.0 - 2.0 * c))
        if c >= 4 and r in (0, 1):
            existing.append((ALDER_CARR, pid, 6.0))
        if 1 <= c <= 3:
            potential.append((pid, WET_MEADOW, 8.0 - c))
        if c >= 3 and r >= 2:
            potential.append((pid, ALDER_CARR, 5.0))
    base = ConservationProblem(
        planning_units=pd.DataFrame({"id": ids, "cost": cost, "status": 0}),
        features=pd.DataFrame({"id": [WET_MEADOW, ALDER_CARR],
                               "name": ["wet_meadow", "alder_carr"],
                               "target": 0.0, "spf": 1.0}),
        pu_vs_features=pd.DataFrame(existing, columns=["species", "pu", "amount"]),
        parameters={"BLM": 0.0},
    )
    p = pd.Series([round(0.9 - 0.1 * c, 2) for c, _ in units_xy], index=ids)
    return RecoveryInputs(
        base=base,
        potential=pd.DataFrame(potential, columns=["pu", "feature", "amount"]),
        p_success=p,
        passive=PassiveAction(),  # η 0.5, cost factor 1.0: placeholders
        protect_prop=0.3,
        restore_prop=0.2,
    )


def main() -> None:
    inputs = build_recovery_inputs()
    problem, mapping = build_recovery_problem(inputs)
    print(f"{len(problem.planning_units)} units, zones {mapping.zones}")
    print("placeholders:", [k for k, v in mapping.provenance.items() if v == "placeholder"])

    exact = evaluate_recovery(solve_recovery(problem), problem, mapping)
    print(f"\nCheapest plan meeting the targets (zone MIP): cost {exact.total_cost:.2f}")
    print(exact.actions["action"].value_counts().to_string())
    print(exact.habitats.to_string(index=False))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        fast = problem.copy_with(parameters={**problem.parameters, "NUMITNS": 50_000,
                                             "NUMTEMP": 500})
        heur = evaluate_recovery(
            solve_recovery(fast, method="heuristic", config=SolverConfig(num_solutions=5, seed=1)),
            problem, mapping,
        )
    print(f"\nHeuristic (zone SA + II): cost {heur.total_cost:.2f}, "
          f"targets met: {heur.targets_met}")

    budget = round(0.7 * exact.total_cost, 2)
    within = evaluate_recovery(solve_recovery(problem, mode="budget", budget=budget),
                               problem, mapping)
    print(f"\nBest plan within budget {budget}: cost {within.total_cost:.2f}")
    print(within.habitats[["name", "restore_shortfall", "protect_shortfall"]]
          .to_string(index=False))

    print("\nSuccess-rate sensitivity of the exact plan:")
    print(success_sensitivity(exact).to_string(index=False))

    with tempfile.TemporaryDirectory() as d:
        save_recovery_project(inputs, d)
        again = load_recovery_project(d)
        same = np.array_equal(solve_recovery(again.problem).zone_assignment, exact.assignment)
        print(f"\nSaved and reloaded: same plan = {same}")

    print("\n" + exact.caveat)


if __name__ == "__main__":
    main()
