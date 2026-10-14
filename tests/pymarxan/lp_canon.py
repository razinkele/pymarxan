"""Order-independent dump of a PuLP model, and a spy that captures models at solve()."""
from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pulp


def _r(v: float) -> float:
    return round(float(v), 12)


def canonical_model(model: pulp.LpProblem) -> dict[str, Any]:
    """Name -> (sense, rhs, sorted coefs) per constraint; sorted objective; variable bounds.

    JSON round-tripped so it compares equal to a dict loaded from a golden file.
    """
    obj = model.objective
    cons = {
        name: {
            "sense": int(c.sense),
            "rhs": _r(-c.constant),
            "coefs": sorted([v.name, _r(a)] for v, a in c.items()),
        }
        for name, c in model.constraints.items()
    }
    out = {
        "sense": int(model.sense),
        "objective": sorted([v.name, _r(a)] for v, a in obj.items()) if obj is not None else [],
        "objective_constant": _r(obj.constant) if obj is not None else 0.0,
        "constraints": cons,
        "variables": sorted([v.name, v.cat, v.lowBound, v.upBound] for v in model.variables()),
    }
    canon: dict[str, Any] = json.loads(json.dumps(out))  # typed: mypy warn_return_any
    return canon


@contextmanager
def captured_models() -> Iterator[list[pulp.LpProblem]]:
    """Record every LpProblem passed to LpProblem.solve (the solve still runs)."""
    models: list[pulp.LpProblem] = []
    orig = pulp.LpProblem.solve

    def spy(self: pulp.LpProblem, *args: Any, **kwargs: Any) -> Any:
        models.append(self)
        return orig(self, *args, **kwargs)

    pulp.LpProblem.solve = spy  # type: ignore[method-assign]
    try:
        yield models
    finally:
        pulp.LpProblem.solve = orig  # type: ignore[method-assign]
