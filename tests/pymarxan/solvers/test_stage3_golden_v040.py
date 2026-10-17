"""v0.40 MIPSolver results that the Stage 3 ``_build`` split must not change (spec §2, §7.1).

GOLDEN was recorded on unmodified v0.40 (8f8ea76 + Stage 3 spec commits) before any source
change. Every case must reproduce exactly: ``mip_objective_value`` always, and the selection
with its cost / objective / targets_met when the recorded optimum is unique. Uniqueness is
probed at record time by re-solving with the found selection forbidden (a no-good cut): a
tie means CBC may legitimately return another optimal plan once LP names or row order change.
The absolute ``min_largest_shortfall`` ghost cases are the one deliberate change (spec §2).

v0.43 (written reason): ``compute_connectivity_penalty`` now mirrors the MIP's
connectivity term, so the ``objective`` (and stored ``connectivity``) of the ``conn_*``
entries were re-recorded in place; every other field is unchanged.

Regenerate only with a written reason:
    PYTHONPATH=src:. /opt/micromamba/envs/shiny/bin/python \
        tests/pymarxan/solvers/test_stage3_golden_v040.py
"""
from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import Solution, SolverConfig
from pymarxan.solvers.mip_solver import MIP_OBJECTIVES, MIPSolver

DATA = Path(__file__).parent.parent.parent / "data"
GOLDEN_PATH = DATA / "golden" / "stage3_v040.json"
SHORTFALLS = ("relative", "absolute")
BUDGET = 33.0  # simple: total cost 83; every budgeted optimum is unique at 33 (enumerated)


def _simple(**params: Any) -> ConservationProblem:
    p = load_project(DATA / "simple")
    return p.copy_with(parameters={**p.parameters, "COSTBUDGET": BUDGET, **params})


def _no_blm() -> ConservationProblem:
    return _simple(BLM=0.0)


def _conn(asymmetric: bool) -> ConservationProblem:
    p = _simple(CONNECTIVITY_WEIGHT=2.0, ASYMMETRIC_CONNECTIVITY=int(asymmetric))
    conn = pd.DataFrame({
        "id1": [1, 2, 3, 2, 5], "id2": [2, 1, 4, 5, 6], "value": [4.0, 3.0, 2.5, 6.0, 1.5],
    })
    return p.copy_with(connectivity=conn)


def _prob(probmode: int | None, weighting: float = 2.0) -> ConservationProblem:
    params: dict[str, Any] = {"PROBABILITYWEIGHTING": weighting}
    if probmode is not None:
        params["PROBMODE"] = probmode
    p = _simple(**params)
    prob = pd.DataFrame({
        "pu": [1, 2, 3, 4, 5, 6], "probability": [0.1, 0.4, 0.0, 0.25, 0.05, 0.3],
    })
    return p.copy_with(probability=prob)


def _locks() -> ConservationProblem:
    p = _simple()
    pu = p.planning_units.copy()
    pu.loc[pu["id"] == 3, "status"] = 2  # locked in
    pu.loc[pu["id"] == 6, "status"] = 3  # locked out
    return p.copy_with(planning_units=pu)


def _ghost(misslevel: float = 0.9) -> ConservationProblem:
    """simple + feature 4 (target 10) with no pu_vs_features rows."""
    p = _simple(MISSLEVEL=misslevel)
    ghost = pd.DataFrame({"id": [4], "name": ["ghost"], "target": [10.0], "spf": [2.0]})
    return p.copy_with(features=pd.concat([p.features, ghost], ignore_index=True))


def _ghost_absolute_mls() -> ConservationProblem:
    """Spec §2: simple + a ghost with target 50 and COSTBUDGET 30."""
    p = load_project(DATA / "simple")
    ghost = pd.DataFrame({"id": [4], "name": ["ghost"], "target": [50.0], "spf": [1.0]})
    return p.copy_with(
        features=pd.concat([p.features, ghost], ignore_index=True),
        parameters={**p.parameters, "COSTBUDGET": 30.0},
    )


NOGOOD = SolverConfig(
    num_solutions=1, metadata={"forbidden_selections": [[0, 1, 0, 1, 0, 1]]},
)
ONE = SolverConfig(num_solutions=1)

VARIANTS: dict[str, tuple[Callable[[], ConservationProblem], SolverConfig]] = {
    "simple": (_simple, ONE),
    "no_blm": (_no_blm, ONE),
    "conn_sym": (lambda: _conn(False), ONE),
    "conn_asym": (lambda: _conn(True), ONE),
    "probmode1": (lambda: _prob(1), ONE),
    "probmode2": (lambda: _prob(2), ONE),
    "prob_no_probmode": (lambda: _prob(None), ONE),
    "locks": (_locks, ONE),
    "nogood": (_simple, NOGOOD),
    "ghost_misslevel09": (_ghost, ONE),
}
CASES = [
    f"{variant}|{objective}|{shortfall}"
    for variant in VARIANTS
    for objective in MIP_OBJECTIVES
    for shortfall in SHORTFALLS
]


def _solve(
    problem: ConservationProblem, config: SolverConfig, **kw: Any
) -> list[Solution]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # PuLP 3.3.2
        return MIPSolver(mip_backend="cbc", **kw).solve(problem, config)


def _key(sol: Solution, objective: str) -> tuple[float, ...]:
    """Lexicographic value the solver optimises, as a minimisation key."""
    v = float(sol.metadata["mip_objective_value"])
    if objective.startswith("max_"):
        return (-v,)
    return (v, float(sol.cost)) if objective == "min_penalties" else (v,)


def _worse(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """a is lexicographically worse than b by more than 1e-9."""
    for ai, bi in zip(a, b, strict=True):
        if abs(ai - bi) > 1e-9:
            return ai > bi
    return False


def _result(
    problem: ConservationProblem, config: SolverConfig, *, record: bool = False, **kw: Any
) -> dict[str, Any]:
    sols = _solve(problem, config, **kw)
    if not sols:
        return {"solutions": 0}
    s = sols[0]
    out: dict[str, Any] = {
        "solutions": len(sols),
        "selected": [int(v) for v in s.selected],
        "cost": float(s.cost),
        "objective": float(s.objective),
        "targets_met": {str(k): bool(v) for k, v in sorted(s.targets_met.items())},
        "mip_objective_value": float(s.metadata["mip_objective_value"]),
    }
    if record:
        forbidden = [*config.metadata.get("forbidden_selections", []), out["selected"]]
        probe = SolverConfig(num_solutions=1, metadata={"forbidden_selections": forbidden})
        other = _solve(problem, probe, **kw)
        objective = kw.get("objective", "min_set")
        out["unique_optimum"] = not other or _worse(
            _key(other[0], objective), _key(s, objective),
        )
    return out


def _run(case: str, *, record: bool = False) -> dict[str, Any]:
    variant, objective, shortfall = case.split("|")
    factory, config = VARIANTS[variant]
    return _result(factory(), config, record=record, objective=objective, shortfall=shortfall)


def record() -> dict[str, Any]:
    out: dict[str, Any] = {case: _run(case, record=True) for case in CASES}
    out["ghost_absolute_mls_v040"] = _result(
        _ghost_absolute_mls(), ONE, record=True,
        objective="min_largest_shortfall", shortfall="absolute",
    )
    return out


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(GOLDEN_PATH.read_text())
    return data


def _assert_same(
    got: dict[str, Any], want: dict[str, Any], keys: list[str] | None = None,
) -> None:
    """Compare ``got`` with a golden record; ``keys`` overrides the uniqueness rule."""
    assert got["solutions"] == want["solutions"]
    if not want["solutions"]:
        return
    if keys is None:
        keys = ["mip_objective_value"]
        if want["unique_optimum"]:
            keys += ["selected", "cost", "objective", "targets_met"]
    for key in keys:
        if isinstance(want[key], float):
            assert got[key] == pytest.approx(want[key], rel=1e-9, abs=1e-9), key
        else:
            assert got[key] == want[key], key


# Spec §2 Fixed: absolute min_largest_shortfall now counts a ghost feature (target > 0, no
# pu_vs_features rows) as m >= R = target × MISSLEVEL (10 × 0.9), as relative mode, the zone
# solver and the v0.39 max_* ghost fix do. v0.40 reported 0.0 (all real targets met).
# Every plan ties at m = R, so only the value is pinned.
STAGE3_FIXED = {"ghost_misslevel09|min_largest_shortfall|absolute": 9.0}

# v0.45 PROBMODE consistency (spec §5): ``build_solution`` judges PROBMODE 2 plans on expected
# amounts Σ amount·(1 − p), as the MIP's target rows do. The MIP's plans are unchanged, but
# the reported ``objective`` and ``targets_met`` of these unique-optimum records move.
# case -> (new objective, new targets_met); the JSON keeps the v0.44 values (not regenerated).
_ALL_MISSED = {"1": False, "2": False, "3": False}
_F3_MET = {"1": False, "2": False, "3": True}
STAGE3_PROBMODE2_FIXED: dict[str, tuple[float, dict[str, bool]]] = {
    **{
        f"probmode2|{objective}|{shortfall}": (59.1, _F3_MET)
        for objective in ("max_features", "max_weighted_features")
        for shortfall in SHORTFALLS
    },
    **{
        f"probmode2|{objective}|{shortfall}": (55.4, _ALL_MISSED)
        for objective in ("min_shortfall", "min_largest_shortfall")
        for shortfall in SHORTFALLS
    },
}
_V044_PROBMODE2 = {
    "max_features": (54.0, _F3_MET),
    "max_weighted_features": (54.0, _F3_MET),
    "min_shortfall": (41.0, {"1": True, "2": True, "3": False}),
    "min_largest_shortfall": (41.0, {"1": True, "2": True, "3": False}),
}


@pytest.mark.parametrize("case", CASES)
def test_mip_solver_result_unchanged(golden: dict[str, Any], case: str) -> None:
    got = _run(case)
    if case in STAGE3_FIXED:
        assert golden[case]["mip_objective_value"] == 0.0  # the v0.40 value
        assert got["mip_objective_value"] == pytest.approx(STAGE3_FIXED[case], abs=1e-9)
        return
    if case in STAGE3_PROBMODE2_FIXED:
        assert golden[case]["unique_optimum"]  # justifies pinning the selection below
        old_objective, old_met = _V044_PROBMODE2[case.split("|")[1]]
        assert golden[case]["objective"] == old_objective  # the v0.44 values
        assert golden[case]["targets_met"] == old_met
        new_objective, new_met = STAGE3_PROBMODE2_FIXED[case]
        assert got["objective"] == pytest.approx(new_objective, rel=1e-9, abs=1e-9)
        assert got["targets_met"] == new_met
        _assert_same(got, golden[case], keys=["selected", "cost", "mip_objective_value"])
        return
    _assert_same(got, golden[case])


def test_simple_min_set_anchor_in_golden(golden: dict[str, Any]) -> None:
    """The correctness anchor (BLM 0): reserve {2, 4, 6} at cost 35.0."""
    for shortfall in SHORTFALLS:
        rec = golden[f"no_blm|min_set|{shortfall}"]
        assert rec["selected"] == [0, 1, 0, 1, 0, 1]
        assert rec["cost"] == 35.0


def test_ghost_absolute_min_largest_shortfall(golden: dict[str, Any]) -> None:
    """Spec §2 Fixed: v0.40 ignored the ghost (target 50, no rows) and reported 5.0.

    The builder counts it (m >= 50). The ghost holds m at 50, so every plan ties and CBC
    returns the empty plan where v0.40 returned a cost-23 one: pin only the value.
    """
    assert golden["ghost_absolute_mls_v040"]["mip_objective_value"] == 5.0
    got = _result(
        _ghost_absolute_mls(), ONE, objective="min_largest_shortfall", shortfall="absolute",
    )
    assert got["mip_objective_value"] == 50.0


if __name__ == "__main__":
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(record(), indent=1, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN_PATH}")
