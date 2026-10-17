"""Representation and gap analysis under PROBMODE 2 use expected amounts (v0.45 §4).

Numerators come from ``compute_held`` (``Σ a·(1 − p)``), denominators from
``expected_feature_totals`` (the same over every planning unit). Every other mode keeps
``feature_amounts()`` exactly, including ``pu_vs_features`` rows for unknown PU ids.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pymarxan.analysis._totals import expected_feature_totals
from pymarxan.analysis.gap_analysis import compute_gap_analysis
from pymarxan.analysis.representation import compute_representation
from pymarxan.io.readers import load_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.utils import build_solution

DATA = Path(__file__).parent.parent.parent / "data"
PROB = pd.DataFrame({"pu": [2, 4, 5, 6], "probability": [0.4, 0.5, 0.25, 1.0]})
KEEP = {1: 1.0, 2: 0.6, 3: 1.0, 4: 0.5, 5: 0.75, 6: 0.0}  # 1 − p per PU


def _simple(
    probability: pd.DataFrame | None = PROB, mode: int | None = 2, **params: Any,
) -> ConservationProblem:
    p = load_project(DATA / "simple")
    merged = {**p.parameters, **params}
    if mode is not None:
        merged["PROBMODE"] = mode
    return p.copy_with(probability=probability, parameters=merged)


def _expected(p: ConservationProblem, pus: set[int]) -> dict[int, float]:
    out = {int(f): 0.0 for f in p.features["id"]}
    for pu, sp, amt in zip(
        p.pu_vs_features["pu"], p.pu_vs_features["species"], p.pu_vs_features["amount"],
    ):
        if int(pu) in pus:
            out[int(sp)] += float(amt) * KEEP[int(pu)]
    return out


ALL = {1, 2, 3, 4, 5, 6}


class TestExpectedTotals:
    def test_mode_2_expected_totals(self) -> None:
        p = _simple()
        got = expected_feature_totals(p)
        want = _expected(p, ALL)
        assert got.keys() == want.keys()
        for fid in want:
            assert got[fid] == pytest.approx(want[fid])

    @pytest.mark.parametrize(("probability", "mode"), [(PROB, 0), (PROB, 1), (None, 2)])
    def test_other_modes_are_feature_amounts(
        self, probability: pd.DataFrame | None, mode: int,
    ) -> None:
        p = _simple(probability, mode)
        assert expected_feature_totals(p) == {
            int(f): float(a) for f, a in p.feature_amounts().items()
        }


class TestRepresentationMode2:
    def test_expected_numerator_and_denominator(self) -> None:
        p = _simple()
        sel = np.array([True, True, False, True, False, True])
        r = compute_representation(p, build_solution(p, sel, 0.0), threshold=0.3)
        held = _expected(p, {1, 2, 4, 6})
        total = _expected(p, ALL)
        for fid in held:
            assert r.represented_amount[fid] == pytest.approx(held[fid])
            assert r.total_amount[fid] == pytest.approx(total[fid])
            assert r.pct_represented[fid] == pytest.approx(100.0 * held[fid] / total[fid])

    def test_selecting_every_unit_is_100_percent(self) -> None:
        p = _simple()
        r = compute_representation(p, build_solution(p, np.ones(6, dtype=bool), 0.0))
        for fid in r.feature_ids:
            assert r.pct_represented[fid] == pytest.approx(100.0)

    def test_binary_and_proportion_plans_agree(self) -> None:
        p = _simple()
        bits = [True, False, True, True, True, False]
        rb = compute_representation(p, build_solution(p, np.array(bits), 0.0))
        rf = compute_representation(p, build_solution(p, np.array(bits, dtype=float), 0.0))
        for fid in rb.feature_ids:
            assert rb.represented_amount[fid] == pytest.approx(rf.represented_amount[fid])
            assert rb.pct_represented[fid] == pytest.approx(rf.pct_represented[fid])

    def test_feature_without_rows_reports_zero(self) -> None:
        p = _simple()
        ghost = pd.DataFrame({"id": [4], "name": ["ghost"], "target": [1.0], "spf": [1.0]})
        p = p.copy_with(features=pd.concat([p.features, ghost], ignore_index=True))
        r = compute_representation(p, build_solution(p, np.ones(6, dtype=bool), 0.0))
        df = r.to_dataframe()
        assert df.loc[df["feature_id"] == 4, "total_amount"].item() == 0.0
        assert df.loc[df["feature_id"] == 4, "pct_represented"].item() == 0.0


def _locked(p: ConservationProblem, ids: list[int]) -> ConservationProblem:
    pu = p.planning_units.copy()
    pu.loc[pu["id"].isin(ids), "status"] = 2
    return p.copy_with(planning_units=pu)


class TestGapMode2:
    def test_expected_protected_and_total(self) -> None:
        p = _locked(_simple(), [2, 4, 5])
        g = compute_gap_analysis(p)
        held = _expected(p, {2, 4, 5})
        total = _expected(p, ALL)
        targets = dict(zip(p.features["id"].astype(int), p.features["target"].astype(float)))
        df = g.to_dataframe().set_index("feature_id")
        for fid in held:
            assert g.protected_amount[fid] == pytest.approx(held[fid])
            assert g.total_amount[fid] == pytest.approx(total[fid])
            assert g.gap[fid] == pytest.approx(max(0.0, targets[fid] - held[fid]))
            assert df.loc[fid, "percent_protected"] == pytest.approx(
                100.0 * held[fid] / total[fid]
            )


def _orphan(p: ConservationProblem) -> ConservationProblem:
    """An extra pu_vs_features row for PU 99, which is not a planning unit."""
    row = pd.DataFrame({"species": [1], "pu": [99], "amount": [40.0]})
    return p.copy_with(pu_vs_features=pd.concat([p.pu_vs_features, row], ignore_index=True))


@pytest.mark.parametrize(("probability", "mode"), [(None, None), (PROB, 0), (PROB, 1)])
class TestOtherModesUnchanged:
    def test_representation_with_orphan_row(
        self, probability: pd.DataFrame | None, mode: int | None,
    ) -> None:
        p = _orphan(_simple(probability, mode))
        sel = np.array([True, True, False, True, False, True])
        r = compute_representation(p, build_solution(p, sel, 0.0))
        # f1: raw total 60 + orphan 40 = 100; selected 10 + 15 + 12 + 10 = 47
        assert r.total_amount[1] == 100.0
        assert r.represented_amount[1] == 47.0
        assert r.pct_represented[1] == pytest.approx(47.0)
        assert r.total_amount[2] == 45.0
        assert r.represented_amount[2] == 25.0

    def test_gap_with_orphan_row(
        self, probability: pd.DataFrame | None, mode: int | None,
    ) -> None:
        p = _locked(_orphan(_simple(probability, mode)), [2, 4])
        df = compute_gap_analysis(p).to_dataframe().set_index("feature_id")
        assert df.loc[1, "total_amount"] == 100.0
        assert df.loc[1, "protected_amount"] == 27.0
        assert df.loc[1, "percent_protected"] == pytest.approx(27.0)
        assert df.loc[3, "protected_amount"] == 10.0


def test_representation_int_selection_reads_as_bool() -> None:
    """An int 0/1 ``selected`` follows the _coerce_selected contract (not fancy indexing)."""
    from pymarxan.solvers.base import Solution

    p = _simple(None, None)
    sel = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    ref = build_solution(p, sel.astype(bool), 0.0)
    as_int = Solution(
        selected=sel, cost=ref.cost, boundary=ref.boundary, objective=ref.objective,
        targets_met=ref.targets_met,
    )
    r = compute_representation(p, as_int)
    assert r.represented_amount == {1: 37.0, 2: 20.0, 3: 18.0}
