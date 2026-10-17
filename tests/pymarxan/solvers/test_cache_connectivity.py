"""ProblemCache connectivity: adjacency arrays and the full-objective term.

The cache's connectivity term must equal ``compute_connectivity_penalty`` (the
reported objective, which equals the MIP's term) on every binary selection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.cache import ProblemCache
from pymarxan.solvers.utils import compute_connectivity_penalty

N_PU = 6


def _problem(
    connectivity: pd.DataFrame | None,
    weight: float = 1.0,
    asymmetric: bool = False,
) -> ConservationProblem:
    """6 PUs (ids 1..6), 2 features; integer data throughout."""
    pu = pd.DataFrame({
        "id": list(range(1, N_PU + 1)),
        "cost": [3.0, 5.0, 2.0, 7.0, 4.0, 6.0],
        "status": [0] * N_PU,
    })
    feat = pd.DataFrame({
        "id": [1, 2], "name": ["a", "b"], "target": [6.0, 4.0], "spf": [10.0, 10.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 1, 2, 2, 2],
        "pu": [1, 3, 5, 2, 4, 6],
        "amount": [3.0, 2.0, 4.0, 2.0, 3.0, 1.0],
    })
    boundary = pd.DataFrame({
        "id1": [1, 2, 3, 4, 5, 1],
        "id2": [2, 3, 4, 5, 6, 1],
        "boundary": [1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
    })
    problem = ConservationProblem(
        planning_units=pu, features=feat, pu_vs_features=puvspr,
        boundary=boundary, connectivity=connectivity,
    )
    problem.parameters["CONNECTIVITY_WEIGHT"] = weight
    if asymmetric:
        problem.parameters["ASYMMETRIC_CONNECTIVITY"] = 1
    return problem


# Symmetric fixture: reversed duplicate with a DIFFERENT value (first row wins),
# an exact duplicate, a self-pair, an unknown id, and an isolated PU (6).
SYM_CONN = pd.DataFrame({
    "id1": [1, 2, 2, 3, 4, 4, 99, 3],
    "id2": [2, 1, 3, 2, 4, 5, 1, 5],
    "value": [4.0, 9.0, 3.0, 8.0, 5.0, 2.0, 7.0, 1.0],
})

# Asymmetric fixture: duplicate (1, 2) rows (sum), the reverse (2, 1), a self-row
# (dropped), an unknown id, and a second source into 2.
ASYM_CONN = pd.DataFrame({
    "id1": [1, 1, 2, 3, 3, 4, 99, 5],
    "id2": [2, 2, 1, 3, 2, 5, 2, 4],
    "value": [4.0, 6.0, 3.0, 9.0, 2.0, 5.0, 7.0, 1.0],
})


def _csr_rows(start: np.ndarray, idx: np.ndarray, val: np.ndarray) -> dict[int, dict[int, float]]:
    out: dict[int, dict[int, float]] = {}
    for i in range(len(start) - 1):
        row = {int(j): float(v) for j, v in zip(idx[start[i]:start[i + 1]],
                                               val[start[i]:start[i + 1]], strict=True)}
        assert len(row) == start[i + 1] - start[i], "duplicate neighbour in CSR row"
        out[i] = row
    return out


class TestSymmetricArrays:
    def test_mode_and_weight(self) -> None:
        cache = ProblemCache.from_problem(_problem(SYM_CONN, weight=2.5))
        assert cache.conn_mode == 1
        assert cache.conn_weight == 2.5

    def test_pairs_deduped_first_row_wins_both_directions(self) -> None:
        cache = ProblemCache.from_problem(_problem(SYM_CONN))
        rows = _csr_rows(cache.conn_out_start, cache.conn_out_idx, cache.conn_out_val)
        # PU index = id - 1. Pair {1,2}: first row (1,2,4) wins over (2,1,9).
        # Pair {2,3}: first row (2,3,3) wins over (3,2,8).
        assert rows == {
            0: {1: 4.0},
            1: {0: 4.0, 2: 3.0},
            2: {1: 3.0, 4: 1.0},
            3: {4: 2.0},
            4: {3: 2.0, 2: 1.0},
            5: {},
        }

    def test_self_pair_goes_to_conn_self(self) -> None:
        cache = ProblemCache.from_problem(_problem(SYM_CONN))
        np.testing.assert_array_equal(cache.conn_self, [0.0, 0.0, 0.0, 5.0, 0.0, 0.0])

    def test_in_lists_empty_for_symmetric(self) -> None:
        cache = ProblemCache.from_problem(_problem(SYM_CONN))
        assert cache.conn_in_idx.size == 0
        assert cache.conn_in_val.size == 0
        np.testing.assert_array_equal(cache.conn_in_start, np.zeros(N_PU + 1))

    def test_csr_dtypes_match_boundary(self) -> None:
        cache = ProblemCache.from_problem(_problem(SYM_CONN))
        assert cache.conn_out_start.dtype == np.int32
        assert cache.conn_out_idx.dtype == np.int32
        assert cache.conn_out_val.dtype == np.float64
        assert cache.conn_self.dtype == np.float64
        assert cache.conn_out_start.shape == (N_PU + 1,)


class TestPrecomputedRowArrays:
    """Row index per CSR entry and per-row out-value sums, built once (no per-call work)."""

    @pytest.mark.parametrize(("conn", "asymmetric"), [(SYM_CONN, False), (ASYM_CONN, True)])
    def test_conn_out_row_matches_csr(self, conn: pd.DataFrame, asymmetric: bool) -> None:
        cache = ProblemCache.from_problem(_problem(conn, asymmetric=asymmetric))
        want = np.repeat(np.arange(N_PU), np.diff(cache.conn_out_start))
        assert cache.conn_out_row.dtype == np.int32
        np.testing.assert_array_equal(cache.conn_out_row, want)

    def test_out_rowsum_asymmetric(self) -> None:
        cache = ProblemCache.from_problem(_problem(ASYM_CONN, asymmetric=True))
        assert cache.conn_out_rowsum.dtype == np.float64
        np.testing.assert_array_equal(cache.conn_out_rowsum, [10.0, 3.0, 2.0, 5.0, 1.0, 0.0])

    def test_out_rowsum_zero_for_symmetric_and_off(self) -> None:
        for problem in (_problem(SYM_CONN), _problem(SYM_CONN, weight=0.0), _problem(None)):
            cache = ProblemCache.from_problem(problem)
            np.testing.assert_array_equal(cache.conn_out_rowsum, np.zeros(N_PU))
            assert cache.conn_out_row.size == cache.conn_out_idx.size


class TestAsymmetricArrays:
    def test_mode(self) -> None:
        cache = ProblemCache.from_problem(_problem(ASYM_CONN, asymmetric=True))
        assert cache.conn_mode == 2
        assert cache.conn_weight == 1.0

    def test_out_edges_duplicates_summed_self_row_dropped(self) -> None:
        cache = ProblemCache.from_problem(_problem(ASYM_CONN, asymmetric=True))
        rows = _csr_rows(cache.conn_out_start, cache.conn_out_idx, cache.conn_out_val)
        assert rows == {
            0: {1: 10.0},
            1: {0: 3.0},
            2: {1: 2.0},
            3: {4: 5.0},
            4: {3: 1.0},
            5: {},
        }

    def test_in_edges(self) -> None:
        cache = ProblemCache.from_problem(_problem(ASYM_CONN, asymmetric=True))
        rows = _csr_rows(cache.conn_in_start, cache.conn_in_idx, cache.conn_in_val)
        assert rows == {
            0: {1: 3.0},
            1: {0: 10.0, 2: 2.0},
            2: {},
            3: {4: 1.0},
            4: {3: 5.0},
            5: {},
        }

    def test_no_self_rewards(self) -> None:
        cache = ProblemCache.from_problem(_problem(ASYM_CONN, asymmetric=True))
        np.testing.assert_array_equal(cache.conn_self, np.zeros(N_PU))


class TestOff:
    @pytest.mark.parametrize(
        ("conn", "weight"),
        [(None, 1.0), (SYM_CONN, 0.0), (SYM_CONN, -1.0), (ASYM_CONN, 0.0)],
    )
    def test_off_cases(self, conn: pd.DataFrame | None, weight: float) -> None:
        problem = _problem(conn, weight=weight, asymmetric=conn is ASYM_CONN)
        cache = ProblemCache.from_problem(problem)
        assert cache.conn_mode == 0
        assert cache.conn_weight == 0.0
        assert cache.conn_out_idx.size == 0
        assert cache.conn_in_idx.size == 0
        np.testing.assert_array_equal(cache.conn_out_start, np.zeros(N_PU + 1))
        np.testing.assert_array_equal(cache.conn_self, np.zeros(N_PU))
        # The term is 0: the objective equals the no-connectivity objective.
        base = ProblemCache.from_problem(_problem(None))
        rng = np.random.default_rng(7)
        for _ in range(10):
            sel = rng.random(N_PU) < 0.5
            held = cache.compute_held(sel)
            assert cache.compute_full_objective(sel, held, 1.0) == (
                base.compute_full_objective(sel, held, 1.0)
            )

    def test_unknown_ids_only_is_empty_but_on(self) -> None:
        conn = pd.DataFrame({"id1": [99, 1], "id2": [1, 98], "value": [5.0, 5.0]})
        cache = ProblemCache.from_problem(_problem(conn))
        assert cache.conn_mode == 1
        assert cache.conn_out_idx.size == 0
        np.testing.assert_array_equal(cache.conn_out_start, np.zeros(N_PU + 1))


class TestFullObjectiveMatchesEvaluator:
    @pytest.mark.parametrize(
        ("conn", "asymmetric", "weight"),
        [(SYM_CONN, False, 1.0), (SYM_CONN, False, 2.5),
         (ASYM_CONN, True, 1.0), (ASYM_CONN, True, 0.75)],
    )
    def test_full_minus_base_equals_connectivity_penalty(
        self, conn: pd.DataFrame, asymmetric: bool, weight: float,
    ) -> None:
        problem = _problem(conn, weight=weight, asymmetric=asymmetric)
        cache = ProblemCache.from_problem(problem)
        base = ProblemCache.from_problem(_problem(None))
        rng = np.random.default_rng(20261017)
        selections = [np.zeros(N_PU, dtype=bool), np.ones(N_PU, dtype=bool)]
        selections += [rng.random(N_PU) < 0.5 for _ in range(30)]
        nonzero = 0
        for sel in selections:
            held = cache.compute_held(sel)
            got = cache.compute_full_objective(sel, held, 1.0) - (
                base.compute_full_objective(sel, held, 1.0)
            )
            want = compute_connectivity_penalty(problem, sel, cache.pu_id_to_idx)
            assert got == pytest.approx(want, abs=1e-9)
            nonzero += want != 0.0
        assert nonzero >= 20  # the fixture actually exercises the term


class TestDeltaExactness:
    @pytest.mark.parametrize(
        ("conn", "asymmetric", "weight"),
        [(SYM_CONN, False, 1.0), (SYM_CONN, False, 2.5),
         (ASYM_CONN, True, 1.0), (ASYM_CONN, True, 0.75)],
    )
    def test_delta_equals_full_difference(
        self, conn: pd.DataFrame, asymmetric: bool, weight: float,
    ) -> None:
        cache = ProblemCache.from_problem(_problem(conn, weight=weight, asymmetric=asymmetric))
        blm = 1.0
        rng = np.random.default_rng(42)
        states = [np.zeros(N_PU, dtype=bool), np.ones(N_PU, dtype=bool)]
        states += [rng.random(N_PU) < 0.5 for _ in range(8)]
        for sel in states:
            held = cache.compute_held(sel)
            before = cache.compute_full_objective(sel, held, blm)
            total_cost = float(np.sum(cache.costs[sel]))
            for idx in range(N_PU):
                delta = cache.compute_delta_objective(idx, sel, held, total_cost, blm)
                flipped = sel.copy()
                flipped[idx] = not flipped[idx]
                after = cache.compute_full_objective(flipped, cache.compute_held(flipped), blm)
                assert delta == pytest.approx(after - before, abs=1e-9), (sel, idx)

    @pytest.mark.parametrize(("conn", "asymmetric"), [(SYM_CONN, False), (ASYM_CONN, True)])
    @pytest.mark.parametrize("dtype", [np.int64, np.uint8])
    def test_integer_selection_dtypes_match_bool(
        self, conn: pd.DataFrame, asymmetric: bool, dtype: type,
    ) -> None:
        """A 0/1 integer selection gives the bool delta (no bitwise ``~`` on ints)."""
        cache = ProblemCache.from_problem(_problem(conn, asymmetric=asymmetric))
        rng = np.random.default_rng(3)
        for sel in [np.zeros(N_PU, dtype=bool)] + [rng.random(N_PU) < 0.5 for _ in range(6)]:
            held = cache.compute_held(sel)
            before = cache.compute_full_objective(sel, held, 1.0)
            cost = float(np.sum(cache.costs[sel]))
            as_int = sel.astype(dtype)
            for idx in range(N_PU):
                want = cache.compute_delta_objective(idx, sel, held, cost, 1.0)
                got = cache.compute_delta_objective(idx, as_int, held, cost, 1.0)
                assert got == pytest.approx(want, abs=1e-9), (dtype, sel, idx)
                flipped = sel.copy()
                flipped[idx] = not flipped[idx]
                after = cache.compute_full_objective(flipped, cache.compute_held(flipped), 1.0)
                assert got == pytest.approx(after - before, abs=1e-9), (dtype, sel, idx)

    def test_off_delta_unchanged(self) -> None:
        """Weight 0 with a table: delta identical to the no-table cache."""
        cache = ProblemCache.from_problem(_problem(SYM_CONN, weight=0.0))
        base = ProblemCache.from_problem(_problem(None))
        sel = np.array([True, False, True, True, False, True])
        held = cache.compute_held(sel)
        cost = float(np.sum(cache.costs[sel]))
        for idx in range(N_PU):
            assert cache.compute_delta_objective(idx, sel, held, cost, 1.0) == (
                base.compute_delta_objective(idx, sel, held, cost, 1.0)
            )
