"""Implicit PROBMODE selection and the probability diagnostics (v0.48.0, spec §3.1).

With no explicit ``PROBMODE``: 1D wanted = a 1D source (pu.dat ``prob`` or prob.dat) and
any ``ptarget1d > 0``; 2D wanted = a puvspr ``prob`` column and any ``ptarget > 0``. Both →
5, 1D → 4, 2D → 3, neither → v0.46's default (1 with prob.dat, else 0). The rule lives in
``_probmode`` (pure), so code-built and ``copy_with`` problems get it too.

``_probability_diagnostics(problem)`` returns the advisory messages; the guarded emitter
issues each as one ``UserWarning`` at ``load_project`` and at the top of every top-level
solve; a solve nested in another (a ``contextvars.ContextVar``) stays silent.
"""
from __future__ import annotations

import contextvars
import warnings

import numpy as np
import pandas as pd
import pytest

from pymarxan.calibration.blm import calibrate_blm
from pymarxan.io.readers import load_project
from pymarxan.io.writers import save_project
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.base import SolverConfig
from pymarxan.solvers.heuristic import HeuristicSolver
from pymarxan.solvers.run_mode import RunModePipeline
from pymarxan.solvers.simulated_annealing import SimulatedAnnealingSolver
from pymarxan.solvers.utils import (
    _emit_probability_diagnostics,
    _in_top_level_solve,
    _probability_diagnostics,
    _probmode,
    _solve_scope,
)


def _problem(*, pu_prob=True, table=None, pv_prob=False, pt1d=0.9, pt2d=-1.0,
             params=None) -> ConservationProblem:
    pu = pd.DataFrame({"id": [1, 2, 3, 4], "cost": [1.0, 2.0, 1.5, 3.0], "status": 0})
    if pu_prob:
        pu["prob"] = [0.1, 0.2, 0.0, 0.3]
    pv = pd.DataFrame({"species": [1, 1, 2, 2], "pu": [1, 2, 3, 4],
                       "amount": [5.0, 4.0, 3.0, 6.0]})
    if pv_prob:
        pv["prob"] = 0.8
    feats = pd.DataFrame({"id": [1, 2], "name": ["a", "b"], "target": [4.0, 3.0],
                          "spf": 5.0})
    if pt1d is not None:
        feats["ptarget1d"] = [pt1d, -1.0]
    if pt2d is not None:
        feats["ptarget"] = [pt2d, pt2d]
    return ConservationProblem(
        planning_units=pu, features=feats, pu_vs_features=pv, probability=table,
        parameters={"NUMITNS": 2000, "NUMTEMP": 20, **(params or {})},
    )


TABLE = pd.DataFrame({"pu": [1, 2, 3, 4], "probability": [0.1, 0.2, 0.0, 0.3]})


class TestImplicitMode:
    def test_1d_only_is_4(self):
        assert _probmode(_problem()) == 4

    def test_prob_dat_source_is_4(self):
        assert _probmode(_problem(pu_prob=False, table=TABLE)) == 4

    def test_2d_only_is_3(self):
        assert _probmode(_problem(pu_prob=False, pt1d=None, pv_prob=True, pt2d=0.7)) == 3

    def test_both_is_5(self):
        assert _probmode(_problem(pv_prob=True, pt2d=0.7)) == 5

    def test_neither_keeps_v046_default(self):
        assert _probmode(_problem(pt1d=-1.0)) == 0
        assert _probmode(_problem(pu_prob=False, table=TABLE, pt1d=-1.0)) == 1
        assert _probmode(_problem(pu_prob=False, pt1d=0.9)) == 0  # targets but no source
        assert _probmode(_problem(pv_prob=False, pt1d=None, pt2d=0.7)) == 0  # no puvspr prob

    def test_missing_target_columns_count_as_disabled(self):
        assert _probmode(_problem(pt1d=None, pt2d=None, pv_prob=True)) == 0

    def test_explicit_mode_wins(self):
        assert _probmode(_problem(params={"PROBMODE": 0})) == 0
        assert _probmode(_problem(params={"PROBMODE": 2})) == 2

    def test_copy_with_and_code_built_problems_get_it(self):
        p = _problem(pt1d=-1.0)
        assert _probmode(p) == 0
        q = p.copy_with(features=p.features.assign(ptarget1d=[0.5, -1.0]))
        assert _probmode(q) == 4

    def test_validate_uses_the_implicit_mode(self):
        # Implicit 4: a non-finite amount is the 1D term's cell data → reported.
        p = _problem()
        p.pu_vs_features.loc[0, "amount"] = np.nan
        assert any("non-finite" in e for e in p.validate())


def _msgs(problem) -> list[str]:
    return _probability_diagnostics(problem)


class TestDiagnostics:
    def test_implicit_mode_differs_from_v046(self):
        (m,) = _msgs(_problem())
        assert "PROBMODE 4" in m and "PROBMODE 0" in m and "v0.46" in m
        (m,) = _msgs(_problem(pu_prob=False, table=TABLE))
        assert "PROBMODE 4" in m and "PROBMODE 1" in m
        assert "premium" in m and 'parameters["PROBMODE"] = 1' in m

    def test_no_message_when_mode_is_unchanged_or_explicit(self):
        assert _msgs(_problem(pt1d=-1.0)) == []
        assert _msgs(_problem(params={"PROBMODE": 4})) == []

    @pytest.mark.parametrize("mode", [1, 2, 3])
    def test_explicit_mode_ignores_1d_targets(self, mode):
        msgs = _msgs(_problem(pu_prob=False, table=TABLE, pv_prob=True,
                              params={"PROBMODE": mode}))
        assert any("ptarget1d" in m and "ignored" in m for m in msgs), msgs

    @pytest.mark.parametrize("mode", [1, 2, 4])
    def test_explicit_mode_ignores_2d_targets(self, mode):
        msgs = _msgs(_problem(pv_prob=True, pt2d=0.7, table=TABLE, params={"PROBMODE": mode}))
        assert any("ptarget" in m and "2D" in m and "ignored" in m for m in msgs), msgs

    @pytest.mark.parametrize("mode", [1, 2])
    def test_pu_dat_prob_with_mode_1_or_2(self, mode):
        msgs = _msgs(_problem(pt1d=-1.0, params={"PROBMODE": mode}))
        assert any("pu.dat" in m and "Probability 1D only" in m for m in msgs), msgs

    @pytest.mark.parametrize("mode", [4, 5])
    def test_1d_term_without_data(self, mode):
        msgs = _msgs(_problem(pu_prob=False, pv_prob=True, params={"PROBMODE": mode}))
        assert any("1D" in m and "no" in m and "off" in m for m in msgs), msgs

    @pytest.mark.parametrize("mode", [3, 5])
    def test_2d_term_without_data(self, mode):
        msgs = _msgs(_problem(params={"PROBMODE": mode}))
        assert any("2D" in m and "puvspr" in m and "off" in m for m in msgs), msgs

    def test_sources_disagree(self):
        table = TABLE.assign(probability=[0.1, 0.2, 0.0, 0.31])
        msgs = _msgs(_problem(table=table, params={"PROBMODE": 4}))
        assert any("disagree" in m and "pu.dat" in m for m in msgs), msgs

    def test_sources_agree_silently(self):
        assert _msgs(_problem(table=TABLE, params={"PROBMODE": 4})) == []

    def test_a_pu_missing_from_prob_dat_counts_as_zero(self):
        # PU 3 has pu.dat prob 0 and no prob.dat row: they agree.
        table = TABLE[TABLE.pu != 3]
        assert _msgs(_problem(table=table, params={"PROBMODE": 4})) == []

    def test_probmode_and_validate_never_warn(self):
        p = _problem()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _probmode(p)
            assert p.validate() == []  # advisory messages never enter validate()


def _record(fn) -> list[str]:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fn()
    return [str(w.message) for w in rec if issubclass(w.category, UserWarning)]


DIFFERS = "differs from v0.46"


class TestEmitter:
    def test_emitter_warns_each_message(self):
        assert sum(DIFFERS in m for m in _record(lambda: _emit_probability_diagnostics(
            _problem()))) == 1

    def test_load_project_emits(self, tmp_path):
        save_project(_problem(), tmp_path)
        msgs = _record(lambda: load_project(tmp_path))
        assert sum(DIFFERS in m for m in msgs) == 1

    def test_ten_run_sa_solve_emits_once(self):
        msgs = _record(lambda: SimulatedAnnealingSolver().solve(
            _problem(), SolverConfig(num_solutions=10, seed=1)))
        assert sum(DIFFERS in m for m in msgs) == 1

    @pytest.mark.parametrize("runmode", [1, 4])
    def test_ten_run_pipeline_emits_once(self, runmode):
        msgs = _record(lambda: RunModePipeline(runmode=runmode).solve(
            _problem(), SolverConfig(num_solutions=10, seed=1)))
        assert sum(DIFFERS in m for m in msgs) == 1

    def test_calibrate_blm_emits_once_per_point(self):
        msgs = _record(lambda: calibrate_blm(
            _problem(), HeuristicSolver(), blm_values=[0.0, 1.0, 2.0, 3.0, 4.0],
            config=SolverConfig(num_solutions=1, seed=1)))
        assert sum(DIFFERS in m for m in msgs) == 5

    def test_multiobjective_mip_emits_once(self):
        from pymarxan.multiobjective import MinCost, MultiObjectiveMIP, WeightedSum
        msgs = _record(lambda: MultiObjectiveMIP(
            [MinCost()], WeightedSum([1.0]), mip_backend="cbc",
        ).solve(_problem(), SolverConfig(num_solutions=1)))
        assert sum(DIFFERS in m for m in msgs) == 1

    def test_nested_solve_is_silent_and_the_guard_resets(self):
        problem = _problem()
        assert not _in_top_level_solve()

        def outer():
            SimulatedAnnealingSolver().solve(problem, SolverConfig(num_solutions=1, seed=1))

        with pytest.raises(RuntimeError), _solve_scope():
            assert _in_top_level_solve()
            raise RuntimeError("boom")
        assert not _in_top_level_solve()  # reset in finally
        assert sum(DIFFERS in m for m in _record(outer)) == 1

    def test_a_copied_context_inside_a_solve_stays_silent(self):
        problem = _problem()
        seen: list[list[str]] = []

        with _solve_scope():
            ctx = contextvars.copy_context()
            seen.append(_record(lambda: ctx.run(
                HeuristicSolver().solve, problem, SolverConfig(num_solutions=1))))
        assert seen == [[]]
        assert not _in_top_level_solve()
