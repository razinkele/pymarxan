"""Tests for raptr-style space/adequacy targets (1 - WSS/TSS)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pymarxan.adequacy import SpaceSpec, compute_space_held
from pymarxan.models.problem import ConservationProblem


def _line_problem(n=5, feature_at=None):
    # n PUs on a line at x=0..n-1 (y=0), all holding feature 1 unless feature_at given.
    ids = np.arange(1, n + 1)
    pu = pd.DataFrame({"id": ids, "cost": 1.0, "status": 0,
                       "xloc": np.arange(n, dtype=float), "yloc": 0.0})
    feats = pd.DataFrame({"id": [1], "name": ["a"], "target": [1.0], "spf": [1.0]})
    occ = ids if feature_at is None else np.asarray(feature_at)
    pvf = pd.DataFrame({"species": 1, "pu": occ, "amount": 1.0})
    return ConservationProblem(pu, feats, pvf)


def test_space_held_full_selection_is_one():
    p = _line_problem(5)
    held = compute_space_held(p, np.ones(5, bool), SpaceSpec())
    assert held[1] == 1.0  # all occupied PUs selected -> WSS 0 -> held 1


def test_space_held_empty_selection_is_zero():
    p = _line_problem(5)
    held = compute_space_held(p, np.zeros(5, bool), SpaceSpec())
    assert held[1] == 0.0


def test_space_held_spread_beats_clustered():
    # 5 PUs on a line; select 2. Spread {ends} covers the range better than clustered {adjacent}.
    p = _line_problem(5)
    spread = np.array([True, False, False, False, True])   # x=0 and x=4
    clustered = np.array([True, True, False, False, False])  # x=0 and x=1
    hs = compute_space_held(p, spread, SpaceSpec())[1]
    hc = compute_space_held(p, clustered, SpaceSpec())[1]
    assert hs > hc


def test_space_held_monotone_in_added_pu():
    p = _line_problem(6)
    base = np.array([True, False, False, False, False, False])
    more = base.copy()
    more[3] = True
    hm = compute_space_held(p, more, SpaceSpec())[1]
    hb = compute_space_held(p, base, SpaceSpec())[1]
    assert hm >= hb


def test_space_held_matches_hand_computed():
    # 3 PUs on a line x=0,1,2, feature everywhere (amount 1). demand pts = the 3 PUs, w=1.
    # centroid c = mean(0,1,2)=1. TSS = 1*(0-1)^2 + 1*(1-1)^2 + 1*(2-1)^2 = 2.
    # Select {x=0}: WSS = 0 + 1 + 4 = 5. held = 1 - 5/2 = -1.5 -> clip 0.
    # Select {x=1} (centre): WSS = 1 + 0 + 1 = 2. held = 1 - 2/2 = 0.0.
    # Select all: WSS 0 -> held 1.
    p = _line_problem(3)
    assert compute_space_held(p, np.array([False, True, False]), SpaceSpec())[1] == 0.0
    assert compute_space_held(p, np.array([True, True, True]), SpaceSpec())[1] == 1.0


def test_space_held_zscored_attribute_columns():
    # A large-unit attribute column must not dominate after z-scoring.
    ids = np.arange(1, 4)
    pu = pd.DataFrame({"id": ids, "cost": 1.0, "status": 0,
                       "env": [0.0, 1000.0, 2000.0]})  # big-unit column
    feats = pd.DataFrame({"id": [1], "name": ["a"], "target": [1.0], "spf": [1.0]})
    pvf = pd.DataFrame({"species": 1, "pu": ids, "amount": 1.0})
    p = ConservationProblem(pu, feats, pvf)
    # Review BUG-A: env-only space (no coords on this problem) -> include_geographic=False.
    spec = SpaceSpec(attribute_columns=["env"], include_geographic=False)
    # centre PU alone -> held 0.0 (same structure as the line, post z-score)
    assert compute_space_held(p, np.array([False, True, False]), spec)[1] == 0.0
