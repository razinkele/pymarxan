"""Phase A — DCI computation (dci_diadromous, dci_potamodromous,
segment_connectivity). Hand-computed fixtures per design §5."""
from __future__ import annotations

import pytest

from pymarxan.rivers import (
    dci_diadromous,
    dci_potamodromous,
    segment_connectivity,
)

# --- linear chain (DCId 58.333, DCIp 61.111) --------------------------


def test_diadromous_chain(chain):
    assert dci_diadromous(chain) == pytest.approx(58.3333333, abs=1e-5)


def test_potamodromous_chain(chain):
    assert dci_potamodromous(chain) == pytest.approx(61.1111111, abs=1e-5)


# --- Y-tree confluence (interior LCA) ---------------------------------


def test_diadromous_ytree(ytree):
    assert dci_diadromous(ytree) == pytest.approx(66.6666667, abs=1e-5)


def test_potamodromous_ytree(ytree):
    assert dci_potamodromous(ytree) == pytest.approx(61.1111111, abs=1e-5)


# --- p=0 mouth-barrier variant: the H1 NaN gate -----------------------


def test_diadromous_ytree_mouth_blocked_is_zero(ytree_mouth_blocked):
    # nothing reaches the sea
    assert dci_diadromous(ytree_mouth_blocked) == pytest.approx(0.0, abs=1e-9)


def test_potamodromous_ytree_mouth_blocked_unchanged(ytree_mouth_blocked):
    # H1: a mouth barrier sits below every in-network pair's LCA, so DCIp is
    # unchanged at 61.111. A division-based c_ij would compute 0/0 = NaN here.
    val = dci_potamodromous(ytree_mouth_blocked)
    assert val == val, "DCIp is NaN — c_ij used the unsafe root_prod division"
    assert val == pytest.approx(61.1111111, abs=1e-5)


# --- sanity properties -------------------------------------------------


def test_all_barriers_removed_is_100(chain):
    passable = {1: 1.0, 2: 1.0}
    assert dci_diadromous(chain, passable) == pytest.approx(100.0, abs=1e-6)
    assert dci_potamodromous(chain, passable) == pytest.approx(100.0, abs=1e-6)


def test_single_segment_is_100(single):
    assert dci_diadromous(single) == pytest.approx(100.0, abs=1e-9)
    assert dci_potamodromous(single) == pytest.approx(100.0, abs=1e-9)


def test_removing_a_barrier_never_decreases_dci(chain):
    before = dci_diadromous(chain)
    after = dci_diadromous(chain, {1: 1.0})  # remove B1
    assert after >= before - 1e-9


# --- direction convention (single_pass default; round_trip deferred) --


def test_direction_default_is_single_pass(chain):
    # default == single_pass == 58.333 on the chain
    assert dci_diadromous(chain) == dci_diadromous(chain, direction="single_pass")


def test_round_trip_not_implemented(chain):
    with pytest.raises(NotImplementedError):
        dci_diadromous(chain, direction="round_trip")


# --- segment_connectivity ---------------------------------------------


def test_segment_connectivity_diadromous(chain):
    c = segment_connectivity(chain, form="diadromous")
    assert c[1] == pytest.approx(1.0)
    assert c[2] == pytest.approx(0.5)
    assert c[3] == pytest.approx(0.25)


def test_segment_connectivity_potamodromous_marginals_reaggregate(chain):
    # marginals m_i = sum_j w_j c_ij must satisfy sum_i w_i m_i == DCIp/100
    m = segment_connectivity(chain, form="potamodromous")
    w = 1.0 / 3.0  # equal lengths
    agg = sum(w * m[sid] for sid in m)
    assert 100.0 * agg == pytest.approx(dci_potamodromous(chain), abs=1e-6)
