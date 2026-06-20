"""Phase A — RiverNetwork model + topology helpers."""
from __future__ import annotations

import pandas as pd
import pytest

from pymarxan.rivers import RiverNetwork

# --- construction & validation ----------------------------------------


def test_chain_builds(chain):
    assert chain.n_segments == 3
    assert chain.n_barriers == 2


def test_outlet_is_unique_segment_with_no_downstream(chain):
    assert chain.outlet == 1


def test_rejects_duplicate_segment_ids():
    with pytest.raises(ValueError, match="segment ids must be unique"):
        RiverNetwork(
            segments=pd.DataFrame(
                {"id": [1, 1], "length": [10.0, 10.0], "down_id": [-1, 1]}
            ),
            barriers=pd.DataFrame(
                {"id": [], "segment": [], "pass_up": [], "pass_down": []}
            ),
        )


def test_rejects_dangling_down_id():
    with pytest.raises(ValueError, match="down_id"):
        RiverNetwork(
            segments=pd.DataFrame(
                {"id": [1, 2], "length": [10.0, 10.0], "down_id": [-1, 99]}
            ),
            barriers=pd.DataFrame(
                {"id": [], "segment": [], "pass_up": [], "pass_down": []}
            ),
        )


def test_rejects_two_outlets():
    with pytest.raises(ValueError, match="exactly one outlet"):
        RiverNetwork(
            segments=pd.DataFrame(
                {"id": [1, 2], "length": [10.0, 10.0], "down_id": [-1, -1]}
            ),
            barriers=pd.DataFrame(
                {"id": [], "segment": [], "pass_up": [], "pass_down": []}
            ),
        )


def test_rejects_cycle():
    # 1->2->1 with no outlet
    with pytest.raises(ValueError, match="outlet|cycle|tree"):
        RiverNetwork(
            segments=pd.DataFrame(
                {"id": [1, 2], "length": [10.0, 10.0], "down_id": [2, 1]}
            ),
            barriers=pd.DataFrame(
                {"id": [], "segment": [], "pass_up": [], "pass_down": []}
            ),
        )


def test_rejects_out_of_range_passability():
    with pytest.raises(ValueError, match="passab"):
        RiverNetwork(
            segments=pd.DataFrame(
                {"id": [1, 2], "length": [10.0, 10.0], "down_id": [-1, 1]}
            ),
            barriers=pd.DataFrame(
                {"id": [1], "segment": [2], "pass_up": [1.5], "pass_down": [1.5]}
            ),
        )


def test_rejects_barrier_on_missing_segment():
    with pytest.raises(ValueError, match="segment"):
        RiverNetwork(
            segments=pd.DataFrame(
                {"id": [1], "length": [10.0], "down_id": [-1]}
            ),
            barriers=pd.DataFrame(
                {"id": [1], "segment": [99], "pass_up": [0.5], "pass_down": [0.5]}
            ),
        )


# --- topology helpers --------------------------------------------------


def test_root_products_chain(chain):
    # root_prod(i) = product of barrier passabilities from i down to mouth.
    rp = chain.root_products()
    assert rp[1] == pytest.approx(1.0)
    assert rp[2] == pytest.approx(0.5)
    assert rp[3] == pytest.approx(0.25)


def test_lca_chain_ancestor_is_self(chain):
    # S2 is an ancestor of S3 → lca(S2,S3) == S2
    assert chain.lca(2, 3) == 2


def test_lca_ytree_is_interior_confluence(ytree):
    # the whole point of the Y fixture: LCA(S2,S3) is the interior segment S1
    assert ytree.lca(2, 3) == 1


def test_path_barriers_chain(chain):
    # path S2<->S3 crosses only B2 (id 2, on segment S3)
    assert set(chain.path_barriers(2, 3)) == {2}


def test_path_barriers_to_mouth_chain(chain):
    # S3 to mouth crosses B2 (on S3) and B1 (on S2)
    assert set(chain.path_barriers_to_mouth(3)) == {1, 2}
    # S1 (outlet, no mouth barrier) crosses nothing
    assert set(chain.path_barriers_to_mouth(1)) == set()


def test_path_barriers_excludes_mouth_barrier_below_lca(ytree_mouth_blocked):
    # H1 guard: the S2<->S3 path must NOT include the mouth barrier B1
    # (it sits below the LCA S1), so c_23 stays p(B2)*p(B3), not 0.
    assert set(ytree_mouth_blocked.path_barriers(2, 3)) == {2, 3}
