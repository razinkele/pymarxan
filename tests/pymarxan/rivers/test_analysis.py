"""Phase E — budget-DCI frontier + barrier selection frequency."""
from __future__ import annotations

import pandas as pd
import pytest

from pymarxan.rivers import (
    RiverNetwork,
    barrier_selection_frequency,
    budget_dci_frontier,
)


def _binary_chain() -> RiverNetwork:
    """S1(outlet) <- S2 <- S3; impassable B1 on S2, B2 on S3 (p=0), cost 1."""
    return RiverNetwork(
        segments=pd.DataFrame(
            {"id": [1, 2, 3], "length": [10.0] * 3, "down_id": [-1, 1, 2]}
        ),
        barriers=pd.DataFrame(
            {
                "id": [1, 2],
                "segment": [2, 3],
                "pass_up": [0.0, 0.0],
                "pass_down": [0.0, 0.0],
                "removal_cost": [1.0, 1.0],
                "status": [0, 0],
            }
        ),
    )


# --- budget_dci_frontier ----------------------------------------------


def test_frontier_columns_and_rows():
    df = budget_dci_frontier(_binary_chain(), [0.0, 1.0, 2.0])
    assert isinstance(df, pd.DataFrame)
    assert list(df["budget"]) == [0.0, 1.0, 2.0]
    for col in ("budget", "dci_before", "dci_after", "gain", "cost", "n_removed"):
        assert col in df.columns


def test_frontier_values_on_binary_chain_mip():
    df = budget_dci_frontier(_binary_chain(), [0.0, 1.0, 2.0], optimizer="mip")
    after = dict(zip(df["budget"], df["dci_after"]))
    assert after[0.0] == pytest.approx(100.0 / 3, abs=1e-4)   # nothing removed
    assert after[1.0] == pytest.approx(200.0 / 3, abs=1e-4)   # remove gating B1
    assert after[2.0] == pytest.approx(100.0, abs=1e-4)       # both removed


def test_frontier_is_monotonic_non_decreasing():
    for optimizer in ("greedy", "mip"):
        df = budget_dci_frontier(
            _binary_chain(), [0.0, 1.0, 2.0], optimizer=optimizer
        )
        vals = list(df["dci_after"])
        assert vals == sorted(vals)  # more budget never hurts


def test_frontier_rejects_unknown_optimizer():
    with pytest.raises(ValueError, match="optimizer"):
        budget_dci_frontier(_binary_chain(), [1.0], optimizer="bogus")


# --- barrier_selection_frequency --------------------------------------


def test_selection_frequency_covers_all_barriers_in_unit_range():
    freq = barrier_selection_frequency(_binary_chain(), budget=1.0, n_runs=10)
    assert set(freq) == {1, 2}
    assert all(0.0 <= v <= 1.0 for v in freq.values())


def test_selection_frequency_ranks_gating_barrier_highest():
    # budget 1 → the gating barrier B1 is the only useful removal → always picked
    freq = barrier_selection_frequency(_binary_chain(), budget=1.0, n_runs=10)
    assert freq[1] == pytest.approx(1.0)
    assert freq[2] < freq[1]


def test_selection_frequency_is_deterministic_with_seed():
    a = barrier_selection_frequency(_binary_chain(), budget=1.0, n_runs=5, base_seed=3)
    b = barrier_selection_frequency(_binary_chain(), budget=1.0, n_runs=5, base_seed=3)
    assert a == b
