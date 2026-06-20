"""Barrier-decision problem & solution for river restoration (Phase B).

A ``BarrierProblem`` poses: *which barriers to remove, subject to a budget, to
maximise the network's DCI?* Decision per removable barrier is binary
(``remove`` / ``keep``). Barrier ``status`` (read from
``network.barriers``) follows the Marxan convention reused here:

- ``STATUS_LOCKED_IN`` (2)  → the barrier **must** be removed (forced ``y = 1``);
- ``STATUS_LOCKED_OUT`` (3) → the barrier **must not** be removed (``y = 0``);
- ``STATUS_NORMAL`` (0)     → free decision.

See ``docs/plans/2026-06-20-phase19-rivers-aquatic-restoration-design.md`` §6.
"""
from __future__ import annotations

from dataclasses import dataclass

from pymarxan.models.problem import STATUS_LOCKED_IN, STATUS_LOCKED_OUT, STATUS_NORMAL
from pymarxan.rivers.network import RiverNetwork

# Default removal cost when the barriers frame has no `removal_cost` column:
# 1.0 makes a budget behave as a cap on the *number* of removals.
_DEFAULT_COST = 1.0


@dataclass
class BarrierProblem:
    """A budget-constrained DCI-maximisation over barrier removals.

    Parameters
    ----------
    network
        The ``RiverNetwork``; barrier ``removal_cost`` and ``status`` are read
        from ``network.barriers`` (defaulted when absent: cost 1.0, status 0).
    budget
        Cost cap on the chosen action set (arbitrary non-negative float).
        ``None`` means unconstrained.
    form
        ``"diadromous"`` or ``"potamodromous"`` — which DCI to maximise.
    objective
        Only ``"max_dci"`` is wired in this release (see design §6).
    """

    network: RiverNetwork
    budget: float | None = None
    form: str = "diadromous"
    objective: str = "max_dci"

    def __post_init__(self) -> None:
        if self.form not in ("diadromous", "potamodromous"):
            raise ValueError(f"unknown form {self.form!r}")
        if self.objective != "max_dci":
            raise ValueError(
                f"objective {self.objective!r} not supported in this release "
                "(only 'max_dci')"
            )
        if self.budget is not None and self.budget < 0:
            raise ValueError("budget must be non-negative")

    # --- barrier metadata (derived from network.barriers) ---------------

    def _meta(self) -> tuple[list[int], list[int], dict[int, float]]:
        """Return (free_ids, forced_ids, cost_by_id).

        ``free`` = STATUS_NORMAL barriers (the decision variables);
        ``forced`` = STATUS_LOCKED_IN (always removed);
        locked-out barriers are simply omitted from both.
        """
        bar = self.network.barriers
        free: list[int] = []
        forced: list[int] = []
        cost: dict[int, float] = {}
        has_status = "status" in bar.columns
        has_cost = "removal_cost" in bar.columns
        for k in range(len(bar)):
            bid = int(bar["id"].iloc[k])
            cost[bid] = float(bar["removal_cost"].iloc[k]) if has_cost else _DEFAULT_COST
            status = int(bar["status"].iloc[k]) if has_status else STATUS_NORMAL
            if status == STATUS_LOCKED_OUT:
                continue
            if status == STATUS_LOCKED_IN:
                forced.append(bid)
            else:
                free.append(bid)
        return free, forced, cost


@dataclass
class BarrierSolution:
    """The outcome of a barrier optimisation.

    ``dci_before`` is the native-network DCI with **no** decision applied
    (locked-in removals are not pre-applied); ``dci_after`` applies every
    removal in ``removed`` (forced + chosen); ``gain = dci_after - dci_before``.
    ``optimal`` is ``True`` only from the (future) exact MIP.
    """

    removed: set[int]
    cost: float
    dci_before: float
    dci_after: float
    gain: float
    optimal: bool = False
