from pathlib import Path

import numpy as np

from pymarxan.zones.objective import (
    check_zone_targets,
    compute_zone_boundary,
    compute_zone_cost,
    compute_zone_penalty,
    compute_zone_objective,
)
from pymarxan.zones.readers import load_zone_project

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "zones"


class TestComputeZoneCost:
    def setup_method(self):
        self.problem = load_zone_project(DATA_DIR)
        self.pu_ids = self.problem.planning_units["id"].tolist()

    def test_all_protected(self):
        # All PUs in zone 1 (protected): costs = 100+150+200+120 = 570
        assignment = np.array([1, 1, 1, 1])
        cost = compute_zone_cost(self.problem, assignment)
        assert cost == 570.0

    def test_all_sustainable(self):
        # All PUs in zone 2: costs = 50+80+100+60 = 290
        assignment = np.array([2, 2, 2, 2])
        cost = compute_zone_cost(self.problem, assignment)
        assert cost == 290.0

    def test_mixed(self):
        # PU1=protected(100), PU2=sustainable(80), PU3=protected(200), PU4=sustainable(60)
        assignment = np.array([1, 2, 1, 2])
        cost = compute_zone_cost(self.problem, assignment)
        assert cost == 440.0

    def test_unassigned_is_zero(self):
        assignment = np.array([0, 0, 0, 0])
        cost = compute_zone_cost(self.problem, assignment)
        assert cost == 0.0


class TestComputeZoneBoundary:
    def setup_method(self):
        self.problem = load_zone_project(DATA_DIR)

    def test_same_zones_no_penalty(self):
        assignment = np.array([1, 1, 1, 1])
        zbc = compute_zone_boundary(self.problem, assignment)
        assert zbc == 0.0

    def test_different_zones_penalty(self):
        # PU1=1,PU2=2: cost=50; PU2=2,PU3=1: cost=50;
        # PU3=1,PU4=2: cost=50
        assignment = np.array([1, 2, 1, 2])
        zbc = compute_zone_boundary(self.problem, assignment)
        assert zbc == 150.0


class TestCheckZoneTargets:
    def setup_method(self):
        self.problem = load_zone_project(DATA_DIR)

    def test_all_protected_targets(self):
        # All in zone 1 (contribution=1.0)
        # F1: 10+8+6+5=29 >= 10 (Z1 target) -> met
        # F2: 5+7+9+4=25 >= 8 (Z1 target) -> met
        assignment = np.array([1, 1, 1, 1])
        targets = check_zone_targets(self.problem, assignment)
        assert targets[(1, 1)] is True
        assert targets[(1, 2)] is True

    def test_none_assigned(self):
        assignment = np.array([0, 0, 0, 0])
        targets = check_zone_targets(self.problem, assignment)
        assert not any(targets.values())


class TestComputeZoneObjective:
    def setup_method(self):
        self.problem = load_zone_project(DATA_DIR)

    def test_objective_components(self):
        assignment = np.array([1, 1, 1, 1])
        obj = compute_zone_objective(self.problem, assignment, blm=0.0)
        # Cost=570, zone boundary=0 (all same zone), blm*boundary=0
        assert obj >= 570.0


class TestComputeZonePenalty:
    """Tests for compute_zone_penalty with unmet and met targets."""

    def setup_method(self):
        self.problem = load_zone_project(DATA_DIR)

    def test_zone_penalty_with_unmet_targets(self):
        """All PUs unassigned => all zone targets unmet => positive penalty."""
        assignment = np.zeros(self.problem.planning_units.shape[0], dtype=int)
        penalty = compute_zone_penalty(self.problem, assignment)
        assert penalty > 0.0, "Unmet zone targets should produce positive penalty"

    def test_zone_penalty_zero_when_all_met(self):
        """Mixed assignment meeting all zone targets => penalty should be zero.

        PU1,PU2 in zone 1 meets Z1 targets (F1: 18>=10, F2: 12>=8).
        PU3,PU4 in zone 2 meets Z2 targets (F1: 5.5>=5, F2: 3.9>=3).
        """
        assignment = np.array([1, 1, 2, 2])
        penalty = compute_zone_penalty(self.problem, assignment)
        assert penalty == 0.0


class TestZonePenaltyMisslevel:
    """Tests for MISSLEVEL support in zone objective functions."""

    def test_zone_penalty_respects_misslevel(self):
        """With MISSLEVEL=0.5, targets should be halved - easier to meet."""
        problem = load_zone_project(DATA_DIR)
        assignment = np.array([1, 0, 0, 0])  # Only PU 0 in zone 1
        penalty_strict = compute_zone_penalty(problem, assignment)
        problem.parameters["MISSLEVEL"] = 0.5
        penalty_relaxed = compute_zone_penalty(problem, assignment)
        assert penalty_relaxed < penalty_strict, (
            "Relaxed MISSLEVEL should produce lower penalty"
        )

    def test_check_zone_targets_respects_misslevel(self):
        """With MISSLEVEL, more targets should appear met."""
        problem = load_zone_project(DATA_DIR)
        assignment = np.array([1, 0, 0, 0])
        strict = check_zone_targets(problem, assignment)
        problem.parameters["MISSLEVEL"] = 0.01
        relaxed = check_zone_targets(problem, assignment)
        assert sum(relaxed.values()) >= sum(strict.values())
