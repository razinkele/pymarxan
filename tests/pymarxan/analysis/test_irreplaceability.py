from pathlib import Path

import pandas as pd

from pymarxan.analysis.irreplaceability import compute_irreplaceability
from pymarxan.io.readers import load_project

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


class TestIrreplaceability:
    def setup_method(self):
        self.problem = load_project(DATA_DIR)

    def test_returns_dict(self):
        result = compute_irreplaceability(self.problem)
        assert isinstance(result, dict)
        assert len(result) == 6

    def test_values_in_range(self):
        result = compute_irreplaceability(self.problem)
        for pid, score in result.items():
            assert 0.0 <= score <= 1.0, f"PU {pid} score {score} out of range"

    def test_pu_with_unique_feature_has_high_score(self):
        """A PU that is the sole provider of a feature should be irreplaceable."""
        problem = self.problem
        extra_feature = pd.DataFrame({
            "id": [99], "name": ["unique_sp"], "target": [5.0], "spf": [1.0],
        })
        problem.features = pd.concat(
            [problem.features, extra_feature], ignore_index=True
        )
        extra_puvspr = pd.DataFrame({
            "species": [99], "pu": [1], "amount": [5.0],
        })
        problem.pu_vs_features = pd.concat(
            [problem.pu_vs_features, extra_puvspr], ignore_index=True
        )
        result = compute_irreplaceability(problem)
        assert result[1] > 0  # PU 1 is critical for species 99


def test_irreplaceability_numerical_values():
    """Verify exact irreplaceability scores for a known problem."""
    import pytest

    from pymarxan.models.problem import ConservationProblem

    pu = pd.DataFrame({"id": [1, 2, 3], "cost": [1.0, 1.0, 1.0], "status": [0, 0, 0]})
    features = pd.DataFrame({
        "id": [1, 2], "name": ["f1", "f2"],
        "target": [8.0, 6.0], "spf": [1.0, 1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 2, 2],
        "pu": [1, 2, 1, 3],
        "amount": [5.0, 5.0, 4.0, 4.0],
    })
    problem = ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
    )
    scores = compute_irreplaceability(problem)
    # Feature 1 total=10, target=8. Remove PU1: remaining=5 < 8 => critical
    # Feature 2 total=8, target=6. Remove PU1: remaining=4 < 6 => critical
    # PU1 critical for 2/2 features => score = 1.0
    assert scores[1] == pytest.approx(1.0)
    # PU2: feature 1 only. Remove PU2: remaining=5 < 8 => critical for f1
    # PU2 not in feature 2. Score = 1/2 = 0.5
    assert scores[2] == pytest.approx(0.5)
    # PU3: feature 2 only. Remove PU3: remaining=4 < 6 => critical for f2
    # Score = 1/2 = 0.5
    assert scores[3] == pytest.approx(0.5)


def test_irreplaceability_excludes_zero_target_features():
    """Score denominator must only count features with positive targets."""
    from pymarxan.models.problem import ConservationProblem

    pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]})
    features = pd.DataFrame({
        "id": [1, 2],
        "name": ["has_target", "zero_target"],
        "target": [10.0, 0.0],
        "spf": [1.0, 1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1, 2, 2],
        "pu": [1, 2, 1, 2],
        "amount": [10.0, 5.0, 5.0, 5.0],
    })
    problem = ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
    )
    scores = compute_irreplaceability(problem)
    # PU 1 is sole provider of enough for feature 1 (10 >= target 10)
    # Removing PU 1: remaining = 5 < 10 => critical for feature 1
    # Only 1 positive-target feature, so score = 1/1 = 1.0
    assert scores[1] == 1.0, f"Expected 1.0, got {scores[1]}"


def test_irreplaceability_locked_out_pus_score_zero():
    """Locked-out PUs cannot be selected — they can never be irreplaceable.

    Previously their amounts inflated the available-total used as the
    'with PU removed' baseline, making the score for OTHER PUs look lower.
    """
    from pymarxan.models.problem import ConservationProblem

    pu = pd.DataFrame({
        "id": [1, 2, 3],
        "cost": [1.0, 1.0, 1.0],
        "status": [0, 0, 3],  # PU 3 locked out
    })
    features = pd.DataFrame({
        "id": [1], "name": ["sp"], "target": [10.0], "spf": [1.0],
    })
    # PU 3 (locked-out) carries the bulk of the feature.
    puvspr = pd.DataFrame({
        "species": [1, 1, 1], "pu": [1, 2, 3], "amount": [6.0, 6.0, 100.0],
    })
    problem = ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
    )
    scores = compute_irreplaceability(problem)
    # Locked-out PU should have score 0 (cannot be selected, never critical)
    assert scores[3] == 0.0
    # With PU 3 excluded, PU 1 and 2 each have 6.0; either one alone gives 6 < 10,
    # so both are critical for the sole feature.
    assert scores[1] == 1.0
    assert scores[2] == 1.0


def test_irreplaceability_applies_misslevel():
    """MISSLEVEL scales the effective target used in criticality test.

    All other target-checking paths (solver, build_solution, export_summary)
    apply MISSLEVEL. Irreplaceability previously compared against raw target,
    over-reporting criticality.
    """
    from pymarxan.models.problem import ConservationProblem

    pu = pd.DataFrame({"id": [1, 2], "cost": [1.0, 1.0], "status": [0, 0]})
    features = pd.DataFrame({
        "id": [1], "name": ["sp"], "target": [10.0], "spf": [1.0],
    })
    puvspr = pd.DataFrame({
        "species": [1, 1], "pu": [1, 2], "amount": [6.0, 6.0],
    })
    # With MISSLEVEL=1.0: total=12, removing PU1 leaves 6 < 10 -> critical
    full = ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
        parameters={"MISSLEVEL": 1.0},
    )
    scores_full = compute_irreplaceability(full)
    assert scores_full[1] == 1.0

    # With MISSLEVEL=0.5: effective target = 5, removing PU1 leaves 6 >= 5 -> NOT critical
    relaxed = ConservationProblem(
        planning_units=pu, features=features, pu_vs_features=puvspr,
        parameters={"MISSLEVEL": 0.5},
    )
    scores_relaxed = compute_irreplaceability(relaxed)
    assert scores_relaxed[1] == 0.0
    assert scores_relaxed[2] == 0.0
