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
