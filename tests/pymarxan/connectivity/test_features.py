from pathlib import Path

import numpy as np
import pandas as pd

from pymarxan.connectivity.features import (
    add_connectivity_features,
    metric_to_feature,
)
from pymarxan.io.readers import load_project
from pymarxan.models.grid import GridGeometry
from pymarxan.models.problem import ConservationProblem

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


class _TaggedProblem(ConservationProblem):
    """A subclass, so the test sees whether the problem's type survives."""


class TestMetricToFeature:
    def test_creates_puvspr_rows(self):
        pu_ids = [1, 2, 3, 4, 5, 6]
        metric_values = np.array([0.8, 0.2, 0.6, 0.9, 0.1, 0.5])
        feature_id = 99
        rows = metric_to_feature(pu_ids, metric_values, feature_id)
        assert len(rows) == 6
        assert set(rows.columns) == {"species", "pu", "amount"}
        assert rows["species"].iloc[0] == 99

    def test_threshold_filters(self):
        pu_ids = [1, 2, 3]
        metric_values = np.array([0.8, 0.2, 0.6])
        rows = metric_to_feature(pu_ids, metric_values, 99, threshold=0.5)
        assert len(rows) == 2


class TestAddConnectivityFeatures:
    def test_adds_feature(self):
        problem = load_project(DATA_DIR)
        metric = np.array([0.5, 0.3, 0.7, 0.2, 0.8, 0.4])
        new_problem = add_connectivity_features(
            problem,
            metrics={"connectivity": metric},
            targets={"connectivity": 2.0},
        )
        assert new_problem.n_features == problem.n_features + 1
        assert len(new_problem.pu_vs_features) > len(problem.pu_vs_features)

    def test_keeps_subclass_and_optional_fields(self):
        """The result keeps the subclass, probability, connectivity and grid.

        The old five-field rebuild dropped all four, so a PROBMODE 1 problem lost its
        chance term (v0.47.0 review V12).
        """
        base = load_project(DATA_DIR)
        probability = pd.DataFrame({"pu": [1, 2, 3, 4, 5, 6], "prob": [0.1] * 6})
        connectivity = pd.DataFrame({"id1": [1], "id2": [2], "value": [1.0]})
        grid = GridGeometry(0.0, 2.0, 1.0, 1.0, np.ones((2, 3), dtype=bool))
        problem = _TaggedProblem(
            planning_units=base.planning_units,
            features=base.features,
            pu_vs_features=base.pu_vs_features,
            boundary=base.boundary,
            parameters={**base.parameters, "PROBMODE": 1},
            probability=probability,
            connectivity=connectivity,
            grid=grid,
        )
        new_problem = add_connectivity_features(
            problem,
            metrics={"deg": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])},
            targets={"deg": 0.0},
        )
        assert type(new_problem) is _TaggedProblem
        assert new_problem.probability is probability
        assert new_problem.connectivity is connectivity
        assert new_problem.grid is grid
        assert new_problem.boundary is problem.boundary
        assert new_problem.n_features == problem.n_features + 1
        assert new_problem.parameters == problem.parameters
        assert new_problem.parameters is not problem.parameters
