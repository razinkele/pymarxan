from pathlib import Path

import numpy as np

from pymarxan.io.readers import load_project
from pymarxan.connectivity.features import (
    metric_to_feature,
    add_connectivity_features,
)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simple"


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
