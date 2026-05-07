from __future__ import annotations

import unittest

import numpy as np

from spread_toolbox.forecasting import MinMaxStateScaler
from spread_toolbox.models.esm import EpidemicSpreadingModel, row_normalize


class ESMTests(unittest.TestCase):
    def test_zero_beta_returns_clipped_baseline(self) -> None:
        adjacency = np.array([[0.0, 1.0], [1.0, 0.0]])
        model = EpidemicSpreadingModel(adjacency)
        baseline = np.array([[0.2, 0.8], [-0.5, 1.5]])
        predicted = model.predict(baseline, np.array([1.0, 2.0]), beta=0.0)
        np.testing.assert_allclose(predicted, np.array([[0.2, 0.8], [0.0, 1.0]]))

    def test_spread_is_bounded_and_increases_connected_empty_region(self) -> None:
        adjacency = np.array([[0.0, 1.0], [1.0, 0.0]])
        model = EpidemicSpreadingModel(adjacency, steps_per_year=24)
        baseline = np.array([[0.0, 0.7]])
        predicted = model.predict(baseline, np.array([1.0]), beta=1.0)
        self.assertGreater(predicted[0, 0], baseline[0, 0])
        self.assertGreaterEqual(predicted.min(), 0.0)
        self.assertLessEqual(predicted.max(), 1.0)

    def test_row_normalize_leaves_disconnected_rows_zero(self) -> None:
        adjacency = np.array([[0.0, 2.0], [0.0, 0.0]])
        normalized = row_normalize(adjacency)
        np.testing.assert_allclose(normalized, np.array([[0.0, 1.0], [0.0, 0.0]]))

    def test_minmax_scaler_uses_training_values(self) -> None:
        scaler = MinMaxStateScaler.fit(np.array([[1.0, 2.0], [3.0, 6.0]]))
        transformed = scaler.transform(np.array([[2.0, 4.0]]))
        np.testing.assert_allclose(transformed, np.array([[0.5, 0.5]]))
        np.testing.assert_allclose(scaler.inverse_transform(transformed), np.array([[2.0, 4.0]]))


if __name__ == "__main__":
    unittest.main()
