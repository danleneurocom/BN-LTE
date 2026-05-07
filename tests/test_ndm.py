from __future__ import annotations

import unittest

import numpy as np

from spread_toolbox.models.ndm import NetworkDiffusionModel


class NDMTests(unittest.TestCase):
    def test_zero_rho_returns_baseline(self) -> None:
        laplacian = np.array([[1.0, -1.0], [-1.0, 1.0]])
        model = NetworkDiffusionModel(laplacian)
        baseline = np.array([[2.0, 4.0], [1.0, 3.0]])
        predicted = model.predict(baseline, np.array([1.0, 2.0]), rho=0.0)
        np.testing.assert_allclose(predicted, baseline)

    def test_diffusion_preserves_total_signal_for_combinatorial_laplacian(self) -> None:
        laplacian = np.array([[1.0, -1.0], [-1.0, 1.0]])
        model = NetworkDiffusionModel(laplacian)
        baseline = np.array([[2.0, 4.0]])
        predicted = model.predict(baseline, np.array([1.0]), rho=0.5)
        np.testing.assert_allclose(predicted.sum(axis=1), baseline.sum(axis=1))
        self.assertGreater(predicted[0, 0], baseline[0, 0])
        self.assertLess(predicted[0, 1], baseline[0, 1])


if __name__ == "__main__":
    unittest.main()
