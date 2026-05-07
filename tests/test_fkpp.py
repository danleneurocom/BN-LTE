from __future__ import annotations

import unittest

import numpy as np

from spread_toolbox.models.fkpp import GraphFKPPModel, normalize_laplacian


class FKPPTests(unittest.TestCase):
    def test_zero_parameters_return_clipped_baseline(self) -> None:
        laplacian = np.array([[1.0, -1.0], [-1.0, 1.0]])
        model = GraphFKPPModel(laplacian, laplacian_normalization="none")
        baseline = np.array([[0.2, 0.8], [-0.5, 1.5]])
        predicted = model.predict(baseline, np.array([1.0, 2.0]), rho=0.0, alpha=0.0)
        np.testing.assert_allclose(predicted, np.array([[0.2, 0.8], [0.0, 1.0]]))

    def test_reaction_only_increases_unsaturated_state(self) -> None:
        laplacian = np.zeros((2, 2))
        model = GraphFKPPModel(laplacian, laplacian_normalization="none", steps_per_year=24)
        baseline = np.array([[0.2, 0.8]])
        predicted = model.predict(baseline, np.array([1.0]), rho=0.0, alpha=1.0)
        self.assertGreater(predicted[0, 0], baseline[0, 0])
        self.assertGreater(predicted[0, 1], baseline[0, 1])
        self.assertLessEqual(predicted.max(), 1.0)

    def test_diffusion_only_preserves_total_signal_without_clipping(self) -> None:
        laplacian = np.array([[1.0, -1.0], [-1.0, 1.0]])
        model = GraphFKPPModel(laplacian, laplacian_normalization="none", steps_per_year=24)
        baseline = np.array([[0.2, 0.8]])
        predicted = model.predict(baseline, np.array([0.5]), rho=0.2, alpha=0.0)
        np.testing.assert_allclose(predicted.sum(axis=1), baseline.sum(axis=1), atol=1.0e-5)
        self.assertGreater(predicted[0, 0], baseline[0, 0])
        self.assertLess(predicted[0, 1], baseline[0, 1])

    def test_spectral_normalization_scales_by_largest_eigenvalue(self) -> None:
        laplacian = np.array([[1.0, -1.0], [-1.0, 1.0]])
        normalized, scale = normalize_laplacian(laplacian, "spectral")
        self.assertAlmostEqual(scale, 2.0)
        np.testing.assert_allclose(normalized, laplacian / 2.0)


if __name__ == "__main__":
    unittest.main()
