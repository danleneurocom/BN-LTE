from __future__ import annotations

import unittest

import numpy as np

from spread_toolbox.models.fkpp import GraphFKPPModel, LocalFKPPModel, fit_local_fkpp_components, normalize_laplacian
from spread_toolbox.models.local_fkpp_bayes import group_indices_by_rid, median_or, predict_by_subject


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

    def test_local_fkpp_zero_parameters_return_clipped_baseline(self) -> None:
        laplacian = np.array([[1.0, -1.0], [-1.0, 1.0]])
        model = LocalFKPPModel(
            laplacian,
            u0=np.array([1.0, 1.2]),
            cc=np.array([2.0, 2.2]),
            laplacian_normalization="none",
        )
        baseline = np.array([[0.5, 1.6], [1.5, 3.0]])
        predicted = model.predict(baseline, np.array([1.0, 2.0]), rho=0.0, alpha=0.0)
        np.testing.assert_allclose(predicted, np.array([[1.0, 1.6], [1.5, 2.2]]))

    def test_local_fkpp_reaction_only_increases_above_floor(self) -> None:
        laplacian = np.zeros((2, 2))
        model = LocalFKPPModel(
            laplacian,
            u0=np.array([1.0, 1.0]),
            cc=np.array([2.0, 2.0]),
            laplacian_normalization="none",
            steps_per_year=24,
        )
        baseline = np.array([[1.2, 1.8]])
        predicted = model.predict(baseline, np.array([1.0]), rho=0.0, alpha=1.0)
        self.assertGreater(predicted[0, 0], baseline[0, 0])
        self.assertGreater(predicted[0, 1], baseline[0, 1])
        self.assertLessEqual(predicted.max(), 2.0)

    def test_local_fkpp_components_find_low_and_high_tau_components(self) -> None:
        rng = np.random.default_rng(7)
        low = rng.normal(1.0, 0.03, size=(80, 1))
        high = rng.normal(2.0, 0.05, size=(80, 1))
        data = np.vstack([low, high])
        components = fit_local_fkpp_components(data, carrying_capacity_quantile=0.99, random_seed=7)
        self.assertAlmostEqual(float(components.u0[0]), 1.0, delta=0.08)
        self.assertGreater(float(components.high_component_mean[0]), 1.8)
        self.assertGreater(float(components.cc[0]), float(components.high_component_mean[0]))

    def test_local_fkpp_bayes_helpers_group_and_predict_with_fallback(self) -> None:
        pairs = [{"RID": "1"}, {"RID": "2"}, {"RID": "1"}]
        grouped = group_indices_by_rid(pairs, np.array([0, 2]))
        np.testing.assert_array_equal(grouped["1"], np.array([0, 2]))

        laplacian = np.zeros((1, 1))
        model = LocalFKPPModel(
            laplacian,
            u0=np.array([1.0]),
            cc=np.array([2.0]),
            laplacian_normalization="none",
        )
        baseline = np.array([[1.2], [1.4], [1.6]])
        predicted = predict_by_subject(
            model,
            pairs,
            baseline,
            np.array([1.0, 1.0, 1.0]),
            baseline.shape,
            {"1": (0.0, 0.0)},
            fallback_rho=0.0,
            fallback_alpha=0.0,
        )
        np.testing.assert_allclose(predicted, baseline)
        self.assertAlmostEqual(median_or(3.0, [1.0, np.nan, 5.0]), 3.0)


if __name__ == "__main__":
    unittest.main()
