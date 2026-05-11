from __future__ import annotations

import unittest

import numpy as np

from spread_toolbox.models.individualized_residual import (
    apply_individualized_residual_correction,
    build_individualized_residual_features,
    choose_residual_shrinkage,
    fit_ridge_residual_model,
)


class IndividualizedResidualTests(unittest.TestCase):
    def test_feature_library_contains_backbone_and_region_features(self) -> None:
        baseline = np.array([[0.1, 0.2], [0.3, 0.4]])
        backbone = baseline + 0.05
        features = build_individualized_residual_features(
            baseline=baseline,
            backbone_prediction=backbone,
            time_years=np.array([1.0, 2.0]),
            laplacian=np.eye(2),
            region_labels=["lh_a", "rh_b"],
            pair_covariates={"apoe": np.array([0.0, 1.0])},
            regional_covariates={"amyloid": np.ones_like(baseline)},
            include_region_bias=True,
        )
        self.assertEqual(features.values.shape[0:2], baseline.shape)
        self.assertIn("fkpp_delta_rate", features.names)
        self.assertIn("apoe*fkpp_growth_drive", features.names)
        self.assertIn("amyloid*baseline_tau", features.names)
        self.assertIn("region_bias:lh_a", features.names)

    def test_ridge_residual_fit_learns_train_residual_rate(self) -> None:
        rng = np.random.default_rng(42)
        baseline = rng.uniform(0.05, 0.8, size=(12, 2))
        backbone = np.clip(baseline + 0.03, 0.0, 1.0)
        features = build_individualized_residual_features(
            baseline=baseline,
            backbone_prediction=backbone,
            time_years=np.ones(12),
            laplacian=np.eye(2),
            region_labels=["a", "b"],
            include_region_bias=False,
        )
        target_rate = 0.2 * baseline - 0.1 * (backbone - baseline)
        fit = fit_ridge_residual_model(
            features,
            target_rate,
            row_indices=np.arange(12),
            pair_groups=np.asarray([str(index) for index in range(12)]),
            alphas=(0.01,),
            cv_folds=3,
        )
        predicted = fit.predict_rate(features.values)
        self.assertLess(float(np.mean((predicted - target_rate) ** 2)), 1.0e-3)

    def test_shrinkage_and_correction_are_train_only_utilities(self) -> None:
        backbone = np.array([[0.2, 0.3], [0.4, 0.5]])
        observed = backbone + 0.1
        rate = np.full_like(backbone, 0.1)
        shrinkage, rows = choose_residual_shrinkage(
            backbone_prediction=backbone,
            observed=observed,
            time_years=np.ones(2),
            residual_rate=rate,
            row_indices=np.array([0, 1]),
            candidates=(0.0, 0.5, 1.0),
        )
        self.assertEqual(shrinkage, 1.0)
        self.assertEqual(len(rows), 3)
        corrected = apply_individualized_residual_correction(backbone, np.ones(2), rate, shrinkage=1.0)
        np.testing.assert_allclose(corrected, observed)


if __name__ == "__main__":
    unittest.main()
