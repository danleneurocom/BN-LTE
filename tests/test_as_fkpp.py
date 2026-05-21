from __future__ import annotations

import unittest

import numpy as np

from spread_toolbox.models.as_fkpp import ASFKPPModel, _r2


class ASFKPPTests(unittest.TestCase):

    def _make(self, n_regions: int = 4, n_eigenmodes: int = 2) -> ASFKPPModel:
        laplacian = np.diag([1.0] * n_regions) - np.ones((n_regions, n_regions)) / n_regions
        return ASFKPPModel(laplacian, laplacian_normalization="none",
                           steps_per_year=12, n_eigenmodes=n_eigenmodes)

    # ------------------------------------------------------------------ eigenmodes
    def test_eigenmode_matrix_shape(self) -> None:
        model = self._make(n_regions=6, n_eigenmodes=3)
        self.assertEqual(model.eigenvectors.shape, (6, 3))
        self.assertEqual(model.eigenvalues.shape, (3,))

    def test_eigenmodes_capped_at_n_regions(self) -> None:
        model = ASFKPPModel(np.eye(3), laplacian_normalization="none", n_eigenmodes=100)
        self.assertEqual(model.n_eigenmodes, 3)

    def test_eigenmode_orthonormality(self) -> None:
        model = self._make(n_regions=6, n_eigenmodes=4)
        VtV = model.eigenvectors.T @ model.eigenvectors
        np.testing.assert_allclose(VtV, np.eye(4), atol=1.0e-10)

    # ------------------------------------------------------------------ build_features
    def test_build_features_shape_no_covariates(self) -> None:
        model = self._make(n_regions=4, n_eigenmodes=3)
        baseline = np.random.default_rng(0).uniform(0.1, 0.8, size=(5, 4))
        X, names = model.build_features(baseline, None, None, None, None)
        # 3 tau burden + 1 hub + 3 eigenmodes = 7
        self.assertEqual(X.shape, (5, 7))
        self.assertEqual(len(names), 7)

    def test_build_features_shape_all_covariates(self) -> None:
        model = self._make(n_regions=4, n_eigenmodes=2)
        n = 8
        rng = np.random.default_rng(1)
        baseline = rng.uniform(0.1, 0.8, size=(n, 4))
        amyloid = rng.normal(0, 1, size=(n, 4))
        thickness = rng.normal(0, 1, size=(n, 4))
        apoe4 = rng.integers(0, 2, size=n).astype(float)
        ptau = rng.normal(200, 50, size=n)
        X, names = model.build_features(baseline, amyloid, thickness, apoe4, ptau)
        # 3 tau burden + 1 hub + 2 eigenmodes + 1 amyloid + 1 thickness + 1 apoe4 + 2 ptau = 11
        self.assertEqual(X.shape, (n, 11))
        self.assertIn("eigenmode_0", names)
        self.assertIn("amyloid_mean", names)
        self.assertIn("tau_burden_mean", names)

    def test_eigenmode_projection_values(self) -> None:
        model = self._make(n_regions=4, n_eigenmodes=2)
        baseline = np.ones((3, 4)) * 0.5   # uniform → loads only onto first eigenmode
        X, names = model.build_features(baseline, None, None, None, None)
        k0 = names.index("eigenmode_0")
        # all rows should give the same projection since baseline is identical
        np.testing.assert_allclose(X[:, k0], X[0, k0], atol=1.0e-10)

    def test_ptau_imputation(self) -> None:
        model = self._make()
        baseline = np.ones((4, 4)) * 0.3
        ptau = np.array([100.0, np.nan, 200.0, np.nan])
        X, names = model.build_features(baseline, None, None, None, ptau, ptau_train_median=150.0)
        i_ptau = names.index("plasma_ptau181")
        i_obs = names.index("plasma_ptau181_observed")
        np.testing.assert_allclose(X[1, i_ptau], 150.0)   # imputed
        np.testing.assert_allclose(X[0, i_ptau], 100.0)   # observed
        np.testing.assert_allclose(X[1, i_obs], 0.0)      # missing flag
        np.testing.assert_allclose(X[0, i_obs], 1.0)      # observed flag

    # ------------------------------------------------------------------ per-pair fitting
    def test_per_pair_fit_finds_higher_alpha(self) -> None:
        """When data was generated with a higher alpha, per-pair fit should find delta_alpha > 0."""
        rng = np.random.default_rng(7)
        laplacian = np.zeros((2, 2))
        model = ASFKPPModel(laplacian, laplacian_normalization="none", steps_per_year=12)
        from spread_toolbox.models.fkpp import GraphFKPPModel
        backbone = GraphFKPPModel(laplacian, laplacian_normalization="none", steps_per_year=12)

        s0 = rng.uniform(0.2, 0.5, size=(1, 2))
        t = 3.0
        # Generate with alpha_true = 0.10 (higher than backbone 0.05)
        s1 = backbone.predict(s0, np.array([t]), rho=0.0, alpha=0.10)[0]

        dr, da, ok = model._fit_one_pair(
            s0[0], s1, t, rho=0.0, alpha=0.05,
            dr_bounds=(-0.04, 0.2), da_bounds=(-0.04, 0.2),
            backbone=backbone, maxiter=50,
        )
        self.assertGreater(da, 0.0)

    def test_per_pair_fit_finds_lower_alpha(self) -> None:
        rng = np.random.default_rng(8)
        laplacian = np.zeros((2, 2))
        model = ASFKPPModel(laplacian, laplacian_normalization="none", steps_per_year=12)
        from spread_toolbox.models.fkpp import GraphFKPPModel
        backbone = GraphFKPPModel(laplacian, laplacian_normalization="none", steps_per_year=12)

        s0 = rng.uniform(0.3, 0.6, size=(1, 2))
        t = 3.0
        # Generate with alpha_true = 0.01 (lower than backbone 0.05)
        s1 = backbone.predict(s0, np.array([t]), rho=0.0, alpha=0.01)[0]

        _, da, _ = model._fit_one_pair(
            s0[0], s1, t, rho=0.0, alpha=0.05,
            dr_bounds=(-0.04, 0.2), da_bounds=(-0.04, 0.2),
            backbone=backbone, maxiter=50,
        )
        self.assertLess(da, 0.0)

    # ------------------------------------------------------------------ full fit
    def test_fit_amortises_alpha_from_tau_burden(self) -> None:
        """Subjects with high tau burden generated with high alpha; amortisation should capture R2 > 0."""
        rng = np.random.default_rng(42)
        n, n_r = 80, 4
        laplacian = np.zeros((n_r, n_r))
        model = ASFKPPModel(laplacian, laplacian_normalization="none",
                            steps_per_year=12, n_eigenmodes=2)
        from spread_toolbox.models.fkpp import GraphFKPPModel
        backbone = GraphFKPPModel(laplacian, laplacian_normalization="none", steps_per_year=12)

        baseline = rng.uniform(0.1, 0.7, size=(n, n_r))
        tau_burden = baseline.mean(axis=1)
        # alpha_true = 0.05 + 0.05 * tau_burden (high burden → faster growth)
        alpha_true = 0.05 + 0.05 * tau_burden
        observed = np.stack([
            backbone.predict(baseline[[i]], np.array([2.0]), rho=0.0, alpha=float(alpha_true[i]))[0]
            for i in range(n)
        ])
        time_years = np.ones(n) * 2.0

        fit = model.fit(
            baseline, observed, time_years,
            amyloid=None, thickness=None,
            train_indices=np.arange(n),
            rho_bounds=(0.0, 0.5), alpha_bounds=(0.0, 0.5),
            per_pair_maxiter=30, backbone_maxiter=40,
        )
        # Amortisation should capture that tau_burden predicts delta_alpha
        self.assertGreater(fit.amortisation_alpha_r2, 0.1)

    def test_fit_stage2a_mse_lower_than_stage1(self) -> None:
        rng = np.random.default_rng(5)
        n, n_r = 40, 2
        laplacian = np.zeros((n_r, n_r))
        model = ASFKPPModel(laplacian, laplacian_normalization="none", steps_per_year=12)
        from spread_toolbox.models.fkpp import GraphFKPPModel
        backbone = GraphFKPPModel(laplacian, laplacian_normalization="none", steps_per_year=12)
        baseline = rng.uniform(0.2, 0.6, size=(n, n_r))
        # Heterogeneous alpha across subjects
        alpha_i = rng.uniform(0.02, 0.12, size=n)
        observed = np.stack([
            backbone.predict(baseline[[i]], np.array([2.0]), rho=0.0, alpha=float(alpha_i[i]))[0]
            for i in range(n)
        ])
        fit = model.fit(
            baseline, observed, np.ones(n) * 2.0,
            amyloid=None, thickness=None,
            train_indices=np.arange(n),
            rho_bounds=(0.0, 0.3), alpha_bounds=(0.0, 0.3),
            per_pair_maxiter=30, backbone_maxiter=40,
        )
        self.assertLess(fit.stage2a_train_mse, fit.stage1_train_mse)

    # ------------------------------------------------------------------ predict
    def test_predict_output_shape(self) -> None:
        rng = np.random.default_rng(3)
        n, n_r = 20, 4
        laplacian = np.zeros((n_r, n_r))
        model = ASFKPPModel(laplacian, laplacian_normalization="none",
                            steps_per_year=12, n_eigenmodes=2)
        baseline = rng.uniform(0.1, 0.5, size=(n, n_r))
        observed = baseline + rng.uniform(0.0, 0.1, size=(n, n_r))
        observed = np.clip(observed, 0.0, 1.0)
        fit = model.fit(
            baseline, observed, np.ones(n) * 2.0,
            amyloid=None, thickness=None,
            train_indices=np.arange(n),
            rho_bounds=(0.0, 0.2), alpha_bounds=(0.0, 0.2),
            per_pair_maxiter=20, backbone_maxiter=30,
        )
        predicted = model.predict(baseline, np.ones(n) * 2.0, fit,
                                  amyloid=None, thickness=None)
        self.assertEqual(predicted.shape, (n, n_r))

    def test_predict_stays_in_unit_interval(self) -> None:
        rng = np.random.default_rng(9)
        n, n_r = 20, 4
        laplacian = np.zeros((n_r, n_r))
        model = ASFKPPModel(laplacian, laplacian_normalization="none",
                            steps_per_year=12, n_eigenmodes=2)
        baseline = rng.uniform(0.1, 0.6, size=(n, n_r))
        observed = np.clip(baseline + rng.uniform(0, 0.1, (n, n_r)), 0, 1)
        fit = model.fit(
            baseline, observed, np.ones(n) * 3.0,
            amyloid=None, thickness=None,
            train_indices=np.arange(n),
            rho_bounds=(0.0, 0.2), alpha_bounds=(0.0, 0.5),
            per_pair_maxiter=20, backbone_maxiter=30,
        )
        pred = model.predict(baseline, np.ones(n) * 3.0, fit,
                             amyloid=None, thickness=None)
        self.assertGreaterEqual(float(pred.min()), 0.0)
        self.assertLessEqual(float(pred.max()), 1.0 + 1.0e-6)

    # ------------------------------------------------------------------ helpers
    def test_r2_perfect(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(_r2(y, y), 1.0)

    def test_r2_mean_prediction(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(_r2(y, np.full_like(y, y.mean())), 0.0)


if __name__ == "__main__":
    unittest.main()
