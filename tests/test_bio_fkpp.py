from __future__ import annotations

import unittest

import numpy as np

from spread_toolbox.models.bio_fkpp import BioFKPPModel


class BioFKPPTests(unittest.TestCase):
    def _make(self, n_regions: int = 2, **kw) -> BioFKPPModel:
        return BioFKPPModel(np.zeros((n_regions, n_regions)), laplacian_normalization="none", **kw)

    def _predict_simple(self, model, baseline, time_years, rho=0.04, alpha=0.08):
        """Helper: predict with scalar rho/alpha and no bio modulation."""
        n = baseline.shape[0] if baseline.ndim == 2 else 1
        n_r = model.laplacian.shape[0]
        re = np.full((n, n_r), rho)
        ae = np.full((n, n_r), alpha)
        cl = np.zeros((n, n_r))
        sd = np.zeros((n, n_r))
        return model.predict(baseline, time_years, rho_eff=re, alpha_eff=ae, clearance=cl, seeding=sd)

    # ------------------------------------------------------------------ core ODE
    def test_zero_params_matches_global_fkpp(self) -> None:
        from spread_toolbox.models.fkpp import GraphFKPPModel
        laplacian = np.array([[1.0, -1.0], [-1.0, 1.0]])
        model = BioFKPPModel(laplacian, laplacian_normalization="none", steps_per_year=12)
        fkpp = GraphFKPPModel(laplacian, laplacian_normalization="none", steps_per_year=12)
        baseline = np.array([[0.2, 0.5], [0.4, 0.3]])
        time_years = np.array([1.5, 2.0])
        np.testing.assert_allclose(
            self._predict_simple(model, baseline, time_years),
            fkpp.predict(baseline, time_years, rho=0.04, alpha=0.08),
            atol=1.0e-10,
        )

    def test_positive_seeding_increases_prediction(self) -> None:
        model = self._make()
        baseline = np.array([[0.3, 0.3]])
        time_years = np.array([2.0])
        ae = np.full((1, 2), 0.05)
        re = np.zeros((1, 2))
        cl = np.zeros((1, 2))
        sd_base = np.zeros((1, 2))
        sd_mod = np.array([[0.05, 0.0]])
        pred_base = model.predict(baseline, time_years, rho_eff=re, alpha_eff=ae, clearance=cl, seeding=sd_base)
        pred_mod = model.predict(baseline, time_years, rho_eff=re, alpha_eff=ae, clearance=cl, seeding=sd_mod)
        self.assertGreater(float(pred_mod[0, 0]), float(pred_base[0, 0]))
        self.assertAlmostEqual(float(pred_mod[0, 1]), float(pred_base[0, 1]), places=8)

    def test_negative_seeding_decreases_prediction(self) -> None:
        model = self._make()
        baseline = np.array([[0.5, 0.5]])
        time_years = np.array([2.0])
        ae = np.full((1, 2), 0.1)
        re = np.zeros((1, 2))
        cl = np.zeros((1, 2))
        sd_no = np.zeros((1, 2))
        sd = np.array([[-0.02, 0.0]])
        pred_no = model.predict(baseline, time_years, rho_eff=re, alpha_eff=ae, clearance=cl, seeding=sd_no)
        pred_sd = model.predict(baseline, time_years, rho_eff=re, alpha_eff=ae, clearance=cl, seeding=sd)
        self.assertLess(float(pred_sd[0, 0]), float(pred_no[0, 0]))

    def test_states_stay_in_unit_interval(self) -> None:
        model = self._make(n_regions=3, steps_per_year=24)
        baseline = np.array([[0.1, 0.5, 0.9]])
        time_years = np.array([10.0])
        ae = np.full((1, 3), 5.0)
        re = np.zeros((1, 3))
        cl = np.zeros((1, 3))
        sd = np.full((1, 3), 1.0)
        predicted = model.predict(baseline, time_years, rho_eff=re, alpha_eff=ae, clearance=cl, seeding=sd)
        self.assertGreaterEqual(float(predicted.min()), 0.0)
        self.assertLessEqual(float(predicted.max()), 1.0 + 1.0e-6)

    def test_positive_clearance_reduces_prediction(self) -> None:
        model = self._make()
        baseline = np.array([[0.4, 0.4]])
        time_years = np.array([2.0])
        ae = np.full((1, 2), 0.05)
        re = np.zeros((1, 2))
        sd = np.zeros((1, 2))
        cl_zero = np.zeros((1, 2))
        cl_pos = np.array([[0.1, 0.0]])
        pred_zero = model.predict(baseline, time_years, rho_eff=re, alpha_eff=ae, clearance=cl_zero, seeding=sd)
        pred_pos = model.predict(baseline, time_years, rho_eff=re, alpha_eff=ae, clearance=cl_pos, seeding=sd)
        self.assertLess(float(pred_pos[0, 0]), float(pred_zero[0, 0]))
        self.assertAlmostEqual(float(pred_pos[0, 1]), float(pred_zero[0, 1]), places=8)

    def test_higher_rho_eff_increases_diffusion(self) -> None:
        laplacian = np.array([[1.0, -1.0], [-1.0, 1.0]])
        model = BioFKPPModel(laplacian, laplacian_normalization="none", steps_per_year=12)
        baseline = np.array([[0.8, 0.2]])   # high left, low right — diffusion flows left→right
        time_years = np.array([2.0])
        ae = np.full((1, 2), 0.0)
        cl = np.zeros((1, 2))
        sd = np.zeros((1, 2))
        re_low = np.full((1, 2), 0.01)
        re_high = np.full((1, 2), 0.1)
        pred_low = model.predict(baseline, time_years, rho_eff=re_low, alpha_eff=ae, clearance=cl, seeding=sd)
        pred_high = model.predict(baseline, time_years, rho_eff=re_high, alpha_eff=ae, clearance=cl, seeding=sd)
        # region 0 loses tau to diffusion; higher rho → lower region 0 value
        self.assertLess(float(pred_high[0, 0]), float(pred_low[0, 0]))

    # ------------------------------------------------------------------ build_rho_eff
    def test_build_rho_eff_no_thickness(self) -> None:
        model = BioFKPPModel(np.eye(2), laplacian_normalization="none")
        re = model.build_rho_eff(3, 0.05, None, delta_rho_thickness=0.1)
        np.testing.assert_allclose(re, np.full((3, 2), 0.05))

    def test_build_rho_eff_with_thickness(self) -> None:
        model = BioFKPPModel(np.eye(2), laplacian_normalization="none")
        thickness = np.array([[1.0, 2.0], [3.0, -1.0]])
        re = model.build_rho_eff(2, 0.1, thickness, delta_rho_thickness=0.05)
        # pair[1,1]: 0.1 + 0.05*(-1) = 0.05, not negative so no clipping
        expected = np.array([[0.15, 0.2], [0.25, 0.05]])
        np.testing.assert_allclose(re, expected, atol=1.0e-12)

    def test_build_rho_eff_clips_to_zero(self) -> None:
        model = BioFKPPModel(np.eye(2), laplacian_normalization="none")
        thickness = np.array([[-5.0, 2.0]])
        re = model.build_rho_eff(1, 0.1, thickness, delta_rho_thickness=0.05)
        # pair[0,0]: 0.1 + 0.05*(-5) = -0.15, clipped to 0
        self.assertEqual(float(re[0, 0]), 0.0)
        np.testing.assert_allclose(re[0, 1], 0.2, atol=1.0e-12)

    # ------------------------------------------------------------------ build_clearance
    def test_build_clearance_with_thickness(self) -> None:
        model = BioFKPPModel(np.eye(2), laplacian_normalization="none")
        thickness = np.array([[1.0, 2.0], [3.0, 4.0]])
        cl = model.build_clearance(2, thickness, lambda_clearance_thickness=0.1)
        np.testing.assert_allclose(cl, 0.1 * thickness, atol=1.0e-12)

    def test_build_clearance_no_thickness(self) -> None:
        model = BioFKPPModel(np.eye(2), laplacian_normalization="none")
        cl = model.build_clearance(3, None, lambda_clearance_thickness=0.5)
        np.testing.assert_allclose(cl, np.zeros((3, 2)))

    # ------------------------------------------------------------------ build_seeding
    def test_seeding_is_covariate_times_baseline(self) -> None:
        model = BioFKPPModel(np.eye(2), laplacian_normalization="none")
        baseline = np.array([[0.2, 0.4], [0.6, 0.8]])
        amyloid = np.array([[1.0, 0.5], [2.0, 1.0]])
        sd = model.build_seeding(baseline, amyloid, None, gamma_seeding_amyloid=0.1, gamma_seeding_thickness=0.0)
        np.testing.assert_allclose(sd, 0.1 * amyloid * baseline, atol=1.0e-12)

    def test_seeding_thickness_term_adds_independently(self) -> None:
        model = BioFKPPModel(np.eye(2), laplacian_normalization="none")
        baseline = np.ones((2, 2)) * 0.5
        thickness = np.array([[1.0, 2.0], [3.0, 4.0]])
        sd = model.build_seeding(baseline, None, thickness, gamma_seeding_amyloid=0.0, gamma_seeding_thickness=0.2)
        np.testing.assert_allclose(sd, 0.2 * thickness * baseline, atol=1.0e-12)

    # ------------------------------------------------------------------ build_alpha_eff
    def test_alpha_eff_adds_all_covariates(self) -> None:
        model = BioFKPPModel(np.eye(2), laplacian_normalization="none")
        amyloid = np.array([[1.0, 0.0], [0.0, 1.0]])
        thickness = np.array([[-1.0, 1.0], [1.0, -1.0]])
        apoe4 = np.array([0.0, 1.0])
        ae = model.build_alpha_eff(2, 0.1, amyloid, thickness, apoe4,
                                   beta_growth_amyloid=0.02, beta_growth_thickness=-0.01, beta_growth_apoe4=0.005)
        # pair 0, region 0: 0.1 + 0.02*1 - 0.01*(-1) + 0.005*0 = 0.13
        # pair 0, region 1: 0.1 + 0.02*0 - 0.01*(1) + 0.005*0 = 0.09
        np.testing.assert_allclose(ae[0, 0], 0.13, atol=1.0e-10)
        np.testing.assert_allclose(ae[0, 1], 0.09, atol=1.0e-10)

    # ------------------------------------------------------------------ fit() signals
    def test_fit_finds_positive_amyloid_growth_beta(self) -> None:
        rng = np.random.default_rng(42)
        n = 60
        model = self._make(steps_per_year=12)
        baseline = rng.uniform(0.1, 0.5, size=(n, 2))
        amyloid = rng.normal(0.0, 1.0, size=(n, 2))
        time_years = np.ones(n) * 2.0
        re = np.zeros((n, 2))
        ae = 0.05 + 0.04 * amyloid
        cl = np.zeros((n, 2))
        sd = np.zeros((n, 2))
        observed = model.predict(baseline, time_years, rho_eff=re, alpha_eff=ae, clearance=cl, seeding=sd)
        fit = model.fit(baseline, observed, time_years, amyloid=amyloid, thickness=None,
                        train_indices=np.arange(n), rho_bounds=(0.0, 0.5), alpha_bounds=(0.0, 0.5))
        self.assertGreater(fit.beta_growth_amyloid, 0.0)

    def test_fit_finds_negative_gamma_seeding_amyloid(self) -> None:
        rng = np.random.default_rng(13)
        n = 60
        model = self._make(steps_per_year=12)
        baseline = rng.uniform(0.2, 0.7, size=(n, 2))
        amyloid = rng.normal(0.0, 1.0, size=(n, 2))
        time_years = np.ones(n) * 2.0
        re = np.zeros((n, 2))
        ae = np.full((n, 2), 0.05)
        cl = np.zeros((n, 2))
        sd = -0.03 * amyloid * baseline
        observed = model.predict(baseline, time_years, rho_eff=re, alpha_eff=ae, clearance=cl, seeding=sd)
        fit = model.fit(baseline, observed, time_years, amyloid=amyloid, thickness=None,
                        train_indices=np.arange(n), rho_bounds=(0.0, 0.5), alpha_bounds=(0.0, 0.5))
        self.assertLess(fit.gamma_seeding_amyloid, 0.0)

    def test_fit_finds_positive_gamma_seeding_thickness(self) -> None:
        # Provide amyloid for growth modulation so the optimizer can cleanly
        # separate amyloid→alpha from thickness→seeding (avoids 4-way collinearity).
        rng = np.random.default_rng(77)
        n = 80
        model = self._make(steps_per_year=12)
        baseline = rng.uniform(0.2, 0.6, size=(n, 2))
        amyloid = rng.normal(0.0, 1.0, size=(n, 2))
        thickness = rng.normal(0.0, 1.0, size=(n, 2))
        time_years = np.ones(n) * 2.0
        re = np.zeros((n, 2))
        ae = 0.05 + 0.04 * amyloid     # amyloid drives growth
        cl = np.zeros((n, 2))
        sd = 0.05 * thickness * baseline  # thickness drives seeding
        observed = model.predict(baseline, time_years, rho_eff=re, alpha_eff=ae, clearance=cl, seeding=sd)
        fit = model.fit(baseline, observed, time_years, amyloid=amyloid, thickness=thickness,
                        train_indices=np.arange(n), rho_bounds=(0.0, 0.5), alpha_bounds=(0.0, 0.5))
        self.assertGreater(fit.gamma_seeding_thickness, 0.0)

    def test_fit_recovers_positive_delta_rho(self) -> None:
        from spread_toolbox.models.fkpp import normalize_laplacian
        rng = np.random.default_rng(99)
        n = 80
        laplacian_raw = np.array([[1.0, -1.0], [-1.0, 1.0]])
        model = BioFKPPModel(laplacian_raw, laplacian_normalization="none", steps_per_year=12)
        baseline = rng.uniform(0.1, 0.6, size=(n, 2))
        thickness = rng.normal(0.0, 1.0, size=(n, 2))
        time_years = np.ones(n) * 2.0
        # Generate data with delta_rho > 0: rho_eff = 0.05 + 0.03*thickness
        re_true = np.maximum(0.05 + 0.03 * thickness, 0.0)
        ae = np.full((n, 2), 0.05)
        cl = np.zeros((n, 2))
        sd = np.zeros((n, 2))
        observed = model.predict(baseline, time_years, rho_eff=re_true, alpha_eff=ae, clearance=cl, seeding=sd)
        fit = model.fit(baseline, observed, time_years, amyloid=None, thickness=thickness,
                        train_indices=np.arange(n), rho_bounds=(0.0, 0.5), alpha_bounds=(0.0, 0.5),
                        delta_rho_bounds=(-1.0, 1.0))
        self.assertGreater(fit.delta_rho_thickness, 0.0)


if __name__ == "__main__":
    unittest.main()
