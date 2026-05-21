from __future__ import annotations

import unittest

import numpy as np

from spread_toolbox.models.bn_lte import (
    bootstrap_edge_probabilities,
    fit_bn_lte_model,
    fit_pseudotime_embedding,
    stable_edges_from_bootstrap,
)


class BNLTEPrototypeTests(unittest.TestCase):
    def test_pseudotime_orients_with_mean_burden(self) -> None:
        rng = np.random.default_rng(10)
        burden = np.linspace(0.0, 1.0, 40)
        baseline = np.column_stack(
            [
                1.0 + burden + rng.normal(0.0, 0.02, burden.size),
                0.5 + 0.5 * burden + rng.normal(0.0, 0.02, burden.size),
            ]
        )
        embedding = fit_pseudotime_embedding(baseline, train_indices=np.arange(40))
        z = embedding.transform(baseline)

        self.assertGreater(float(np.corrcoef(z, np.mean(baseline, axis=1))[0, 1]), 0.9)
        self.assertGreaterEqual(float(np.min(z)), 0.0)
        self.assertLessEqual(float(np.max(z)), 1.0)

    def test_progression_ordered_model_learns_early_to_late_signal(self) -> None:
        baseline, observed, time_years, groups = synthetic_tau_progression()
        fit = fit_bn_lte_model(
            baseline=baseline,
            observed=observed,
            time_years=time_years,
            region_labels=["early", "late", "noise"],
            train_indices=np.arange(45),
            pair_groups=groups,
            model_name="toy_bn_lte",
            parent_mode="progression_ordered",
            max_parents_per_child=2,
            n_knots=1,
            ridge_alphas=(0.01, 0.1, 1.0),
            cv_folds=3,
            edge_effect_threshold=0.02,
        )
        predicted = fit.predict(baseline, time_years)
        edge_rows = fit.edge_rows()
        early_to_late = [row for row in edge_rows if row["parent"] == "early" and row["child"] == "late"]

        self.assertEqual(predicted.shape, baseline.shape)
        self.assertTrue(early_to_late)
        self.assertTrue(early_to_late[0]["included_by_effect_threshold"])
        self.assertLess(float(np.mean((predicted[:45, 1] - observed[:45, 1]) ** 2)), 0.01)

    def test_bootstrap_rows_can_define_stable_edges(self) -> None:
        baseline, observed, time_years, groups = synthetic_tau_progression()
        rows = bootstrap_edge_probabilities(
            baseline=baseline,
            observed=observed,
            time_years=time_years,
            region_labels=["early", "late", "noise"],
            train_indices=np.arange(45),
            pair_groups=groups,
            iterations=3,
            random_seed=123,
            model_name="toy_bootstrap",
            parent_mode="progression_ordered",
            include_roots=False,
            root_names=[],
            root_values=None,
            include_self_history=True,
            max_parents_per_child=2,
            n_knots=1,
            spline_degree=0,
            ridge_alphas=(0.1,),
            edge_effect_threshold=0.02,
        )
        stable = stable_edges_from_bootstrap(rows, pip_threshold=0.0)

        self.assertTrue(rows)
        self.assertIn(("early", "late"), stable)


def synthetic_tau_progression() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    n = 60
    early = rng.normal(1.2, 0.12, n)
    late = rng.normal(0.7, 0.10, n)
    noise = rng.normal(0.8, 0.08, n)
    baseline = np.column_stack([early, late, noise])
    time_years = rng.uniform(0.8, 1.5, n)
    rate = np.zeros_like(baseline)
    rate[:, 0] = 0.01 * early
    rate[:, 1] = 0.18 * early - 0.03 * late
    rate[:, 2] = rng.normal(0.0, 0.005, n)
    observed = baseline + time_years[:, None] * rate + rng.normal(0.0, 0.005, baseline.shape)
    groups = np.asarray([str(index) for index in range(n)], dtype=object)
    return baseline, observed, time_years, groups


if __name__ == "__main__":
    unittest.main()
