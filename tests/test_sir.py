from __future__ import annotations

import unittest

import numpy as np

from spread_toolbox.models.sir import GraphSIRModel, normalize_compartments


class GraphSIRTests(unittest.TestCase):
    def test_zero_parameters_return_clipped_infected_baseline(self) -> None:
        model = GraphSIRModel(np.array([[0.0, 1.0], [1.0, 0.0]]))
        baseline = np.array([[0.2, 0.8], [-0.5, 1.5]])
        predicted = model.predict(baseline, np.array([1.0, 2.0]), beta=0.0, gamma=0.0)
        np.testing.assert_allclose(predicted, np.array([[0.2, 0.8], [0.0, 1.0]]))

    def test_beta_spreads_infection_to_connected_susceptible_region(self) -> None:
        model = GraphSIRModel(np.array([[0.0, 1.0], [1.0, 0.0]]), steps_per_year=24)
        baseline = np.array([[0.0, 0.7]])
        predicted = model.predict(baseline, np.array([1.0]), beta=1.0, gamma=0.0)
        self.assertGreater(float(predicted[0, 0]), baseline[0, 0])
        self.assertGreaterEqual(float(predicted.min()), 0.0)
        self.assertLessEqual(float(predicted.max()), 1.0)

    def test_gamma_reduces_infection_without_network_drive(self) -> None:
        model = GraphSIRModel(np.zeros((1, 1)), steps_per_year=24)
        baseline = np.array([[0.7]])
        predicted = model.predict(baseline, np.array([1.0]), beta=0.0, gamma=1.0)
        self.assertLess(float(predicted[0, 0]), baseline[0, 0])

    def test_normalize_compartments_clips_and_preserves_simplex(self) -> None:
        s, i, r = normalize_compartments(np.array([0.8]), np.array([0.8]), np.array([0.4]))
        self.assertLessEqual(float(s[0] + i[0] + r[0]), 1.0)
        self.assertGreaterEqual(float(s[0]), 0.0)
        self.assertGreaterEqual(float(i[0]), 0.0)
        self.assertGreaterEqual(float(r[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
