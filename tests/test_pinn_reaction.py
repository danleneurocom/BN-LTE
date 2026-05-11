from __future__ import annotations

import unittest

import numpy as np

from spread_toolbox.models.pinn_reaction import (
    GraphKPPReactionModel,
    fisher_reaction,
    kpp_auxiliary_loss,
    symbolic_reaction_from_expression,
)


class PINNReactionTests(unittest.TestCase):
    def test_fisher_reaction_satisfies_kpp_boundaries(self) -> None:
        reaction = fisher_reaction(hidden_layers=(2,))
        c = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        predicted = reaction.predict(c)
        np.testing.assert_allclose(predicted, c * (1.0 - c), atol=1.0e-12)
        self.assertEqual(float(predicted[0]), 0.0)
        self.assertEqual(float(predicted[-1]), 0.0)
        self.assertLessEqual(float(np.max(predicted)), 0.25)

    def test_symbolic_reaction_is_hard_clipped_to_kpp_range(self) -> None:
        reaction = symbolic_reaction_from_expression("2*c*(1-c)")
        predicted = reaction.predict(np.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(predicted, np.array([0.0, 0.25, 0.0]))

    def test_graph_kpp_reaction_model_integrates_reaction(self) -> None:
        model = GraphKPPReactionModel(
            np.zeros((1, 1)),
            u0=np.array([0.0]),
            cc=np.array([1.0]),
            reaction=fisher_reaction(hidden_layers=(2,)),
            laplacian_normalization="none",
            steps_per_year=12,
        )
        predicted = model.predict(np.array([[0.1]]), np.array([1.0]), rho=0.0, alpha=1.0)
        self.assertGreater(float(predicted[0, 0]), 0.1)
        self.assertLess(float(predicted[0, 0]), 1.0)

    def test_kpp_auxiliary_loss_is_small_for_fisher_reaction(self) -> None:
        loss = kpp_auxiliary_loss(fisher_reaction(hidden_layers=(2,)), alpha=1.0)
        self.assertLess(loss, 1.0e-12)


if __name__ == "__main__":
    unittest.main()
