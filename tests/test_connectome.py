from __future__ import annotations

import unittest

import numpy as np

from spread_toolbox.connectome import build_laplacian, clean_adjacency


class ConnectomeTests(unittest.TestCase):
    def test_clean_adjacency_symmetrizes_and_zeroes_diagonal(self) -> None:
        adjacency = np.array(
            [
                [5.0, 1.0, np.nan],
                [3.0, 2.0, 4.0],
                [6.0, 8.0, 9.0],
            ]
        )

        cleaned, report = clean_adjacency(
            adjacency,
            symmetrize=True,
            zero_diagonal=True,
            edge_weight_transform="none",
        )

        self.assertEqual(report["nan_count_before"], 1)
        self.assertAlmostEqual(float(np.max(np.abs(cleaned - cleaned.T))), 0.0)
        self.assertAlmostEqual(float(np.max(np.abs(np.diag(cleaned)))), 0.0)
        self.assertAlmostEqual(cleaned[0, 1], 2.0)
        self.assertAlmostEqual(cleaned[0, 2], 3.0)

    def test_build_combinatorial_laplacian(self) -> None:
        adjacency = np.array(
            [
                [0.0, 2.0, 1.0],
                [2.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        laplacian = build_laplacian(adjacency)

        expected = np.array(
            [
                [3.0, -2.0, -1.0],
                [-2.0, 2.0, 0.0],
                [-1.0, 0.0, 1.0],
            ]
        )
        np.testing.assert_allclose(laplacian, expected)
        np.testing.assert_allclose(laplacian.sum(axis=1), np.zeros(3))


if __name__ == "__main__":
    unittest.main()
