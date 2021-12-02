import unittest
import numpy as np
from src.utils.stop_crit import _design_matrix, _det_gram, _cofactor_matrix


class TestSlopeEst(unittest.TestCase):
    def test_design_matrix(self):
        X = _design_matrix(2)
        true_X = np.array([[1, 1], [1, 2]])
        self.assertTrue(np.allclose(X, true_X))

    def test_gram_matrix(self):
        n = 8
        N = _gram_matrix(n)
        X = _design_matrix(n)
        self.assertTrue(np.allclose(N, X.T @ X))

    def test_gram_det(self):
        n = 13
        N = _gram_matrix(n)
        det_N = _det_gram(n)
        a, b, c, d = N[0, 0], N[0, 1], N[1, 0], N[1, 1]
        true_det = a * d - b * c
        self.assertEqual(det_N, true_det)

    def test_cofactor_matrix(self):
        n = 175
        Q = _cofactor_matrix(n)
        X = _design_matrix(n)
        self.assertTrue(np.allclose(Q, np.linalg.inv(X.T @ X)))


def _gram_matrix(n: int):
    sum_i = n * (n + 1) // 2
    return np.array([[n, sum_i], [sum_i, (n * (n + 1) * (2 * n + 1)) // 6]])


if __name__ == "__main__":
    unittest.main()
