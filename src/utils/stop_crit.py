import numpy as np
from scipy.stats import norm


def prob_of_positive_slope(ys):
    n = ys.shape[0]
    standard_norm_cdf = norm().cdf
    k_hat = slope(ys)
    s_sq_k_hat = (12 * _s_sq_eps(ys)) / (n ** 3 - n)
    return 1 - standard_norm_cdf(-k_hat / np.sqrt(s_sq_k_hat))


def _design_matrix(n: int):
    ones_column = np.ones((n, 1))
    i_column = np.arange(1, n + 1)
    return np.column_stack((ones_column, i_column))


def _det_gram(n: int):
    return (n ** 4 - n ** 2) // 12


def _cofactor_matrix(n: int):
    a = n * (n + 1) * (2 * n + 1) // 6
    b = -(n * (n + 1)) // 2
    c = b
    d = n
    return np.array([[a, b], [c, d]]) / _det_gram(n)


def _s_sq_eps(ys):
    n = ys.shape[0]
    X = _design_matrix(n)
    beta = estimate_line(ys)
    y_hats = X @ beta
    sum_sq_residuals = np.sum((ys - y_hats) ** 2)
    return sum_sq_residuals / (n - 2)


def slope(ys):
    beta = estimate_line(ys)
    return beta[1]


def estimate_line(ys):
    n = ys.shape[0]
    Q = _cofactor_matrix(n)
    X = _design_matrix(n)
    return Q @ X.T @ ys
