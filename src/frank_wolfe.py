import numpy as np
import torch


def pseudo_frank_wolfe(obj, obj_grad, extreme_point_finder, init_guess, num_iter):
    extr_points = _gen_extreme_points()
    x_k = init_guess
    for i in np.arange(num_iter):
        search_dirs = extr_points - x_k


def isosceles_triangular_extreme_points(grad_f, budget, num_samples):
    extreme_points = _gen_extreme_points(budget, num_samples)


def _gen_extreme_points(budget, num_samples):
    extreme_points = torch.zeros(num_samples, num_samples + 1)
    for i in np.arange(1, num_samples + 1):
        prot = torch.zeros(num_samples)
        prot[i - 1] = 1.0
        extreme_points[:, i] = prot
    return extreme_points
