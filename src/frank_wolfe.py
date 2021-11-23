from copy import deepcopy
import numpy as np
import torch


def pseudo_frank_wolfe(
    obj_fn, obj_grad_fn, extreme_point_finder, step_length_selector, termination_condition, init_guess, num_iter
):
    # The constraints are assumed to be a gen. tetrahedron, aka a simplex.
    # Instead of properly solving the linear problem we can generate all extreme points
    # and check select the optimum value for the approximate linear problem.
    extr_points = extreme_point_finder()
    alpha_k = deepcopy(init_guess)
    alpha_store = deepcopy(alpha_k)
    gamma = 1.0
    iter_ = 0
    while not termination_condition(alpha_k, obj_fn, obj_grad_fn, gamma, iter_):
        iter_ += 1
        search_dirs = extr_points - alpha_k
        grad_f = obj_grad_fn(alpha_k)
        extr_points_lin_val = grad_f.T @ (search_dirs)
        min_idx = extr_points_lin_val.argmin()
        search_dir = extr_points[:, min_idx].reshape(alpha_k.size(0), 1) - alpha_k
        gamma = step_length_selector(search_dir, alpha_k)
        alpha_k += gamma * search_dir
        alpha_store = torch.column_stack((alpha_store, alpha_k))  # Broadcast magic requires transpose.
    return alpha_k, alpha_store


def term(alpha_k, _obj_fn, obj_grad_fn, _gamma, iter_):
    grad_f = obj_grad_fn(alpha_k)
    return iter_ > 5 or torch.allclose(grad_f, torch.zeros(grad_f.size(), dtype=grad_f.dtype))


def isosceles_triangular_extreme_points(grad_f, budget, num_samples):
    extreme_points = _gen_extreme_points(budget, num_samples)


def _gen_extreme_points(budget, num_samples):
    extreme_points = torch.zeros(num_samples, num_samples + 1)
    for i in np.arange(1, num_samples + 1):
        prot = torch.zeros(num_samples)
        prot[i - 1] = 1.0
        extreme_points[:, i] = prot
    return extreme_points
