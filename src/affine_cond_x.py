import torch
from functools import partial


def objective_template(precisions: torch.Tensor, xs: torch.Tensor, sigma_sq_k: float, sigma_sq_m: float):
    xs_elem_sq = xs ** 2
    num_samples = xs.size(0)
    term_1 = (precisions * xs_elem_sq).sum() * precisions.sum()
    print("term_1", term_1)
    term_2 = num_samples / sigma_sq_m * (precisions * xs_elem_sq).sum()
    print("term_2", term_2)
    term_3 = num_samples / sigma_sq_k * precisions.sum()
    print("term_3", term_3)
    term_4 = ((precisions * xs).sum()) ** 2
    print("term_4", term_4)
    return term_1 + term_2 + term_3 - term_4


def objective_gradient(precisions: torch.Tensor, xs: torch.Tensor, sigma_sq_k: float, sigma_sq_m: float):
    xs_elem_sq = xs ** 2
    num_samples = xs.size(0)
    term_1 = xs_elem_sq * precisions.sum() - xs_elem_sq * precisions
    print(term_1)
    term_2 = xs_elem_sq.T @ precisions - xs_elem_sq * precisions
    term_3 = num_samples / sigma_sq_m * xs_elem_sq
    term_4 = num_samples / sigma_sq_k
    return term_1 + term_2  # + term_3 + term_4


def obj_matrix_form(precisions: torch.Tensor, xs: torch.Tensor, sigma_sq_k: float, sigma_sq_m: float):
    num_samples = xs.size(0)
    xs_elem_sq = xs ** 2
    first_quadr = xs_elem_sq.repeat((1, num_samples))
    second_quadr = xs @ xs.T
    quadr_mat = first_quadr - second_quadr
    linear = num_samples / sigma_sq_m * xs_elem_sq + num_samples / sigma_sq_k * torch.ones(xs.size())
    return precisions.T @ quadr_mat @ precisions + linear.T @ precisions


def grad_matrix_form(precisions: torch.Tensor, xs: torch.Tensor, sigma_sq_k: float, sigma_sq_m: float):
    num_samples = xs.size(0)
    xs_elem_sq = xs ** 2
    first_quadr = xs_elem_sq.repeat((1, num_samples))
    second_quadr = xs @ xs.T
    linear = num_samples / sigma_sq_m * xs_elem_sq + num_samples / sigma_sq_k * torch.ones(xs.size())
    asym = first_quadr.T @ precisions + first_quadr @ precisions
    sym = second_quadr @ precisions
    return asym + sym  # + linear


def grad_fn(x):
    return torch.tensor([200 * x[0], 2 * x[1], 2 * (x[2] - 20)]).reshape(x.size(0), 1)


def extreme_points():
    return torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 20]]).T


def opt_step_len(alpha, search_dir, quadr_mat, linear):
    scale = search_dir.T @ quadr_mat @ search_dir
    return (linear - quadr_mat @ alpha).T @ search_dir / scale
