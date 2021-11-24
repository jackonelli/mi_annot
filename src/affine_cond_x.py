import torch


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
    # Gradient of the quadratic term is A + A^T = [ (x_1 - xs)^2  (x_2 - xs)**2 ... (x_n - xs)**2 ]
    grad_mat = (xs - xs.T) ** 2
    linear = num_samples / sigma_sq_m * xs_elem_sq + num_samples / sigma_sq_k * torch.ones(xs.size())
    return grad_mat @ precisions + linear


def objective_template(precisions: torch.Tensor, xs: torch.Tensor, sigma_sq_k: float, sigma_sq_m: float):
    """Reference implementation"""
    xs_elem_sq = xs ** 2
    num_samples = xs.size(0)
    term_1 = (precisions * xs_elem_sq).sum() * precisions.sum()
    term_2 = num_samples / sigma_sq_m * (precisions * xs_elem_sq).sum()
    term_3 = num_samples / sigma_sq_k * precisions.sum()
    term_4 = ((precisions * xs).sum()) ** 2
    return term_1 + term_2 + term_3 - term_4


def objective_gradient(precisions: torch.Tensor, xs: torch.Tensor, sigma_sq_k: float, sigma_sq_m: float):
    """Reference implementation"""
    xs_elem_sq = xs ** 2
    num_samples = xs.size(0)
    term_1 = xs_elem_sq * precisions.sum() - xs_elem_sq * precisions
    term_2 = xs_elem_sq.T @ precisions - xs_elem_sq * precisions
    term_3 = 2 * xs * (xs.T @ precisions - xs * precisions)
    term_4 = num_samples / sigma_sq_m * xs_elem_sq
    term_5 = num_samples / sigma_sq_k
    return term_1 + term_2 - term_3 + term_4 + term_5


def opt_step_len(alpha, search_dir, quadr_mat, linear):
    scale = search_dir.T @ quadr_mat @ search_dir
    return (linear - quadr_mat @ alpha).T @ search_dir / scale


# Grad check
# xs = torch.randn((2,1))
# precisions = torch.randn((2,1))
# sigma_sq_k, sigma_sq_m = 1, 1
# full = objective_gradient(precisions, xs, sigma_sq_k, sigma_sq_m)
# mat = grad_matrix_form(precisions, xs, sigma_sq_k, sigma_sq_m)

# Fn check
# precisions = torch.randn((2,1))
# #xs = torch.tensor([[2.0, 3]]).T
# #precisions = torch.tensor([[1.0, 2]]).T
# sigma_sq_k, sigma_sq_m = 1, 1
# full = objective_template(precisions, xs, sigma_sq_k, sigma_sq_m)
# mat = obj_matrix_form(precisions, xs, sigma_sq_k, sigma_sq_m)
