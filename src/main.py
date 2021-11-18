"""Main entry point"""
from functools import partial
import numpy as np
import torch
import torch.nn as torch_nn
from frank_wolfe import pseudo_frank_wolfe


def main():
    budget = 4.0
    xs = torch.randn(5, 1)
    precisions = torch_nn.parameter.Parameter(torch.randn(xs.size()))
    obj = partial(objective_template, xs=xs, sigma_sq_k=1, sigma_sq_m=1)
    obj_grad = partial(objective_gradient, xs=xs, sigma_sq_k=1, sigma_sq_m=1)
    pseudo_frank_wolfe =


def gradient_descent(obj, initial_precisions, lr, num_iter):
    precisions = initial_precisions
    for i in np.arange(num_iter):
        val = obj(precisions)
        val.backward()
        with torch.no_grad():
            precisions -= lr * precisions.grad
            precisions.grad.zero_()
    return precisions


def objective_template(precisions: torch.Tensor, xs: torch.Tensor, sigma_sq_k: float, sigma_sq_m: float):
    xs_elem_sq = xs ** 2
    num_samples = xs.size(0)
    term_1 = (precisions * xs_elem_sq).sum() * precisions.sum()
    term_2 = num_samples / sigma_sq_m * (precisions * xs_elem_sq).sum()
    term_3 = num_samples / sigma_sq_k * precisions.sum()
    term_4 = ((precisions * xs).sum()) ** 2
    return term_1 + term_2 + term_3 - term_4


def objective_gradient(precisions: torch.Tensor, xs: torch.Tensor, sigma_sq_k: float, sigma_sq_m: float):
    xs_elem_sq = xs ** 2
    num_samples = xs.size(0)
    term_1 = xs_elem_sq * precisions.sum()
    term_2 = xs_elem_sq * precisions
    term_3 = num_samples / sigma_sq_m * xs_elem_sq
    term_4 = num_samples / sigma_sq_k
    return term_1 - term_2 + term_3 - term_4


if __name__ == "__main__":
    main()
