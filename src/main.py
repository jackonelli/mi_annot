"""Main entry point"""
from functools import partial
import torch
import torch.nn as torch_nn
from frank_wolfe import pseudo_frank_wolfe


def main():
    budget = 4.0
    xs = torch.randn(5, 1)
    precisions = torch_nn.parameter.Parameter(torch.randn(xs.size()))
    obj = partial(objective_template, xs=xs, sigma_sq_k=1, sigma_sq_m=1)
    obj_grad = partial(objective_gradient, xs=xs, sigma_sq_k=1, sigma_sq_m=1)
    pseudo_frank_wolfe(example_f, example_grad, example_extreme_points, torch.tensor([1, 0, 0]), 5)


if __name__ == "__main__":
    main()
