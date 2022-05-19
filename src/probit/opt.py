import torch
from torch import nn
from src.probit.probit import entropy_of_avg, avg_entropy


class MiOpt(nn.Module):
    """Custom Pytorch model for gradient optimization."""

    def __init__(self, xs, starting_alphas, cost_fn, budget, sampling_fn):
        super().__init__()
        opt_param = inverse_alpha_mapping(starting_alphas)
        self.opt_param = nn.Parameter(opt_param)
        self._sampler = sampling_fn
        self.cost_fn = cost_fn
        self.budget = budget
        self.xs = xs

    def compute_mi(self):
        alphas = alpha_mapping(self.opt_param)
        # print("alphas: ", alphas)
        L_1 = entropy_of_avg(self.xs, alphas, self._sampler)
        # print("L_1", L_1)
        L_2 = avg_entropy(self.xs, alphas, self._sampler)
        # print("L_2", L_2)
        # print(f"L_1: {L_1.item()}, L_2: {L_2.item()}, constr: {constraint}")

        return L_1 - L_2

    def constraint(self):
        alphas = alpha_mapping(self.opt_param)
        return interior_penalty(self.cost_fn(alphas), self.budget)

    def alphas(self):
        return alpha_mapping(self.opt_param)


def linear_cost(alphas):
    return alphas.sum()


def interior_penalty(cost, budget):
    """Interior penalty barrier

    Emulates constrained opt. in an unconstrained framework
    by adding a continuous and differentiable proxy for the function
    Chi(alphas) = 0, if cost < budget
                  inf, else.
    """
    nu = 1e-2
    chi_hat_S = 1 / (budget - cost)
    return nu * chi_hat_S


def alpha_mapping(zs):
    """Map unbounded proxy variable to [1/2, 1]"""
    return 1 / 2 + 1 / 2 * torch.sigmoid(zs)


def inverse_alpha_mapping(alphas):
    """Map alpha values in (1/2, 1) to unbounded proxy variables"""
    return -torch.log(1 / (2 * alphas - 1) - 1)
