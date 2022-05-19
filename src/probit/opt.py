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
        self.xs = xs

    def forward(self, _x):
        alphas = alpha_mapping(self.opt_param)
        # print("alphas: ", alphas)
        L_1 = entropy_of_avg(self.xs, alphas, self._sampler)
        # print("L_1", L_1)
        L_2 = avg_entropy(self.xs, alphas, self._sampler)
        # print("L_2", L_2)
        return L_1 - L_2

    def alphas(self):
        return alpha_mapping(self.opt_param)


def linear_cost(alphas):
    return alphas.sum()


def alpha_mapping(zs):
    """Map unbounded proxy variable to [1/2, 1]"""
    return 1 / 2 + 1 / 2 * torch.sigmoid(zs)


def inverse_alpha_mapping(alphas):
    """Map alpha values in (1/2, 1) to unbounded proxy variables"""
    return -torch.log(1 / (2 * alphas - 1) - 1)
