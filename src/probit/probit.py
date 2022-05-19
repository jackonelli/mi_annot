"""MI computation for probit model"""
import torch
from itertools import product
from scipy.stats import norm, bernoulli
import torch
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
import matplotlib.pyplot as plt


def p(z, alpha_i):
    """Likelihood p(Ỹ_i = ỹ_i | Theta = theta, X_i = x_i),
    with z = theta^T x_i - beta
    """
    return (2 * alpha_i - 1) * Normal(loc=0, scale=1).cdf(z) + 1 - alpha_i


def h(p):
    """Entropy of a Bernoulli variable"""
    # Set h = 0 in the limit p -> 0, and p -> 1
    entropy = torch.empty(p.shape)
    one_lim = torch.isclose(p, torch.ones(p.shape))
    zero_lim = torch.isclose(p, torch.zeros(p.shape))
    limit_inds = torch.logical_or(one_lim, zero_lim)
    entropy[limit_inds] = 0
    # For the rest, compute it the usual way.
    ok_inds = torch.logical_not(limit_inds)
    ok_p = p[ok_inds]
    entropy[ok_inds] = -ok_p * torch.log2(ok_p) - (1 - ok_p) * torch.log2(1 - ok_p)
    return entropy


def p_y_tilde_seq_given_params(y_tilde_seq, x_seq, params, alpha_seq):
    """Likelihood p(Ỹ_1:n = ỹ_1:n | Theta = theta, X_1:n = x_1:n)

    Args:
        y_tilde_seq (np.array): Seq. of Bernoulli variables (n,): {0, 1}^n
        x_seq (np.array): Seq. of np.arrays (n, D_x)
        theta (np.array): Vector (D_x,)
        alpha_seq (np.array): Seq of prec. values (n,): [0.5, 1]^n
    """
    theta, beta = params
    true_inds = y_tilde_seq == 1
    false_inds = y_tilde_seq == 0
    z = theta * (x_seq - beta)
    alpha_seq[true_inds]
    true_prod = p(z[true_inds], alpha_seq[true_inds])
    false_prod = 1 - p(z[false_inds], alpha_seq[false_inds])
    return true_prod.prod() * false_prod.prod()


def entropy_of_avg(x_seq, alpha_seq, param_sampling):
    """H[Ỹ_1:n | X_1:n]
    Compute the probability for every ỹ in {0,1}^n
    and evaluate the bernoulli entropy.
    """
    elements = binary_labels_combination(x_seq.size(0))
    entropy = torch.tensor(0.0)
    for y_tilde_seq in elements:
        thetas, betas = param_sampling()
        ps = torch.tensor(0.0)
        for params in zip(thetas, betas):
            ps += p_y_tilde_seq_given_params(y_tilde_seq, x_seq, params, alpha_seq)
            # print("ps", ps, params)
        prob = ps / thetas.size(0)
        # print(y_tilde_seq, prob)
        entropy -= prob * torch.log2(prob)
    return entropy / elements.size(0)


def avg_entropy(x_seq, alpha_seq, param_sampling):
    """E_p(theta) H[Ỹ_1:n | Theta, Beta, X_1:n]"""
    thetas, betas = param_sampling()

    for params in zip(thetas, betas):
        theta, beta = params
        zs = theta * (x_seq - beta)
        ps = p(zs, alpha_seq)
        entropies = h(ps)
    return entropies.mean()


def sample_joint_normal_uniform(mu_theta, sigma_sq_theta, a_B, b_B, num_mc_samples):
    """Sample from a joint (independent) Normal-Uniform distr"""
    theta = Normal(loc=mu_theta, scale=sigma_sq_theta).sample(sample_shape=(num_mc_samples,))
    beta = Uniform(a_B, b_B).sample(sample_shape=(num_mc_samples,))
    return (theta, beta)


def binary_labels_combination(num_labels):
    """Generate tensor with all combinations {0, 1}^n

    Args:
        num_labels (int): number of labels/samples, n

    Returns:
        combinations (torch.Tensor): tensor with shape (2^n, n)
    """
    return torch.tensor(list(product([0, 1], repeat=num_labels)))
