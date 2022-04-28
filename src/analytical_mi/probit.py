"""MI computation for probit model"""
import torch
from scipy.stats import norm

def likelihood_noisy_labels(theta, ys, xs, alphas):
    true_inds = ys == 1
    false_inds = ys == 0
    (2 * alphas[true_inds] - 1)*norm.cdf(theta * xs[true_inds])
def log_joint_prob_taylor_approx(theta, ys, xs, alphas):
    pass
