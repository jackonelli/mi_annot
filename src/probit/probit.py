"""MI computation for probit model"""
import torch
from scipy.stats import norm

def likelihood_noisy_labels(theta, ys, xs, alphas):
    true_inds = ys == 1
    false_inds = ys == 0
    (2 * alphas[true_inds] - 1)*norm.cdf(theta * xs[true_inds])

def log_joint_prob_taylor_approx(theta, ys, xs, alphas):
    pass

def bald_mi(mean_fx, sigma_sq_fx):
    """The BALD (Houlsby et al., 2011) MI for the probit model is a special case,
    with a single data point x = 1 and alpha = 1.
    Then our parameter theta can be interpreted as their f_x, the Gaussian approx of the GP
    """
    return bald_entropy_pred_distr(mean_fx, sigma_sq_fx) - bald_avg_entropy(mean_fx, sigma_sq_fx)
