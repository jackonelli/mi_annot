import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

class OneDimGaussian:

    def __init__(self, mean, sigma_sq):
        self.mean = mean
        self.sigma_sq = sigma_sq

    def sample(self, num_samples):
        return np.random.normal(self.mean, np.sqrt(self.sigma_sq), num_samples)

class MultivarGaussian:

    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov


def mi_gaussian_gaussian(params_1, params_2):
    pass


def onedim_gaussian_pdf(x, params):
    mu, sigma_sq = params
    diff_sq = (x - mu)**2
    factor = 1 / ( np.sqrt(2 * np.pi * sigma_sq ))
    return factor * np.exp(-diff_sq / (2 * sigma_sq))

def plot_sigma_level(ax, mean, cov, level, label, color, resolution=50):
    fmt = "--"
    ellips = ellipse_points(mean, cov, level, resolution)
    if ax is None:
        _, ax = plt.subplots()
    (handle,) = ax.plot(ellips[:, 0], ellips[:, 1], fmt)
    handle.set_color(color)
    # ax.axis("equal")
    return handle


def ellipse_points(center, transf, scale, resolution):
    """Generate points along an ellipse"""
    angles = np.linspace(0, 2 * np.pi, resolution)
    curve_parameter = np.row_stack((np.cos(angles), np.sin(angles)))

    print(transf.shape)
    level_sigma_offsets = scale * sqrtm(transf) @ curve_parameter

    return center + level_sigma_offsets.T
