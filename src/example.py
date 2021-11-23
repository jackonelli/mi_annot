import torch


def fn(x):
    return 100 * x[0] ** 2 + x[1] ** 2 + (x[2] - 20) ** 2


def grad_fn(x):
    return torch.tensor([200 * x[0], 2 * x[1], 2 * (x[2] - 20)]).reshape(x.size(0), 1)


def extreme_points():
    return torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 20]]).T


def opt_step_len(x, search_dir):
    p_1, p_2, p_3 = search_dir
    x_1, x_2, x_3 = x
    num = 200 * p_1 * x_1 + 2 * p_2 * x_2 + 2 * p_3 * x_3 - 40 * p_3
    den = -200 * p_1 ** 2 - 2 * p_2 ** 2 - 2 * p_3 ** 2
    return num / den


def appr_step_len(x, search_dir):
    p_1, p_2, p_3 = search_dir
    x_1, x_2, x_3 = x
    num = 200 * p_1 * x_1 + 2 * p_2 * x_2 + 2 * p_3 * x_3 - 40 * p_3
    den = -200 * p_1 ** 2 - 2 * p_2 ** 2 - 2 * p_3 ** 2
    return num / den
