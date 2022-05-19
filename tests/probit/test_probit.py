import unittest
from functools import partial
import torch
from src.probit.probit import (
    p_y_tilde_seq_given_params,
    binary_labels_combination,
    sample_joint_normal_uniform,
    avg_entropy,
)


class TestProbabilities(unittest.TestCase):
    def test_conditional_prob(self):
        xs = torch.tensor([-2, -1 / 2, 0, 1 / 2])
        y_tilde_seq = torch.zeros(xs.size())
        y_tilde_seq = torch.tensor([1, 0, 0, 0])

        # Perfect annotations
        alphas = torch.ones(xs.size())
        prob = p_y_tilde_seq_given_params(y_tilde_seq, xs, (1e6, 0.5), alphas).item()
        self.assertAlmostEqual(prob, 0.0)
        # All noise annotations
        alphas = 1 / 2 * torch.ones(xs.size())
        prob = p_y_tilde_seq_given_params(y_tilde_seq, xs, (1e6, 0.5), alphas).item()
        self.assertAlmostEqual(prob, (1 / 2) ** 4)

    def test_combinations(self):
        combs = binary_labels_combination(5)
        self.assertEqual(combs.size(0), 2 ** 5)

    def test_prob_combinations(self):
        xs = torch.tensor([-2, -1 / 2, 0, 1 / 2])
        alphas = 1 / 2 * torch.ones(xs.size())
        y_elements = binary_labels_combination(xs.size(0))
        probs = [p_y_tilde_seq_given_params(y_tilde_seq, xs, (1e6, 0.5), alphas).item() for y_tilde_seq in y_elements]
        self.assertAlmostEqual(sum(probs), 1.0)


class TestEntropies(unittest.TestCase):
    def test_avg_entropy(self):
        xs = torch.tensor([-2, -1 / 2, 0, 1 / 2])

        a_B = -1
        b_B = 1
        mu_theta = 1e6
        sigma_sq_theta = 0.1
        num_mc_samples = 10
        sampler = partial(sample_joint_normal_uniform, mu_theta, sigma_sq_theta, a_B, b_B, num_mc_samples)

        # Perfect annotations
        alphas = torch.ones(xs.size())
        avg_h = avg_entropy(xs, alphas, sampler)
        self.assertAlmostEqual(avg_h, 0.0)


if __name__ == "__main__":
    unittest.main()
