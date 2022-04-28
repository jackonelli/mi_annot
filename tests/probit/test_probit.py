import unittest
import torch
from src.probit.probit import likelihood_noisy_labels


class TestSymmetricLabelNoise(unittest.TestCase):
    def test_likelihood_noisy_labels(self):
        ys = torch.tensor([1, 0, 1])
        xs = torch.zeros(ys.shape)
        theta = 0.0


if __name__ == "__main__":
    unittest.main()

