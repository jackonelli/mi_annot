import unittest
import torch
from src.probit.opt import alpha_mapping, inverse_alpha_mapping


class TestOpt(unittest.TestCase):
    def test_alpha_mapping(self):
        alpha = torch.distributions.Uniform(1 / 2, 1).sample((10,))
        mapped_alpha = alpha_mapping(inverse_alpha_mapping(alpha))
        a = torch.allclose(alpha, mapped_alpha)
        print(a)
        self.assertTrue(torch.allclose(alpha, mapped_alpha))


if __name__ == "__main__":
    unittest.main()
