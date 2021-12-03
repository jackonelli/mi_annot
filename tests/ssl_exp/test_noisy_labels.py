import unittest
import torch
from src.ssl_exp.data_loader import symmetric_label_noise


class TestSymmetricLabelNoise(unittest.TestCase):
    def est_perfect_quality(self):
        annot_quality = 1.0  # Perfect quality
        num_classes = 3
        distr = symmetric_label_noise(0, annot_quality, num_classes)
        true_distr = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        self.assertTrue(torch.allclose(distr, true_distr))
        self.assertAlmostEqual(distr.sum(), 1.0)

    def test_max_entropy(self):
        annot_quality = 0.0  # Perfect quality
        num_classes = 1000
        distr = symmetric_label_noise(0, annot_quality, num_classes)
        true_distr = torch.ones((num_classes,), dtype=torch.float64) / num_classes
        self.assertAlmostEqual(distr.sum().item(), 1.0)
        self.assertTrue(torch.allclose(distr, true_distr))


if __name__ == "__main__":
    unittest.main()
