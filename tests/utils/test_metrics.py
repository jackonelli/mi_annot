import unittest
from pathlib import Path
import torch
import numpy as np
from src.utils.metrics import Metrics, Metric, Accuracy, output_to_label, TopXAccuracy


class TestMetrics(unittest.TestCase):
    def test_unique_construct(self):
        try:
            Metrics([Accuracy("acc"), Accuracy("random")])
        except KeyError as err:
            self.fail(f"Failed, construting Metrics: {err}")

    def test_nonunique_construct(self):
        raised = False
        try:
            Metrics([Accuracy("acc"), Accuracy("acc")])
        except KeyError:
            raised = True
        self.assertTrue(raised, "Non-unique metric id's should raise an exception")

    def test_key_access(self):
        metrics = Metrics([Accuracy("accuracy"), Accuracy("random")])
        self.assertIsNotNone(metrics["accuracy"])
        self.assertTrue(isinstance(metrics["accuracy"], Accuracy))

    def test_multi_assign(self):
        metrics = Metrics([Accuracy("acc_1"), Accuracy("acc_2")])
        y_true = torch.Tensor([1, 0, 0, 1])
        y_hat = torch.Tensor([0.9, 0.9, 0.3, 0.75])
        metrics.add_sample(y_true, y_hat, None)
        for metric in metrics:
            mean, std = metric.mean_and_std()
            self.assertAlmostEqual(mean, 0.75)


class TestMetric(unittest.TestCase):
    def test_checkpoints(self):
        acc = Accuracy("acc")
        timesteps = np.array([12, 18, 67])
        for ts in range(len(timesteps)):
            for _ in range(5):
                y_true = torch.randint(0, 1, (4,))
                y_hat = torch.randn((4,))
                acc.add_sample(y_true, y_hat, None)
            acc.checkpoint()
        checkpoints = list(acc.list_checkpoints())
        self.assertEqual(len(checkpoints), 3)

        ch_with_ext_times = np.column_stack((timesteps, checkpoints))
        self.assertEqual(ch_with_ext_times.shape, (3, 4))
        # acc.save_checkpoints(Path.cwd(), "test_save_acc", timesteps)


class TestAccuracy(unittest.TestCase):
    def test_batch(self):
        acc = Accuracy("acc")
        y_true = torch.Tensor([1, 0, 0, 1])
        y_hat = torch.Tensor([0.9, 0.9, 0.3, 0.75])
        acc.add_sample(y_true, y_hat, None)
        mean, std = acc.mean_and_std()
        self.assertAlmostEqual(mean, 0.75)

    def test_output_to_label(self):
        prob_vec = torch.randn((100, 10))
        hard_preds = output_to_label(prob_vec)
        self.assertEqual(hard_preds.size(), torch.Size((100,)))


class TestTopXAccuracy(unittest.TestCase):
    def test_rank_bigger_than_num_classes_always_100_percent(self):
        top_x = 5
        acc = TopXAccuracy("top-5-acc", top_x)
        y_true = torch.Tensor([1, 0, 0, 1])
        y_hat = torch.randn((4, top_x))
        acc.add_sample(y_true, y_hat, None)
        mean, std = acc.mean_and_std()
        self.assertAlmostEqual(mean, 1.0)

    def test_mixed_preds(self):
        acc = TopXAccuracy("top-5-acc", 2)
        y_true = torch.Tensor([1, 2, 0, 2])
        y_hat = torch.Tensor([[0.8, 0.11, 0.09], [0.2, 0.7, 0.1], [0.3, 0.25, 0.4], [0.6, 0.3, 0.1]])
        acc.add_sample(y_true, y_hat, None)
        mean, std = acc.mean_and_std()
        self.assertAlmostEqual(mean, 0.5)


if __name__ == "__main__":
    unittest.main()
