import unittest

import torch

from Evaluate.evaluate import binary_accuracy, binary_precision, binary_recall, dice, miou


class MetricsTest(unittest.TestCase):
    def test_binary_metrics_are_one_for_perfect_prediction(self):
        target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        pred = torch.tensor([[[[0.9, 0.1], [0.2, 0.8]]]])

        self.assertAlmostEqual(float(dice(pred, target)), 1.0, places=5)
        self.assertAlmostEqual(float(miou(pred, target)), 1.0, places=5)
        self.assertAlmostEqual(float(binary_accuracy(pred, target)), 1.0, places=5)
        self.assertAlmostEqual(float(binary_precision(pred, target)), 1.0, places=5)
        self.assertAlmostEqual(float(binary_recall(pred, target)), 1.0, places=5)

    def test_binary_metrics_penalize_wrong_prediction(self):
        target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        pred = torch.tensor([[[[0.1, 0.9], [0.8, 0.2]]]])

        self.assertLess(float(dice(pred, target)), 0.1)
        self.assertLess(float(miou(pred, target)), 0.1)
        self.assertLess(float(binary_accuracy(pred, target)), 0.1)


if __name__ == "__main__":
    unittest.main()
