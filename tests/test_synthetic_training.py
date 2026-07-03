import tempfile
import unittest

import torch

from model.model_loader import load_model


class SyntheticTrainingTest(unittest.TestCase):
    def test_one_synthetic_training_step(self):
        torch.manual_seed(42)

        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "device": "cpu",
                "model": {"save_path": tmp, "pretrained_encoder": False},
                "datasets": {"isic2018": {"in_channel": 3, "class_num": 1}},
            }
            model = load_model(config, "bnet", "isic2018", load_weights=False)
            model.train()

            images = torch.randn(2, 3, 32, 32)
            masks = torch.zeros(2, 1, 32, 32)
            masks[:, :, 8:24, 8:24] = 1.0

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            loss_fn = torch.nn.BCEWithLogitsLoss()

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()

            grad_norm = 0.0
            for parameter in model.parameters():
                if parameter.grad is not None:
                    grad_norm += float(parameter.grad.abs().sum())

            optimizer.step()

            self.assertEqual(tuple(outputs.shape), (2, 1, 32, 32))
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(grad_norm, 0.0)


if __name__ == "__main__":
    unittest.main()
