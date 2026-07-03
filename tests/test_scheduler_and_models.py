import importlib.util
import tempfile
import unittest

import torch

from Evaluate.LearningRate import PolyWarmupScheduler
from model.model_loader import load_model


class SchedulerAndModelTest(unittest.TestCase):
    def test_poly_warmup_scheduler_changes_lr_over_time(self):
        model = torch.nn.Conv2d(1, 1, kernel_size=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        scheduler = PolyWarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=2,
            total_epochs=5,
            initial_lr=1.0,
            power=1.0,
            eta_min=0.1,
        )

        lrs = []
        for _ in range(6):
            scheduler.step()
            lrs.append(round(scheduler.get_lr(), 4))

        self.assertEqual(lrs[0], 0.1)
        self.assertEqual(lrs[1], 0.55)
        self.assertEqual(lrs[2], 1.0)
        self.assertLess(lrs[3], lrs[2])
        self.assertEqual(lrs[-1], 0.1)

    def test_model_loader_can_build_core_models_without_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "device": "cpu",
                "model": {"save_path": tmp, "pretrained_encoder": False},
                "datasets": {"isic2018": {"in_channel": 3, "class_num": 1}},
            }

            for model_name in ["unet", "unetpp", "bnet", "bnet34", "duck", "pham", "dga"]:
                with self.subTest(model_name=model_name):
                    model = load_model(config, model_name, "isic2018", load_weights=False)
                    model.eval()
                    with torch.no_grad():
                        output = model(torch.randn(1, 3, 64, 64))

                    self.assertEqual(tuple(output.shape), (1, 1, 64, 64))

    def test_unext_is_optional(self):
        if importlib.util.find_spec("timm") is None:
            with tempfile.TemporaryDirectory() as tmp:
                config = {
                    "device": "cpu",
                    "model": {"save_path": tmp, "pretrained_encoder": False},
                    "datasets": {"isic2018": {"in_channel": 3, "class_num": 1}},
                }
                with self.assertRaises(ModuleNotFoundError):
                    load_model(config, "unext", "isic2018", load_weights=False)
            return

        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "device": "cpu",
                "model": {"save_path": tmp, "pretrained_encoder": False},
                "datasets": {"isic2018": {"in_channel": 3, "class_num": 1}},
            }

            model = load_model(config, "unext", "isic2018", load_weights=False)
            model.eval()
            with torch.no_grad():
                output = model(torch.randn(1, 3, 64, 64))

            self.assertEqual(tuple(output.shape), (1, 1, 64, 64))


if __name__ == "__main__":
    unittest.main()
