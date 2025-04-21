import os
import torch
import numpy as np
import random
from torchvision.utils import save_image

from Evaluate.evaluate import dice, miou, binary_accuracy, binary_recall, binary_precision, binary_jaccard_index
from LoadData.data import get_dataset
from LoadData.utils import load_config
from Evaluate.LossChoose import LossFunctionHub
from model.model_loader import load_model


# import warnings
# warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")

class Tester:

    def __init__(self, config_path, _model_name, _dataset_name):
        # config setting
        self.config = load_config(config_path)
        self.device = torch.device(self.config['device'] if torch.cuda.is_available() else "cpu")
        self.class_num = self.config["datasets"][_dataset_name]["class_num"]
        self.model_name = _model_name
        self.dataset_name = _dataset_name

        # load the test dataset, assuming get_dataset supports the "test" mode
        self.test_dataset = get_dataset(self.config, self.dataset_name, 'test')

        # Load the model and move it to the device.
        # Note that special settings may be required during testing, such as adjusting the batch_size.
        self.net = load_model(self.config, _model_name, _dataset_name).to(self.device)

        self.save_image_path = self.config['save_image_path']

    def test(self) -> float:
        self.net.eval()
        num_batches = 0

        dice_scores = []
        miou_scores = []
        accuracy_scores = []
        recall_scores = []
        precision_scores = []

        with torch.no_grad():
            for i, (image, segment_image) in enumerate(self.test_dataset):
                image, segment_image = image.to(self.device), segment_image.to(self.device)
                out_image = self.net(image)

                dice_scores.append(dice(pred=out_image, target=segment_image))
                miou_scores.append(miou(pred=out_image, target=segment_image))
                accuracy_scores.append(binary_accuracy(pred=out_image, target=segment_image))
                recall_scores.append(binary_recall(pred=out_image, target=segment_image))
                precision_scores.append(binary_precision(pred=out_image, target=segment_image))
                # jaccard_scores.append(binary_jaccard_index(pred=out_image,target=segment_image))

                num_batches += 1

            ####################################################################################
            # Save test result images for visualization
            # _image = image[0]
            # _segment_image = segment_image[0]
            # _out_image = out_image[0]

            # If the label and output are single-channel, repeat to 3 channels for visualization
            # _segment_image = _segment_image.repeat(3, 1, 1)
            # _out_image = _out_image.repeat(3, 1, 1)

            # Stack the original image, label image, and predicted image
            # img = torch.stack([_image, _segment_image, _out_image], dim=0)
            # save_path = os.path.join(self.save_image_path,
            # f"{self.model_name}_{self.dataset_name}_{i}.png")
            # save_image(img, save_path)
            ####################################################################################

        _dice = sum(dice_scores) / num_batches
        _miou = sum(miou_scores) / num_batches
        _accuracy = sum(accuracy_scores) / num_batches
        _precision = sum(precision_scores) / num_batches
        _recall = sum(recall_scores) / num_batches

        return _dice, _miou, _accuracy, _precision, _recall


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    model_hub = ["bnet34"]
    dataset_hub = ['isic2018', 'clinicdb', 'kvasir']

    test_config_path = 'configs/config.yaml'

    repeat_times = 1
    seeds = [random.randint(1, 10000) for _ in range(repeat_times)]
    print(seeds)
    for model_name in model_hub:
        for dataset_name in dataset_hub:

            print("------------------------------------------")
            print(model_name + ' || ' + dataset_name)
            tester = Tester(test_config_path, _model_name=model_name, _dataset_name=dataset_name)
            dice_avg, miou_avg, acc_avg, prec_avg, recall_avg = 0, 0, 0, 0, 0

            for run in range(repeat_times):
                seed = seeds[run]
                set_random_seed(seed)
                print(f"Running {model_name} on {dataset_name}, Seed: {seed}")
                a, b, c, d, e = tester.test()

                dice_avg += a
                miou_avg += b
                acc_avg += c
                prec_avg += d
                recall_avg += e

            print(f"Average Results for {model_name} on {dataset_name}:")
            print(f"Dice Avg: {dice_avg / repeat_times:.6f}")
            print(f"Miou Avg: {miou_avg / repeat_times:.6f}")
            print(f"Accuracy Avg: {acc_avg / repeat_times:.6f}")
            print(f"Precision Avg: {prec_avg / repeat_times:.6f}")
            print(f"Recall Avg: {recall_avg / repeat_times:.6f}")
