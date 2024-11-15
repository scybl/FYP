import random
import numpy as np
from torch import nn
from LoadData.data import get_test_dataset
from LoadData.utils import load_config
from model_defination.model_loader import load_model_test
import os
import csv
import torch


def set_seed(seed):
    """设置随机种子以确保结果一致性"""
    random.seed(seed)  # 设置 Python 原生的随机数生成器的种子
    np.random.seed(seed)  # 设置 NumPy 随机数生成器的种子
    torch.manual_seed(seed)  # 设置 PyTorch 随机数生成器的种子（CPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 设置当前 GPU 设备的随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有 GPU 设备设置随机种子
        torch.backends.cudnn.deterministic = True  # 确保 CUDA 使用确定性算法
        torch.backends.cudnn.benchmark = False  # 关闭优化以确保一致性


class SegmentationEvaluator:
    def __init__(self, config_path):
        # 加载配置文件
        self.net = None
        self.config = load_config(config_path)
        set_seed(self.config["test_setting"]['seed'])
        self.device = torch.device(self.config['device'] if torch.cuda.is_available() else "cpu")
        self.model_path = self.config['model_path']
        self.model_name = self.config['model_name']
        self.data_loader = get_test_dataset(self.config)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def load_model(self, model_path):
        """加载模型权重"""
        model = load_model_test(self.config)
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        return model.to(self.device)

    @staticmethod
    def dice_coefficient(pred, target):
        intersection = (pred * target).sum()
        return (2. * intersection) / (pred.sum() + target.sum() + 1e-6)

    @staticmethod
    def iou_coefficient(pred, target):
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return intersection / (union + 1e-6)

    @staticmethod
    def pixel_accuracy(pred, target):
        correct = (pred == target).sum()
        return correct / target.numel()

    def evaluate(self):
        """评估指定路径中的所有模型"""
        results = []

        # 获取路径中所有符合命名规则的.pth文件
        model_files = [f for f in os.listdir(self.model_path) if f.startswith(self.model_name) and f.endswith(".pth")]

        if not model_files:
            print(f"No model files found for {self.model_name} in {self.model_path}")
            return

        best_model = None
        best_dice = -1  # 用于记录最佳模型的 Dice 指标
        print(model_files)

        for model_file in model_files:
            model_path = os.path.join(self.model_path, model_file)
            print(f"Evaluating model: {model_file}")

            # 加载模型
            self.net = self.load_model(model_path)
            self.net.eval()

            total_loss = 0
            total_dice = 0
            total_iou = 0
            total_pixel_acc = 0
            num_batches = 0

            with torch.no_grad():
                for i, (image, segment_image) in enumerate(self.data_loader):
                    image, segment_image = image.to(self.device), segment_image.to(self.device)

                    out_image = self.net(image)

                    loss = self.loss_fn(out_image, segment_image)
                    total_loss += loss.item()

                    pred = torch.sigmoid(out_image) > 0.5
                    pred = pred.float()

                    dice = self.dice_coefficient(pred, segment_image)
                    total_dice += dice

                    iou = self.iou_coefficient(pred, segment_image)
                    total_iou += iou

                    pixel_acc = self.pixel_accuracy(pred, segment_image)
                    total_pixel_acc += pixel_acc

                    num_batches += 1

            avg_loss = total_loss / num_batches
            avg_dice = total_dice / num_batches
            avg_iou = total_iou / num_batches
            avg_pixel_acc = total_pixel_acc / num_batches

            # 保存评估结果
            results.append({
                "model_file": model_file,
                "avg_loss": avg_loss,
                "avg_dice": avg_dice,
                "avg_iou": avg_iou,
                "avg_pixel_acc": avg_pixel_acc
            })

            # 更新最佳模型
            if avg_dice > best_dice:
                best_dice = avg_dice
                best_model = model_file

        # 保存结果到CSV
        csv_path = os.path.join(self.config['save_image_path'], "evaluation_results.csv")
        with open(csv_path, mode="w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file,
                                    fieldnames=["model_file", "avg_loss", "avg_dice", "avg_iou", "avg_pixel_acc"])
            writer.writeheader()
            writer.writerows(results)

        print(f"Evaluation results saved to {csv_path}")

        # 将最佳模型重命名为 model_best.pth
        if best_model:
            best_model_path = os.path.join(self.model_path, best_model)
            best_model_dest = os.path.join(self.model_path, f"{self.model_name}_best.pth")
            os.rename(best_model_path, best_model_dest)
            print(f"Best model '{best_model}' renamed to '{self.model_name}_best.pth'")


if __name__ == "__main__":
    CONFIG_NAME = "config.yaml"
    CONFIG_PATH = os.path.join("../configs/", CONFIG_NAME)
    # 创建评估类的实例，并运行评估
    evaluator = SegmentationEvaluator(CONFIG_PATH)
    evaluator.evaluate()
