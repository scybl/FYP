import os
import random

import numpy as np
from torchvision.utils import save_image
from LoadData.data import get_test_dataset
from LoadData.utils import load_config
import torch
import torch.nn as nn
from model_defination.model_loader import load_model


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
        self.config = load_config(config_path)
        set_seed(self.config["test_setting"]['seed'])
        self.device = torch.device(self.config['device'] if torch.cuda.is_available() else "cpu")
        self.model_path = os.path.join(self.config['model_path'], self.config['model_name'] + ".pth")
        self.data_loader = get_test_dataset(self.config)
        # 使用外部的 load_model_from_config 函数来加载模型
        self.net = load_model(self.config)
        self.loss_fn = nn.BCEWithLogitsLoss()

    @staticmethod
    def dice_coefficient(pred, target):
        """计算Dice系数"""
        smooth = 1e-5
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return dice.item()

    @staticmethod
    def pixel_accuracy(pred, target):
        """计算像素准确率"""
        correct = (pred == target).sum().item()
        total = pred.numel()
        return correct / total

    @staticmethod
    def iou_coefficient(pred, target):
        """计算IoU系数"""
        smooth = 1e-5
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou.item()

    def evaluate(self):
        """评估模型性能"""
        total_loss = 0
        total_dice = 0
        total_iou = 0
        total_pixel_acc = 0
        num_batches = 0

        # 设置模型为评估模式
        self.net.eval()

        with torch.no_grad():
            for i, (image, segment_image) in enumerate(self.data_loader):
                image, segment_image = image.to(self.device), segment_image.to(self.device)

                # 获取模型输出
                out_image = self.net(image)

                # 计算损失
                loss = self.loss_fn(out_image, segment_image)
                total_loss += loss.item()

                # 将模型输出转为二值化的掩码
                pred = torch.sigmoid(out_image) > 0.5
                pred = pred.float()

                # 计算各项评估指标
                dice = self.dice_coefficient(pred, segment_image)
                total_dice += dice

                iou = self.iou_coefficient(pred, segment_image)
                total_iou += iou

                pixel_acc = self.pixel_accuracy(pred, segment_image)
                total_pixel_acc += pixel_acc

                # 保存图像
                _image = image[0]
                _segment_image = segment_image[0]
                _out_image = pred[0]
                img = torch.stack([_image, _segment_image, _out_image], dim=0)
                save_image(img, f"{self.config['save_image_path']}/test_{i}.png")

                print(
                    f"Batch {i} --- Loss: {loss.item():.4f}, Dice: {dice:.4f}, IoU: {iou:.4f}, Pixel Acc: {pixel_acc:.4f}")
                num_batches += 1

        # 计算平均指标
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        avg_iou = total_iou / num_batches
        avg_pixel_acc = total_pixel_acc / num_batches

        # 返回评估结果
        _results = {
            "avg_loss": avg_loss,
            "avg_dice": avg_dice,
            "avg_iou": avg_iou,
            "avg_pixel_acc": avg_pixel_acc
        }

        print(f"Average Loss: {avg_loss:.4f}, ")
        print(f"Average Dice: {avg_dice:.4f}, ")
        print(f"Average IoU: {avg_iou:.4f}, ")
        print(f"Average Pixel Accuracy: {avg_pixel_acc:.4f}")
        # 计算并打印模型的参数量大小（单位：MB）
        total_params = sum(p.numel() for p in self.net.parameters())
        trainable_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)

        # 假设参数为 float32，每个参数占 4 字节
        total_params_mb = total_params * 4 / (1024 * 1024)
        trainable_params_mb = trainable_params * 4 / (1024 * 1024)

        print(f"Total Parameters: {total_params_mb:.2f} MB")
        print(f"Trainable Parameters: {trainable_params_mb:.2f} MB")


if __name__ == "__main__":
    CONFIG_NAME = "config.yaml"
    CONFIG_PATH = os.path.join("../configs/", CONFIG_NAME)
    # 创建评估类的实例，并运行评估
    evaluator = SegmentationEvaluator(CONFIG_PATH)
    evaluator.evaluate()
