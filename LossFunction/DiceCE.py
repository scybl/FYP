import torch
import torch.nn as nn

from LossFunction.CrossEntropy import WCELoss
from LossFunction.diceLoss import DiceLoss


class DiceCE(nn.Module):
    """组合 Dice Loss 和 Weighted Cross Entropy (WCE) Loss。"""

    def __init__(self, n_classes, alpha=0.7, beta=0.3, weight=None):
        """
        初始化组合损失函数。

        参数:
        n_classes (int): 类别数量。
        alpha (float): Dice Loss 的权重。
        beta (float): WCE Loss 的权重。
        weight (list or torch.Tensor, optional): 类别权重（用于 WCE）。
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.dice_loss = DiceLoss(n_classes) # 假设你已定义 `DiceLoss`
        self.wce_loss = WCELoss(n_classes, weight=weight) # 假设你已定义 `WCELoss`

    def forward(self, inputs, target):
        """
        计算加权 Dice + WCE 组合损失。

        参数:
        inputs (torch.Tensor): 预测 logits，形状 (N, C, H, W)。
        target (torch.Tensor): 真实标签，形状 (N, H, W) 或 one-hot (N, C, H, W)。

        返回:
        torch.Tensor: 组合损失值（标量）。
        """
        dice = self.dice_loss(inputs, target)
        wce = self.wce_loss(inputs, target)
        return self.alpha * dice + self.beta * wce