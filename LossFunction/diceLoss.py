import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """Dice 损失函数，用于二分类和多分类的语义分割任务。

    Dice Loss 主要用于评估预测结果与真实目标区域的重叠程度。该损失函数
    适用于二分类（单通道）和多分类（多通道）场景，并支持类别权重调整。

    属性:
        n_classes (int): 任务中的类别数量。
        weight (torch.Tensor or None): 类别权重张量，若为 None，则不进行加权。
        eps (float): 一个小常数，用于避免除零错误，提高数值稳定性。
    """

    def __init__(self, n_classes: int, weight=None, eps: float = 1e-7):
        """初始化 DiceLoss 模块。

        参数:
            n_classes (int): 类别数，用于确定 one-hot 编码的通道数。
            weight (list 或 torch.Tensor, 可选): 类别权重，用于调整不同类别对损失的影响。
            如果为 None，则不进行加权，默认值为 None。
            eps (float, 可选): 一个小数值，用于防止除零错误，默认值为 1e-7。
        """
        super().__init__()
        self.n_classes = n_classes
        self.register_buffer('weight', torch.tensor(weight) if weight else None)
        self.eps = eps

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算 Dice 损失。

        参数:
            inputs (torch.Tensor): 预测的 logits 张量，形状为 (N, C, H, W)，
            其中 N 是 batch_size，C 是类别数，H 和 W 是输入图像的高度和宽度。
            logits 需要转换为概率（softmax 或 sigmoid）。
            target (torch.Tensor): 真实标签张量，形状为 (N, C, H, W)，
            其中 C 是类别的 one-hot 编码表示，确保与 `inputs` 形状匹配。

        返回:
            torch.Tensor: Dice 损失值（标量）。
        """

        # 处理二分类任务（C=1），需要将 logits 变成两通道概率
        if inputs.shape[1] == 1:
            # 使用 sigmoid 激活并转换为二分类形式 (N, 2, H, W)
            inputs = torch.sigmoid(inputs)
            inputs = torch.cat([1 - inputs, inputs], dim=1) # 第一通道为 1 - p，第二通道为 p
        else:
        # 多分类任务，使用 softmax 归一化
            inputs = torch.softmax(inputs, dim=1)

        # 确保输入 `inputs` 和 `target` 形状匹配
        assert inputs.shape == target.shape, f"形状不匹配: {inputs.shape} vs {target.shape}"

        # 计算每个类别的 Dice 系数
        dims = (2, 3) # 在 H 和 W 维度上求和
        intersection = torch.sum(inputs * target, dim=dims) # 计算交集（true positives）
        union = torch.sum(inputs, dim=dims) + torch.sum(target, dim=dims) # 计算并集（预测 + 目标）

        # 计算每个类别的 Dice 得分
        dice_scores = (2. * intersection + self.eps) / (union + self.eps)

        # 如果提供了类别权重，则进行加权计算
        if self.weight is not None:
            dice_scores = dice_scores * self.weight.view(1, -1)

        # 返回 Dice 损失，取均值后进行反向优化（1 - Dice）
        return 1 - torch.mean(dice_scores)