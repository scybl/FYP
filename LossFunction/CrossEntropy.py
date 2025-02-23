import torch
import torch.nn as nn
import torch.nn.functional as F

class WCELoss(nn.Module):
    """加权交叉熵损失 (WCE, Weighted Cross Entropy)。

    该损失用于类别不平衡的多分类或二分类任务，
    通过给不同类别赋予权重，以减少数据不均衡的影响。

    属性:
    n_classes (int): 类别数量。
    weight (torch.Tensor or None): 类别权重张量，若为 None，则所有类别权重相等。
    """

    def __init__(self, n_classes: int, weight=None):
        """
        参数:
        n_classes (int): 类别数量。
        weight (list 或 torch.Tensor, 可选): 类别权重，用于调整不同类别对损失的影响。
        如果为 None，则不进行加权，默认值为 None。
        """
        super().__init__()
        self.n_classes = n_classes
        if weight is not None:
            self.register_buffer('weight', torch.tensor(weight, dtype=torch.float32))
        else:
            self.weight = None # 无权重时，默认权重为 1

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算加权交叉熵损失 (WCE)。

        参数:
        inputs (torch.Tensor): 预测的 logits 张量，形状为 (N, C, H, W)。
        其中 N 是 batch_size，C 是类别数，H 和 W 是输入图像的高度和宽度。
        logits 需要转换为概率（softmax）。
        target (torch.Tensor): 真实标签张量，形状为 (N, H, W) 或 (N, C, H, W)。
        - 若形状为 (N, H, W)，表示类别索引 (int)，需转换为 one-hot。
        - 若形状为 (N, C, H, W)，表示 one-hot 形式，可直接使用。

        返回:
        torch.Tensor: 计算得到的加权交叉熵损失（标量）。
        """

        # 处理输入，转换 logits 为概率分布
        inputs = torch.softmax(inputs, dim=1)

        # 处理 target：如果是类别索引 (N, H, W)，转换为 one-hot (N, C, H, W)
        if target.dim() == 3: # (N, H, W)
            target = F.one_hot(target.long(), num_classes=self.n_classes).permute(0, 3, 1, 2).float()

        # 确保输入 `inputs` 和 `target` 形状匹配
        assert inputs.shape == target.shape, f"形状不匹配: {inputs.shape} vs {target.shape}"

        # 计算交叉熵损失: - y * log(p)
        ce_loss = - target * torch.log(inputs + 1e-7) # 避免 log(0)

        # 如果提供了类别权重，则应用权重
        if self.weight is not None:
            ce_loss = ce_loss * self.weight.view(1, -1, 1, 1)

        # 计算均值损失
        return torch.mean(ce_loss)