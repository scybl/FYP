import torch
import torch.nn as nn
import torch.nn.functional as F


class IoULoss(nn.Module):
    def __init__(self, n_classes=1, weight=None, smooth=1e-6, reduction='weighted_mean'):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth
        self.reduction = reduction

        if weight is not None:
            self.register_buffer('weight', torch.tensor(weight, dtype=torch.float32))
        else:
            self.weight = None

    def forward(self, inputs, target):
        # 自动处理二分类情况
        if inputs.shape[1] == 1 and self.n_classes == 1:
            inputs = torch.sigmoid(inputs)# 激活，所以不用外部激活
            inputs = torch.cat([1 - inputs, inputs], dim=1)  # 扩展为双通道
            self.n_classes = 2

        # 多分类处理
        if inputs.shape[1] != self.n_classes:
            inputs = torch.softmax(inputs, dim=1)# 激活，所以不用外部激活

        # 转换 target 为 one-hot
        target_onehot = F.one_hot(target.long(), self.n_classes).permute(0, 3, 1, 2).float()

        assert inputs.shape == target_onehot.shape, f"Shape mismatch: {inputs.shape} vs {target_onehot.shape}"

        # 计算 IoU（0，1，2，3），(N,C,H,W)
        dims = (2, 3)  # 假设输入为 2D 图像 (H,W)
        intersection = torch.sum(inputs * target_onehot, dim=dims)
        union = torch.sum(inputs, dim=dims) + torch.sum(target_onehot, dim=dims) - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)
        loss = 1 - iou

        # 加权策略
        if self.weight is not None:
            loss = loss * self.weight.view(1, -1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # weighted_mean
            if self.weight is None:
                return loss.mean()
            else:
                return loss.sum() / self.weight.sum()