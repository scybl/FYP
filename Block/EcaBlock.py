import torch
from torch import nn
import math


class EcaBlock(nn.Module):
    """
    ECA（Efficient Channel Attention）Block 是对 SE（Squeeze-and-Excitation）Block 的一种改进，
    其主要目标是减少计算开销，同时保留或提升通道注意力机制的有效性。
    ECA Block 去除了 SE Block 中的**全连接层和显式的降维操作**，通过更轻量级的方式建模通道间的关系。
    1. 使用一维卷积代替全连接层，降低计算复杂度
    2. 提升模型的鲁棒性，减少过拟合风险
    """

    def __init__(self, channels, gamma=2, b=1):
        super(EcaBlock, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(1, 1, kernel_size=1, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        v = self.avg_pool(x)
        v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)  # 将全链接层改为卷积
        v = self.sigmoid(v)
        return x * v
