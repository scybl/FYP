import torch
import torch.nn as nn


class ECAB(nn.Module):
    def __init__(self, channels, reduction_ratio=4, dropout_rate=0.3):
        super(ECAB, self).__init__()
        reduced_channels = channels // reduction_ratio

        # AMP 路径（全局平均池化）
        self.AMP = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 (C, H, W) -> (C, 1, 1)
            nn.Conv2d(channels, reduced_channels, 1),  # 1x1 卷积: (C, 1, 1) -> (C/r, 1, 1)
            nn.ReLU(inplace=True),  # ReLU 激活函数
            nn.Dropout(p=dropout_rate),  # Dropout 层
            nn.Conv2d(reduced_channels, channels, 1)  # 1x1 卷积: (C/r, 1, 1) -> (C, 1, 1)
        )

        # APP 路径（全局最大池化）
        self.APP = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),  # 全局最大池化 (C, H, W) -> (C, 1, 1)
            nn.Conv2d(channels, reduced_channels, 1),  # 1x1 卷积: (C, 1, 1) -> (C/r, 1, 1)
            nn.ReLU(inplace=True),  # ReLU 激活函数
            nn.Dropout(p=dropout_rate),  # Dropout 层
            nn.Conv2d(reduced_channels, channels, 1)  # 1x1 卷积: (C/r, 1, 1) -> (C, 1, 1)
        )

        # 空间 H * W 卷积路径
        self.layer3 = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1),  # 1x1 卷积: (C, H, W) -> (C/r, H, W)
            nn.ReLU(inplace=True),  # ReLU 激活函数
            nn.Dropout(p=dropout_rate),  # Dropout 层
            nn.Conv2d(reduced_channels, channels, 1)  # 1x1 卷积: (C/r, H, W) -> (C, H, W)
        )

        # **新增：可学习参数**
        self.weights = nn.Parameter(torch.ones(3))  # 初始化三个路径的权重

        # Sigmoid 激活函数用于计算注意力缩放因子
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        amp_out = self.AMP(x)  # 计算 AMP 注意力
        app_out = self.APP(x)  # 计算 APP 注意力
        hw_out = self.layer3(x)  # 计算空间 H * W 卷积注意力

        # **使用 softmax 归一化可学习参数**
        weight = torch.softmax(self.weights, dim=0)  # 归一化权重，确保总和为 1

        # **动态调整路径比重**
        combined = weight[0] * amp_out + weight[1] * app_out + weight[2] * hw_out

        # 通过 Sigmoid 计算缩放因子
        scale = self.sigmoid(combined)

        # 通过注意力映射对输入进行缩放
        out = x * scale
        return out
