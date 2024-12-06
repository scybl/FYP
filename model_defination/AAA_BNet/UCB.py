import torch
import torch.nn as nn


class UCB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UCB, self).__init__()

        # 上采样（2倍，双线性插值）
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 深度可分离卷积 (Depthwise Separable Convolution, DWC)
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # Batch Normalization 和 ReLU
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 执行上采样
        x = self.upsample(x)

        # 深度可分离卷积
        x = self.dw_conv(x)
        x = self.pointwise_conv(x)

        # BN 和 ReLU
        x = self.bn(x)
        x = self.relu(x)

        return x
