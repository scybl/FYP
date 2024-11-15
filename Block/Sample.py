import torch
from torch import nn
from torch.nn import functional as F


class DownSample(nn.Module):
    """
    下采样（池化）,最大池化没有特征提取能力，会丢很多特征，所以只使用3*3卷积池化采样
    输出特征图大小 = (输入尺寸 - 卷积核尺寸 + 2* padding) / stride + 1
    """

    def __init__(self, input_channels):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, 2, 1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(input_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):
    """
    上采样模块: 包含最近邻插值上采样、卷积操作及特征拼接。
    也可以使用n*1*1缩到n纬

    适用于 U-Net 等模型中的解码器部分，用于逐步恢复图像分辨率。

    参数:
        channels (int): 输入特征图的通道数。
    """

    def __init__(self, channels):
        super(UpSample, self).__init__()
        # 定义一个卷积层，保持输入和输出通道数相同。
        # 这里使用 2x2 的卷积核，步长为 1，填充为 1，保证输出尺寸与输入尺寸一致。
        self.layer = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1, 1),  # 核为1，步长1，特种通道减半
        )

    def forward(self, x, feature_map):
        """
        前向传播函数:

        输入:
            x (Tensor): 需要上采样的特征图，来自网络的较低分辨率层。
            feature_map (Tensor): 需要拼接的特征图，来自网络编码路径中的高分辨率特征。

        返回:
            Tensor: 上采样后的特征图和编码器特征图的拼接结果。
        """

        # 使用最近邻插值法对输入特征图 x 进行上采样，放大 2 倍 (scale_factor=2)
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        # 对上采样后的特征图进行卷积处理，提取新特征

        out = self.layer(up)
        # 将经过上采样和卷积后的特征图 (out) 与来自编码器的特征图 (feature_map) 进行拼接
        # 拼接操作沿通道维度 (dim=1) 进行，拼接后的通道数为两者通道数的总和

        return torch.cat((out, feature_map), dim=1)
