import torch
from torch import nn
from torch.nn import functional as F


class DownSample(nn.Module):
    """
    Downsampling (pooling), Max pooling has no feature extraction ability and will lose a lot of features,
    so only 3*3 convolutional pooling sampling is used
    Output feature map size = (input size - kernel size + 2* padding)/stride + 1
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
    Upsampling module: includes nearest neighbor interpolation upsampling,
    convolution operation and feature concatenation.
    You can also use n* 1* 1 to shrink to n dimensions

    It is suitable for the decoder part in U-Net and other models to gradually restore the image resolution.

    Parameters:
    channels (int): The number of channels to input the feature map.
    """

    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1, 1),  # 核为1，步长1，特种通道减半
        )

    def forward(self, x, feature_map):
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        out = self.layer(up)
        return torch.cat((out, feature_map), dim=1)
