import torch
from torch import nn

from Block.ConvBlock import ConvBlock


class InceptionBlockV1(nn.Module):
    """
    this is inception V2 block,
    use two 3*3 conv core to replace the 5*5 core
    """

    def __init__(self, channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        """
        Inception Block with dimension reduction.
        Args:
            in_channels: Number of input channels
            out_1x1: Number of output channels for the 1x1 convolution branch
            red_3x3: Number of output channels for 1x1 reduction before 3x3 convolution
            out_3x3: Number of output channels for the 3x3 convolution branch
            red_5x5: Number of output channels for 1x1 reduction before 5x5 convolution
            out_5x5: Number of output channels for the 5x5 convolution branch
            out_pool: Number of output channels for the 1x1 convolution after max pooling
        """
        super(InceptionBlockV1, self).__init__()
        self.branch1 = nn.Sequential(
            ConvBlock(channels, ch5x5red, 1, 1, 0),
            ConvBlock(channels, ch5x5, 5, 1, 2),
        )

        self.branch2 = nn.Sequential(
            ConvBlock(channels, ch3x3red, 1, 1, 0),
            ConvBlock(channels, ch3x3, 3, 1, 1)
        )

        self.branch3 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            ConvBlock(channels, pool_proj, 1, 1, 0)
        )

        self.branch4 = nn.Sequential(
            ConvBlock(channels, ch1x1, 1, 1, 0)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat([out1, out2, out3, out4], 1)
